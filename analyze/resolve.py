"""Layer 1 — feature ↔ config mapping.

Pure functions over DataFrames (no CubeStore, no SQL). Given a
``configs_df`` and ``features_df`` (from ``analyze.data``), answer:

  * which config is the "add-one" version of feature F over base?
    (``simple_effect_configs``)
  * which configs contain feature F? (``configs_containing_feature``)
  * is feature F a "base feature" — i.e. all its primitives already
    live in the base config? (``is_base_feature``)
"""
from __future__ import annotations

from typing import Dict, Set


def base_func_ids(configs_df_in, base_config_id) -> frozenset:
    """Return the func_ids set for the given base_config_id.

    The lookup is dtype-tolerant: ``base_config_id`` is coerced to int
    for comparison, and the ``config_id`` column is coerced to int on
    the way in. This defends against cubes where the column was
    round-tripped as float/str and a naive ``== int`` comparison
    silently misses.

    Raises ``ValueError`` with the list of available config_ids (capped
    at 20) plus the smallest-func_ids row as the likely base when the
    lookup genuinely misses.
    """
    if base_config_id is None:
        raise ValueError(
            "base_config_id is None. method='simple' needs an explicit "
            "base config. Call Pipeline.scope(base_config_id=<int>)."
        )

    # Dtype-tolerant comparison. Cast both sides to int; fall back to
    # raw equality if coercion fails on an exotic column type.
    try:
        target = int(base_config_id)
        col_int = configs_df_in["config_id"].astype(int)
        mask = col_int == target
    except (TypeError, ValueError):
        mask = configs_df_in["config_id"] == base_config_id

    row = configs_df_in[mask]
    if row.empty:
        available = sorted(configs_df_in["config_id"].tolist())
        shown = available[:20]
        more = f" (+{len(available) - 20} more)" if len(available) > 20 else ""
        likely = None
        if not configs_df_in.empty:
            sizes = configs_df_in["func_ids"].map(len)
            likely = int(configs_df_in.loc[sizes.idxmin(), "config_id"])
        hint = f" Smallest-func_ids config is {likely}." if likely is not None else ""
        raise ValueError(
            f"base_config_id {base_config_id!r} not found in cube. "
            f"Available config_ids: {shown}{more}.{hint}"
        )
    return row.iloc[0]["func_ids"]


def simple_effect_configs(
    configs_df_in,
    features_df_in,
    base_config_id: int,
) -> Dict[str, int]:
    """Map ``canonical_id → config_id`` for "base + exactly this feature".

    Two-pass resolution, in priority order:

      1. **``config.meta.canonical_id``** — the authoritative, explicit
         mapping written by ``add_one_feature`` and other generators.
         Cheap, robust against materializer side-effects that add
         extra func_ids beyond the feature's primitive_spec.
      2. **Func_ids delta** — for configs whose meta lacks a
         canonical_id (e.g. coalitions, legacy cubes), match by
         ``c.func_ids − base.func_ids == feature.add_funcs``.

    The delta fallback is necessary but fragile: if the materializer
    injects primitives (section nodes, output fields) that aren't
    present in the feature's primitive_spec, the equality check misses
    and the feature silently vanishes from the analysis. Pass 1 avoids
    that failure mode for any config generator that writes
    canonical_id into meta.

    Features whose primitives are entirely contained in the base config
    are skipped (no separable "with" semantics).
    """
    base_fids = base_func_ids(configs_df_in, base_config_id)
    out: Dict[str, int] = {}

    # Canonical IDs known to the feature registry — guard against a
    # cube where config.meta.canonical_id points to a feature that
    # isn't currently registered (stale runs).
    known_canon = set(features_df_in["canonical_id"].tolist()) \
        if not features_df_in.empty else set()

    # Base's feature_ids — used in pass 1 to verify strict add-one
    # structure. Without this, a cube with two batches (e.g. "feat_X
    # alone" AND "CoT + feat_X") both carrying canonical_id=feat_X
    # would resolve ambiguously.
    #
    # Sourced from the ``feature_ids_set`` column on configs_df (v8
    # config_feature join table). Falls back to ``meta.feature_ids``
    # if that column isn't present (e.g. hand-built test DataFrame).
    base_row = configs_df_in[configs_df_in["config_id"] == base_config_id]
    base_feat_ids: set = set()
    if not base_row.empty:
        base_row_obj = base_row.iloc[0]
        if "feature_ids_set" in configs_df_in.columns:
            base_feat_ids = set(base_row_obj["feature_ids_set"] or set())
        else:
            base_meta = base_row_obj.get("meta") or {}
            if isinstance(base_meta, dict):
                base_feat_ids = set(base_meta.get("feature_ids", []) or [])

    # Map canonical_id → feature_id (content hash) for add-one equality check.
    cid_to_fid = (dict(zip(features_df_in["canonical_id"], features_df_in["feature_id"]))
                  if not features_df_in.empty and "feature_id" in features_df_in.columns
                  else {})

    # Pass 1a (strict): canonical_id match AND feature_ids == base ∪ {feat_X}.
    # Disambiguates cubes that contain multiple configs sharing a
    # canonical_id under different bases (e.g. "feat_x solo" vs
    # "CoT + feat_x" both labeled feat_x).
    #
    # Prefers ``configs_df.feature_ids_set`` (v8 config_feature-sourced,
    # authoritative) when present; falls back to ``meta.feature_ids``.
    has_fids_col = "feature_ids_set" in configs_df_in.columns
    used_cids: set = set()
    for _, crow in configs_df_in.iterrows():
        cid = int(crow["config_id"])
        if cid == base_config_id:
            continue
        meta = crow.get("meta") or {}
        if not isinstance(meta, dict):
            continue
        canon = meta.get("canonical_id")
        if not canon or canon not in known_canon or canon in out:
            continue
        # Resolve this config's feature_ids — authoritative source first.
        if has_fids_col and crow.get("feature_ids_set") is not None:
            cfg_feat_ids = set(crow["feature_ids_set"])
            has_strict_signal = True
        elif "feature_ids" in meta:
            cfg_feat_ids = set(meta.get("feature_ids") or [])
            has_strict_signal = True
        else:
            cfg_feat_ids = set()
            has_strict_signal = False
        if not has_strict_signal:
            continue  # leave for pass 1b
        expected_fid = cid_to_fid.get(canon)
        if expected_fid is None:
            continue
        if cfg_feat_ids == base_feat_ids | {expected_fid}:
            out[canon] = cid
            used_cids.add(cid)

    # Pass 1b (relaxed): canonical_id match alone, for cubes where the
    # generator didn't write feature_ids into meta (legacy / minimal
    # metadata). Skips configs already claimed by pass 1a.
    for _, crow in configs_df_in.iterrows():
        cid = int(crow["config_id"])
        if cid == base_config_id or cid in used_cids:
            continue
        meta = crow.get("meta") or {}
        if not isinstance(meta, dict):
            continue
        canon = meta.get("canonical_id")
        if not canon or canon not in known_canon or canon in out:
            continue
        out[canon] = cid
        used_cids.add(cid)

    # Pass 2: func_ids delta for any feature still missing.
    missing = [row for _, row in features_df_in.iterrows()
               if row["canonical_id"] not in out] if not features_df_in.empty else []
    if missing:
        deltas = []
        for _, crow in configs_df_in.iterrows():
            cid = int(crow["config_id"])
            if cid == base_config_id:
                continue
            # Skip configs already claimed by a pass-1 match.
            if cid in out.values():
                continue
            deltas.append((cid, crow["func_ids"] - base_fids))

        for frow in missing:
            add_funcs = frow["func_ids"] - base_fids
            if not add_funcs:
                continue
            for cid, delta in deltas:
                if delta == add_funcs:
                    out[frow["canonical_id"]] = cid
                    break
    return out


def configs_containing_feature(
    configs_df_in,
    features_df_in,
) -> Dict[str, Set[int]]:
    """Map ``canonical_id → set of config_ids whose func_ids ⊇ feature.func_ids``.

    Empty-feature edge case: returns an empty set (no config "contains"
    a no-op feature in any meaningful sense).
    """
    out: Dict[str, Set[int]] = {}
    cfg_pairs = [
        (int(c["config_id"]), c["func_ids"])
        for _, c in configs_df_in.iterrows()
    ]
    for _, frow in features_df_in.iterrows():
        fids = frow["func_ids"]
        if not fids:
            out[frow["canonical_id"]] = set()
            continue
        out[frow["canonical_id"]] = {
            cid for cid, cfids in cfg_pairs if fids.issubset(cfids)
        }
    return out


def is_base_feature(feature_func_ids, base_func_ids_set) -> bool:
    """True if the feature's complete primitive set lies inside base.

    Such a feature has no separable "with vs without" semantics — under
    add-one it has no target config, under marginal it's in every config.
    Either way the lift is undefined.
    """
    if not feature_func_ids:
        return True
    return feature_func_ids.issubset(base_func_ids_set)
