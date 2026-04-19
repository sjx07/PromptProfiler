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


def base_func_ids(configs_df_in, base_config_id: int) -> frozenset:
    """Return the func_ids set for the given base_config_id."""
    row = configs_df_in[configs_df_in["config_id"] == base_config_id]
    if row.empty:
        raise ValueError(f"base_config_id {base_config_id} not found")
    return row.iloc[0]["func_ids"]


def simple_effect_configs(
    configs_df_in,
    features_df_in,
    base_config_id: int,
) -> Dict[str, int]:
    """Map ``canonical_id → config_id`` for "base + exactly this feature".

    A config ``c`` matches feature ``f`` iff ``c.func_ids − base.func_ids``
    equals the feature's add_funcs (its ``func_ids`` minus base).

    Features whose add_funcs is empty (everything they contribute already
    lives in base) are skipped.
    """
    base_fids = base_func_ids(configs_df_in, base_config_id)
    out: Dict[str, int] = {}

    # Precompute deltas for non-base configs.
    deltas = []
    for _, crow in configs_df_in.iterrows():
        cid = int(crow["config_id"])
        if cid == base_config_id:
            continue
        deltas.append((cid, crow["func_ids"] - base_fids))

    for _, frow in features_df_in.iterrows():
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
