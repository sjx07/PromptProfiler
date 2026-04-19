"""Layer 2 — lift / DiD arithmetic.

Pure pandas. No SQL, no CubeStore. Given a ``scores_df`` and a
canonical_id-to-config(s) mapping, produce the long-format effect
table that everything downstream consumes.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set


def _default_reference_value(predicate_values: List[str]) -> Optional[str]:
    """For binary predicates, return the alphabetically-first value.
    For multi-value, return None (caller must supply explicitly)."""
    uniq = sorted(set(predicate_values))
    if len(uniq) == 2:
        return uniq[0]
    return None


def lift_simple(
    scores_df_in,
    *,
    base_cid: int,
    canonicals: List[str],
    canonical_to_cid: Dict[str, int],
    predicate_names: Optional[List[str]] = None,
    reference_values: Optional[Dict[str, str]] = None,
    progress: bool = False,
):
    """Long-format effect table under add-one (paired) semantics.

    For every (canonical_id × predicate_name × predicate_value) compute:
      n_with, mean_with, n_without, mean_without, lift = mean_with − mean_without.

    A canonical_id that has no entry in ``canonical_to_cid`` still
    appears with n_with=0, mean_with=NaN, lift=NaN — caller decides
    whether to include unmatched rows.
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas required") from e

    if predicate_names is None:
        if scores_df_in.empty:
            predicate_names = []
        else:
            predicate_names = sorted(scores_df_in["predicate_name"].unique().tolist())

    reference_values = reference_values or {}
    records = []

    from analyze._progress import progress_iter
    for pname in progress_iter(predicate_names, enable=progress,
                               total=len(predicate_names),
                               desc="lift/simple"):
        sub = scores_df_in[scores_df_in["predicate_name"] == pname]
        if sub.empty:
            continue
        per_cfg = sub.groupby(
            ["config_id", "predicate_value"], as_index=False,
        ).agg(mean_score=("score", "mean"), n=("score", "count"))
        pivot_score = per_cfg.pivot(index="config_id", columns="predicate_value",
                                    values="mean_score")
        pivot_n = per_cfg.pivot(index="config_id", columns="predicate_value",
                                values="n")
        predicate_values = sorted(per_cfg["predicate_value"].unique().tolist())
        ref_value = reference_values.get(pname) or _default_reference_value(predicate_values)

        for canonical in canonicals:
            target_cid = canonical_to_cid.get(canonical)
            for pv in predicate_values:
                s_with = (pivot_score.loc[target_cid, pv]
                          if target_cid is not None
                             and target_cid in pivot_score.index
                             and pv in pivot_score.columns
                          else float("nan"))
                n_with = (int(pivot_n.loc[target_cid, pv])
                          if target_cid is not None
                             and target_cid in pivot_n.index
                             and pv in pivot_n.columns
                             and pd.notna(pivot_n.loc[target_cid, pv])
                          else 0)
                s_without = (pivot_score.loc[base_cid, pv]
                             if base_cid in pivot_score.index
                                and pv in pivot_score.columns
                             else float("nan"))
                n_without = (int(pivot_n.loc[base_cid, pv])
                             if base_cid in pivot_n.index
                                and pv in pivot_n.columns
                                and pd.notna(pivot_n.loc[base_cid, pv])
                             else 0)
                lift = (s_with - s_without) if pd.notna(s_with) and pd.notna(s_without) \
                    else float("nan")
                records.append({
                    "canonical_id":     canonical,
                    "predicate_name":   pname,
                    "predicate_value":  pv,
                    "n_with":           n_with,
                    "mean_with":        s_with,
                    "n_without":        n_without,
                    "mean_without":     s_without,
                    "lift":             lift,
                    "_ref":             ref_value,
                })
    return pd.DataFrame(records)


def lift_marginal(
    scores_df_in,
    *,
    canonicals: List[str],
    canonical_to_with_cids: Dict[str, Set[int]],
    all_cids: List[int],
    predicate_names: Optional[List[str]] = None,
    reference_values: Optional[Dict[str, str]] = None,
    progress: bool = False,
):
    """Long-format effect table under marginal (per-config pooling) semantics.

    For each feature: average per-config means across configs containing
    the feature vs configs not containing it. Per-config pooling makes
    the estimate robust to imbalanced per-config run sizes.
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas required") from e

    if predicate_names is None:
        if scores_df_in.empty:
            predicate_names = []
        else:
            predicate_names = sorted(scores_df_in["predicate_name"].unique().tolist())

    reference_values = reference_values or {}
    records = []
    all_cids_set = set(all_cids)

    from analyze._progress import progress_iter
    for pname in progress_iter(predicate_names, enable=progress,
                               total=len(predicate_names),
                               desc="lift/marginal"):
        sub = scores_df_in[scores_df_in["predicate_name"] == pname]
        if sub.empty:
            continue
        per_cfg = sub.groupby(
            ["config_id", "predicate_value"], as_index=False,
        ).agg(mean_score=("score", "mean"), n=("score", "count"))
        pivot_score = per_cfg.pivot(index="config_id", columns="predicate_value",
                                    values="mean_score")
        predicate_values = sorted(per_cfg["predicate_value"].unique().tolist())
        ref_value = reference_values.get(pname) or _default_reference_value(predicate_values)

        for canonical in canonicals:
            with_cfgs = canonical_to_with_cids.get(canonical, set())
            without_cfgs = all_cids_set - with_cfgs
            for pv in predicate_values:
                if pv in pivot_score.columns:
                    with_vals = pivot_score.reindex(list(with_cfgs))[pv].dropna()
                    without_vals = pivot_score.reindex(list(without_cfgs))[pv].dropna()
                else:
                    with_vals = pd.Series(dtype=float)
                    without_vals = pd.Series(dtype=float)
                s_with = with_vals.mean() if len(with_vals) else float("nan")
                s_without = without_vals.mean() if len(without_vals) else float("nan")
                n_with = len(with_vals)
                n_without = len(without_vals)
                lift = (s_with - s_without) if pd.notna(s_with) and pd.notna(s_without) \
                    else float("nan")
                records.append({
                    "canonical_id":     canonical,
                    "predicate_name":   pname,
                    "predicate_value":  pv,
                    "n_with":           n_with,
                    "mean_with":        s_with,
                    "n_without":        n_without,
                    "mean_without":     s_without,
                    "lift":             lift,
                    "_ref":             ref_value,
                })
    return pd.DataFrame(records)


def did(lift_df_in, *, drop_ref_col: bool = True):
    """Add a ``did`` column to a lift table.

    DiD value at each non-reference predicate_value is
    ``lift[value] − lift[reference]``. The reference row gets did=0.

    The reference is read from the ``_ref`` column embedded by
    ``lift_simple`` / ``lift_marginal``. When ``_ref`` is None
    (multi-value predicate without explicit reference), did stays NaN.
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas required") from e

    df = lift_df_in.copy()
    if df.empty:
        if drop_ref_col:
            df = df.drop(columns=["_ref"], errors="ignore")
        return df

    df["did"] = float("nan")
    for (canon, pname), group in df.groupby(["canonical_id", "predicate_name"]):
        ref = group["_ref"].iloc[0]
        if ref is None:
            continue
        ref_lift_series = group.loc[group["predicate_value"] == ref, "lift"]
        if ref_lift_series.empty or pd.isna(ref_lift_series.iloc[0]):
            continue
        ref_lift = ref_lift_series.iloc[0]
        for idx in group.index:
            pv = df.at[idx, "predicate_value"]
            if pv == ref:
                df.at[idx, "did"] = 0.0
            else:
                lv = df.at[idx, "lift"]
                if pd.notna(lv):
                    df.at[idx, "did"] = lv - ref_lift
    if drop_ref_col:
        df = df.drop(columns=["_ref"], errors="ignore")
    return df
