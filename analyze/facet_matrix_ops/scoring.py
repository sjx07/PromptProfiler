"""Scoring operators for FACET feature/context rules."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from .itemsets import context_transactions, feature_transactions
from .stats import bootstrap_mean_delta_ci, trimmed_mean, wilson_interval


def _ids_for_itemset(transactions: Mapping[Any, frozenset[str]], itemset: Sequence[str]) -> List[Any]:
    want = frozenset(itemset)
    return sorted(key for key, items in transactions.items() if want.issubset(items))


def _label(items: Sequence[str], sep: str) -> str:
    return sep.join(items) if items else "(none)"


def score_feature_itemsets(
    delta_df,
    feature_itemsets: Sequence[Sequence[str]],
    *,
    n_bootstrap: int = 500,
    seed: int = 42,
    ci_level: float = 0.95,
):
    """Score global paired deltas for feature/coalition itemsets."""
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas required for score_feature_itemsets") from e

    ftx = feature_transactions(delta_df)
    records: List[Dict[str, Any]] = []
    for itemset in feature_itemsets:
        items = tuple(sorted(itemset))
        config_ids = _ids_for_itemset(ftx, items)
        if not config_ids:
            continue
        sub = delta_df[delta_df["config_id"].isin(config_ids)]
        if sub.empty:
            continue
        query_means = sub.groupby("query_id")["delta"].mean()
        ci_lo, ci_hi, p_gt_zero = bootstrap_mean_delta_ci(
            query_means.tolist(),
            n_bootstrap=n_bootstrap,
            seed=seed,
            ci_level=ci_level,
        )
        values = query_means.tolist()
        records.append({
            "feature_items": items,
            "feature_rule": _label(items, "+"),
            "feature_order": len(items),
            "n_configs": int(sub["config_id"].nunique()),
            "n_queries": int(sub["query_id"].nunique()),
            "n_observations": int(len(sub)),
            "mean_delta": float(query_means.mean()),
            "trimmed_mean_delta_5_95": trimmed_mean(values, 0.05),
            "median_delta": float(query_means.median()),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p_gt_zero": p_gt_zero,
        })
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(
            ["mean_delta", "n_queries", "feature_rule"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
    return df


def score_contextual_feature_itemsets(
    delta_df,
    context_df,
    feature_itemsets: Sequence[Sequence[str]],
    context_itemsets: Sequence[Sequence[str]],
    *,
    min_query_support: int = 20,
    min_observation_support: int = 20,
    n_bootstrap: int = 500,
    seed: int = 42,
    ci_level: float = 0.95,
):
    """Score feature coalitions under context conjunctions.

    The confidence interval targets:

    ``mean_delta(F within C) - mean_delta(F globally)``.

    Query-level means are used before bootstrapping so one query is not
    overweighted simply because multiple configs contain the same itemset.
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas required for score_contextual_feature_itemsets") from e

    ftx = feature_transactions(delta_df)
    ctx = context_transactions(context_df)
    records: List[Dict[str, Any]] = []
    for f_itemset in feature_itemsets:
        f_items = tuple(sorted(f_itemset))
        config_ids = _ids_for_itemset(ftx, f_items)
        if not config_ids:
            continue
        f_rows = delta_df[delta_df["config_id"].isin(config_ids)]
        if f_rows.empty:
            continue
        f_query_means = f_rows.groupby("query_id")["delta"].mean()
        feature_mean = float(f_query_means.mean())
        for c_itemset in context_itemsets:
            c_items = tuple(sorted(c_itemset))
            query_ids = set(_ids_for_itemset(ctx, c_items))
            if len(query_ids) < min_query_support:
                continue
            sub = f_rows[f_rows["query_id"].isin(query_ids)]
            n_queries = int(sub["query_id"].nunique())
            if n_queries < min_query_support or len(sub) < min_observation_support:
                continue
            query_means = sub.groupby("query_id")["delta"].mean()
            values = query_means.tolist()
            ci_lo, ci_hi, p_gt_feature = bootstrap_mean_delta_ci(
                values,
                center=feature_mean,
                n_bootstrap=n_bootstrap,
                seed=seed,
                ci_level=ci_level,
            )
            mean_delta = float(query_means.mean())
            records.append({
                "feature_items": f_items,
                "feature_rule": _label(f_items, "+"),
                "feature_order": len(f_items),
                "context_items": c_items,
                "context_rule": _label(c_items, " AND "),
                "context_order": len(c_items),
                "n_configs": int(sub["config_id"].nunique()),
                "n_queries": n_queries,
                "n_observations": int(len(sub)),
                "mean_delta": mean_delta,
                "trimmed_mean_delta_5_95": trimmed_mean(values, 0.05),
                "median_delta": float(query_means.median()),
                "feature_mean_delta": feature_mean,
                "conditional_lift": mean_delta - feature_mean,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "p_gt_feature": p_gt_feature,
            })
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(
            ["conditional_lift", "n_queries", "feature_rule", "context_rule"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
    return df


def score_binary_contextual_rules(
    delta_df,
    context_df,
    feature_itemsets: Sequence[Sequence[str]],
    context_itemsets: Sequence[Sequence[str]],
    *,
    min_query_support: int = 20,
    min_observation_support: int = 20,
    positive_delta_threshold: float = 0.0,
    ci_level: float = 0.95,
):
    """Score binary useful-rate rules for cross-benchmark comparison."""
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas required for score_binary_contextual_rules") from e

    ftx = feature_transactions(delta_df)
    ctx = context_transactions(context_df)
    records: List[Dict[str, Any]] = []
    for f_itemset in feature_itemsets:
        f_items = tuple(sorted(f_itemset))
        config_ids = _ids_for_itemset(ftx, f_items)
        if not config_ids:
            continue
        f_rows = delta_df[delta_df["config_id"].isin(config_ids)]
        if f_rows.empty:
            continue
        feature_rate = float((f_rows["delta"] > positive_delta_threshold).mean())
        for c_itemset in context_itemsets:
            c_items = tuple(sorted(c_itemset))
            query_ids = set(_ids_for_itemset(ctx, c_items))
            if len(query_ids) < min_query_support:
                continue
            sub = f_rows[f_rows["query_id"].isin(query_ids)]
            n_queries = int(sub["query_id"].nunique())
            n_obs = int(len(sub))
            if n_queries < min_query_support or n_obs < min_observation_support:
                continue
            successes = int((sub["delta"] > positive_delta_threshold).sum())
            useful_rate = successes / n_obs
            ci_lo, ci_hi = wilson_interval(successes, n_obs, ci_level=ci_level)
            half_width = max((ci_hi - ci_lo) / 2.0, 1e-12)
            lift = useful_rate - feature_rate
            records.append({
                "feature_items": f_items,
                "feature_rule": _label(f_items, "+"),
                "feature_order": len(f_items),
                "context_items": c_items,
                "context_rule": _label(c_items, " AND "),
                "context_order": len(c_items),
                "n_configs": int(sub["config_id"].nunique()),
                "n_queries": n_queries,
                "n_observations": n_obs,
                "successes": successes,
                "useful_rate": useful_rate,
                "feature_useful_rate": feature_rate,
                "rate_lift_over_feature": lift,
                "wilson_ci_lo": ci_lo,
                "wilson_ci_hi": ci_hi,
                "confidence_ratio": lift / half_width,
            })
    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(
            ["confidence_ratio", "rate_lift_over_feature", "n_queries"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return df


def compare_binary_rules_across_benchmarks(rule_frames: Mapping[str, Any]):
    """Compare identical binary rule keys found in multiple benchmarks."""
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas required for compare_binary_rules_across_benchmarks") from e

    frames = []
    for dataset, frame in rule_frames.items():
        if frame is None or frame.empty:
            continue
        tmp = frame.copy()
        tmp["dataset"] = dataset
        tmp["rule_key"] = tmp["feature_rule"] + " :: " + tmp["context_rule"]
        frames.append(tmp)
    if not frames:
        return pd.DataFrame(columns=[
            "rule_key", "datasets", "n_datasets", "mean_rate_lift_over_feature",
            "min_rate_lift_over_feature", "max_rate_lift_over_feature",
            "mean_confidence_ratio", "total_observations",
        ])

    all_rules = pd.concat(frames, ignore_index=True)
    out = all_rules.groupby("rule_key", as_index=False).agg(
        datasets=("dataset", lambda values: tuple(sorted(set(values)))),
        n_datasets=("dataset", "nunique"),
        mean_rate_lift_over_feature=("rate_lift_over_feature", "mean"),
        min_rate_lift_over_feature=("rate_lift_over_feature", "min"),
        max_rate_lift_over_feature=("rate_lift_over_feature", "max"),
        mean_confidence_ratio=("confidence_ratio", "mean"),
        total_observations=("n_observations", "sum"),
    )
    return out.sort_values(
        ["n_datasets", "mean_confidence_ratio", "mean_rate_lift_over_feature"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
