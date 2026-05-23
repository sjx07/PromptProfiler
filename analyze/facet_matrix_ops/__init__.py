"""Composable FACET matrix operators.

This package decomposes the feature/context matrix analyzer into small
operators that can be used independently or through ``FacetOperatorChain``.
The legacy ``analyze.facet_matrix`` module re-exports this package.
"""
from __future__ import annotations

from .config import config_feature_sets, infer_baseline_config_id
from .context import query_context_sets
from .itemsets import (
    context_itemsets_from_context,
    context_transactions,
    feature_itemsets_from_delta,
    feature_transactions,
    frequent_itemsets,
)
from .outcomes import paired_delta_rows
from .pipeline import (
    FacetChainState,
    FacetOperatorChain,
    FacetScope,
    discover_benchmark_subgroups,
)
from .scoring import (
    compare_binary_rules_across_benchmarks,
    score_binary_contextual_rules,
    score_contextual_feature_itemsets,
    score_feature_itemsets,
)
from .stats import bootstrap_mean_delta_ci, wilson_interval


__all__ = [
    "FacetChainState",
    "FacetOperatorChain",
    "FacetScope",
    "bootstrap_mean_delta_ci",
    "compare_binary_rules_across_benchmarks",
    "config_feature_sets",
    "context_itemsets_from_context",
    "context_transactions",
    "discover_benchmark_subgroups",
    "feature_itemsets_from_delta",
    "feature_transactions",
    "frequent_itemsets",
    "infer_baseline_config_id",
    "paired_delta_rows",
    "query_context_sets",
    "score_binary_contextual_rules",
    "score_contextual_feature_itemsets",
    "score_feature_itemsets",
    "wilson_interval",
]
