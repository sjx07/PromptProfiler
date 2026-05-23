"""Analysis & monitoring primitives over a CubeStore.

Four layers:

  * ``analyze.meta``    — describe-what-exists (configs, models, scorers, ...).
  * ``analyze.query``   — ``ExecutionQuery`` chainable filter + projection.
  * ``analyze.compare`` — cross-config deltas and rankings.
  * ``analyze.monitor`` — ``ProgressMonitor`` live-view facade.

See ``docs/analysis.md`` for the full taxonomy and recipes.
"""
from __future__ import annotations

# ── meta (describe) ───────────────────────────────────────────────────
from analyze.meta import (
    list_configs,
    list_configs_with_features,
    list_configs_with_func_types,
    list_models,
    list_scorers,
    list_phases,
    list_datasets,
    list_predicates,
    list_features_in_cube,
    summary,
)
from analyze.data import predicate_kinds, predicate_overlap

# ── FACET feature/context matrix operators ───────────────────────────
from analyze.facet_matrix_ops import (
    FacetChainState,
    FacetOperatorChain,
    FacetScope,
    bootstrap_mean_delta_ci,
    compare_binary_rules_across_benchmarks,
    config_feature_sets,
    context_itemsets_from_context,
    context_transactions,
    discover_benchmark_subgroups,
    feature_itemsets_from_delta,
    feature_transactions,
    frequent_itemsets,
    infer_baseline_config_id,
    paired_delta_rows,
    query_context_sets,
    score_binary_contextual_rules,
    score_contextual_feature_itemsets,
    score_feature_itemsets,
    wilson_interval,
)

# ── chain-of-ops pipeline (R2) ────────────────────────────────────────
from analyze.pipeline import Pipeline

# ── SQL-native source handle (R5 / option D) ─────────────────────────
from analyze.source import SourceHandle

# ── query (filter + project + aggregate) ──────────────────────────────
from analyze.query import ExecutionQuery

# ── compare (compose) ─────────────────────────────────────────────────
from analyze.compare import (
    score_diff,
    feature_effect_ranking,
    predicate_slice,
    add_one_deltas,
    flip_rows,
    harm_cases,
    help_cases,
    feature_predicate_table,
    feature_profile,
)

# ── export (error reports for downstream LLM loops) ───────────────────
from analyze.export import flipped_responses

# ── cube visualizer operations ────────────────────────────────────────
from analyze.cube_ops import (
    cube_summary,
    list_configs_detailed,
    list_query_meta_fields,
    slice_scores,
    compare_configs,
    comparison_examples,
    examples,
    execution_artifact,
    diagnostics,
    feature_summary,
    plan_delete,
)

# ── monitor ───────────────────────────────────────────────────────────
from analyze.monitor import ProgressMonitor


__all__ = [
    # meta
    "list_configs",
    "list_configs_with_features",
    "list_configs_with_func_types",
    "list_models",
    "list_scorers",
    "list_phases",
    "list_datasets",
    "list_predicates",
    "list_features_in_cube",
    "summary",
    "predicate_kinds",
    "predicate_overlap",
    # FACET matrix
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
    # query
    "ExecutionQuery",
    # compare
    "score_diff",
    "feature_effect_ranking",
    "predicate_slice",
    "add_one_deltas",
    "flip_rows",
    "harm_cases",
    "help_cases",
    "feature_predicate_table",
    "feature_profile",
    "Pipeline",
    "SourceHandle",
    # export
    "flipped_responses",
    # cube visualizer operations
    "cube_summary",
    "list_configs_detailed",
    "list_query_meta_fields",
    "slice_scores",
    "compare_configs",
    "comparison_examples",
    "examples",
    "execution_artifact",
    "diagnostics",
    "feature_summary",
    "plan_delete",
    # monitor
    "ProgressMonitor",
]
