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
    "plan_delete",
    # monitor
    "ProgressMonitor",
]
