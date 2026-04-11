"""WTQ predicate extractors — register with query_cohorts for seed_predicates."""
from __future__ import annotations

import json
from typing import Dict

from prompt_profiler.experiment.query_cohorts import register_extractor
from prompt_profiler.tasks.predicates import compute_base_predicates


def compute_predicates(meta: dict) -> Dict[str, str]:
    """Compute all predicates for a WTQ query."""
    raw = meta.get("_raw", {})
    table = raw.get("table", {})
    header = table.get("header", [])
    rows = table.get("rows", [])
    question = raw.get("question", "")

    preds = compute_base_predicates(header, rows, question)

    # NOTE: Do NOT add predicates derived from gold answers (n_answers,
    # answer_type, is_yesno, qa_value_freq). These are hindsight-
    # contaminated — unavailable at inference time.

    return preds


# ── Register each predicate as a query_cohorts extractor ──────────────
# This allows seed_predicates(store) to compute all WTQ predicates.


def _make_extractor(pred_name: str):
    """Create an extractor closure for a single predicate name."""
    def extractor(query: dict) -> str:
        meta = query.get("meta", {})
        if isinstance(meta, str):
            meta = json.loads(meta)
        preds = compute_predicates(meta)
        return preds.get(pred_name, "unknown")
    return extractor


# Register all predicates that compute_base_predicates + compute_predicates produce.
# Done at import time so seed_predicates() can find them.
_PRED_NAMES = [
    "n_rows", "n_cols", "has_numeric_cols", "n_numeric_cols", "t_shape",
    "has_aggregation", "has_superlative", "has_comparison", "has_temporal",
    "has_negation", "has_arithmetic", "has_filter",
    "operation_type", "question_length", "is_count",
]

for _name in _PRED_NAMES:
    register_extractor(_name)(_make_extractor(_name))
