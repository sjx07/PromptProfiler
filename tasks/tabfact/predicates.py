"""TabFact predicate extractors — register with query_cohorts for seed_predicates."""
from __future__ import annotations

import json
import re
from typing import Dict

from prompt_profiler.experiment.query_cohorts import register_extractor
from prompt_profiler.tasks.predicates import compute_base_predicates


def compute_predicates(meta: dict) -> Dict[str, str]:
    """Compute all predicates for a TabFact query."""
    raw = meta.get("_raw", {})
    table_text = raw.get("table_text", "")
    statement = raw.get("statement", "")

    from prompt_profiler.tasks.tabfact.loaders import parse_table_text
    parsed = parse_table_text(table_text)
    header = parsed.get("headers", [])
    rows = parsed.get("rows", [])

    # Statement used as "question" for intent predicates
    preds = compute_base_predicates(header, rows, statement)

    # TabFact-specific
    preds["statement_length"] = str(len(statement.split()))
    preds["has_numeric_claim"] = "yes" if re.search(r"\b\d+\.?\d*\b", statement) else "no"

    return preds


def _make_extractor(pred_name: str):
    def extractor(query: dict) -> str:
        meta = query.get("meta", {})
        if isinstance(meta, str):
            meta = json.loads(meta)
        preds = compute_predicates(meta)
        return preds.get(pred_name, "unknown")
    return extractor


_PRED_NAMES = [
    "n_rows", "n_cols", "has_numeric_cols", "n_numeric_cols", "t_shape",
    "has_aggregation", "has_superlative", "has_comparison", "has_temporal",
    "has_negation", "has_arithmetic", "has_filter",
    "operation_type", "question_length", "is_count",
    "statement_length", "has_numeric_claim",
]

# Prefix with task to avoid collisions if both tasks are imported.
# WTQ registers bare names; TabFact prefixes with "tabfact_".
# seed_predicates uses the task-specific import, so only one set is active.
for _name in _PRED_NAMES:
    register_extractor(f"tabfact_{_name}")(_make_extractor(_name))
