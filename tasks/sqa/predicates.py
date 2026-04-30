"""SQA predicate extractors — register with query_cohorts for seed_predicates."""
from __future__ import annotations

import json
import re
from typing import Dict

from experiment.query_cohorts import register_extractor
from tasks.predicates import compute_base_predicates


def compute_predicates(meta: dict) -> Dict[str, str]:
    """Compute all predicates for an SQA query.

    Mixes generic table predicates (n_rows, has_aggregation, ...) with
    SQA-specific ones (position, n_history, has_reference). All values
    are categorical strings.

    NOTE: Avoid gold-derived predicates (n_answers count etc.) at
    inference time. position, n_history, has_reference are input-only.
    """
    raw = meta.get("_raw", {})
    question = raw.get("question", "")
    position = raw.get("position", 0)
    table = raw.get("table", {})
    history = raw.get("history", [])

    # SQA stores table with 'headers' (not 'header')
    header = table.get("headers", table.get("header", []))
    rows = table.get("rows", [])

    preds = compute_base_predicates(header, rows, question)

    # ── SQA-specific (input-only) ──────────────────────────────────────
    preds["position"] = str(position)
    preds["n_history"] = str(len(history))

    ref_pats = (
        r"\b(those|that|these|them|they|it|its|their|the same|"
        r"which one|which ones|above|previous|mentioned)\b"
    )
    preds["has_reference"] = "yes" if re.search(ref_pats, question.lower()) else "no"

    # Override operation_type for follow-up questions
    if preds.get("operation_type") == "lookup" and preds["has_reference"] == "yes":
        preds["operation_type"] = "follow_up"

    return preds


# ── Register each predicate with query_cohorts ─────────────────────────
# Prefixed with `sqa_` to avoid bare-name collision with WTQ.


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
    "position", "n_history", "has_reference",
]

for _name in _PRED_NAMES:
    register_extractor(f"sqa_{_name}")(_make_extractor(_name))
