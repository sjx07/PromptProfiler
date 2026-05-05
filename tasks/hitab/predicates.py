"""HiTab predicate extractors."""
from __future__ import annotations

import json
from typing import Dict

from experiment.query_cohorts import register_extractor
from tasks.predicates import compute_base_predicates


def compute_predicates(meta: dict) -> Dict[str, str]:
    raw = meta.get("_raw", {})
    question = raw.get("question", "")
    table_content = raw.get("table_content", {})
    aggregation = raw.get("aggregation", "none") or "none"
    table_source = raw.get("table_source", "") or ""

    texts = table_content.get("texts", []) if isinstance(table_content, dict) else []
    header = texts[0] if texts else []
    rows = texts[1:] if len(texts) > 1 else []

    preds = compute_base_predicates(header, rows, question)

    agg_lower = str(aggregation).lower()
    if agg_lower in ("none", ""):
        preds["agg_type"] = "none"
    elif "argmax" in agg_lower or "argmin" in agg_lower:
        preds["agg_type"] = "argmax_argmin"
    elif "diff" in agg_lower or "greater_than" in agg_lower:
        preds["agg_type"] = "comparison"
    elif "topk" in agg_lower or "rank" in agg_lower:
        preds["agg_type"] = "ranking"
    elif "ratio" in agg_lower or "percent" in agg_lower:
        preds["agg_type"] = "ratio"
    else:
        preds["agg_type"] = "other"

    src = str(table_source).lower()
    if "statcan" in src or "nsf" in src:
        preds["table_source"] = "statistical_report"
    elif "totto" in src or "wikipedia" in src or "wiki" in src:
        preds["table_source"] = "wikipedia"
    else:
        preds["table_source"] = "other"

    return preds


def _make_extractor(pred_name: str):
    def extractor(query: dict) -> str:
        meta = query.get("meta", {})
        if isinstance(meta, str):
            meta = json.loads(meta)
        return compute_predicates(meta).get(pred_name, "unknown")
    return extractor


_PRED_NAMES = [
    "n_rows", "n_cols", "has_numeric_cols", "n_numeric_cols", "t_shape",
    "has_aggregation", "has_superlative", "has_comparison", "has_temporal",
    "has_negation", "has_arithmetic", "has_filter",
    "operation_type", "question_length", "is_count",
    "agg_type", "table_source",
]

for _name in _PRED_NAMES:
    register_extractor(f"hitab_{_name}")(_make_extractor(_name))
