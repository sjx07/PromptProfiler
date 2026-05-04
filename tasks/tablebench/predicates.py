"""TableBench predicates — register with query_cohorts for seed_predicates.

Free dataset-supplied categoricals (qtype, qsubtype) plus the same surface-level
predicates extracted from question/table for parity with WTQ/TabFact.
"""
from __future__ import annotations

import json
from typing import Dict

from experiment.query_cohorts import register_extractor
from tasks.predicates import compute_base_predicates


def compute_predicates(meta: dict) -> Dict[str, str]:
    raw = meta.get("_raw", {})
    table = raw.get("table", {})
    header = table.get("header", [])
    rows = table.get("rows", [])
    question = raw.get("question", "")

    preds = compute_base_predicates(header, rows, question)
    # Free dataset-supplied labels:
    preds["qtype"] = raw.get("qtype", meta.get("qtype", "unknown"))
    preds["qsubtype"] = raw.get("qsubtype", meta.get("qsubtype", "unknown"))
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
    # Free from dataset
    "qtype", "qsubtype",
    # Surface-level (parity with wtq/tabfact)
    "n_rows", "n_cols", "has_numeric_cols", "n_numeric_cols", "t_shape",
    "has_aggregation", "has_superlative", "has_comparison", "has_temporal",
    "has_negation", "has_arithmetic", "has_filter",
    "operation_type", "question_length", "is_count",
]

for _name in _PRED_NAMES:
    register_extractor(_name)(_make_extractor(_name))
