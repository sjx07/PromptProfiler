"""Shared predicate extractors for table tasks.

Computes observable context features from table structure and question/statement
text. Used by both WTQ and TabFact (and future table tasks).

All predicate values are categorical strings.
"""
from __future__ import annotations

import re
from typing import Dict, List


# ── regex patterns ────────────────────────────────────────────────────

_AGG = re.compile(
    r"\b(total|sum|average|mean|count|how many|number of|all|every|combined|altogether)\b",
    re.IGNORECASE,
)
_SUP = re.compile(
    r"\b(most|least|highest|lowest|best|worst|largest|smallest|greatest|fewest"
    r"|maximum|minimum|top|bottom|longest|shortest|oldest|newest|first|last)\b",
    re.IGNORECASE,
)
_CMP = re.compile(
    r"\b(more|less|greater|fewer|higher|lower|better|worse|bigger|smaller"
    r"|longer|shorter|older|newer|earlier|later)\s+than\b",
    re.IGNORECASE,
)
_TEMPORAL = re.compile(
    r"\b(before|after|during|between|year|month|day|date|season|century|decade|when"
    r"|january|february|march|april|may|june|july|august|september|october|november|december|\d{4})\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(not|no|never|neither|none|nobody|nothing|nowhere|nor|without"
    r"|didn't|doesn't|don't|wasn't|weren't|isn't|aren't|hasn't|haven't"
    r"|hadn't|won't|wouldn't|couldn't|shouldn't)\b",
    re.IGNORECASE,
)
_ARITH = re.compile(
    r"\b(difference|subtract|add|multiply|divide|ratio|percent|percentage"
    r"|how much more|how much less|how many more|how many fewer)\b",
    re.IGNORECASE,
)
_FILTER = re.compile(
    r"\b(which|where|who|what)\b.*\b(with|in|at|from|for|on|during|after|before)\b",
    re.IGNORECASE,
)


def _is_numeric(v: str) -> bool:
    v = v.strip().replace(",", "").replace("%", "").replace("$", "")
    try:
        float(v)
        return True
    except ValueError:
        return False


# ── question / statement intent ───────────────────────────────────────


def question_intent_predicates(text: str) -> Dict[str, str]:
    """Binary intent predicates from question or statement text."""
    return {
        "has_aggregation": "yes" if _AGG.search(text) else "no",
        "has_superlative": "yes" if _SUP.search(text) else "no",
        "has_comparison": "yes" if _CMP.search(text) else "no",
        "has_temporal": "yes" if _TEMPORAL.search(text) else "no",
        "has_negation": "yes" if _NEGATION.search(text) else "no",
        "has_arithmetic": "yes" if _ARITH.search(text) else "no",
        "has_filter": "yes" if _FILTER.search(text) else "no",
    }


def operation_type_predicate(text: str) -> str:
    """Heuristic operation type from question text."""
    if re.search(r"\bhow many\b", text, re.IGNORECASE):
        return "count"
    if _SUP.search(text):
        return "superlative"
    if _ARITH.search(text):
        return "arithmetic"
    if _CMP.search(text):
        return "comparison"
    if _AGG.search(text):
        return "aggregation"
    return "lookup"


# ── table structure ───────────────────────────────────────────────────


def table_structure_predicates(
    header: List[str], rows: List[List[str]],
) -> Dict[str, str]:
    """Core table structure predicates."""
    n_rows = len(rows)
    n_cols = len(header)
    preds: Dict[str, str] = {
        "n_rows": str(n_rows),
        "n_cols": str(n_cols),
    }

    # Numeric column analysis
    numeric_cols = 0
    if rows and n_cols > 0:
        for ci in range(n_cols):
            vals = [str(rows[ri][ci]).strip() for ri in range(min(n_rows, 50))
                    if ci < len(rows[ri]) and str(rows[ri][ci]).strip()]
            if vals and sum(_is_numeric(v) for v in vals) / len(vals) > 0.5:
                numeric_cols += 1

    preds["has_numeric_cols"] = "yes" if numeric_cols > 0 else "no"
    preds["n_numeric_cols"] = str(numeric_cols)

    # Shape
    if n_rows > 0 and n_cols > 0:
        ratio = n_cols / n_rows
        preds["t_shape"] = "wide" if ratio > 0.5 else ("tall" if ratio < 0.1 else "balanced")
    else:
        preds["t_shape"] = "empty"

    return preds


# ── combined ──────────────────────────────────────────────────────────


def compute_base_predicates(
    header: List[str],
    rows: List[List[str]],
    text: str = "",
) -> Dict[str, str]:
    """Compute all shared predicates from table + question/statement text."""
    preds = table_structure_predicates(header, rows)
    if text:
        preds.update(question_intent_predicates(text))
        preds["operation_type"] = operation_type_predicate(text)
        preds["question_length"] = str(len(text.split()))
        preds["is_count"] = "yes" if re.search(r"\bhow many\b", text, re.IGNORECASE) else "no"
    return preds
