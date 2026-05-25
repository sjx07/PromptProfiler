#!/usr/bin/env python3
"""Lightweight context-attribute extractors for registry v1.2.

The extractor intentionally stays primitive: lexical intent, table shape,
header/cell value surfaces, and token overlap. It avoids semantic entity-role
ontologies such as country/team/player/name-role categories.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

DATASETS = ("wtq", "sqa", "tablebench", "tab_fact", "hitab")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "their", "there", "this", "to", "was",
    "were", "what", "when", "where", "which", "who", "with",
}

TEXT_PATTERNS: dict[str, re.Pattern[str]] = {
    "intent.count": re.compile(r"\b(how many|number of|count(?:s|ed|ing)?|total number)\b", re.I),
    "intent.superlative_rank": re.compile(
        r"\b(most|least|highest|lowest|best|worst|largest|smallest|greatest|fewest|"
        r"maximum|minimum|top|bottom|longest|shortest|oldest|newest|first|last)\b",
        re.I,
    ),
    "intent.comparison": re.compile(
        r"\b(more|less|greater|fewer|higher|lower|better|worse|bigger|smaller|"
        r"longer|shorter|older|newer|earlier|later)\s+than\b",
        re.I,
    ),
    "intent.arithmetic": re.compile(
        r"\b(difference|subtract|add|multiply|divide|ratio|percent|percentage|"
        r"how much more|how much less|how many more|how many fewer)\b",
        re.I,
    ),
    "intent.aggregate_noncount": re.compile(r"\b(total|sum|average|mean|combined|altogether)\b", re.I),
    "intent.temporal": re.compile(
        r"\b(before|after|during|between|year|month|day|date|season|century|decade|when|"
        r"january|february|march|april|may|june|july|august|september|october|november|december|"
        r"\d{4})\b",
        re.I,
    ),
    "intent.negation_marker": re.compile(
        r"\b(not|no|never|neither|none|nobody|nothing|nowhere|nor|without|"
        r"didn't|doesn't|don't|wasn't|weren't|isn't|aren't|hasn't|haven't|"
        r"hadn't|won't|wouldn't|couldn't|shouldn't)\b",
        re.I,
    ),
    "text.has_number": re.compile(r"\b\d+(?:\.\d+)?\b"),
    "text.has_year": re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b"),
    "text.has_ordinal": re.compile(r"\b(?:first|second|third|fourth|fifth|last|\d+(?:st|nd|rd|th))\b", re.I),
    "text.has_quoted_span": re.compile(r"(['\"]).+?\1"),
}

DIALOG_REFERENCE_PATTERN = re.compile(
    r"\b(those|that|these|them|they|it|its|their|the same|which one|which ones|above|previous|mentioned|of these|of those|among them)\b",
    re.I,
)

HEADER_PATTERNS: dict[str, re.Pattern[str]] = {
    "schema.has_date_col": re.compile(r"\b(date|year|month|day|season|week|time)\b", re.I),
    "schema.has_rank_col": re.compile(r"^(#|rank|place|position|seed|round|pos|no\.?|draw)$|\brank\b|\bposition\b", re.I),
    "schema.has_score_col": re.compile(r"\b(score|points?|goals?|result|record|won|lost|drawn|difference|bonus|win-loss)\b", re.I),
    "schema.has_unit_col": re.compile(r"[%$£€¥]|\b(km|kg|mph|miles?|metres?|meters?|ft|feet|area|population|density|rate|percent|percentage|usd|eur|gbp)\b", re.I),
}

COMMA_NUMBER_PATTERN = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")
PERCENT_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*%")
CURRENCY_PATTERN = re.compile(r"[$£€¥]")
DATE_LIKE_PATTERN = re.compile(
    r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b",
    re.I,
)
NUMERIC_SPAN_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\b")
AFL_SCORE_PATTERN = re.compile(r"\b\d+\.\d+\s*\(\d+\)")
MISSING_MARKERS = {"", "-", "--", "?", "n/a", "na", "none", "null"}


def as_meta(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        return json.loads(raw or "{}")
    if isinstance(raw, Mapping):
        return dict(raw)
    return {}


def tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    }


def question_text(raw: dict[str, Any], dataset: str) -> str:
    if dataset == "tab_fact":
        return str(raw.get("statement", "") or "")
    return str(raw.get("question", "") or "")


def flatten_table(raw: dict[str, Any], dataset: str) -> tuple[list[str], list[list[str]]]:
    if dataset == "sqa":
        table = raw.get("table", {}) if isinstance(raw.get("table", {}), Mapping) else {}
        return list(table.get("headers", table.get("header", [])) or []), list(table.get("rows", []) or [])
    if dataset == "tab_fact":
        rows = []
        for line in str(raw.get("table_text", "") or "").splitlines():
            parts = [part for part in line.split("#") if part != ""]
            if parts:
                rows.append(parts)
        if not rows:
            return [], []
        return rows[0], rows[1:]
    if dataset == "hitab":
        table_content = raw.get("table_content", {})
        texts = table_content.get("texts", []) if isinstance(table_content, Mapping) else []
        if not texts:
            return [], []
        return list(texts[0] or []), list(texts[1:] or [])
    table = raw.get("table", {}) if isinstance(raw.get("table", {}), Mapping) else {}
    return list(table.get("header", table.get("headers", [])) or []), list(table.get("rows", []) or [])


def bin_count(n: int, cuts: tuple[int, ...]) -> str:
    previous = 0
    for cut in cuts:
        if n <= cut:
            return f"{previous + 1}_{cut}" if previous else f"0_{cut}"
        previous = cut
    return f"gt_{cuts[-1]}"


def numeric_like(value: str) -> bool:
    cleaned = re.sub(r"[$£€¥,%]", "", value.strip())
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned))


def split_range_and_score_atoms(flat_text: str, has_score_col: bool) -> tuple[str, str]:
    """Split score-like surfaces from generic numeric ranges.

    This is intentionally conservative: score surfaces require either a score-like
    header or an AFL-style value such as ``17.9 (111)``. Date/year ranges can
    still fire ``cell.has_numeric_range``.
    """
    has_span = bool(NUMERIC_SPAN_PATTERN.search(flat_text))
    has_afl_score = bool(AFL_SCORE_PATTERN.search(flat_text))
    has_score_surface = has_afl_score or (has_score_col and has_span)
    has_numeric_range = has_span and not (has_score_col and not DATE_LIKE_PATTERN.search(flat_text))
    return "yes" if has_numeric_range else "no", "yes" if has_score_surface else "no"


def canonical_atoms(meta: dict[str, Any], dataset: str) -> dict[str, str]:
    """Return registry v1.2 canonical atoms for one query.

    All returned values are strings so they can be converted into indicator
    atoms like ``atom=value`` without ambiguity.
    """
    raw = as_meta(meta.get("_raw", {}))
    text = question_text(raw, dataset)
    headers, rows = flatten_table(raw, dataset)
    header_text = " ".join(map(str, headers))
    visible_cells = [str(cell).strip() for row in rows[:80] for cell in list(row)[:50]]
    flat_text = " ".join(visible_cells)

    out: dict[str, str] = {}
    for atom, pattern in TEXT_PATTERNS.items():
        out[atom] = "yes" if pattern.search(text) else "no"

    normalized_headers = [str(header).strip().lower() for header in headers]
    for atom, pattern in HEADER_PATTERNS.items():
        out[atom] = "yes" if any(pattern.search(str(header)) for header in headers) else "no"
    out["schema.header_repetition_marker"] = "yes" if len(normalized_headers) != len(set(normalized_headers)) else "no"

    n_rows = len(rows)
    n_cols = len(headers)
    out["table.rows_bin"] = bin_count(n_rows, (5, 10, 25, 50))
    out["table.cols_bin"] = bin_count(n_cols, (3, 6, 10))
    if not n_rows or not n_cols:
        out["table.shape"] = "empty"
    else:
        ratio = n_cols / n_rows
        out["table.shape"] = "wide" if ratio > 0.5 else ("tall" if ratio < 0.1 else "balanced")

    numeric_cols = 0
    for ci in range(n_cols):
        vals = [
            str(rows[ri][ci]).strip()
            for ri in range(min(n_rows, 50))
            if ci < len(rows[ri]) and str(rows[ri][ci]).strip()
        ]
        if vals and sum(numeric_like(v) for v in vals) / len(vals) > 0.5:
            numeric_cols += 1
    out["table.numeric_cols_bin"] = "0" if numeric_cols == 0 else ("1" if numeric_cols == 1 else ("2_3" if numeric_cols <= 3 else "ge_4"))
    if not n_cols or numeric_cols == 0:
        out["table.numeric_density_bin"] = "none"
    else:
        density = numeric_cols / n_cols
        out["table.numeric_density_bin"] = "low" if density < 0.25 else ("mid" if density < 0.6 else "high")

    out["cell.has_comma_number"] = "yes" if COMMA_NUMBER_PATTERN.search(flat_text) else "no"
    out["cell.has_percent"] = "yes" if PERCENT_PATTERN.search(flat_text) else "no"
    out["cell.has_currency"] = "yes" if CURRENCY_PATTERN.search(flat_text) else "no"
    out["cell.has_date_like"] = "yes" if DATE_LIKE_PATTERN.search(flat_text) else "no"
    out["cell.missing_value_marker"] = "yes" if any(cell.strip().lower() in MISSING_MARKERS for cell in visible_cells) else "no"
    numeric_range, score_surface = split_range_and_score_atoms(flat_text, out["schema.has_score_col"] == "yes")
    out["cell.has_numeric_range"] = numeric_range
    out["cell.has_score_surface"] = score_surface

    question_tokens = tokens(text)
    out["grounding.header_overlap"] = "yes" if question_tokens & tokens(header_text) else "no"
    out["grounding.cell_overlap"] = "yes" if question_tokens & tokens(flat_text) else "no"

    if dataset == "sqa":
        position = int(raw.get("position", meta.get("position", 0)) or 0)
        out["dialog.turn_bin"] = "first" if position == 0 else ("early" if position <= 2 else "late")
        out["dialog.has_reference_marker"] = "yes" if DIALOG_REFERENCE_PATTERN.search(text) else "no"
    else:
        out["dialog.turn_bin"] = "na"
        out["dialog.has_reference_marker"] = "na"

    if dataset == "tablebench":
        out["native.tablebench_qtype"] = str(raw.get("qtype", meta.get("qtype", "unknown")) or "unknown")
        out["native.tablebench_qsubtype"] = str(raw.get("qsubtype", meta.get("qsubtype", "unknown")) or "unknown")
    if dataset == "hitab":
        out["native.hitab_agg_type"] = str(raw.get("aggregation", meta.get("aggregation", "unknown")) or "unknown")
        out["native.hitab_source_family"] = str(raw.get("table_source", meta.get("table_source", "unknown")) or "unknown")

    return out


def indicator_atoms(atoms: dict[str, str], registry_atoms: dict[str, dict[str, Any]]) -> set[str]:
    """Convert raw atom values into binary indicator strings for mining."""
    indicators: set[str] = set()
    for atom, value in atoms.items():
        spec = registry_atoms.get(atom, {})
        if spec.get("value_type") == "binary":
            if value == "yes":
                indicators.add(atom)
        elif value not in {"", "na", "unknown"}:
            indicators.add(f"{atom}={value}")
    return indicators
