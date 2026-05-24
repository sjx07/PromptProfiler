"""Canonical context attributes for FACET analysis.

The cube's ``predicate`` table remains the raw, task-local extractor output.
This module provides an offline canonicalization layer that maps query metadata
into a smaller, shared context vocabulary for cross-benchmark mining.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


DATASETS = ("wtq", "sqa", "tablebench", "tab_fact", "hitab")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "their", "there", "this", "to", "was",
    "were", "what", "when", "where", "which", "who", "with",
}


TEXT_PATTERNS = {
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
    "intent.arithmetic_op_marker": re.compile(
        r"\b(difference|subtract|add|multiply|divide|how much more|how much less|"
        r"how many more|how many fewer)\b",
        re.I,
    ),
    "intent.percent_or_ratio": re.compile(r"\b(ratio|percent|percentage)\b|%", re.I),
    "intent.reduction_marker": re.compile(r"\b(total|sum|average|mean|combined|altogether)\b", re.I),
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
    "dialog.has_reference_marker": re.compile(
        r"\b(those|that|these|them|they|it|its|their|the same|which one|which ones|above|previous|mentioned)\b",
        re.I,
    ),
}

HEADER_PATTERNS = {
    "schema.has_date_col": re.compile(r"\b(date|year|month|day|season|week|time)\b", re.I),
    "schema.has_rank_col": re.compile(r"^(#|rank|place|position|seed|round)$|\brank\b", re.I),
    "schema.has_score_col": re.compile(r"\b(score|points?|goals?|result|won|lost|drawn|difference|bonus)\b", re.I),
    "schema.has_unit_col": re.compile(
        r"[%$]|\b(km|kg|miles?|area|population|density|rate|percent|percentage)\b",
        re.I,
    ),
    "schema.has_entity_col": re.compile(
        r"\b(name|player|team|country|city|club|person|film|title|municipality|province|state|school|company)\b",
        re.I,
    ),
}

CELL_PATTERNS = {
    "cell.has_comma_number": re.compile(r"\b\d{1,3}(?:,\d{3})+\b"),
    "cell.has_percent": re.compile(r"\d+(?:\.\d+)?\s*%"),
    "cell.has_range_or_score": re.compile(r"\b\d+\s*[-\u2013]\s*\d+\b|\b\d+\.\d+\s*\(\d+\)"),
    "cell.has_date_like": re.compile(
        r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b|"
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b",
        re.I,
    ),
}

SHARED_ATTRIBUTES = frozenset(
    {
        "intent.count",
        "intent.superlative_rank",
        "intent.comparison",
        "intent.temporal",
        "intent.negation_marker",
        "intent.reduction_marker",
        "intent.arithmetic_op_marker",
        "intent.percent_or_ratio",
        "text.has_number",
        "text.has_year",
        "text.has_ordinal",
        "schema.has_date_col",
        "schema.has_entity_col",
        "schema.has_score_col",
        "schema.has_unit_col",
        "schema.has_rank_col",
        "schema.header_repetition_marker",
        "table.rows_bin",
        "table.cols_bin",
        "table.numeric_cols_bin",
        "table.numeric_density_bin",
        "table.shape",
        "cell.has_date_like",
        "cell.has_percent",
        "cell.has_range_or_score",
        "cell.has_comma_number",
        "cell.missing_value_marker",
        "grounding.header_overlap",
        "grounding.cell_overlap",
    }
)

SCOPED_ATTRIBUTES = frozenset(
    {
        "dialog.turn_bin",
        "dialog.has_reference_marker",
        "native.tablebench_qtype",
        "native.tablebench_qsubtype",
        "native.hitab_agg_type",
        "native.hitab_source_family",
    }
)


@dataclass(frozen=True)
class ContextAtomRecord:
    """Normalized context atom plus provenance metadata."""

    query_id: str
    dataset: str
    split: str | None
    atom_name: str
    atom_value: str
    canonical_atom: str
    family: str
    view: str
    reliability: str
    extractor_kind: str


def load_json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return json.loads(value)
    return value or {}


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    }


def flatten_table(raw: Mapping[str, Any], dataset: str) -> tuple[list[str], list[list[Any]]]:
    """Return a header/rows view from task-local query metadata."""
    if dataset == "sqa":
        table = raw.get("table", {})
        if isinstance(table, Mapping):
            return list(table.get("headers", table.get("header", [])) or []), list(table.get("rows", []) or [])
        return [], []
    if dataset == "tab_fact":
        table_text = raw.get("table_text", "")
        rows = []
        for line in str(table_text).splitlines():
            parts = [part for part in line.split("#") if part != ""]
            if parts:
                rows.append(parts)
        return (rows[0], rows[1:]) if rows else ([], [])
    if dataset == "hitab":
        table_content = raw.get("table_content", {})
        texts = table_content.get("texts", []) if isinstance(table_content, Mapping) else []
        return (list(texts[0] or []), list(texts[1:] or [])) if texts else ([], [])
    table = raw.get("table", {})
    if isinstance(table, Mapping):
        return list(table.get("header", table.get("headers", [])) or []), list(table.get("rows", []) or [])
    return [], []


def query_text(raw: Mapping[str, Any], dataset: str) -> str:
    if dataset == "tab_fact":
        return str(raw.get("statement", "") or "")
    return str(raw.get("question", "") or "")


def bin_count(n: int, cuts: tuple[int, ...]) -> str:
    prev = 0
    for cut in cuts:
        if n <= cut:
            return f"{prev + 1}_{cut}" if prev else f"0_{cut}"
        prev = cut
    return f"gt_{cuts[-1]}"


def numeric_like(value: str) -> bool:
    cleaned = re.sub(r"[$,%]", "", value.strip())
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned))


def canonical_context_values(meta: Mapping[str, Any] | str, dataset: str) -> dict[str, str]:
    """Compute canonical, input-only context attributes for one query."""
    meta_obj = load_json_obj(meta)
    raw = load_json_obj(meta_obj.get("_raw", {}))
    text = query_text(raw, dataset)
    header, rows = flatten_table(raw, dataset)
    header_text = " ".join(map(str, header))
    flat_cells = [str(cell).strip() for row in rows[:80] for cell in list(row)[:50]]
    flat_text = " ".join(flat_cells)

    out: dict[str, str] = {}

    for name, pattern in TEXT_PATTERNS.items():
        if name.startswith("dialog."):
            continue
        out[name] = "yes" if pattern.search(text) else "no"

    norm_headers = [str(h).strip().lower() for h in header]
    for name, pattern in HEADER_PATTERNS.items():
        out[name] = "yes" if any(pattern.search(str(h)) for h in header) else "no"
    out["schema.header_repetition_marker"] = "yes" if len(norm_headers) != len(set(norm_headers)) else "no"

    n_rows = len(rows)
    n_cols = len(header)
    out["table.rows_bin"] = bin_count(n_rows, (5, 10, 25, 50))
    out["table.cols_bin"] = bin_count(n_cols, (3, 6, 10))
    out["table.shape"] = "empty"
    if n_rows and n_cols:
        ratio = n_cols / n_rows
        out["table.shape"] = "wide" if ratio > 0.5 else ("tall" if ratio < 0.1 else "balanced")

    numeric_cols = 0
    for col_idx in range(n_cols):
        vals = [
            str(rows[row_idx][col_idx]).strip()
            for row_idx in range(min(n_rows, 50))
            if col_idx < len(rows[row_idx]) and str(rows[row_idx][col_idx]).strip()
        ]
        if vals and sum(numeric_like(v) for v in vals) / len(vals) > 0.5:
            numeric_cols += 1
    out["table.numeric_cols_bin"] = (
        "0" if numeric_cols == 0 else "1" if numeric_cols == 1 else "2_3" if numeric_cols <= 3 else "ge_4"
    )
    density = numeric_cols / n_cols if n_cols else 0.0
    out["table.numeric_density_bin"] = (
        "none" if density == 0 else "low" if density < 0.25 else "mid" if density < 0.6 else "high"
    )

    out["cell.missing_value_marker"] = (
        "yes" if any(str(cell).strip() == "" for row in rows[:80] for cell in list(row)[:50]) else "no"
    )
    for name, pattern in CELL_PATTERNS.items():
        out[name] = "yes" if pattern.search(flat_text) else "no"

    text_tokens = tokenize(text)
    out["grounding.header_overlap"] = "yes" if text_tokens & tokenize(header_text) else "no"
    out["grounding.cell_overlap"] = "yes" if text_tokens & tokenize(flat_text) else "no"

    if dataset == "sqa":
        pos = int(raw.get("position", meta_obj.get("position", 0)) or 0)
        out["dialog.turn_bin"] = "first" if pos == 0 else "early" if pos <= 2 else "late"
        out["dialog.has_reference_marker"] = "yes" if TEXT_PATTERNS["dialog.has_reference_marker"].search(text) else "no"
    else:
        out["dialog.turn_bin"] = "na"
        out["dialog.has_reference_marker"] = "na"

    if dataset == "tablebench":
        out["native.tablebench_qtype"] = str(raw.get("qtype", meta_obj.get("qtype", "unknown")) or "unknown")
        out["native.tablebench_qsubtype"] = str(raw.get("qsubtype", meta_obj.get("qsubtype", "unknown")) or "unknown")
    elif dataset == "hitab":
        out["native.hitab_agg_type"] = str(raw.get("aggregation", meta_obj.get("aggregation", "unknown")) or "unknown")
        out["native.hitab_source_family"] = str(raw.get("table_source", meta_obj.get("table_source", "unknown")) or "unknown")

    return out


def attribute_view(name: str) -> str:
    if name in SHARED_ATTRIBUTES:
        return "shared"
    if name in SCOPED_ATTRIBUTES:
        return "scoped"
    return "debug"


def attribute_family(name: str) -> str:
    return name.split(".", 1)[0]


def attribute_reliability(name: str) -> str:
    if name in {"intent.negation_marker", "intent.reduction_marker", "schema.has_rank_col", "schema.header_repetition_marker"}:
        return "medium"
    if name.startswith("native."):
        return "native"
    return "high"


def extractor_kind(name: str) -> str:
    if name.startswith("table."):
        return "table_scan"
    if name.startswith("schema.") or name.startswith("cell.") or name.startswith("grounding."):
        return "surface_scan"
    if name.startswith("dialog.") or name.startswith("native."):
        return "metadata_or_regex"
    return "regex"


def canonical_context_items(
    meta: Mapping[str, Any] | str,
    dataset: str,
    *,
    view: str = "shared",
    include_negative: bool = True,
    include_na: bool = False,
) -> tuple[str, ...]:
    """Return sorted ``name=value`` atoms for mining."""
    values = canonical_context_values(meta, dataset)
    items = []
    for name, value in values.items():
        atom_view = attribute_view(name)
        if view == "shared" and atom_view != "shared":
            continue
        if view == "scoped" and atom_view not in {"shared", "scoped"}:
            continue
        if view not in {"shared", "scoped", "all"}:
            raise ValueError(f"unknown context view: {view}")
        if value == "na" and not include_na:
            continue
        if value == "no" and not include_negative:
            continue
        items.append(f"{name}={value}")
    return tuple(sorted(items))


def canonical_context_records(
    query_id: str,
    dataset: str,
    split: str | None,
    meta: Mapping[str, Any] | str,
    *,
    view: str = "shared",
    include_negative: bool = True,
    include_na: bool = False,
) -> tuple[ContextAtomRecord, ...]:
    values = canonical_context_values(meta, dataset)
    records: list[ContextAtomRecord] = []
    for name, value in values.items():
        atom_view = attribute_view(name)
        if view == "shared" and atom_view != "shared":
            continue
        if view == "scoped" and atom_view not in {"shared", "scoped"}:
            continue
        if view not in {"shared", "scoped", "all"}:
            raise ValueError(f"unknown context view: {view}")
        if value == "na" and not include_na:
            continue
        if value == "no" and not include_negative:
            continue
        records.append(
            ContextAtomRecord(
                query_id=query_id,
                dataset=dataset,
                split=split,
                atom_name=name,
                atom_value=value,
                canonical_atom=f"{name}={value}",
                family=attribute_family(name),
                view=atom_view,
                reliability=attribute_reliability(name),
                extractor_kind=extractor_kind(name),
            )
        )
    return tuple(records)


def atom_set_from_values(
    values: Mapping[str, str],
    *,
    view: str = "shared",
    include_negative: bool = True,
    include_na: bool = False,
) -> frozenset[str]:
    items = []
    for name, value in values.items():
        atom_view = attribute_view(name)
        if view == "shared" and atom_view != "shared":
            continue
        if view == "scoped" and atom_view not in {"shared", "scoped"}:
            continue
        if value == "na" and not include_na:
            continue
        if value == "no" and not include_negative:
            continue
        items.append(f"{name}={value}")
    return frozenset(items)


def collect_atom_names(records: Iterable[ContextAtomRecord]) -> tuple[str, ...]:
    return tuple(sorted({record.canonical_atom for record in records}))
