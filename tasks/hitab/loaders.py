"""HiTab loaders for hierarchical table QA."""
from __future__ import annotations

import ast
import json
import logging
from typing import Any, Dict, List

from core.schema import make_query_id
from core.store import CubeStore, OnConflict

logger = logging.getLogger(__name__)


def _parse_str_field(raw_val: Any, default: Any) -> Any:
    if not isinstance(raw_val, str):
        return raw_val if raw_val is not None else default
    raw_val = raw_val.strip()
    if not raw_val:
        return default
    try:
        return ast.literal_eval(raw_val)
    except (ValueError, SyntaxError):
        pass
    try:
        return json.loads(raw_val)
    except (json.JSONDecodeError, TypeError):
        return default


def seed_queries_hitab(
    store: CubeStore,
    split: str,
    *,
    max_queries: int = 0,
    sample_seed: int = 0,
    on_conflict: OnConflict = OnConflict.SKIP,
) -> int:
    """Load HiTab from Hugging Face and seed queries."""
    from datasets import load_dataset

    ds = load_dataset("kasnerz/hitab", split=split, trust_remote_code=True)
    if max_queries > 0 and max_queries < len(ds):
        if sample_seed > 0:
            ds = ds.shuffle(seed=sample_seed)
        ds = ds.select(range(max_queries))

    queries: List[Dict[str, Any]] = []
    for i, row in enumerate(ds):
        question = row["question"]
        answer = row["answer"]
        table_content = _parse_str_field(row.get("table_content", "{}"), {})
        linked_cells = _parse_str_field(row.get("linked_cells", "{}"), {})
        answer_formulas = _parse_str_field(row.get("answer_formulas", "[]"), [])

        query_id = make_query_id("hitab", question, context=f"{split}:{i}")
        queries.append({
            "query_id": query_id,
            "dataset": "hitab",
            "content": question,
            "meta": {
                "gold_answer": answer,
                "split": split,
                "table_id": row.get("table_id", ""),
                "table_source": row.get("table_source", ""),
                "aggregation": row.get("aggregation", "none"),
                "_raw": {
                    "id": row.get("id", ""),
                    "question": question,
                    "answer": answer,
                    "aggregation": row.get("aggregation", "none"),
                    "table_id": row.get("table_id", ""),
                    "table_source": row.get("table_source", ""),
                    "sentence_id": row.get("sentence_id", ""),
                    "sub_sentence_id": row.get("sub_sentence_id", ""),
                    "sub_sentence": row.get("sub_sentence", ""),
                    "answer_formulas": answer_formulas,
                    "reference_cells_map": _parse_str_field(row.get("reference_cells_map", "{}"), {}),
                    "table_content": table_content,
                    "linked_cells": linked_cells,
                },
            },
        })

    store.upsert_queries(queries, on_conflict=on_conflict)
    logger.info("Seeded %d HiTab queries (split=%s)", len(queries), split)
    return len(queries)


def table_content_to_grid(table_content: dict) -> tuple[str, list[list[str]], int]:
    """Return title, merged-cell-propagated grid, and header row count."""
    if not table_content:
        return "", [], 0

    title = table_content.get("title", "")
    texts = table_content.get("texts", [])
    if not texts:
        return title, [], 0

    n_rows = len(texts)
    n_cols = max((len(row) for row in texts), default=0)
    grid = []
    for row in texts:
        padded = [str(c) for c in row] + [""] * (n_cols - len(row))
        grid.append(padded[:n_cols])

    for region in table_content.get("merged_regions", []):
        r0 = region["first_row"]
        r1 = min(region["last_row"], n_rows - 1)
        c0 = region["first_column"]
        c1 = min(region["last_column"], n_cols - 1)
        value = grid[r0][c0] if r0 < n_rows and c0 < n_cols else ""
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if not grid[r][c].strip():
                    grid[r][c] = value

    top_header_n = min(int(table_content.get("top_header_rows_num", 1) or 1), n_rows)
    return title, grid, top_header_n


def table_content_to_markdown(table_content: dict) -> str:
    title, grid, top_header_n = table_content_to_grid(table_content)
    if not grid:
        return title or ""

    n_cols = max((len(row) for row in grid), default=0)
    lines = []
    if title:
        lines.append(f"Table: {title}")
    for i in range(top_header_n):
        lines.append("| " + " | ".join(grid[i]) + " |")
    lines.append("| " + " | ".join("---" for _ in range(n_cols)) + " |")
    for i in range(top_header_n, len(grid)):
        lines.append("| " + " | ".join(grid[i]) + " |")
    return "\n".join(lines)


def table_content_to_records(table_content: dict) -> tuple[list[str], list[list[str]]]:
    """Flatten hierarchical headers into DataFrame-ready columns and rows."""
    _title, grid, top_header_n = table_content_to_grid(table_content)
    if not grid:
        return [], []

    n_cols = max((len(row) for row in grid), default=0)
    headers: list[str] = []
    for c in range(n_cols):
        parts: list[str] = []
        for r in range(top_header_n):
            value = grid[r][c].strip() if c < len(grid[r]) else ""
            if value and value not in parts:
                parts.append(value)
        headers.append(" / ".join(parts) if parts else f"col_{c + 1}")

    rows = [row + [""] * (n_cols - len(row)) for row in grid[top_header_n:]]
    return headers, [row[:n_cols] for row in rows]
