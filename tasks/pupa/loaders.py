"""PUPA/PAPILLON local data loader.

The loader intentionally expects a local JSON/JSONL file. This keeps the task
reproducible and avoids silently depending on a remote dataset layout.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

from core.schema import make_query_id
from core.store import CubeStore, OnConflict

logger = logging.getLogger(__name__)


def seed_queries_pupa(
    store: CubeStore,
    data_path: str | Path,
    split: str,
    *,
    max_queries: int = 0,
    on_conflict: OnConflict = OnConflict.SKIP,
) -> int:
    """Seed PUPA/PAPILLON queries from a local JSON or JSONL file.

    Accepted row fields are intentionally permissive because public/local
    exports vary: ``user_query``/``query``/``question``/``input`` for the
    private request, optional ``reference``/``answer``/``target`` for scoring,
    and optional ``forbidden_terms``/``private_terms``/``sensitive_terms`` for
    leakage checks.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"PUPA data file not found: {path}")

    queries: List[Dict[str, Any]] = []
    for row in _iter_rows(path):
        row_split = str(row.get("split") or split)
        if row_split != split:
            continue

        user_query = _first_text(row, ["user_query", "query", "question", "input", "prompt"])
        if not user_query:
            raise ValueError("PUPA row missing user query field")

        source_id = str(row.get("query_id") or row.get("id") or row.get("uid") or len(queries))
        reference = _first_text(row, ["reference", "target", "answer", "response", "gold", "gold_response"])
        forbidden_terms = _list_field(row, ["forbidden_terms", "private_terms", "sensitive_terms"])

        meta: Dict[str, Any] = {
            "split": split,
            "source_id": source_id,
            "_raw": row,
        }
        if reference:
            meta["reference"] = reference
        if forbidden_terms:
            meta["forbidden_terms"] = forbidden_terms

        query_id = str(row.get("query_id") or make_query_id("pupa", user_query, context=f"{split}:{source_id}"))
        queries.append({
            "query_id": query_id,
            "dataset": "pupa",
            "content": user_query,
            "meta": meta,
        })
        if max_queries > 0 and len(queries) >= max_queries:
            break

    store.upsert_queries(queries, on_conflict=on_conflict)
    logger.info("Seeded %d PUPA queries (split=%s)", len(queries), split)
    return len(queries)


def _iter_rows(path: Path) -> Iterable[dict]:
    if path.suffix.lower() == ".jsonl":
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("PUPA JSONL rows must be objects")
                    yield row
        return

    with path.open() as f:
        payload = json.load(f)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("data") or payload.get("examples") or payload.get("rows")
        if rows is None:
            rows = [
                row
                for value in payload.values()
                if isinstance(value, list)
                for row in value
            ]
    else:
        raise ValueError("PUPA JSON file must contain a list or object")

    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("PUPA JSON rows must be objects")
        yield row


def _first_text(row: dict, keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _list_field(row: dict, keys: list[str]) -> list[str]:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if str(value).strip():
            return [str(value).strip()]
    return []

