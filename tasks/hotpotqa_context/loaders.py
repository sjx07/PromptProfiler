"""HotpotQA-context loader.

Rows can come from a local JSON/JSONL export or, when no data_path is supplied,
from Hugging Face's ``hotpot_qa/fullwiki`` dataset.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

from core.schema import make_query_id
from core.store import CubeStore, OnConflict

logger = logging.getLogger(__name__)


def seed_queries_hotpotqa_context(
    store: CubeStore,
    split: str,
    *,
    data_path: str | Path | None = None,
    max_queries: int = 0,
    on_conflict: OnConflict = OnConflict.SKIP,
) -> int:
    """Seed HotpotQA questions with their provided context passages."""
    queries: List[Dict[str, Any]] = []
    for row in _iter_rows(split, data_path):
        row_split = row.get("split")
        if row_split is not None and str(row_split) != split:
            continue

        question = _first_text(row, ["question", "query", "input"])
        if not question:
            raise ValueError("HotpotQA-context row missing question field")

        context = row.get("context") or row.get("contexts") or row.get("paragraphs")
        if context is None:
            raise ValueError("HotpotQA-context row missing context field")

        source_id = str(row.get("id") or row.get("_id") or row.get("query_id") or len(queries))
        answer = _first_text(row, ["answer", "gold", "target", "reference"])
        supporting_facts = row.get("supporting_facts")

        meta: Dict[str, Any] = {
            "split": split,
            "source_id": source_id,
            "context": _json_safe(context),
            "_raw": _json_safe(row),
        }
        if answer:
            meta["answer"] = answer
        if supporting_facts is not None:
            meta["supporting_facts"] = _json_safe(supporting_facts)
        if row.get("level") is not None:
            meta["level"] = row.get("level")
        if row.get("type") is not None:
            meta["type"] = row.get("type")

        query_id = str(
            row.get("query_id")
            or make_query_id("hotpotqa_context", question, context=f"{split}:{source_id}")
        )
        queries.append({
            "query_id": query_id,
            "dataset": "hotpotqa_context",
            "content": question,
            "meta": meta,
        })
        if max_queries > 0 and len(queries) >= max_queries:
            break

    store.upsert_queries(queries, on_conflict=on_conflict)
    logger.info("Seeded %d HotpotQA-context queries (split=%s)", len(queries), split)
    return len(queries)


def _iter_rows(split: str, data_path: str | Path | None) -> Iterable[dict]:
    if data_path:
        yield from _iter_local_rows(Path(data_path))
        return

    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "fullwiki", split=split, trust_remote_code=True)
    for row in ds:
        yield dict(row)


def _iter_local_rows(path: Path) -> Iterable[dict]:
    if not path.exists():
        raise FileNotFoundError(f"HotpotQA-context data file not found: {path}")

    if path.suffix.lower() == ".jsonl":
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("HotpotQA-context JSONL rows must be objects")
                yield row
        return

    with path.open() as f:
        payload = json.load(f)

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and "question" in payload:
        rows = [payload]
    elif isinstance(payload, dict):
        rows = (
            payload.get("data")
            or payload.get("examples")
            or payload.get("rows")
            or [row for value in payload.values() if isinstance(value, list) for row in value]
        )
    else:
        raise ValueError("HotpotQA-context JSON file must contain a list or object")

    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("HotpotQA-context JSON rows must be objects")
        yield row


def _first_text(row: dict, keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        return str(value)
