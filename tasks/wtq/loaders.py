"""WTQ loaders — seed WikiTableQuestions queries into the unified CubeStore."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from core.schema import make_query_id
from core.store import CubeStore, OnConflict

logger = logging.getLogger(__name__)


def seed_queries_wtq(
    store: CubeStore,
    split: str,
    *,
    max_queries: int = 0,
    on_conflict: OnConflict = OnConflict.SKIP,
) -> int:
    """Load WikiTableQuestions and seed queries into the unified store.

    Args:
        split: HuggingFace split name ("train", "validation", "test").
        max_queries: Limit number of queries (0 = all).
    """
    from datasets import load_dataset

    ds = load_dataset(
        "stanfordnlp/wikitablequestions",
        name="random-split-1",
        split=split,
        trust_remote_code=True,
    )
    if max_queries > 0:
        ds = ds.select(range(min(max_queries, len(ds))))

    queries: List[Dict[str, Any]] = []
    for i, row in enumerate(ds):
        question = row["question"]
        table = row["table"]
        answers = row["answers"]

        query_id = make_query_id("wtq", question, context=f"{split}:{i}")
        queries.append({
            "query_id": query_id,
            "dataset": "wtq",
            "content": question,
            "meta": {
                "gold_answers": answers,
                "split": split,
                "table_name": table.get("name", ""),
                "_raw": {
                    "id": row["id"],
                    "question": question,
                    "answers": answers,
                    "table": {
                        "header": table["header"],
                        "rows": table["rows"],
                        "name": table.get("name", ""),
                    },
                },
            },
        })

    store.upsert_queries(queries, on_conflict=on_conflict)
    logger.info("Seeded %d WTQ queries (split=%s)", len(queries), split)
    return len(queries)


def table_to_markdown(header: List[str], rows: List[List[str]], name: str = "") -> str:
    """Convert WTQ table (header + rows) to markdown format."""
    if not header:
        return ""

    lines = []
    if name:
        # Extract readable name from path like "csv/204-csv/590.csv"
        lines.append(f"Table: {name}")

    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows:
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded[:len(header)]) + " |")

    return "\n".join(lines)
