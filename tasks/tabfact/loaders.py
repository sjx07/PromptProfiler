"""TabFact loaders — seed fact verification queries into the unified CubeStore."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from prompt_profiler.core.schema import make_query_id
from prompt_profiler.core.store import CubeStore, OnConflict

logger = logging.getLogger(__name__)


def seed_queries_tabfact(
    store: CubeStore,
    split: str,
    *,
    max_queries: int = 0,
    on_conflict: OnConflict = OnConflict.SKIP,
) -> int:
    """Load TabFact dataset and seed queries into the unified store.

    Args:
        split: HuggingFace split name ("train", "validation", "test").
        max_queries: Limit number of queries (0 = all).
    """
    from datasets import load_dataset

    ds = load_dataset("tab_fact", "tab_fact", split=split, trust_remote_code=True)
    if max_queries > 0:
        ds = ds.select(range(min(max_queries, len(ds))))

    queries: List[Dict[str, Any]] = []
    for i, row in enumerate(ds):
        table_text = row["table_text"]
        statement = row["statement"]

        query_id = make_query_id("tab_fact", statement, context=f"{split}:{i}")
        queries.append({
            "query_id": query_id,
            "dataset": "tab_fact",
            "content": statement,
            "meta": {
                "gold_label": row["label"],
                "split": split,
                "table_id": row.get("table_id", ""),
                "table_caption": row.get("table_caption", ""),
                "_raw": {
                    "table_text": table_text,
                    "statement": statement,
                    "label": row["label"],
                    "table_id": row.get("table_id", ""),
                    "table_caption": row.get("table_caption", ""),
                },
            },
        })

    store.upsert_queries(queries, on_conflict=on_conflict)
    logger.info("Seeded %d TabFact queries (split=%s)", len(queries), split)
    return len(queries)


def parse_table_text(table_text: str) -> Dict[str, Any]:
    """Parse TabFact's table_text format into structured data.

    Format: columns separated by #, rows separated by newlines.
    First row is the header.

    Returns:
        {"headers": [...], "rows": [[...], ...], "n_rows": int, "n_cols": int}
    """
    lines = [line.strip() for line in table_text.strip().split("\n") if line.strip()]
    if not lines:
        return {"headers": [], "rows": [], "n_rows": 0, "n_cols": 0}

    headers = [h.strip() for h in lines[0].split("#")]
    rows = []
    for line in lines[1:]:
        cells = [c.strip() for c in line.split("#")]
        rows.append(cells)

    return {
        "headers": headers,
        "rows": rows,
        "n_rows": len(rows),
        "n_cols": len(headers),
    }
