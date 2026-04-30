"""NL2SQL loaders — seed BIRD and Spider queries into the unified CubeStore.

Loads queries directly from JSON files (no transfer_pipeline dependency).
Task-specific fields (db_id, gold_sql, difficulty, evidence, db_path, schema)
are stored in query.meta._raw for downstream evaluation.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from core.schema import make_query_id
from core.store import CubeStore, OnConflict

logger = logging.getLogger(__name__)


def _build_schema_ddl(db_path: str) -> str:
    """Build CREATE TABLE DDL text from a sqlite DB via PRAGMA introspection.

    Args:
        db_path: Absolute path to the .sqlite file.

    Returns:
        Multi-line DDL string with one CREATE TABLE block per table.
        Returns empty string if the file cannot be opened.

    Raises:
        FileNotFoundError: If db_path does not exist.
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {db_path}")

    uri = f"file:{db_path}?mode=ro"
    ddl_parts: List[str] = []

    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()

        for table_row in tables:
            table_name = table_row["name"]
            # PRAGMA does not support ? placeholders; escape by doubling quotes.
            safe_name = table_name.replace('"', '""')
            columns = conn.execute(
                f'PRAGMA table_info("{safe_name}")'
            ).fetchall()
            # PRAGMA table_info may fail on system tables; skip if empty
            if not columns:
                continue

            col_defs: List[str] = []
            for col in columns:
                col_type = col["type"] or "TEXT"
                not_null = " NOT NULL" if col["notnull"] else ""
                pk = " PRIMARY KEY" if col["pk"] == 1 else ""
                col_defs.append(f"    `{col['name']}` {col_type}{not_null}{pk}")

            ddl_parts.append(
                f"CREATE TABLE `{table_name}` (\n"
                + ",\n".join(col_defs)
                + "\n);"
            )

    return "\n\n".join(ddl_parts)


def seed_queries_bird(
    store: CubeStore,
    bird_dir: str,
    split: str = "dev",
    *,
    max_queries: int = 0,
    sample_seed: int = 0,
    on_conflict: OnConflict = OnConflict.SKIP,
) -> int:
    """Load BIRD queries and seed into the unified store.

    Directory layout expected under bird_dir:
        dev/dev.json                           (split="dev")
        dev/dev_databases/{db_id}/{db_id}.sqlite
        train/train.json                       (split="train")
        train/train_databases/{db_id}/{db_id}.sqlite

    Each record in query.meta contains:
        db_id, gold_sql, difficulty, evidence, split,
        _raw (including db_path and schema DDL text).

    Args:
        store: Target CubeStore instance.
        bird_dir: Root directory containing dev/ and train/ subdirs.
        split: "dev" or "train".
        max_queries: Cap on number of queries loaded (0 = all).
        on_conflict: Conflict resolution policy for duplicate query_ids.

    Returns:
        Number of queries seeded.

    Raises:
        FileNotFoundError: If the JSON data file does not exist.
        ValueError: If split is not "dev" or "train".
    """
    if split not in ("dev", "train"):
        raise ValueError(f"Unsupported BIRD split '{split}'. Must be 'dev' or 'train'.")

    base = Path(bird_dir)

    if split == "dev":
        json_path = base / "dev" / "dev.json"
        db_root = base / "dev" / "dev_databases"
    else:
        json_path = base / "train" / "train.json"
        db_root = base / "train" / "train_databases"

    if not json_path.exists():
        raise FileNotFoundError(f"BIRD {split} data not found: {json_path}")

    with open(json_path) as f:
        raw_records: List[Dict[str, Any]] = json.load(f)

    if max_queries > 0 and max_queries < len(raw_records):
        if sample_seed > 0:
            import random
            rng = random.Random(sample_seed)
            raw_records = rng.sample(raw_records, max_queries)
        else:
            raw_records = raw_records[:max_queries]

    queries: List[Dict[str, Any]] = []
    schema_errors = 0

    for i, r in enumerate(raw_records):
        question = r.get("question", "")
        db_id = r.get("db_id", "")
        # question_id present in dev but absent in train; fall back to index
        qid = r.get("question_id", i)
        gold_sql = r.get("SQL", "")
        difficulty = r.get("difficulty", "")
        evidence = r.get("evidence", "")

        db_path = str(db_root / db_id / f"{db_id}.sqlite")

        try:
            schema_ddl = _build_schema_ddl(db_path)
        except FileNotFoundError:
            logger.warning("DB not found for query %s (db_id=%s): %s", qid, db_id, db_path)
            schema_ddl = ""
            schema_errors += 1

        query_id = make_query_id("bird", question, context=f"{split}:{db_id}:{qid}")
        queries.append({
            "query_id": query_id,
            "dataset": "bird",
            "content": question,
            "meta": {
                "db_id": db_id,
                "gold_sql": gold_sql,
                "difficulty": difficulty,
                "evidence": evidence,
                "split": split,
                "_raw": {
                    "question_id": qid,
                    "question": question,
                    "db_id": db_id,
                    "SQL": gold_sql,
                    "difficulty": difficulty,
                    "evidence": evidence,
                    "db_path": db_path,
                    "schema": schema_ddl,
                },
            },
        })

    store.upsert_queries(queries, on_conflict=on_conflict)
    if schema_errors:
        logger.warning(
            "%d/%d BIRD queries had missing DB files (split=%s)",
            schema_errors, len(queries), split,
        )
    logger.info("Seeded %d BIRD queries (split=%s)", len(queries), split)
    return len(queries)


def seed_queries_spider(
    store: CubeStore,
    spider_dir: str,
    split: str = "dev",
    *,
    max_queries: int = 0,
    sample_seed: int = 0,
    on_conflict: OnConflict = OnConflict.SKIP,
) -> int:
    """Load Spider queries and seed into the unified store.

    Directory layout expected under spider_dir:
        dev.json                               (split="dev")
        train_spider.json                      (split="train")
        database/{db_id}/{db_id}.sqlite

    Each record in query.meta contains:
        db_id, gold_sql, split,
        _raw (including db_path and schema DDL text).

    Args:
        store: Target CubeStore instance.
        spider_dir: Root directory of the Spider dataset.
        split: "dev" or "train".
        max_queries: Cap on number of queries loaded (0 = all).
        on_conflict: Conflict resolution policy for duplicate query_ids.

    Returns:
        Number of queries seeded.

    Raises:
        FileNotFoundError: If the JSON data file does not exist.
        ValueError: If split is not "dev" or "train".
    """
    if split not in ("dev", "train"):
        raise ValueError(f"Unsupported Spider split '{split}'. Must be 'dev' or 'train'.")

    base = Path(spider_dir)

    json_filename = "dev.json" if split == "dev" else "train_spider.json"
    json_path = base / json_filename
    db_root = base / "database"

    if not json_path.exists():
        raise FileNotFoundError(f"Spider {split} data not found: {json_path}")

    with open(json_path) as f:
        raw_records: List[Dict[str, Any]] = json.load(f)

    if max_queries > 0 and max_queries < len(raw_records):
        if sample_seed > 0:
            import random
            rng = random.Random(sample_seed)
            raw_records = rng.sample(raw_records, max_queries)
        else:
            raw_records = raw_records[:max_queries]

    queries: List[Dict[str, Any]] = []
    schema_errors = 0

    for i, r in enumerate(raw_records):
        question = r.get("question", "")
        db_id = r.get("db_id", "")
        gold_sql = r.get("query", "")

        db_path = str(db_root / db_id / f"{db_id}.sqlite")

        try:
            schema_ddl = _build_schema_ddl(db_path)
        except FileNotFoundError:
            logger.warning("DB not found for Spider query %d (db_id=%s): %s", i, db_id, db_path)
            schema_ddl = ""
            schema_errors += 1

        query_id = make_query_id("spider", question, context=f"{split}:{db_id}:{i}")
        queries.append({
            "query_id": query_id,
            "dataset": "spider",
            "content": question,
            "meta": {
                "db_id": db_id,
                "gold_sql": gold_sql,
                "split": split,
                "_raw": {
                    "question": question,
                    "db_id": db_id,
                    "query": gold_sql,
                    "db_path": db_path,
                    "schema": schema_ddl,
                },
            },
        })

    store.upsert_queries(queries, on_conflict=on_conflict)
    if schema_errors:
        logger.warning(
            "%d/%d Spider queries had missing DB files (split=%s)",
            schema_errors, len(queries), split,
        )
    logger.info("Seeded %d Spider queries (split=%s)", len(queries), split)
    return len(queries)
