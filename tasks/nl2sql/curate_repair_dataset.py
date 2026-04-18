"""Curate SQL repair dataset from existing experiment DBs.

Reads wrong SQL predictions from multiple experiment DBs, deduplicates,
recovers metadata, and seeds into a new CubeStore for the repair task.

Usage:
    python -m tasks.nl2sql.curate_repair_dataset \
        --source_dbs /path/to/db1.db /path/to/db2.db \
        --output_db /path/to/sql_repair.db \
        --scorer ex_acc
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── SQL extraction (mirrors sql_generation, no import to stay standalone) ──

_MARKDOWN_FENCE_RE = re.compile(r"^```[a-z]*\n?", re.IGNORECASE | re.MULTILINE)


def _extract_sql_from_prediction(prediction: str) -> str:
    """Extract SQL from a raw model prediction string."""
    if not prediction:
        return ""
    # Try JSON with sql_query key
    try:
        parsed = json.loads(prediction)
        if isinstance(parsed, dict) and "sql_query" in parsed:
            return parsed["sql_query"].strip()
    except (json.JSONDecodeError, TypeError, ValueError):
        m = re.search(r'"sql_query"\s*:\s*"((?:[^"\\]|\\.)*)"', prediction, re.DOTALL)
        if m:
            return m.group(1).replace("\\'", "'").replace('\\"', '"').strip()

    # Try markdown code block
    lines = prediction.strip().splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```"):
        for end in range(len(lines) - 1, 0, -1):
            if lines[end].strip() == "```":
                inner = "\n".join(lines[1:end]).strip()
                # Inner might itself be JSON
                try:
                    parsed = json.loads(inner)
                    if isinstance(parsed, dict) and "sql_query" in parsed:
                        return parsed["sql_query"].strip()
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
                return inner

    return prediction


def _normalize_sql(sql: str) -> str:
    """Normalize SQL for deduplication: lowercase, collapse whitespace, strip semicolons."""
    return re.sub(r"\s+", " ", sql.lower().strip().rstrip(";")).strip()


# ── DB path resolution ─────────────────────────────────────────────────

_BIRD_DB_ROOTS = [
    "/data/bird/dev_20240627/dev_databases",
    "/data/bird/train/train_databases",
    "/data/bird/databases",
]
_SPIDER_DB_ROOTS = [
    "/data/spider/database",
    "/data/spider2/database",
]


def _resolve_db_path(db_id: str, dataset: str) -> str:
    """Resolve sqlite path for a given db_id and dataset name."""
    if "bird" in dataset:
        roots = _BIRD_DB_ROOTS
    elif "spider" in dataset:
        roots = _SPIDER_DB_ROOTS
    else:
        roots = _BIRD_DB_ROOTS + _SPIDER_DB_ROOTS

    for root in roots:
        candidate = Path(root) / db_id / f"{db_id}.sqlite"
        if candidate.exists():
            return str(candidate)
    return ""


# ── Source DB extraction ───────────────────────────────────────────────

_EXTRACT_SQL = """
SELECT e.query_id, e.config_id, e.model, e.prediction,
       ev.score, ev.metrics,
       q.content, q.dataset, q.meta
FROM execution e
JOIN evaluation ev ON e.execution_id = ev.execution_id
JOIN query q ON e.query_id = q.query_id
WHERE ev.score = 0 AND ev.scorer = ?
  AND e.prediction IS NOT NULL AND e.prediction != ''
"""


def _extract_from_source(
    db_path: str,
    scorer: str,
    max_rows: int,
) -> List[Dict[str, Any]]:
    """Extract failed predictions from one source DB."""
    uri = f"file:{db_path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.OperationalError as exc:
        logger.error("Cannot open source DB %s: %s", db_path, exc)
        return []

    try:
        rows = conn.execute(_EXTRACT_SQL, (scorer,)).fetchall()
    except sqlite3.OperationalError as exc:
        logger.error("Query failed on %s: %s", db_path, exc)
        conn.close()
        return []
    finally:
        conn.close()

    results = []
    for row in rows:
        if max_rows and len(results) >= max_rows:
            break

        meta_raw = row["meta"]
        try:
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
        except (json.JSONDecodeError, TypeError):
            meta = {}

        metrics_raw = row["metrics"]
        try:
            metrics = json.loads(metrics_raw) if isinstance(metrics_raw, str) else (metrics_raw or {})
        except (json.JSONDecodeError, TypeError):
            metrics = {}

        wrong_sql = _extract_sql_from_prediction(row["prediction"])
        error_type = metrics.get("error_type", "unknown") or "unknown"
        error_message = metrics.get("error_message", "") or ""

        results.append({
            "query_id": row["query_id"],
            "config_id": row["config_id"],
            "model": row["model"],
            "wrong_sql": wrong_sql,
            "error_type": error_type,
            "error_message": error_message,
            "content": row["content"],
            "dataset": row["dataset"],
            "meta": meta,
            "source_db": db_path,
        })

    return results


# ── Deduplication ──────────────────────────────────────────────────────

def _dedup_rows(
    all_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Deduplicate by (query_id, normalized_wrong_sql), merging source info."""
    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    counts: Dict[Tuple[str, str], int] = defaultdict(int)
    models: Dict[Tuple[str, str], set] = defaultdict(set)

    for row in all_rows:
        norm = _normalize_sql(row["wrong_sql"])
        key = (row["query_id"], norm)
        counts[key] += 1
        models[key].add(row["model"])
        if key not in seen:
            seen[key] = row

    deduped = []
    for key, row in seen.items():
        row = dict(row)
        row["n_source_configs"] = counts[key]
        row["source_models"] = sorted(models[key])
        deduped.append(row)

    return deduped


# ── Repair query construction ──────────────────────────────────────────

def _build_repair_query(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a deduplicated failure row into a repair query dict."""
    meta = row["meta"]
    raw = meta.get("_raw", {})

    question = raw.get("question", "") or row["content"]
    gold_sql = meta.get("gold_sql", raw.get("gold_sql", raw.get("gold_sql_query", "")))
    db_id = meta.get("db_id", raw.get("db_id", ""))
    dataset = row["dataset"]
    evidence = meta.get("evidence", raw.get("evidence", ""))
    difficulty = meta.get("difficulty", raw.get("difficulty", ""))
    wrong_sql = row["wrong_sql"]
    error_type = row["error_type"]
    error_message = row["error_message"]

    # Resolve db_path
    db_path = raw.get("db_path", "")
    if not db_path and db_id:
        db_path = _resolve_db_path(db_id, dataset)

    # Schema text
    schema = raw.get("full_schema", raw.get("schema", raw.get("schema_text", "")))

    if not question or not gold_sql or not wrong_sql:
        logger.debug("Skipping row missing question/gold/wrong_sql: query_id=%s", row["query_id"])
        return None

    norm = _normalize_sql(wrong_sql)
    wrong_sql_hash = hashlib.sha256(norm.encode()).hexdigest()[:8]

    from core.schema import make_query_id

    repair_query_id = make_query_id(
        "sql_repair",
        question,
        context=f"{db_id}:{wrong_sql_hash}",
    )

    # Infer source_dataset label
    if "bird" in dataset:
        source_dataset = "bird"
        repair_dataset = "sql_repair_bird"
    elif "spider" in dataset:
        source_dataset = "spider"
        repair_dataset = "sql_repair_spider"
    else:
        source_dataset = dataset
        repair_dataset = f"sql_repair_{dataset}"

    return {
        "query_id": repair_query_id,
        "dataset": repair_dataset,
        "content": question,
        "meta": {
            "db_id": db_id,
            "gold_sql": gold_sql,
            "wrong_sql": wrong_sql,
            "error_type": error_type,
            "error_message": error_message,
            "source_model": row["model"],
            "source_dataset": source_dataset,
            "difficulty": difficulty,
            "evidence": evidence,
            "n_source_configs": row["n_source_configs"],
            "split": "train",
            "_raw": {
                "db_path": db_path,
                "gold_sql": gold_sql,
                "wrong_sql": wrong_sql,
                "question": question,
                "schema": schema,
            },
        },
    }


# ── Summary stats ──────────────────────────────────────────────────────

def _print_summary(
    total_extracted: int,
    after_dedup: int,
    queries: List[Dict[str, Any]],
) -> None:
    by_dataset: Dict[str, int] = defaultdict(int)
    by_error_type: Dict[str, int] = defaultdict(int)
    by_model: Dict[str, int] = defaultdict(int)

    for q in queries:
        m = q["meta"]
        by_dataset[q["dataset"]] += 1
        by_error_type[m.get("error_type", "unknown")] += 1
        by_model[m.get("source_model", "unknown")] += 1

    print(f"\n=== SQL Repair Dataset Curation Summary ===")
    print(f"Total extracted from source DBs : {total_extracted}")
    print(f"After deduplication             : {after_dedup}")
    print(f"Built as repair queries         : {len(queries)}")
    print("\nBy dataset:")
    for k, v in sorted(by_dataset.items()):
        print(f"  {k}: {v}")
    print("\nBy error_type:")
    for k, v in sorted(by_error_type.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print("\nBy source_model:")
    for k, v in sorted(by_model.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Curate SQL repair dataset from existing experiment DBs."
    )
    parser.add_argument(
        "--source_dbs",
        nargs="+",
        required=True,
        metavar="DB_PATH",
        help="One or more source CubeStore SQLite files.",
    )
    parser.add_argument(
        "--output_db",
        required=True,
        metavar="DB_PATH",
        help="Output CubeStore SQLite file for repair queries.",
    )
    parser.add_argument(
        "--scorer",
        default="ex_acc",
        help="Scorer name used in source evaluation table (default: ex_acc).",
    )
    parser.add_argument(
        "--max_per_source",
        type=int,
        default=0,
        metavar="N",
        help="Max rows to extract per source DB (0 = all).",
    )
    args = parser.parse_args()

    # Extract from all source DBs
    all_rows: List[Dict[str, Any]] = []
    for db_path in args.source_dbs:
        logger.info("Extracting from %s", db_path)
        rows = _extract_from_source(db_path, args.scorer, args.max_per_source)
        logger.info("  -> %d failed predictions", len(rows))
        all_rows.extend(rows)

    total_extracted = len(all_rows)
    logger.info("Total extracted: %d", total_extracted)

    # Deduplicate
    deduped = _dedup_rows(all_rows)
    logger.info("After dedup: %d", len(deduped))

    # Build repair queries
    repair_queries: List[Dict[str, Any]] = []
    for row in deduped:
        q = _build_repair_query(row)
        if q is not None:
            repair_queries.append(q)

    logger.info("Repair queries built: %d", len(repair_queries))

    # Write to output CubeStore
    from core.store import CubeStore, OnConflict

    store = CubeStore(args.output_db)
    n_inserted = store.upsert_queries(repair_queries, on_conflict=OnConflict.REPLACE)
    store.close()
    logger.info("Inserted/updated %d queries in %s", n_inserted, args.output_db)

    _print_summary(total_extracted, len(deduped), repair_queries)


if __name__ == "__main__":
    main()
