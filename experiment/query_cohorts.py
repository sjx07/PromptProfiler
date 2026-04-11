"""Query cohorts — group queries by precomputed C-predicates.

Workflow:
    1. Register extractors (task-specific modules)
    2. seed_predicates(store) — compute all predicates, store in predicate table
    3. build_cohorts(store, "difficulty") — lookup from predicate table

Extractors compute predicate values. The predicate table stores them.
The cohort builder just reads the table — no computation at query time.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Callable, Dict, List, Optional

from prompt_profiler.core.store import CubeStore

logger = logging.getLogger(__name__)

Cohorts = Dict[str, List[str]]  # {cohort_label: [query_ids]}

# ── extractor registry ────────────────────────────────────────────────
# Extractors: (query_row: dict) → str
# Used only during seed_predicates(), not during build_cohorts().

ExtractorFn = Callable[..., str]  # (query, **context) -> str
EXTRACTORS: Dict[str, ExtractorFn] = {}


def register_extractor(name: str):
    """Decorator to register a C-predicate extractor.

    The extractor receives a query row dict with keys:
        query_id, dataset, content, meta (already parsed as dict)
    and optional **context kwargs (e.g., data_dir for DB introspection).
    Returns a string value for the predicate.
    """
    def wrapper(fn: ExtractorFn) -> ExtractorFn:
        EXTRACTORS[name] = fn
        return fn
    return wrapper


# ── seed predicates ───────────────────────────────────────────────────

def seed_predicates(
    store: CubeStore,
    *,
    dataset: Optional[str] = None,
    extractors: Optional[List[str]] = None,
    context: Optional[Dict] = None,
) -> int:
    """Compute all registered predicates for each query and store them.

    Args:
        dataset: Filter to this dataset. None = all queries.
        extractors: Subset of extractor names to run. None = all registered.
        context: Extra kwargs passed to extractors (e.g., data_dir for DB introspection).

    Returns:
        Number of predicate rows inserted.
    """
    ctx = context or {}
    ext_names = extractors or list(EXTRACTORS.keys())
    ext_fns = {name: EXTRACTORS[name] for name in ext_names if name in EXTRACTORS}
    if not ext_fns:
        logger.warning("No extractors to run")
        return 0

    conn = store._get_conn()
    if dataset:
        rows = conn.execute("SELECT * FROM query WHERE dataset = ?", (dataset,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM query").fetchall()

    records = []
    for row in rows:
        q = dict(row)
        if isinstance(q.get("meta"), str):
            q["meta"] = json.loads(q["meta"])

        for name, fn in ext_fns.items():
            try:
                value = fn(q, **ctx)
            except TypeError:
                # Extractor doesn't accept context kwargs — call without
                value = fn(q)
            except Exception as e:
                logger.debug("Extractor %s failed on %s: %s", name, q["query_id"], e)
                value = "unknown"
            records.append((q["query_id"], name, value))

    with store._cursor() as cur:
        cur.executemany(
            "INSERT OR REPLACE INTO predicate (query_id, name, value) VALUES (?, ?, ?)",
            records,
        )

    logger.info("seed_predicates: %d predicates × %d queries = %d rows",
                len(ext_fns), len(rows), len(records))
    return len(records)


# ── cohort builder (reads predicate table) ────────────────────────────

def build_cohorts(
    store: CubeStore,
    by: str,
    *,
    dataset: Optional[str] = None,
    min_size: int = 1,
    max_queries: int = 0,
) -> Cohorts:
    """Group queries by a precomputed predicate.

    Reads from the predicate table — predicates must be seeded first
    via seed_predicates().

    Args:
        by: Predicate name (must exist in predicate table).
        dataset: Filter to this dataset. None = all.
        min_size: Drop cohorts smaller than this.
        max_queries: Cap each cohort (0 = unlimited).

    Returns:
        {predicate_value: [query_ids]}
    """
    conn = store._get_conn()
    if dataset:
        rows = conn.execute(
            """SELECT p.query_id, p.value
               FROM predicate p JOIN query q ON p.query_id = q.query_id
               WHERE p.name = ? AND q.dataset = ?""",
            (by, dataset),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT query_id, value FROM predicate WHERE name = ?",
            (by,),
        ).fetchall()

    if not rows:
        logger.warning("No predicate values found for %r — did you run seed_predicates()?", by)
        return {}

    groups: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        groups[row["value"]].append(row["query_id"])

    cohorts: Cohorts = {}
    for label, qids in sorted(groups.items()):
        if len(qids) < min_size:
            continue
        cohorts[label] = qids[:max_queries] if max_queries > 0 else qids

    logger.info("build_cohorts(by=%s): %d cohorts, %d total queries",
                by, len(cohorts), sum(len(v) for v in cohorts.values()))
    return cohorts


def build_cohorts_multi(
    store: CubeStore,
    predicates: List[str],
    *,
    dataset: Optional[str] = None,
    min_size: int = 1,
) -> Dict[str, Cohorts]:
    """Build cohorts for multiple predicates at once."""
    return {
        pred: build_cohorts(store, pred, dataset=dataset, min_size=min_size)
        for pred in predicates
    }


def build_cohorts_compound(
    store: CubeStore,
    predicates: List[str],
    *,
    dataset: Optional[str] = None,
    min_size: int = 1,
) -> Cohorts:
    """Group queries by the intersection of multiple predicates.

    Each cohort label is a tuple of values joined by "|".
    E.g. predicates=["difficulty", "asks_distinct"] →
         {"challenging|True": [...], "simple|False": [...], ...}
    """
    conn = store._get_conn()
    if dataset:
        qids = {r["query_id"] for r in conn.execute(
            "SELECT query_id FROM query WHERE dataset = ?", (dataset,),
        ).fetchall()}
    else:
        qids = None

    # Build per-query predicate vectors from the predicate table
    rows = conn.execute(
        "SELECT query_id, name, value FROM predicate WHERE name IN ({})".format(
            ",".join("?" for _ in predicates)),
        predicates,
    ).fetchall()

    # query_id → {pred_name: value}
    vectors: Dict[str, Dict[str, str]] = defaultdict(dict)
    for row in rows:
        if qids is not None and row["query_id"] not in qids:
            continue
        vectors[row["query_id"]][row["name"]] = row["value"]

    # Group by compound key
    groups: Dict[str, List[str]] = defaultdict(list)
    for query_id, preds in vectors.items():
        if len(preds) < len(predicates):
            continue  # missing predicates, skip
        key = "|".join(preds.get(p, "?") for p in predicates)
        groups[key].append(query_id)

    cohorts: Cohorts = {}
    for label, qids_list in sorted(groups.items()):
        if len(qids_list) < min_size:
            continue
        cohorts[label] = qids_list

    logger.info("build_cohorts_compound(by=%s): %d cohorts, %d total queries",
                predicates, len(cohorts), sum(len(v) for v in cohorts.values()))
    return cohorts


def predicate_table(
    store: CubeStore,
    predicates: Optional[List[str]] = None,
    *,
    dataset: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build the flat view: one row per query, one column per predicate.

    Returns list of dicts: {query_id, pred_1, pred_2, ...}
    """
    conn = store._get_conn()

    # Get predicate names
    if predicates is None:
        pred_names = [r["name"] for r in conn.execute(
            "SELECT DISTINCT name FROM predicate ORDER BY name"
        ).fetchall()]
    else:
        pred_names = predicates

    # Get all predicate values
    rows = conn.execute("SELECT query_id, name, value FROM predicate").fetchall()
    vectors: Dict[str, Dict[str, str]] = defaultdict(dict)
    for row in rows:
        if row["name"] in pred_names:
            vectors[row["query_id"]][row["name"]] = row["value"]

    # Filter by dataset
    if dataset:
        ds_qids = {r["query_id"] for r in conn.execute(
            "SELECT query_id FROM query WHERE dataset = ?", (dataset,),
        ).fetchall()}
    else:
        ds_qids = None

    result = []
    for query_id, preds in vectors.items():
        if ds_qids is not None and query_id not in ds_qids:
            continue
        row = {"query_id": query_id}
        row.update({p: preds.get(p, "") for p in pred_names})
        result.append(row)

    return result


def list_predicates(store: CubeStore) -> List[Dict[str, int]]:
    """List all predicate names and their distinct value counts."""
    conn = store._get_conn()
    rows = conn.execute(
        """SELECT name, COUNT(DISTINCT value) as n_values, COUNT(*) as n_queries
           FROM predicate GROUP BY name ORDER BY name""",
    ).fetchall()
    return [dict(r) for r in rows]
