"""Lazy evaluation — score executions that haven't been evaluated yet.

Two entry points:
    evaluate_config  — score unevaluated executions for one config (parallel)
    evaluate_all     — score ALL unevaluated executions across configs (parallel)
"""
from __future__ import annotations

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.store import CubeStore, OnConflict

logger = logging.getLogger(__name__)

_EVAL_WORKERS = os.cpu_count() or 8
_EVAL_TIMEOUT_SEC = 180


def _score_with_timeout(task: Any, prediction: str, query_meta: dict,
                         timeout: int = _EVAL_TIMEOUT_SEC) -> Tuple[float, dict]:
    """Run task.score() with a timeout to prevent hanging on bad SQL."""
    container: Dict[str, Any] = {}

    def _target():
        try:
            container["result"] = task.score(prediction, query_meta)
        except Exception as e:
            container["error"] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        raise TimeoutError(f"SQL eval timed out ({timeout}s)")
    if "error" in container:
        raise container["error"]
    return container["result"]


def _score_items(
    task: Any,
    items: List[Tuple[int, str, dict]],
    store: CubeStore,
    scorer: str,
    num_workers: int,
    on_conflict: OnConflict,
) -> Tuple[int, int]:
    """Score a list of (execution_id, prediction, query_meta) in parallel.

    Returns (evaluated, errors).
    """
    def _score_one(item):
        eid, prediction, query_meta = item
        try:
            score, metrics = _score_with_timeout(task, prediction, query_meta)
        except Exception as e:
            score = 0.0
            metrics = {"status": "scoring_error", "error": str(e)[:500]}
            return eid, score, metrics, True
        return eid, score, metrics, False

    evaluated = 0
    errors = 0
    workers = min(num_workers, len(items))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_score_one, item): item for item in items}
        for fut in as_completed(futures):
            eid, score, metrics, is_err = fut.result()
            store.upsert_evaluation(eid, scorer, score,
                                    metrics=metrics, on_conflict=on_conflict)
            if is_err:
                errors += 1
            else:
                evaluated += 1

    return evaluated, errors


def evaluate_config(
    store: CubeStore,
    config_id: int,
    model: str,
    task: Any,
    *,
    scorer: str = "",
    dataset: str = "",
    num_workers: int = _EVAL_WORKERS,
    on_conflict: OnConflict = OnConflict.SKIP,
) -> Dict[str, Any]:
    """Score all unevaluated executions for one config (parallel).

    When `dataset` is non-empty, only executions whose query.dataset
    matches are scored. Required when multiple tasks share a config_id
    (e.g. tasks with identical structural section sets) — without it,
    the first task to evaluate scores all cross-dataset executions
    using its own task.score(), which produces wrong scores.
    """
    scorer = scorer or task.scorer
    conn = store._get_conn()

    if dataset:
        executions = conn.execute(
            """SELECT e.execution_id, e.query_id, e.prediction, e.error
               FROM execution e
               JOIN query q ON q.query_id = e.query_id
               LEFT JOIN evaluation ev ON e.execution_id = ev.execution_id AND ev.scorer = ?
               WHERE e.config_id = ? AND e.model = ? AND q.dataset = ? AND ev.eval_id IS NULL""",
            (scorer, config_id, model, dataset),
        ).fetchall()
    else:
        executions = conn.execute(
            """SELECT e.execution_id, e.query_id, e.prediction, e.error
               FROM execution e
               LEFT JOIN evaluation ev ON e.execution_id = ev.execution_id AND ev.scorer = ?
               WHERE e.config_id = ? AND e.model = ? AND ev.eval_id IS NULL""",
            (scorer, config_id, model),
        ).fetchall()

    if not executions:
        return {"evaluated": 0, "skipped": 0, "errors": 0}

    # Pre-load query metas
    query_ids = list({ex["query_id"] for ex in executions})
    placeholders = ",".join("?" for _ in query_ids)
    meta_rows = conn.execute(
        f"SELECT query_id, meta FROM query WHERE query_id IN ({placeholders})",
        query_ids,
    ).fetchall()
    query_metas = {
        r["query_id"]: json.loads(r["meta"]) if isinstance(r["meta"], str) else r["meta"]
        for r in meta_rows
    }

    # Separate errored from scoreable
    to_score = []
    errors = 0
    for ex in executions:
        if ex["error"]:
            store.upsert_evaluation(
                ex["execution_id"], scorer, 0.0,
                metrics={"status": "execution_error", "error": ex["error"]},
                on_conflict=on_conflict,
            )
            errors += 1
        else:
            meta = query_metas.get(ex["query_id"])
            if meta:
                to_score.append((ex["execution_id"], ex["prediction"], meta))

    evaluated, score_errors = _score_items(
        task, to_score, store, scorer, num_workers, on_conflict,
    )
    errors += score_errors

    logger.info("Config %d (%s/%s): evaluated=%d, errors=%d",
                config_id, model, scorer, evaluated, errors)
    return {"evaluated": evaluated, "skipped": 0, "errors": errors}


def evaluate_all(
    store: CubeStore,
    model: str,
    task: Any,
    *,
    scorer: str = "",
    num_workers: int = _EVAL_WORKERS,
    on_conflict: OnConflict = OnConflict.REPLACE,
) -> Dict[str, Any]:
    """Score ALL unevaluated executions across all configs (parallel)."""
    scorer = scorer or task.scorer
    conn = store._get_conn()

    executions = conn.execute(
        """SELECT e.execution_id, e.query_id, e.prediction, e.error
           FROM execution e
           LEFT JOIN evaluation ev ON e.execution_id = ev.execution_id AND ev.scorer = ?
           WHERE e.model = ? AND ev.eval_id IS NULL""",
        (scorer, model),
    ).fetchall()

    if not executions:
        logger.info("All executions already evaluated")
        return {"evaluated": 0, "skipped": 0, "errors": 0}

    # Pre-load all query metas
    meta_rows = conn.execute("SELECT query_id, meta FROM query").fetchall()
    query_metas = {
        r["query_id"]: json.loads(r["meta"]) if isinstance(r["meta"], str) else r["meta"]
        for r in meta_rows
    }

    to_score = []
    errors = 0
    for ex in executions:
        if ex["error"]:
            store.upsert_evaluation(
                ex["execution_id"], scorer, 0.0,
                metrics={"status": "execution_error", "error": ex["error"]},
                on_conflict=on_conflict,
            )
            errors += 1
        else:
            meta = query_metas.get(ex["query_id"])
            if meta:
                to_score.append((ex["execution_id"], ex["prediction"], meta))

    logger.info("Evaluating %d executions (%d errored) with %d workers",
                len(to_score), errors, num_workers)

    evaluated, score_errors = _score_items(
        task, to_score, store, scorer, num_workers, on_conflict,
    )
    errors += score_errors

    logger.info("evaluate_all: evaluated=%d, errors=%d", evaluated, errors)
    return {"evaluated": evaluated, "skipped": 0, "errors": errors}
