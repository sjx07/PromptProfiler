"""Unified runner — task-agnostic execution engine.

Lifecycle:
    1. apply_config → PromptBuildState
    2. task.bind(state)
    3. For each uncached query:
       task.build_prompt(query) → LLM call → task.parse_response → store execution
    4. Evaluation is separate (lazy).

TODO: Pipeline LLM inference and evaluation — currently sequential per config.
      Could overlap: evaluate config N while running config N+1.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List

from core.store import CubeStore, OnConflict

logger = logging.getLogger(__name__)


def run_config(
    store: CubeStore,
    config_id: int,
    queries: List[Dict[str, Any]],
    task: Any,
    model: str,
    llm_call: Callable[[str, str], Dict[str, Any]],
    *,
    num_workers: int = 1,
    dry_run: bool = False,
    on_conflict: OnConflict = OnConflict.SKIP,
    phase: str | None = None,
) -> Dict[str, Any]:
    """Run a config against queries using a task.

    Args:
        store: Unified CubeStore.
        config_id: Config to execute.
        queries: List of query dicts (from store).
        task: Stage instance, already bound via task.bind(state).
        model: Model identifier string.
        llm_call: Callable(system_prompt, user_content) → dict with keys:
                  raw_response, prompt_tokens, completion_tokens.
        num_workers: Thread pool size (1 = sequential).
        dry_run: If True, skip inference and just report cache status.
        on_conflict: Conflict policy for storing executions.

    Returns:
        Progress dict with done/total/remaining/newly_executed.
    """
    # Check cache — intersect with current query set
    all_cached = store.get_cached_query_ids(config_id, model)
    query_ids = {q["query_id"] for q in queries}
    cached = all_cached & query_ids
    uncached = [q for q in queries if q["query_id"] not in cached]

    # Tag cached executions with current phase
    if phase and cached:
        for qid in cached:
            existing = store.get_cached_execution(config_id, qid, model)
            if existing:
                store.tag_phase(existing["execution_id"], phase)

    if not uncached:
        logger.info("Config %d: fully cached (%d/%d)", config_id, len(cached), len(queries))
        return _progress(config_id, len(cached), len(queries), 0)

    logger.info("Config %d: %d cached, %d remaining", config_id, len(cached), len(uncached))

    if dry_run:
        return _progress(config_id, len(cached), len(queries), 0)

    # Execute uncached queries
    err_count = 0
    t0 = time.time()

    def _run_one(query: Dict[str, Any]) -> None:
        nonlocal err_count
        t_start = time.time()
        p_tokens = None
        c_tokens = None
        try:
            system_prompt, user_content = task.build_prompt(query)
            result = llm_call(system_prompt, user_content)
            raw_response = result.get("raw_response", "")
            if not isinstance(raw_response, str):
                raw_response = str(raw_response)
            prediction = task.parse_response(raw_response)
            p_tokens = result.get("prompt_tokens")
            c_tokens = result.get("completion_tokens")
            error = None
        except Exception as e:
            system_prompt = ""
            user_content = ""
            raw_response = ""
            prediction = ""
            error = str(e)[:500]
            err_count += 1

        latency_ms = (time.time() - t_start) * 1000

        store.insert_execution(
            config_id=config_id,
            query_id=query["query_id"],
            model=model,
            system_prompt=system_prompt,
            user_content=user_content,
            raw_response=raw_response,
            prediction=prediction,
            latency_ms=latency_ms,
            prompt_tokens=p_tokens,
            completion_tokens=c_tokens,
            error=error,
            phase=phase,
            on_conflict=on_conflict,
        )

    if num_workers <= 1:
        for q in uncached:
            _run_one(q)
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_run_one, q): q for q in uncached}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    q = futures[future]
                    logger.error("Query %s failed: %s", q["query_id"], exc)

    elapsed = time.time() - t0
    rate = len(uncached) / max(elapsed, 0.1)
    logger.info(
        "Config %d: %d queries in %.1fs (%.1f q/s), %d errors",
        config_id, len(uncached), elapsed, rate, err_count,
    )

    return _progress(config_id, len(cached) + len(uncached), len(queries), len(uncached))


def _progress(config_id: int, done: int, total: int, newly_executed: int) -> Dict[str, Any]:
    return {
        "config_id": config_id,
        "done": done,
        "total": total,
        "remaining": total - done,
        "newly_executed": newly_executed,
    }
