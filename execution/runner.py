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
from contextlib import nullcontext
from typing import Any, Callable, Dict, List

from core.store import CubeStore, OnConflict
from task import CompoundTask, ModuleRuntime

logger = logging.getLogger(__name__)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional runtime sugar.
    tqdm = None


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
    retry_errors: bool = False,
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
    all_cached = store.get_cached_query_ids(
        config_id,
        model,
        include_errors=not retry_errors,
    )
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
    t0 = time.time()
    err_count = 0

    progress = _make_progress_bar(
        total=len(uncached),
        desc=f"config {config_id}",
    )
    if num_workers <= 1:
        with progress as pbar:
            for q in uncached:
                if _run_one_execution(
                    store, config_id, q, task, model, llm_call,
                    on_conflict=on_conflict, phase=phase, retry_errors=retry_errors,
                ):
                    err_count += 1
                if pbar is not None:
                    pbar.update(1)
    else:
        with progress as pbar, ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(
                    _run_one_execution,
                    store, config_id, q, task, model, llm_call,
                    on_conflict=on_conflict, phase=phase, retry_errors=retry_errors,
                ): q
                for q in uncached
            }
            for future in as_completed(futures):
                try:
                    if future.result():
                        err_count += 1
                except Exception as exc:
                    q = futures[future]
                    logger.error("Query %s failed: %s", q["query_id"], exc)
                finally:
                    if pbar is not None:
                        pbar.update(1)

    elapsed = time.time() - t0
    rate = len(uncached) / max(elapsed, 0.1)
    logger.info(
        "Config %d: %d queries in %.1fs (%.1f q/s), %d errors",
        config_id, len(uncached), elapsed, rate, err_count,
    )

    return _progress(config_id, len(cached) + len(uncached), len(queries), len(uncached))


def run_config_batch(
    store: CubeStore,
    work_items: List[Dict[str, Any]],
    model: str,
    llm_call: Callable[[str, str], Dict[str, Any]],
    *,
    num_workers: int = 1,
    on_conflict: OnConflict = OnConflict.SKIP,
    phase: str | None = None,
    retry_errors: bool = False,
) -> Dict[str, Any]:
    """Run uncached query/config pairs through one shared worker pool.

    Each work item must contain ``config_id``, ``query``, and a task instance
    already bound for that config. This keeps prompt construction config-aware
    while letting the LLM queue batch across configs.
    """
    if not work_items:
        logger.info("Batched config execution: no uncached work")
        return {"total": 0, "newly_executed": 0, "errors": 0}

    err_count = 0
    t0 = time.time()
    progress = _make_progress_bar(total=len(work_items), desc="configs batched")

    def _run_item(item: Dict[str, Any]) -> bool:
        return _run_one_execution(
            store,
            int(item["config_id"]),
            item["query"],
            item["task"],
            model,
            llm_call,
            on_conflict=on_conflict,
            phase=phase,
            retry_errors=retry_errors,
        )

    if num_workers <= 1:
        with progress as pbar:
            for item in work_items:
                if _run_item(item):
                    err_count += 1
                if pbar is not None:
                    pbar.update(1)
    else:
        with progress as pbar, ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_run_item, item): item for item in work_items}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    if future.result():
                        err_count += 1
                except Exception as exc:
                    query_id = item.get("query", {}).get("query_id")
                    logger.error(
                        "Batched query failed: config=%s query=%s error=%s",
                        item.get("config_id"), query_id, exc,
                    )
                finally:
                    if pbar is not None:
                        pbar.update(1)

    elapsed = time.time() - t0
    rate = len(work_items) / max(elapsed, 0.1)
    logger.info(
        "Batched configs: %d executions in %.1fs (%.1f q/s), %d errors",
        len(work_items), elapsed, rate, err_count,
    )
    return {"total": len(work_items), "newly_executed": len(work_items), "errors": err_count}


def _run_one_execution(
    store: CubeStore,
    config_id: int,
    query: Dict[str, Any],
    task: Any,
    model: str,
    llm_call: Callable[[str, str], Dict[str, Any]],
    *,
    on_conflict: OnConflict,
    phase: str | None,
    retry_errors: bool,
) -> bool:
    """Run one query for one bound config and persist its execution row.

    Returns True if task execution/parsing raised and was stored as an error.
    """
    t_start = time.time()
    p_tokens = None
    c_tokens = None
    raw_reasoning = ""
    finish_reason = None
    meta = None
    runtime = None
    try:
        if isinstance(task, CompoundTask):
            runtime = ModuleRuntime(llm_call)
            run_result = task.run(query, runtime)
            prediction = str(run_result)
            last_trace = runtime.last_trace()
            system_prompt = last_trace.system_prompt if last_trace else ""
            user_content = last_trace.user_content if last_trace else ""
            raw_response = last_trace.raw_response if last_trace else ""
            raw_reasoning = last_trace.raw_reasoning if last_trace else ""
            finish_reason = last_trace.finish_reason if last_trace else None
            p_tokens = runtime.total_prompt_tokens()
            c_tokens = runtime.total_completion_tokens()
            meta = {
                "compound": True,
                "module_traces": runtime.trace_dicts(),
            }
        else:
            system_prompt, user_content = task.build_prompt(query)
            result = llm_call(system_prompt, user_content)
            raw_response = result.get("raw_response", "")
            if not isinstance(raw_response, str):
                raw_response = str(raw_response)
            raw_reasoning = result.get("raw_reasoning", "")
            if not isinstance(raw_reasoning, str):
                raw_reasoning = str(raw_reasoning)
            finish_reason = result.get("finish_reason")
            prediction = task.parse_response(raw_response)
            p_tokens = result.get("prompt_tokens")
            c_tokens = result.get("completion_tokens")
        error = None
    except Exception as e:
        if runtime is not None:
            last_trace = runtime.last_trace()
            system_prompt = last_trace.system_prompt if last_trace else ""
            user_content = last_trace.user_content if last_trace else ""
            raw_response = last_trace.raw_response if last_trace else ""
            raw_reasoning = last_trace.raw_reasoning if last_trace else ""
            finish_reason = last_trace.finish_reason if last_trace else None
            p_tokens = runtime.total_prompt_tokens()
            c_tokens = runtime.total_completion_tokens()
            meta = {
                "compound": True,
                "module_traces": runtime.trace_dicts(),
            }
        else:
            system_prompt = ""
            user_content = ""
            raw_response = ""
            raw_reasoning = ""
            finish_reason = None
        prediction = ""
        error = str(e)[:500]

    latency_ms = (time.time() - t_start) * 1000

    store.insert_execution(
        config_id=config_id,
        query_id=query["query_id"],
        model=model,
        system_prompt=system_prompt,
        user_content=user_content,
        raw_response=raw_response,
        raw_reasoning=raw_reasoning,
        prediction=prediction,
        latency_ms=latency_ms,
        prompt_tokens=p_tokens,
        completion_tokens=c_tokens,
        finish_reason=finish_reason,
        error=error,
        meta=meta,
        phase=phase,
        on_conflict=OnConflict.REPLACE if retry_errors else on_conflict,
    )
    return error is not None


def _make_progress_bar(total: int, desc: str):
    if tqdm is None:
        return nullcontext(None)
    return tqdm(total=total, desc=desc, unit="q", dynamic_ncols=True)


def _progress(config_id: int, done: int, total: int, newly_executed: int) -> Dict[str, Any]:
    return {
        "config_id": config_id,
        "done": done,
        "total": total,
        "remaining": total - done,
        "newly_executed": newly_executed,
    }
