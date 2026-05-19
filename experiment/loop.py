"""Experiment loop — iterative plan → run → evaluate → analyze cycle.

The store (db) is the single source of state. Each step reads from
and writes to it. No intermediate data structures between stages.

Key optimization: run and eval are pipelined per config — eval for
config N happens in the background while config N+1 is running.

Usage:
    from experiment_loop import run_experiment, AnalysisResult

    def my_analyzer(store, model, scorer, iteration):
        # read results, compute effects, decide next plan
        ...
        return AnalysisResult(insights={...}, next_plan=next_entries or None)

    run_experiment(
        store=store,
        initial_plan=plan,
        task_cls=SchemaLinking,
        model="Qwen/Qwen2.5-Coder-32B-Instruct",
        llm_call=llm_call,
        analyze_fn=my_analyzer,
    )
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

from execution.evaluate import evaluate_config
from experiment.planner import RunEntry
from core.func_registry import apply_config
from execution.runner import run_config, run_config_batch
from core.store import CubeStore, OnConflict

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Output of an analysis step."""
    insights: Dict[str, Any] = field(default_factory=dict)
    next_plan: Optional[List[RunEntry]] = None  # None = done


AnalyzeFn = Callable[[CubeStore, str, str, int], AnalysisResult]


def run_experiment(
    store: CubeStore,
    initial_plan: List[RunEntry],
    task_cls: Type,
    model: str,
    llm_call: Callable,
    analyze_fn: AnalyzeFn,
    *,
    num_workers: int = 40,
    eval_pool_size: int = 4,
    max_iterations: int = 10,
    on_conflict: OnConflict = OnConflict.SKIP,
    example_pool: Optional[list] = None,
    dataset: str = "",
) -> List[Dict[str, Any]]:
    """Run the iterative experiment loop.

    Args:
        store: CubeStore (db is the state).
        initial_plan: First set of RunEntries.
        task_cls: Task class (SqlGeneration, SchemaLinking, etc.).
        model: Model name.
        llm_call: Callable for LLM inference.
        analyze_fn: User-defined analysis function.
            Signature: (store, model, scorer, iteration) → AnalysisResult.
        num_workers: Thread pool size for runner.
        max_iterations: Safety bound on loop iterations.
        on_conflict: Conflict policy for inserts.
        example_pool: Train-split query dicts for few-shot. Passed to task.bind().

    Returns:
        List of insights dicts from each iteration.
    """
    scorer = task_cls.scorer
    all_insights: List[Dict[str, Any]] = []
    plan = initial_plan

    for iteration in range(1, max_iterations + 1):
        if not plan:
            logger.info("Iteration %d: empty plan, stopping", iteration)
            break

        logger.info("Iteration %d: %d entries, %d total LLM calls",
                     iteration, len(plan),
                     sum(len(e.query_ids) for e in plan))

        # ── run + eval (pipelined) ────────────────────────────────
        _run_and_eval_plan(store, plan, task_cls, model, llm_call,
                           num_workers=num_workers, eval_pool_size=eval_pool_size,
                           on_conflict=on_conflict, example_pool=example_pool,
                           dataset=dataset)

        # ── analyze ───────────────────────────────────────────────
        result = analyze_fn(store, model, scorer, iteration)
        all_insights.append(result.insights)
        logger.info("Iteration %d: analysis complete, next_plan=%s",
                     iteration,
                     f"{len(result.next_plan)} entries" if result.next_plan else "None (done)")

        plan = result.next_plan

    return all_insights


def _run_and_eval_plan(
    store: CubeStore,
    plan: List[RunEntry],
    task_cls: Type,
    model: str,
    llm_call: Callable,
    *,
    num_workers: int = 40,
    eval_pool_size: int = 4,
    on_conflict: OnConflict = OnConflict.SKIP,
    example_pool: Optional[list] = None,
    phase: str | None = None,
    dataset: str = "",
    retry_errors: bool = False,
    batch_configs: bool = False,
) -> None:
    """Run configs and evaluate them in a pipelined fashion.

    After config N finishes running, its evaluation is submitted to a
    background pool (eval_pool_size workers). Config N+1 starts running
    immediately. Multiple evals can overlap; each uses
    cpu_count // eval_pool_size threads to avoid thrashing.
    """
    # Deduplicate configs — same config may appear with different query sets
    config_queue: List[tuple] = []  # (cid, func_ids, query_ids)
    config_queries: Dict[int, List[str]] = {}
    config_func_ids: Dict[int, List[str]] = {}
    for entry in plan:
        if entry.config_id not in config_queries:
            config_queries[entry.config_id] = []
            config_func_ids[entry.config_id] = entry.func_ids
        config_queries[entry.config_id].extend(entry.query_ids)

    for cid in config_queries:
        config_queue.append((cid, config_func_ids[cid], config_queries[cid]))

    # eval_workers_per_config: divide cpu budget across concurrent eval jobs
    eval_workers_per_config = max(4, (os.cpu_count() or 8) // eval_pool_size)
    eval_pool = ThreadPoolExecutor(max_workers=eval_pool_size, thread_name_prefix="eval")
    eval_futures: List[Future] = []

    try:
        if batch_configs:
            def submit_eval(cid: int) -> None:
                eval_task = task_cls()
                fut = eval_pool.submit(
                    evaluate_config, store, cid, model, eval_task,
                    num_workers=eval_workers_per_config,
                    on_conflict=OnConflict.REPLACE,
                    dataset=dataset,
                )
                eval_futures.append(fut)
                logger.info("Config %d eval submitted to background", cid)

            _run_config_queue_batched(
                store,
                config_queue,
                task_cls,
                model,
                llm_call,
                num_workers=num_workers,
                on_conflict=on_conflict,
                example_pool=example_pool,
                phase=phase,
                retry_errors=retry_errors,
                on_config_complete=submit_eval,
            )

            for fut in eval_futures:
                fut.result()
            return

        for i, (cid, func_ids, query_ids) in enumerate(config_queue):
            # ── update labels if llm_call supports it ─────────────
            if hasattr(llm_call, "set_labels"):
                n_rules = len([f for f in func_ids if len(f) == 12]) - len(config_queue[0][1])
                llm_call.set_labels(config_id=cid, n_rules=max(n_rules, 0))

            # ── run config ────────────────────────────────────────
            task = task_cls()
            if hasattr(task, "bind_modules"):
                from core.func_registry import apply_config_modules
                state_by_module = apply_config_modules(
                    func_ids,
                    store,
                    module_names=task.module_names(),
                )
                task.bind_modules(state_by_module, example_pool=example_pool)
            else:
                state = apply_config(func_ids, store)
                task.bind(state, example_pool=example_pool)

            conn = store._get_conn()
            placeholders = ",".join("?" for _ in query_ids)
            rows = conn.execute(
                f"SELECT * FROM query WHERE query_id IN ({placeholders})",
                query_ids,
            ).fetchall()
            queries = [dict(r) for r in rows]

            logger.info("Running config %d/%d (id=%d, %d queries) ...",
                        i + 1, len(config_queue), cid, len(queries))
            run_config(store, cid, queries, task, model, llm_call,
                       num_workers=num_workers, on_conflict=on_conflict,
                       phase=phase, retry_errors=retry_errors)

            # ── submit eval to background ─────────────────────────
            eval_task = task_cls()  # fresh instance for thread safety
            fut = eval_pool.submit(
                evaluate_config, store, cid, model, eval_task,
                num_workers=eval_workers_per_config,
                on_conflict=OnConflict.REPLACE,
                dataset=dataset,
            )
            eval_futures.append(fut)
            logger.info("Config %d eval submitted to background", cid)

        # ── wait for all evals to finish ──────────────────────────
        for fut in eval_futures:
            fut.result()  # raises if eval failed

    finally:
        eval_pool.shutdown(wait=True)


def _run_config_queue_batched(
    store: CubeStore,
    config_queue: List[tuple],
    task_cls: Type,
    model: str,
    llm_call: Callable,
    *,
    num_workers: int,
    on_conflict: OnConflict,
    example_pool: Optional[list],
    phase: str | None,
    retry_errors: bool,
    on_config_complete: Callable[[int], None] | None = None,
) -> None:
    """Prepare all configs, then batch uncached executions across configs.

    When ``on_config_complete`` is provided, it is called as soon as all
    expected execution attempts for a config have completed. This lets the
    caller pipeline evaluation with remaining batched LLM work.
    """
    work_items: List[Dict[str, Any]] = []
    remaining_by_config: Dict[int, int] = {}
    submitted_configs: set[int] = set()
    total_cached = 0
    total_queries = 0

    def mark_config_complete(cid: int) -> None:
        if cid in submitted_configs:
            return
        submitted_configs.add(cid)
        if on_config_complete is not None:
            on_config_complete(cid)

    for cid, func_ids, query_ids in config_queue:
        task = _bind_task_for_config(store, task_cls, func_ids, example_pool)
        query_ids = _dedupe_preserve_order(query_ids)
        total_queries += len(query_ids)

        all_cached = store.get_cached_query_ids(
            cid,
            model,
            include_errors=not retry_errors,
        )
        query_id_set = set(query_ids)
        cached = all_cached & query_id_set
        total_cached += len(cached)

        if phase and cached:
            for qid in cached:
                existing = store.get_cached_execution(cid, qid, model)
                if existing:
                    store.tag_phase(existing["execution_id"], phase)

        uncached_ids = [qid for qid in query_ids if qid not in cached]
        remaining_by_config[cid] = len(uncached_ids)
        queries = _load_queries_by_ids(store, uncached_ids)
        logger.info(
            "Prepared config id=%d for batched execution: %d cached, %d remaining",
            cid, len(cached), len(queries),
        )
        if not uncached_ids:
            mark_config_complete(cid)
        work_items.extend({
            "config_id": cid,
            "query": query,
            "task": task,
        } for query in queries)

    logger.info(
        "Batched plan: %d configs, %d total queries, %d cached, %d remaining",
        len(config_queue), total_queries, total_cached, len(work_items),
    )

    def mark_item_done(cid: int) -> None:
        remaining_by_config[cid] -= 1
        if remaining_by_config[cid] <= 0:
            mark_config_complete(cid)

    run_config_batch(
        store,
        work_items,
        model,
        llm_call,
        num_workers=num_workers,
        on_conflict=on_conflict,
        phase=phase,
        retry_errors=retry_errors,
        on_item_done=mark_item_done,
    )


def _bind_task_for_config(
    store: CubeStore,
    task_cls: Type,
    func_ids: List[str],
    example_pool: Optional[list],
) -> Any:
    task = task_cls()
    if hasattr(task, "bind_modules"):
        from core.func_registry import apply_config_modules
        state_by_module = apply_config_modules(
            func_ids,
            store,
            module_names=task.module_names(),
        )
        task.bind_modules(state_by_module, example_pool=example_pool)
    else:
        state = apply_config(func_ids, store)
        task.bind(state, example_pool=example_pool)
    return task


def _dedupe_preserve_order(query_ids: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for query_id in query_ids:
        if query_id not in seen:
            seen.add(query_id)
            out.append(query_id)
    return out


def _load_queries_by_ids(store: CubeStore, query_ids: List[str]) -> List[dict]:
    if not query_ids:
        return []

    conn = store._get_conn()
    rows_by_id: Dict[str, dict] = {}
    chunk_size = 900
    for i in range(0, len(query_ids), chunk_size):
        chunk = query_ids[i:i + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT * FROM query WHERE query_id IN ({placeholders})",
            chunk,
        ).fetchall()
        rows_by_id.update({r["query_id"]: dict(r) for r in rows})
    return [rows_by_id[qid] for qid in query_ids if qid in rows_by_id]
