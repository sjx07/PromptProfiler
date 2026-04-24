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
from execution.runner import run_config
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
                           on_conflict=on_conflict, example_pool=example_pool)

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
                       num_workers=num_workers, on_conflict=on_conflict)

            # ── submit eval to background ─────────────────────────
            eval_task = task_cls()  # fresh instance for thread safety
            fut = eval_pool.submit(
                evaluate_config, store, cid, model, eval_task,
                num_workers=eval_workers_per_config,
                on_conflict=OnConflict.REPLACE,
            )
            eval_futures.append(fut)
            logger.info("Config %d eval submitted to background", cid)

        # ── wait for all evals to finish ──────────────────────────
        for fut in eval_futures:
            fut.result()  # raises if eval failed

    finally:
        eval_pool.shutdown(wait=True)
