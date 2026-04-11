"""Experiment runner — config-first with CLI overrides.

Usage:
    python3 -m prompt_profiler.run_experiment \
        prompt_profiler/runs/exp_wtq_7b_code.json

    # CLI overrides:
    python3 -m prompt_profiler.run_experiment \
        prompt_profiler/runs/exp_wtq_7b_code.json \
        --model Qwen/Qwen2.5-7B-Instruct --ports 8000 --num_workers 64
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from autovllm.store import TrajectoryStore as VLLMStore

from prompt_profiler.common import seed_funcs
from prompt_profiler.experiment.config_generators import generate, REGISTRY as GEN_REGISTRY
from prompt_profiler.experiment.loop import _run_and_eval_plan
from prompt_profiler.experiment.planner import RunEntry
from prompt_profiler.core.func_registry import make_func_id
from prompt_profiler.core.store import CubeStore, OnConflict
from prompt_profiler.execution.pooled_llm import PooledLLMCall
from prompt_profiler.task_registry import get_registry
from prompt_profiler.common import seed_pool, resolve_node_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _import_predicate_extractors(task_name: str) -> None:
    """Import task-specific predicate modules (registers extractors on import)."""
    if task_name == "table_qa":
        import prompt_profiler.tasks.wtq.predicates  # noqa: F401
    elif task_name == "fact_verification":
        import prompt_profiler.tasks.tabfact.predicates  # noqa: F401
    else:
        logger.warning("No predicate extractors for task %s", task_name)


def _load_config(config_path: str, cli_overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Load experiment config JSON, then apply CLI overrides."""
    with open(config_path) as f:
        cfg = json.load(f)
    # Resolve pool path relative to config file
    if "pool" in cfg:
        pool_rel = Path(config_path).parent / cfg["pool"]
        if pool_rel.exists():
            cfg["pool"] = str(pool_rel)
    # CLI overrides win (only for non-None values)
    for k, v in cli_overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def main():
    registry = get_registry()

    parser = argparse.ArgumentParser(description="Unified cube experiment runner")
    parser.add_argument("config", help="Experiment config JSON path")
    parser.add_argument("--task", default=None, choices=list(registry.keys()))
    parser.add_argument("--experiment_type", default=None,
                        choices=list(GEN_REGISTRY.keys()) + ["iterative"])
    parser.add_argument("--db_path", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--ports", type=int, nargs="+", default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--max_rules", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--example_split", default=None)
    parser.add_argument("--vllm_db", default=None)
    args = parser.parse_args()

    # Build overrides dict (only explicitly provided CLI args)
    cli_overrides = {k: v for k, v in vars(args).items()
                     if k != "config" and v is not None}
    cfg = _load_config(args.config, cli_overrides)

    # ── Resolve required fields (config + defaults) ───────────────────
    task_name = cfg.get("task")
    if not task_name:
        parser.error("'task' must be in config or --task")
    if task_name not in registry:
        parser.error(f"Unknown task: {task_name}. Choose from: {list(registry.keys())}")
    entry = registry[task_name]
    task_cls = entry.task_cls

    experiment_type = cfg.get("experiment_type")
    if not experiment_type:
        parser.error("'experiment_type' must be in config or --experiment_type")

    db_path = cfg.get("db_path")
    if not db_path:
        parser.error("'db_path' must be in config or --db_path")

    model = cfg.get("model", "Qwen/Qwen2.5-Coder-32B-Instruct")
    ports = cfg.get("ports", [8000, 8001])
    num_workers = cfg.get("num_workers", 48)
    base_url = cfg.get("base_url", None)
    api_key = os.environ.get(cfg["api_key_env"], "") if "api_key_env" in cfg else None
    max_queries = cfg.get("max_queries", 0)
    max_rules = cfg.get("max_rules", 0)
    n_samples = cfg.get("n_samples", 200)
    seed = cfg.get("seed", 42)
    split = cfg.get("split", "dev")
    example_split = cfg.get("example_split")
    vllm_db = cfg.get("vllm_db", "/data/users/jsu323/autovllm/trajectory.db")
    pool_path = cfg.get("pool", "")
    phase_base = cfg.get("phase")
    if phase_base:
        from datetime import datetime
        phase = f"{phase_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        phase = None

    # ── Setup ─────────────────────────────────────────────────────────
    if phase:
        logger.info("Phase: %s", phase)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    store = CubeStore(db_path)

    # ── Seed ──────────────────────────────────────────────────────────
    if pool_path:
        seed_pool(store, pool_path)

    entry.seeder_fn(store, cfg, split)

    # Base funcs — resolve node_id refs to full pool params (no-op if no node_ids)
    base_specs = cfg.get("base", [])
    if base_specs and pool_path:
        base_specs = resolve_node_ids(base_specs, pool_path)
    if base_specs:
        seed_funcs(store, base_specs)

    # Experiment funcs (the "experiment" key in config is the func list)
    exp_specs = cfg.get("experiment", [])
    if exp_specs and pool_path:
        exp_specs = resolve_node_ids(exp_specs, pool_path)
    if exp_specs:
        seed_funcs(store, exp_specs)

    # ── Build base config ─────────────────────────────────────────────
    all_funcs = store.list_funcs()
    section_ids = [f["func_id"] for f in all_funcs if f["func_type"] == "define_section"]
    base_ids = list(section_ids)
    for spec in base_specs:
        base_ids.append(make_func_id(spec["func_type"], spec.get("params", {})))

    base_cid = store.get_or_create_config(base_ids)
    logger.info("Base config: id=%d, %d funcs", base_cid, len(base_ids))

    # Experiment pool: from pool.json + experiment specs
    base_set = set(base_ids)
    pool_func_ids = set()
    if pool_path:
        with open(pool_path) as _f:
            pool_data = json.load(_f)
        for section in pool_data.get("sections", []):
            for child in section.get("children", []):
                if child.get("node_type") == "rule":
                    fid = make_func_id("add_rule", {"content": child["content"]})
                    if fid not in base_set:
                        pool_func_ids.add(fid)
    exp_func_ids = [make_func_id(s["func_type"], s.get("params", {})) for s in exp_specs]
    if exp_func_ids:
        rule_ids = exp_func_ids  # experiment specs override pool
    else:
        rule_ids = sorted(pool_func_ids)  # no experiment specs → use full pool
    logger.info("Experiment pool: %d funcs (%d pool rules, %d experiment)",
                len(rule_ids), len(pool_func_ids), len(exp_func_ids))

    # ── Load queries ──────────────────────────────────────────────────
    dataset_key = entry.dataset_key_fn(cfg)
    conn = store._get_conn()
    rows = conn.execute(
        "SELECT * FROM query WHERE dataset = ? AND json_extract(meta, '$.split') = ?",
        (dataset_key, split),
    ).fetchall()
    queries = [dict(r) for r in rows]
    if max_queries > 0:
        queries = queries[:max_queries]
    query_ids = [q["query_id"] for q in queries]
    logger.info("Queries: %d", len(queries))

    # ── LLM call setup ───────────────────────────────────────────────
    slots_per_port = max(1, num_workers // len(ports))

    llm_call = PooledLLMCall(
        model, ports, slots_per_port=slots_per_port,
        labels={"task": task_name, "experiment_type": experiment_type,
                "dataset": dataset_key, "split": split,
                "num_workers": num_workers},
        vllm_store=VLLMStore(vllm_db),
        base_url=base_url,
        api_key=api_key,
    )
    logger.info("Port pool: %s, %d workers, %d slots/port",
                ports, num_workers, slots_per_port)

    # ── Example pool ──────────────────────────────────────────────────
    all_specs = base_specs + exp_specs
    has_examples = any(s.get("func_type") == "add_example" for s in all_specs)
    example_pool = None
    if has_examples:
        if not example_split:
            raise ValueError("'example_split' required when using add_example funcs")
        if example_split == split:
            raise ValueError(f"example_split must differ from split (both are '{example_split}')")
        pool_rows = conn.execute(
            "SELECT * FROM query WHERE dataset = ? AND json_extract(meta, '$.split') = ?",
            (dataset_key, example_split),
        ).fetchall()
        example_pool = [dict(r) for r in pool_rows]
        logger.info("Example pool: %d queries (split=%s)", len(example_pool), example_split)

    # ══════════════════════════════════════════════════════════════════
    # Iterative experiment: seed → analyze → target → repeat
    # ══════════════════════════════════════════════════════════════════
    if experiment_type == "iterative":
        from prompt_profiler.experiment.analysis import (
            PrimitiveSpec, make_predicate_aware_analyzer, make_seed_plan,
        )
        from prompt_profiler.experiment.loop import run_experiment as run_loop
        from prompt_profiler.experiment.query_cohorts import seed_predicates

        # Seed predicates if empty
        pred_count = conn.execute("SELECT COUNT(*) FROM predicate").fetchone()[0]
        if pred_count == 0:
            # Import task-specific extractors (registers them on import)
            _import_predicate_extractors(task_name)
            n_seeded = seed_predicates(store, dataset=dataset_key)
            logger.info("Seeded %d predicate rows", n_seeded)
        else:
            logger.info("Predicates already seeded: %d rows", pred_count)

        # Config from JSON
        predicate_names = cfg.get("predicate_names", [
            "operation_type", "is_count", "has_aggregation", "has_superlative",
        ])
        seed_n_primitives = cfg.get("seed_n_primitives", 3)
        seed_n_queries = cfg.get("seed_n_queries", 200)
        seed_predicate = cfg.get("seed_predicate", None)
        max_iterations = cfg.get("max_iterations", 5)
        iter_top_k = cfg.get("top_k", 10)
        iter_budget = cfg.get("query_budget_per_cell", 50)
        iter_max_new = cfg.get("max_new_queries", 500)

        # Wrap rule_ids as single-func PrimitiveSpecs
        prim_specs = [PrimitiveSpec(name=rid, func_ids=[rid]) for rid in rule_ids]

        initial_plan = make_seed_plan(
            store, base_ids, prim_specs, query_ids,
            n_primitives=seed_n_primitives,
            n_queries=seed_n_queries,
            seed=seed,
            predicate_name=seed_predicate,
        )

        analyze_fn = make_predicate_aware_analyzer(
            base_cid=base_cid, base_ids=base_ids,
            primitive_specs=prim_specs,
            predicate_names=predicate_names,
            top_k=iter_top_k,
            query_budget_per_cell=iter_budget,
            max_new_queries=iter_max_new,
        )

        all_insights = run_loop(
            store, initial_plan, task_cls, model, llm_call, analyze_fn,
            num_workers=num_workers, max_iterations=max_iterations,
            on_conflict=OnConflict.SKIP, example_pool=example_pool,
        )

        # ── Print iterative summary ──────────────────────────────
        scorer = task_cls.scorer
        scores = store.scores_by_config(model, scorer)
        base_avg = next((s["avg_score"] for s in scores if s["config_id"] == base_cid), 0.0)

        print(f"\n=== Iterative Results ({len(all_insights)} iterations, base avg={base_avg:.3f}) ===")
        for i, ins in enumerate(all_insights, 1):
            n_q = ins.get("n_queries", 0)
            n_p = ins.get("n_primitives", 0)
            cov = ins.get("coverage", 0)
            print(f"  Iter {i}: {n_q} queries, {n_p} primitives, coverage={cov:.2f}")
            print(f"    always_on:  {ins.get('always_on', [])}")
            print(f"    always_off: {ins.get('always_off', [])}")
            print(f"    gated:      {ins.get('gated', [])}")

        llm_call.close()
        store.close()
        return

    # ══════════════════════════════════════════════════════════════════
    # Single-pass experiment: generate all configs, run once
    # ══════════════════════════════════════════════════════════════════
    configs = generate(experiment_type, store, base_ids, rule_ids,
                       max_rules=max_rules, n_samples=n_samples, seed=seed)
    logger.info("Generated %d configs (%s)", len(configs), experiment_type)

    plan = [RunEntry(config_id=base_cid, func_ids=base_ids, query_ids=query_ids)]
    for cid, func_ids, meta in configs:
        plan.append(RunEntry(config_id=cid, func_ids=func_ids, query_ids=query_ids, meta=meta))

    logger.info("Plan: %d configs, %d total LLM calls",
                len(plan), sum(len(e.query_ids) for e in plan))

    _run_and_eval_plan(store, plan, task_cls, model, llm_call,
                       num_workers=num_workers, on_conflict=OnConflict.SKIP,
                       example_pool=example_pool, phase=phase)

    # ── Results ───────────────────────────────────────────────────────
    scorer = task_cls.scorer
    scores = store.scores_by_config(model, scorer)
    base_avg = next((s["avg_score"] for s in scores if s["config_id"] == base_cid), 0.0)

    experiment_cids = {cid for cid, _, _ in configs}
    config_meta = {cid: m for cid, _, m in configs}

    print(f"\n=== {experiment_type} Results (base avg={base_avg:.3f}) ===")
    results = []
    for s in scores:
        if s["config_id"] == base_cid:
            results.append((-999, f"  BASE:           avg={s['avg_score']:.3f}  n={s['n']}"))
        elif s["config_id"] in experiment_cids:
            meta = config_meta[s["config_id"]]
            delta = s["avg_score"] - base_avg
            sign = "+" if delta >= 0 else ""
            if "added_rule" in meta:
                label = meta["added_rule"][:12]
            elif "removed_rule" in meta:
                label = f"-{meta['removed_rule'][:11]}"
            elif "n_rules" in meta:
                label = f"{meta['n_rules']}rules"
            else:
                label = f"c{s['config_id']}"
            results.append((delta, f"  {label:14s}  avg={s['avg_score']:.3f}  delta={sign}{delta:.3f}  n={s['n']}"))

    for _, line in sorted(results, key=lambda x: -x[0]):
        print(line)

    llm_call.close()
    store.close()


if __name__ == "__main__":
    main()
