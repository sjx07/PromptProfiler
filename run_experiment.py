"""Experiment runner — feature-first config composition.

Config shape (example: prompt_profiler/runs/example_add_one.json):

    {
      "task":                "table_qa",                 // required
      "experiment_type":     "add_one_feature",          // add_one_feature | leave_one_out_feature | coalition_feature | base_only
      "db_path":             "runs/wtq_mvp.db",          // required

      "base_features":       ["_section_role", ...],     // canonical_ids that form the base config
      "experiment_features": ["enable_cot", ...],        // canonical_ids, one bundle per config in add_one

      "model":               "meta-llama/Llama-3.1-8B-Instruct",
      "ports":               [8000],
      "num_workers":         32,
      "base_url":            null,                       // optional vLLM override
      "api_key_env":         null,                       // optional

      "split":               "dev",
      "example_split":       "train",                    // required iff any feature uses add_example
      "max_queries":         200,
      "seed":                42,
      "phase":               "wtq_mvp",                  // optional; timestamp auto-appended

      "vllm_db":             "/data/.../autovllm/trajectory.db"
    }

Run:
    python3 -m run_experiment prompt_profiler/runs/example_add_one.json

    # CLI overrides (only for top-level scalars):
    python3 -m run_experiment config.json --model ... --ports 8000 8001
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from autovllm.store import TrajectoryStore as VLLMStore

from common import seed_funcs
from core.feature_registry import FeatureRegistry
from core.store import CubeStore, OnConflict
from execution.pooled_llm import PooledLLMCall
from experiment.config_generators import generate, REGISTRY as GEN_REGISTRY
from experiment.loop import _run_and_eval_plan
from experiment.planner import RunEntry
from task_registry import get_registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ── predicate extractor import (for iterative runs) ──────────────────

def _import_predicate_extractors(task_name: str) -> None:
    """Import task-specific predicate modules (registers extractors on import)."""
    if task_name in ("table_qa", "wtq"):
        import tasks.wtq.predicates  # noqa: F401
    elif task_name in ("fact_verification", "tabfact"):
        import tasks.tabfact.predicates  # noqa: F401
    elif task_name in ("sequential_qa", "sqa"):
        import tasks.sqa.predicates  # noqa: F401
    elif task_name in ("sql_generation", "bird", "spider"):
        try:
            import tasks.nl2sql.predicates  # noqa: F401
        except ImportError:
            pass
    else:
        logger.warning("No predicate extractors for task %s", task_name)


# ── config loader ─────────────────────────────────────────────────────

def _load_config(config_path: str, cli_overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Load experiment config JSON, then apply CLI overrides (non-None only)."""
    with open(config_path) as f:
        cfg = json.load(f)
    for k, v in cli_overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


_GENERATOR_OPTION_KEYS = ("min_features", "max_features", "min_rules", "max_rules", "coalitions")


def _generator_kwargs(cfg: Dict[str, Any], *, n_samples: int, seed: int) -> Dict[str, Any]:
    """Build kwargs forwarded from run config to experiment config generators."""
    kwargs: Dict[str, Any] = {"n_samples": n_samples, "seed": seed}
    for key in _GENERATOR_OPTION_KEYS:
        if cfg.get(key) is not None:
            kwargs[key] = cfg[key]
    return kwargs


def _llm_sampling_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build decoding kwargs for PooledLLMCall without overloading analysis top_k."""
    kwargs: Dict[str, Any] = {}
    if cfg.get("temperature") is not None:
        kwargs["temperature"] = float(cfg["temperature"])
    if cfg.get("top_p") is not None:
        kwargs["top_p"] = float(cfg["top_p"])
    sampling_top_k = cfg.get("sampling_top_k", cfg.get("llm_top_k"))
    if sampling_top_k is not None:
        kwargs["top_k"] = int(sampling_top_k)
    return kwargs


def _example_seed_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build a seeder config for example_split without reusing eval max_queries.

    Eval configs often cap ``max_queries`` for fast runs. Few-shot/example pools
    should be controlled independently, otherwise a 10-query smoke run silently
    leaves only 10 train examples.
    """
    out = dict(cfg)
    out["max_queries"] = int(
        cfg.get("max_example_queries",
                cfg.get("max_train_queries",
                        cfg.get("example_max_queries",
                                cfg.get("max_queries", 0)))) or 0
    )
    out["sample_seed"] = int(
        cfg.get("example_sample_seed",
                cfg.get("train_sample_seed",
                        cfg.get("sample_seed", 0))) or 0
    )
    return out


# ── feature materialization ───────────────────────────────────────────

def _build_feature_bundles(
    feat_reg: FeatureRegistry,
    base_features: List[str],
    experiment_features: List[str],
) -> Tuple[
    List[dict],
    List[str],
    Dict[str, Tuple[str, List[str]]],
    Dict[str, str],
    Dict[str, frozenset],
]:
    """Materialize base + experiment features through the FeatureRegistry.

    Each experiment feature is materialized *independently* alongside the
    base set, so that mutually-conflicting experiment features (e.g.
    ``enable_code`` vs ``enable_sql``) can coexist as separate add-one
    bundles.  Conflict enforcement happens at config-composition time
    inside the generator.

    Returns:
        full_specs:    deduped list of all func specs to seed into the store.
        base_ids:      func_ids that compose the base config.
        bundles:       canonical_id -> (feature_id_hash, incremental func_ids).
        base_feature_hashes: canonical_id -> feature_id_hash for base features.
        conflicts:     canonical_id -> frozenset[canonical_id] of declared
                       conflicts_with relations for experiment features,
                       for feature-aware generators to consult.
    """
    base_specs, _base_f2f = feat_reg.materialize(base_features)
    base_ids = [s["func_id"] for s in base_specs]
    base_set = set(base_ids)

    all_specs_by_fid: Dict[str, dict] = {s["func_id"]: s for s in base_specs}
    bundles: Dict[str, Tuple[str, List[str]]] = {}
    conflicts: Dict[str, frozenset] = {}

    for cid in experiment_features:
        # Materialize this feature with base; validates requires against base,
        # and skips cross-sibling conflict validation.
        feat_specs, feat_f2f = feat_reg.materialize(base_features + [cid])
        for s in feat_specs:
            all_specs_by_fid[s["func_id"]] = s
        feat_hash = feat_reg.feature_id_for(cid)
        feat_fids = feat_f2f[feat_hash]
        add_funcs = [f for f in feat_fids if f not in base_set]
        bundles[cid] = (feat_hash, add_funcs)

        spec = feat_reg._by_canonical[cid]
        conflicts[cid] = frozenset(spec.get("conflicts_with", []))

    full_specs = list(all_specs_by_fid.values())
    base_feature_hashes = {cid: feat_reg.feature_id_for(cid) for cid in base_features}
    return full_specs, base_ids, bundles, base_feature_hashes, conflicts


# ── main ──────────────────────────────────────────────────────────────

def main():
    task_registry = get_registry()

    parser = argparse.ArgumentParser(description="Feature-first cube experiment runner")
    parser.add_argument("config", help="Experiment config JSON path")
    parser.add_argument("--task", default=None, choices=list(task_registry.keys()))
    parser.add_argument("--experiment_type", default=None,
                        choices=list(GEN_REGISTRY.keys()) + ["iterative"])
    parser.add_argument("--db_path", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--ports", type=int, nargs="+", default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--sampling_top_k", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--min_features", type=int, default=None)
    parser.add_argument("--max_features", type=int, default=None)
    parser.add_argument("--min_rules", type=int, default=None)
    parser.add_argument("--max_rules", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--example_split", default=None)
    parser.add_argument("--phase", default=None)
    parser.add_argument("--vllm_db", default=None)
    args = parser.parse_args()

    cli_overrides = {k: v for k, v in vars(args).items()
                     if k != "config" and v is not None}
    cfg = _load_config(args.config, cli_overrides)

    # ── required fields ──────────────────────────────────────────────
    task_name = cfg.get("task")
    if not task_name:
        parser.error("'task' must be in config or --task")
    if task_name not in task_registry:
        parser.error(f"Unknown task: {task_name}. Choose from: {list(task_registry.keys())}")
    task_entry = task_registry[task_name]
    task_cls = task_entry.task_cls
    if hasattr(task_cls, "configure_from_cfg"):
        task_cls.configure_from_cfg(cfg)

    experiment_type = cfg.get("experiment_type", "add_one_feature")
    db_path = cfg.get("db_path")
    if not db_path:
        parser.error("'db_path' must be in config or --db_path")

    base_features: List[str] = cfg.get("base_features", [])
    experiment_features: List[str] = cfg.get("experiment_features", [])
    if not base_features:
        parser.error("'base_features' required (list of canonical_ids forming the base config)")

    # ── scalar params with defaults ──────────────────────────────────
    model = cfg.get("model", "meta-llama/Llama-3.1-8B-Instruct")
    ports = cfg.get("ports", [8000])
    num_workers = cfg.get("num_workers", 32)
    base_url = cfg.get("base_url")
    api_key = os.environ.get(cfg["api_key_env"], "") if cfg.get("api_key_env") else None
    max_queries = cfg.get("max_queries", 0)
    max_tokens = int(cfg.get("max_tokens", 2048))
    llm_sampling_kwargs = _llm_sampling_kwargs(cfg)
    n_samples = cfg.get("n_samples", 200)
    seed = cfg.get("seed", 42)
    split = cfg.get("split", "dev")
    example_split = cfg.get("example_split")
    vllm_db = cfg.get("vllm_db", "/data/users/jsu323/autovllm/trajectory.db")

    phase_base = cfg.get("phase")
    phase = f"{phase_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if phase_base else None

    # ── store + feature registry ─────────────────────────────────────
    if phase:
        logger.info("Phase: %s", phase)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    store = CubeStore(db_path)

    try:
        feat_reg = FeatureRegistry.load(task=task_name)
    except FileNotFoundError as e:
        raise SystemExit(
            f"No feature directory for task={task_name!r}: {e}\n"
            f"Expected: {Path(__file__).parent / 'features' / task_name}"
        )

    sync_result = feat_reg.sync_to_cube(store)
    logger.info("Feature registry synced: %s", sync_result)

    # ── seed dataset queries ─────────────────────────────────────────
    task_entry.seeder_fn(store, cfg, split)

    # ── materialize features → func specs ────────────────────────────
    full_specs, base_ids, bundles, base_feature_hashes, conflicts = _build_feature_bundles(
        feat_reg, base_features, experiment_features,
    )
    seed_funcs(store, full_specs)
    logger.info("Seeded %d func specs (base_features=%d, experiment_features=%d)",
                len(full_specs), len(base_features), len(experiment_features))

    has_examples = any(s["func_type"] == "add_example" for s in full_specs)
    if has_examples:
        if not example_split:
            raise ValueError("'example_split' required when any feature uses add_example")
        if example_split == split:
            raise ValueError(f"example_split must differ from split (both are {example_split!r})")
        task_entry.seeder_fn(store, _example_seed_cfg(cfg), example_split)

    # ── base config ──────────────────────────────────────────────────
    base_cid = store.get_or_create_config(
        base_ids,
        meta={
            "kind": "base",
            "canonical_ids": list(base_features),
            "feature_ids": [base_feature_hashes[c] for c in base_features],
        },
    )
    logger.info("Base config: id=%d, %d funcs, %d features",
                base_cid, len(base_ids), len(base_features))

    # ── queries ──────────────────────────────────────────────────────
    dataset_key = task_entry.dataset_key_fn(cfg)
    conn = store._get_conn()
    rows = conn.execute(
        "SELECT * FROM query WHERE dataset = ? AND json_extract(meta, '$.split') = ?",
        (dataset_key, split),
    ).fetchall()
    queries = [dict(r) for r in rows]
    if max_queries > 0:
        queries = queries[:max_queries]
    query_ids = [q["query_id"] for q in queries]
    logger.info("Queries: %d (dataset=%s split=%s)", len(queries), dataset_key, split)

    # ── example pool ─────────────────────────────────────────────────
    example_pool = None
    if has_examples:
        pool_rows = conn.execute(
            "SELECT * FROM query WHERE dataset = ? AND json_extract(meta, '$.split') = ?",
            (dataset_key, example_split),
        ).fetchall()
        example_pool = [dict(r) for r in pool_rows]
        logger.info("Example pool: %d queries (split=%s)", len(example_pool), example_split)

    # ── LLM call setup ───────────────────────────────────────────────
    slots_per_port = max(1, num_workers // len(ports))
    llm_call = PooledLLMCall(
        model, ports, slots_per_port=slots_per_port,
        labels={"task": task_name, "experiment_type": experiment_type,
                "dataset": dataset_key, "split": split,
                "num_workers": num_workers, "max_tokens": max_tokens,
                **llm_sampling_kwargs},
        vllm_store=VLLMStore(vllm_db),
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        **llm_sampling_kwargs,
    )
    logger.info(
        "Port pool: %s, %d workers, %d slots/port, max_tokens=%d, sampling=%s",
        ports,
        num_workers,
        slots_per_port,
        max_tokens,
        llm_sampling_kwargs,
    )

    # ══════════════════════════════════════════════════════════════════
    # Iterative experiment (unchanged from pre-refactor; rule_ids path).
    # ══════════════════════════════════════════════════════════════════
    if experiment_type == "iterative":
        _run_iterative(
            cfg, store, task_cls, model, llm_call,
            base_cid=base_cid, base_ids=base_ids, bundles=bundles,
            query_ids=query_ids, dataset_key=dataset_key,
            num_workers=num_workers, example_pool=example_pool, seed=seed,
        )
        llm_call.close()
        store.close()
        return

    # ══════════════════════════════════════════════════════════════════
    # Single-pass feature experiment.
    # ══════════════════════════════════════════════════════════════════
    configs = generate(
        experiment_type, store,
        base_ids=base_ids, bundles=bundles, conflicts=conflicts,
        base_canonical_ids=list(base_features),
        base_feature_ids=[base_feature_hashes[c] for c in base_features],
        **_generator_kwargs(cfg, n_samples=n_samples, seed=seed),
    )
    logger.info("Generated %d configs (%s)", len(configs), experiment_type)

    plan = [RunEntry(config_id=base_cid, func_ids=base_ids, query_ids=query_ids)]
    for cid, func_ids, meta in configs:
        plan.append(RunEntry(config_id=cid, func_ids=func_ids, query_ids=query_ids, meta=meta))

    logger.info("Plan: %d configs, %d total LLM calls",
                len(plan), sum(len(e.query_ids) for e in plan))

    dataset_key = task_entry.dataset_key_fn(cfg) or ""
    _run_and_eval_plan(store, plan, task_cls, model, llm_call,
                       num_workers=num_workers, on_conflict=OnConflict.SKIP,
                       example_pool=example_pool, phase=phase,
                       dataset=dataset_key)

    # ── print summary ────────────────────────────────────────────────
    _print_results(store, task_cls, model, base_cid, configs, experiment_type, task_name)

    llm_call.close()
    store.close()


# ── iterative branch (kept for advanced usage; converts bundles → rule_ids) ──

def _run_iterative(
    cfg: Dict[str, Any],
    store: CubeStore,
    task_cls: type,
    model: str,
    llm_call: PooledLLMCall,
    *,
    base_cid: int,
    base_ids: List[str],
    bundles: Dict[str, Tuple[str, List[str]]],
    query_ids: List[str],
    dataset_key: str,
    num_workers: int,
    example_pool: Any,
    seed: int,
) -> None:
    """Iterative loop: treats each feature bundle as a single PrimitiveSpec."""
    from experiment.analysis import (
        PrimitiveSpec, make_predicate_aware_analyzer, make_seed_plan,
    )
    from experiment.loop import run_experiment as run_loop
    from experiment.query_cohorts import seed_predicates

    conn = store._get_conn()
    pred_count = conn.execute("SELECT COUNT(*) FROM predicate").fetchone()[0]
    if pred_count == 0:
        _import_predicate_extractors(cfg.get("task"))
        n_seeded = seed_predicates(store, dataset=dataset_key)
        logger.info("Seeded %d predicate rows", n_seeded)
    else:
        logger.info("Predicates already seeded: %d rows", pred_count)

    predicate_names = cfg.get("predicate_names", [
        "operation_type", "is_count", "has_aggregation", "has_superlative",
    ])
    seed_n_primitives = cfg.get("seed_n_primitives", 3)
    seed_n_queries = cfg.get("seed_n_queries", 200)
    seed_predicate = cfg.get("seed_predicate")
    max_iterations = cfg.get("max_iterations", 5)
    iter_top_k = cfg.get("top_k", 10)
    iter_budget = cfg.get("query_budget_per_cell", 50)
    iter_max_new = cfg.get("max_new_queries", 500)

    prim_specs = [
        PrimitiveSpec(name=cid, func_ids=add_funcs)
        for cid, (_fid, add_funcs) in bundles.items()
    ]

    initial_plan = make_seed_plan(
        store, base_ids, prim_specs, query_ids,
        n_primitives=seed_n_primitives, n_queries=seed_n_queries,
        seed=seed, predicate_name=seed_predicate,
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


# ── results printer ─────────────────────────────────────────────────

def _print_results(
    store: CubeStore,
    task_cls: type,
    model: str,
    base_cid: int,
    configs: List[tuple],
    experiment_type: str,
    task_name: str,
) -> None:
    scorer = task_cls.scorer
    scores = store.scores_by_config(model, scorer)
    base_avg = next((s["avg_score"] for s in scores if s["config_id"] == base_cid), 0.0)

    experiment_cids = {cid for cid, _, _ in configs}
    config_meta = {cid: m for cid, _, m in configs}

    print(f"\n=== {experiment_type} Results on {task_name} (base avg={base_avg:.3f}) ===")
    results = []
    for s in scores:
        if s["config_id"] == base_cid:
            results.append((-999, f"  BASE:                    avg={s['avg_score']:.3f}  n={s['n']}"))
        elif s["config_id"] in experiment_cids:
            meta = config_meta[s["config_id"]]
            delta = s["avg_score"] - base_avg
            sign = "+" if delta >= 0 else ""
            label = (
                meta.get("canonical_id")
                or (f"-{meta['removed_canonical_id']}" if "removed_canonical_id" in meta else None)
                or (f"{meta['n_features']}feats" if "n_features" in meta else None)
                or f"c{s['config_id']}"
            )
            results.append((
                delta,
                f"  {label:24s} avg={s['avg_score']:.3f}  delta={sign}{delta:.3f}  n={s['n']}",
            ))
    for _, line in sorted(results, key=lambda x: -x[0]):
        print(line)


if __name__ == "__main__":
    main()
