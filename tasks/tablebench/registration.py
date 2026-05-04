"""TableBench task registrations."""
from __future__ import annotations

import json
from typing import Any, Dict

from task_registry import register_task, seed_predicates_for_dataset
from tasks.tablebench.repro import TableBenchRepro
from tasks.tablebench.table_bench import TableBench


def _seed_tablebench(store: Any, cfg: Dict[str, Any], split: str) -> None:
    from tasks.tablebench.loaders import seed_queries_tablebench
    import tasks.tablebench.predicates  # noqa: F401

    max_queries = int(cfg.get("max_queries", 0) or 0)
    sample_seed = int(cfg.get("sample_seed", 0) or 0)
    dataset_revision = cfg.get("dataset_revision")
    hf_cache_dir = cfg.get("hf_cache_dir") or cfg.get("cache_dir")
    train_revision = cfg.get("train_dataset_revision")
    train_instruction_types = cfg.get("train_instruction_types")
    train_data_path = cfg.get("train_data_path")
    include_visualization = bool(cfg.get("include_visualization", False))
    seed_queries_tablebench(
        store,
        split,
        revision=dataset_revision,
        cache_dir=hf_cache_dir,
        train_revision=train_revision,
        train_instruction_types=train_instruction_types,
        train_data_path=train_data_path,
        include_visualization=include_visualization,
        max_queries=max_queries,
        sample_seed=sample_seed,
    )
    seed_predicates_for_dataset(store, dataset="tablebench")


def _seed_tablebench_repro(store: Any, cfg: Dict[str, Any], split: str) -> None:
    from experiment.query_cohorts import register_extractor
    from tasks.tablebench.repro import seed_queries_tablebench_repro
    import tasks.tablebench.predicates  # noqa: F401

    @register_extractor("variant")
    def _variant_extractor(query: dict) -> str:
        meta = query.get("meta", {})
        if isinstance(meta, str):
            meta = json.loads(meta)
        return meta.get("variant", "unknown")

    max_queries = int(cfg.get("max_queries", 0) or 0)
    sample_seed = int(cfg.get("sample_seed", 0) or 0)
    dataset_revision = cfg.get("dataset_revision")
    hf_cache_dir = cfg.get("hf_cache_dir") or cfg.get("cache_dir")
    include_visualization = bool(cfg.get("include_visualization", False))
    variants = cfg.get("variants") or ["DP", "TCoT", "SCoT", "PoT"]
    if isinstance(variants, str):
        variants = [variants]
    for variant in variants:
        seed_queries_tablebench_repro(
            store,
            split,
            variant=variant,
            revision=dataset_revision,
            cache_dir=hf_cache_dir,
            include_visualization=include_visualization,
            max_queries=max_queries,
            sample_seed=sample_seed,
        )
    seed_predicates_for_dataset(store, dataset="tablebench_repro")


register_task(
    "tablebench",
    seeder_fn=_seed_tablebench,
    dataset_key_fn=lambda cfg: "tablebench",
)(TableBench)

register_task(
    "tablebench_repro",
    seeder_fn=_seed_tablebench_repro,
    dataset_key_fn=lambda cfg: "tablebench_repro",
)(TableBenchRepro)
