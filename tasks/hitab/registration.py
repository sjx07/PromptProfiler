"""HiTab task registrations."""
from __future__ import annotations

from typing import Any, Dict

from task_registry import register_task, seed_predicates_for_dataset
from tasks.hitab.table_qa import HiTabQA


def _seed_hitab(store: Any, cfg: Dict[str, Any], split: str) -> None:
    from tasks.hitab.loaders import seed_queries_hitab
    import tasks.hitab.predicates  # noqa: F401

    max_queries = int(cfg.get("max_queries", 0) or 0)
    sample_seed = int(cfg.get("sample_seed", 0) or 0)
    seed_queries_hitab(
        store,
        split,
        max_queries=max_queries,
        sample_seed=sample_seed,
    )
    seed_predicates_for_dataset(store, dataset="hitab")


register_task(
    "hitab_qa",
    seeder_fn=_seed_hitab,
    dataset_key_fn=lambda cfg: "hitab",
)(HiTabQA)

register_task(
    "hitab",
    seeder_fn=_seed_hitab,
    dataset_key_fn=lambda cfg: "hitab",
)(HiTabQA)
