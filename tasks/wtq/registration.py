"""WTQ task registrations."""
from __future__ import annotations

from typing import Any, Dict

from task_registry import register_task, seed_predicates_for_dataset
from tasks.wtq.table_qa import TableQA


def _seed_table_qa(store: Any, cfg: Dict[str, Any], split: str) -> None:
    from tasks.wtq.loaders import seed_queries_wtq
    import tasks.wtq.predicates  # noqa: F401

    max_queries = int(cfg.get("max_queries", 0) or 0)
    sample_seed = int(cfg.get("sample_seed", 0) or 0)
    seed_queries_wtq(store, split, max_queries=max_queries, sample_seed=sample_seed)
    seed_predicates_for_dataset(store, dataset="wtq")


register_task(
    "table_qa",
    seeder_fn=_seed_table_qa,
    dataset_key_fn=lambda cfg: "wtq",
)(TableQA)

register_task(
    "wtq",
    seeder_fn=_seed_table_qa,
    dataset_key_fn=lambda cfg: "wtq",
)(TableQA)
