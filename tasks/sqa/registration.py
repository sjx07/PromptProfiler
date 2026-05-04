"""Sequential QA task registrations."""
from __future__ import annotations

from typing import Any, Dict

from task_registry import register_task, seed_predicates_for_dataset
from tasks.sqa.sequential_qa import SequentialQA


def _seed_sequential_qa(store: Any, cfg: Dict[str, Any], split: str) -> None:
    from tasks.sqa.loaders import DEFAULT_DATA_DIR, seed_queries_sqa
    import tasks.sqa.predicates  # noqa: F401

    data_dir = cfg.get("data_dir") or cfg.get("sqa_data_dir") or DEFAULT_DATA_DIR
    max_queries = int(cfg.get("max_queries", 0) or 0)
    sample_seed = int(cfg.get("sample_seed", 0) or 0)
    seed_queries_sqa(
        store,
        split,
        data_dir=data_dir,
        max_queries=max_queries,
        sample_seed=sample_seed,
    )
    seed_predicates_for_dataset(store, dataset="sqa")


register_task(
    "sequential_qa",
    seeder_fn=_seed_sequential_qa,
    dataset_key_fn=lambda cfg: "sqa",
)(SequentialQA)

register_task(
    "sqa",
    seeder_fn=_seed_sequential_qa,
    dataset_key_fn=lambda cfg: "sqa",
)(SequentialQA)
