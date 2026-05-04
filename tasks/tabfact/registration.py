"""TabFact task registrations."""
from __future__ import annotations

from typing import Any, Dict

from task_registry import register_task, seed_predicates_for_dataset
from tasks.tabfact.fact_verification import FactVerification


def _seed_tabfact(store: Any, cfg: Dict[str, Any], split: str) -> None:
    from tasks.tabfact.loaders import seed_queries_tabfact
    import tasks.tabfact.predicates  # noqa: F401

    max_queries = int(cfg.get("max_queries", 0) or 0)
    sample_seed = int(cfg.get("sample_seed", 0) or 0)
    seed_queries_tabfact(store, split, max_queries=max_queries, sample_seed=sample_seed)
    seed_predicates_for_dataset(store, dataset="tab_fact")


register_task(
    "fact_verification",
    seeder_fn=_seed_tabfact,
    dataset_key_fn=lambda cfg: "tab_fact",
)(FactVerification)

register_task(
    "tabfact",
    seeder_fn=_seed_tabfact,
    dataset_key_fn=lambda cfg: "tab_fact",
)(FactVerification)
