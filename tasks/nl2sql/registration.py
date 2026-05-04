"""NL2SQL task registrations."""
from __future__ import annotations

from typing import Any, Dict

from task_registry import register_task, seed_predicates_for_dataset
from tasks.nl2sql.sql_generation import SqlGeneration
from tasks.nl2sql.sql_repair import SqlRepair


def _seed_sql_generation(store: Any, cfg: Dict[str, Any], split: str) -> None:
    from tasks.nl2sql.loaders import seed_queries_bird, seed_queries_spider

    dataset = cfg.get("dataset", "bird")
    data_dir = cfg.get("data_dir", "")
    max_queries = int(cfg.get("max_queries", 0) or 0)
    sample_seed = int(cfg.get("sample_seed", 0) or 0)
    if dataset == "bird":
        seed_queries_bird(
            store,
            data_dir,
            split,
            max_queries=max_queries,
            sample_seed=sample_seed,
        )
    elif dataset == "spider":
        seed_queries_spider(
            store,
            data_dir,
            split,
            max_queries=max_queries,
            sample_seed=sample_seed,
        )

    try:
        import tasks.nl2sql.predicates  # noqa: F401
        seed_predicates_for_dataset(store, dataset=dataset)
    except ImportError:
        pass


def _seed_sql_repair(store: Any, cfg: Dict[str, Any], split: str) -> None:
    _ = cfg, split
    conn = store._get_conn()
    n = conn.execute("SELECT COUNT(*) FROM query").fetchone()[0]
    if n == 0:
        raise ValueError(
            "Repair dataset not curated yet. Run curate_repair_dataset.py first."
        )


def _seed_bird(store: Any, cfg: Dict[str, Any], split: str) -> None:
    cfg_b = dict(cfg)
    cfg_b["dataset"] = "bird"
    cfg_b.setdefault("data_dir", "/data/users/jsu323/nl2sql/bird")
    _seed_sql_generation(store, cfg_b, split)


def _seed_spider(store: Any, cfg: Dict[str, Any], split: str) -> None:
    cfg_s = dict(cfg)
    cfg_s["dataset"] = "spider"
    cfg_s.setdefault("data_dir", "/data/users/jsu323/nl2sql/spider")
    _seed_sql_generation(store, cfg_s, split)


register_task(
    "sql_generation",
    seeder_fn=_seed_sql_generation,
    dataset_key_fn=lambda cfg: cfg.get("dataset", "bird"),
)(SqlGeneration)

register_task(
    "sql_repair",
    seeder_fn=_seed_sql_repair,
    dataset_key_fn=lambda cfg: cfg.get("dataset", "sql_repair_bird"),
)(SqlRepair)

register_task(
    "bird",
    seeder_fn=_seed_bird,
    dataset_key_fn=lambda cfg: "bird",
)(SqlGeneration)

register_task(
    "spider",
    seeder_fn=_seed_spider,
    dataset_key_fn=lambda cfg: "spider",
)(SqlGeneration)
