"""Task registry — maps task names to classes, seeders, and dataset keys."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class TaskEntry:
    task_cls: type
    seeder_fn: Callable  # (store, cfg, split) -> None
    dataset_key_fn: Callable  # (cfg) -> str


# ── Seeders ──────────────────────────────────────────────────────────
# Each seeder inserts queries AND seeds predicates (via registered
# extractors). Predicate seeding is idempotent — calls on an already-
# seeded cube insert zero new rows. This removes the manual
# `seed_predicates` step that previously had to run before any predicate-
# aware analysis.


def _seed_predicates(store: Any, dataset: str) -> None:
    """Run all registered predicate extractors for a dataset (idempotent)."""
    from experiment.query_cohorts import seed_predicates
    n = seed_predicates(store, dataset=dataset)
    if n:
        import logging
        logging.getLogger(__name__).info(
            "Seeded %d predicate rows for dataset=%s", n, dataset,
        )


def _seed_table_qa(store: Any, cfg: Dict[str, Any], split: str) -> None:
    from tasks.wtq.loaders import seed_queries_wtq
    import tasks.wtq.predicates  # noqa: F401 — registers extractors on import
    seed_queries_wtq(store, split)
    _seed_predicates(store, dataset="wtq")


def _seed_tabfact(store: Any, cfg: Dict[str, Any], split: str) -> None:
    from tasks.tabfact.loaders import seed_queries_tabfact
    import tasks.tabfact.predicates  # noqa: F401 — registers extractors on import
    seed_queries_tabfact(store, split)
    _seed_predicates(store, dataset="tab_fact")


def _seed_sql_generation(store: Any, cfg: Dict[str, Any], split: str) -> None:
    from tasks.nl2sql.loaders import seed_queries_bird, seed_queries_spider
    dataset = cfg.get("dataset", "bird")
    data_dir = cfg.get("data_dir", "")
    if dataset == "bird":
        seed_queries_bird(store, data_dir, split)
    elif dataset == "spider":
        seed_queries_spider(store, data_dir, split)
    # nl2sql predicates are DB-introspection based; if a predicates module
    # exists, importing it registers extractors. Skip gracefully otherwise.
    try:
        import tasks.nl2sql.predicates  # noqa: F401
        _seed_predicates(store, dataset=dataset)
    except ImportError:
        pass


def _seed_sql_repair(store: Any, cfg: Dict[str, Any], split: str) -> None:
    # Repair dataset is pre-curated via curate_repair_dataset.py.
    # The seeder just checks that queries exist.
    conn = store._get_conn()
    n = conn.execute("SELECT COUNT(*) FROM query").fetchone()[0]
    if n == 0:
        raise ValueError(
            "Repair dataset not curated yet. Run curate_repair_dataset.py first."
        )


def _seed_pupa(store: Any, cfg: Dict[str, Any], split: str) -> None:
    from tasks.pupa.loaders import seed_queries_pupa
    data_path = cfg.get("data_path") or cfg.get("pupa_data_path")
    if not data_path:
        raise ValueError(
            "PUPA requires cfg.data_path pointing to a local PAPILLON/PUPA JSON or JSONL file."
        )
    seed_queries_pupa(store, data_path, split)


# ── Registry ─────────────────────────────────────────────────────────

def _build_registry() -> Dict[str, TaskEntry]:
    from tasks.wtq.table_qa import TableQA
    from tasks.tabfact.fact_verification import FactVerification
    from tasks.nl2sql.sql_generation import SqlGeneration
    from tasks.nl2sql.sql_repair import SqlRepair
    from tasks.pupa.pupa import PupaPrivacyDelegationTask

    return {
        "table_qa": TaskEntry(
            task_cls=TableQA,
            seeder_fn=_seed_table_qa,
            dataset_key_fn=lambda cfg: "wtq",
        ),
        "fact_verification": TaskEntry(
            task_cls=FactVerification,
            seeder_fn=_seed_tabfact,
            dataset_key_fn=lambda cfg: "tab_fact",
        ),
        "sql_generation": TaskEntry(
            task_cls=SqlGeneration,
            seeder_fn=_seed_sql_generation,
            dataset_key_fn=lambda cfg: cfg.get("dataset", "bird"),
        ),
        "sql_repair": TaskEntry(
            task_cls=SqlRepair,
            seeder_fn=_seed_sql_repair,
            dataset_key_fn=lambda cfg: cfg.get("dataset", "sql_repair_bird"),
        ),
        "pupa": TaskEntry(
            task_cls=PupaPrivacyDelegationTask,
            seeder_fn=_seed_pupa,
            dataset_key_fn=lambda cfg: "pupa",
        ),
    }


_registry: Dict[str, TaskEntry] | None = None


def get_registry() -> Dict[str, TaskEntry]:
    global _registry
    if _registry is None:
        _registry = _build_registry()
    return _registry
