"""Task registry core.

This module is intentionally task-agnostic. Concrete tasks register themselves
from ``tasks.<task_package>.registration`` via the ``register_task`` decorator.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskEntry:
    task_cls: type
    seeder_fn: Callable  # (store, cfg, split) -> None
    dataset_key_fn: Callable  # (cfg) -> str


_REGISTRY: Dict[str, TaskEntry] = {}
_DISCOVERED = False


def register_task(
    name: str,
    *,
    seeder_fn: Callable,
    dataset_key_fn: Callable,
) -> Callable[[type], type]:
    """Register a task class under ``name``.

    Used by task-owned registration modules:

    ``register_task("wtq", seeder_fn=..., dataset_key_fn=...)(TableQA)``
    """
    def decorator(task_cls: type) -> type:
        entry = TaskEntry(
            task_cls=task_cls,
            seeder_fn=seeder_fn,
            dataset_key_fn=dataset_key_fn,
        )
        existing = _REGISTRY.get(name)
        if existing is not None and existing != entry:
            raise ValueError(f"duplicate task registration for {name!r}")
        _REGISTRY[name] = entry
        return task_cls

    return decorator


def seed_predicates_for_dataset(store: Any, dataset: str) -> None:
    """Run registered predicate extractors for a dataset, idempotently."""
    from experiment.query_cohorts import seed_predicates

    n = seed_predicates(store, dataset=dataset)
    if n:
        logger.info("Seeded %d predicate rows for dataset=%s", n, dataset)


def discover_task_registrations() -> None:
    """Import all task-owned registration modules exactly once."""
    global _DISCOVERED
    if _DISCOVERED:
        return

    import tasks

    for module_info in pkgutil.iter_modules(tasks.__path__, prefix="tasks."):
        if not module_info.ispkg:
            continue
        module_name = f"{module_info.name}.registration"
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                continue
            raise

    _DISCOVERED = True


def get_registry() -> Dict[str, TaskEntry]:
    discover_task_registrations()
    return _REGISTRY


__all__ = [
    "TaskEntry",
    "discover_task_registrations",
    "get_registry",
    "register_task",
    "seed_predicates_for_dataset",
]
