"""PUPA task registrations."""
from __future__ import annotations

from typing import Any, Dict

from task_registry import register_task
from tasks.pupa.pupa import PupaPrivacyDelegationTask


def _seed_pupa(store: Any, cfg: Dict[str, Any], split: str) -> None:
    from tasks.pupa.loaders import seed_queries_pupa

    data_path = cfg.get("data_path") or cfg.get("pupa_data_path")
    if not data_path:
        raise ValueError(
            "PUPA requires cfg.data_path pointing to a local PAPILLON/PUPA JSON or JSONL file."
        )
    seed_queries_pupa(store, data_path, split)


register_task(
    "pupa",
    seeder_fn=_seed_pupa,
    dataset_key_fn=lambda cfg: "pupa",
)(PupaPrivacyDelegationTask)
