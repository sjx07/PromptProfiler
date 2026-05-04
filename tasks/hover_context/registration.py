"""HoVer context task registrations."""
from __future__ import annotations

from typing import Any, Dict

from task_registry import register_task
from tasks.hover_context.hover_context import HoverContextTask


def _seed_hover_context(store: Any, cfg: Dict[str, Any], split: str) -> None:
    from tasks.hover_context.loaders import seed_queries_hover

    data_path = cfg.get("data_path") or cfg.get("hover_data_path")
    max_queries = int(cfg.get("max_queries", 0) or 0)
    sample_seed = int(cfg.get("hover_sample_seed") or cfg.get("sample_seed") or 1)
    seed_queries_hover(
        store,
        split,
        data_path=data_path,
        max_queries=max_queries,
        sample_seed=sample_seed,
    )


register_task(
    "hover_context",
    seeder_fn=_seed_hover_context,
    dataset_key_fn=lambda cfg: "hover_context",
)(HoverContextTask)
