"""HotpotQA context task registrations."""
from __future__ import annotations

from typing import Any, Dict

from task_registry import register_task
from tasks.hotpotqa_context.hotpotqa_context import HotpotQAContextTask


def _seed_hotpotqa_context(store: Any, cfg: Dict[str, Any], split: str) -> None:
    from tasks.hotpotqa_context.loaders import seed_queries_hotpotqa_context

    data_path = cfg.get("data_path") or cfg.get("hotpotqa_data_path")
    max_queries = int(cfg.get("max_queries", 0) or 0)
    split_mode = str(cfg.get("split_mode") or cfg.get("hotpotqa_split_mode") or "dataset")
    sample_seed = int(cfg.get("hotpotqa_sample_seed") or 1)
    seed_queries_hotpotqa_context(
        store,
        split,
        data_path=data_path,
        max_queries=max_queries,
        split_mode=split_mode,
        sample_seed=sample_seed,
    )


register_task(
    "hotpotqa_context",
    seeder_fn=_seed_hotpotqa_context,
    dataset_key_fn=lambda cfg: "hotpotqa_context",
)(HotpotQAContextTask)
