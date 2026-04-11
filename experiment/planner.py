"""Experiment planner — cross configs × query cohorts into a run plan.

The planner produces a flat list of RunEntry items. Each entry pairs
a config with a set of query_ids to execute. The runner iterates over them.

Usage:
    from config_generators import generate
    from query_cohorts import build_cohorts

    configs = generate("add_one", store, base_ids, rule_ids)
    cohorts = build_cohorts(store, "difficulty", dataset="bird")

    plan = cross(configs, cohorts)
    # → one RunEntry per (config, cohort) pair

    plan = cross(configs)
    # → cohorts=None means run all queries (no stratification)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from prompt_profiler.experiment.config_generators import ConfigEntry
from prompt_profiler.experiment.query_cohorts import Cohorts

logger = logging.getLogger(__name__)


@dataclass
class RunEntry:
    """A single unit of work: one config × one query cohort."""
    config_id: int
    func_ids: List[str]
    query_ids: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)


def cross(
    configs: List[ConfigEntry],
    cohorts: Optional[Cohorts] = None,
    *,
    all_query_ids: Optional[List[str]] = None,
) -> List[RunEntry]:
    """Cross configs × cohorts into a flat run plan.

    Args:
        configs: From a config generator.
        cohorts: From build_cohorts. None = use all_query_ids as single cohort.
        all_query_ids: Required when cohorts is None.

    Returns:
        Flat list of RunEntry.
    """
    if cohorts is None:
        if all_query_ids is None:
            raise ValueError("Either cohorts or all_query_ids must be provided")
        cohorts = {"all": all_query_ids}

    plan: List[RunEntry] = []
    for config_id, func_ids, config_meta in configs:
        for cohort_label, query_ids in cohorts.items():
            meta = {
                **config_meta,
                "cohort": cohort_label,
                "n_queries": len(query_ids),
            }
            plan.append(RunEntry(
                config_id=config_id,
                func_ids=func_ids,
                query_ids=query_ids,
                meta=meta,
            ))

    total_calls = sum(len(e.query_ids) for e in plan)
    logger.info("Experiment plan: %d configs × %d cohorts = %d entries, %d total LLM calls",
                len(configs), len(cohorts), len(plan), total_calls)
    return plan
