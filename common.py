"""Common utilities — task-agnostic helpers for the unified CubeStore.

Provides generic func seeding, pool.json parsing, and node_id resolution.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from prompt_profiler.core.func_registry import make_func_id
from prompt_profiler.core.store import CubeStore, OnConflict

logger = logging.getLogger(__name__)


def seed_funcs(
    store: CubeStore,
    specs: List[Dict[str, Any]],
    on_conflict: OnConflict = OnConflict.SKIP,
) -> int:
    """Auto-ID and insert funcs from (func_type, params) specs.

    Each spec is a dict with:
        func_type: str          — registered handler name
        params: dict            — handler-specific parameters
        func_id: str (optional) — override auto-generated ID
        meta: dict (optional)   — arbitrary metadata

    If func_id is not provided, it is generated via make_func_id
    (content-addressed from func_type + params).

    Any loader (pool.json, YAML, manual, etc.) produces specs,
    this handles ID generation and insertion.
    """
    funcs: List[Dict[str, Any]] = []
    for spec in specs:
        func_id = spec.get("func_id") or make_func_id(
            spec["func_type"], spec.get("params", {})
        )
        funcs.append({
            "func_id": func_id,
            "func_type": spec["func_type"],
            "params": spec.get("params", {}),
            "meta": spec.get("meta", {}),
        })

    store.upsert_funcs(funcs, on_conflict=on_conflict)
    logger.info("Seeded %d funcs", len(funcs))
    return len(funcs)


# ── pool.json utilities ──────────────────────────────────────────────


def _strip_list_prefix(text: str) -> str:
    """Strip leading list markers (- , * , 1. , 2. , etc.) from rule content."""
    return re.sub(r"^(?:[-*]\s+|\d+\.\s+)", "", text.strip())


def seed_pool(
    store: CubeStore,
    pool_path: str | Path,
    on_conflict: OnConflict = OnConflict.SKIP,
) -> Dict[str, int]:
    """Parse pool.json into func specs and seed via seed_funcs.

    Returns {"sections": N, "rules": M} counts.
    """
    with open(pool_path) as f:
        pool = json.load(f)

    sections_data = pool.get("sections", pool if isinstance(pool, list) else [])
    specs: List[Dict[str, Any]] = []

    for i, sec in enumerate(sections_data):
        sec_params = {
            "title": sec["title"],
            "ordinal": i,
            "is_system": sec.get("is_system", False),
            "min_rules": sec.get("min_rules", 0),
            "max_rules": sec.get("max_rules", 10),
        }
        sec_func_id = make_func_id("define_section", sec_params)
        specs.append({
            "func_type": "define_section",
            "func_id": sec_func_id,
            "params": sec_params,
            "meta": {"pool_id": sec.get("id", f"s{i}")},
        })

        for j, child in enumerate(sec.get("children", [])):
            if child.get("node_type") != "rule":
                continue
            rule_params = {
                "section_id": sec_func_id,
                "content": _strip_list_prefix(child.get("content", "")),
            }
            specs.append({
                "func_type": "add_rule",
                "params": rule_params,
                "meta": {
                    "pool_id": child.get("id", f"s{i}/r{j}"),
                    "rule_kind": child.get("rule_kind", ""),
                },
            })

    seed_funcs(store, specs, on_conflict=on_conflict)

    section_count = sum(1 for s in specs if s["func_type"] == "define_section")
    rule_count = sum(1 for s in specs if s["func_type"] == "add_rule")
    logger.info("Seeded %d sections, %d rules from %s", section_count, rule_count, pool_path)
    return {"sections": section_count, "rules": rule_count}


def resolve_node_ids(
    specs: List[Dict[str, Any]],
    pool_path: str | Path,
) -> List[Dict[str, Any]]:
    """Resolve node_id references in specs to full pool params.

    Specs with {"func_type": "add_rule", "params": {"node_id": "s2/r0"}}
    are expanded to match the pool's registration params (section_id + content),
    ensuring they hash to the same func_id as the pool version.
    """
    with open(pool_path) as f:
        pool = json.load(f)

    sections_data = pool.get("sections", pool if isinstance(pool, list) else [])

    node_map: Dict[str, Dict[str, Any]] = {}
    for i, sec in enumerate(sections_data):
        sec_params = {
            "title": sec["title"],
            "ordinal": i,
            "is_system": sec.get("is_system", False),
            "min_rules": sec.get("min_rules", 0),
            "max_rules": sec.get("max_rules", 10),
        }
        sec_func_id = make_func_id("define_section", sec_params)
        for j, child in enumerate(sec.get("children", [])):
            if child.get("node_type") != "rule":
                continue
            node_id = child.get("id", f"s{i}/r{j}")
            node_map[node_id] = {
                "section_id": sec_func_id,
                "content": _strip_list_prefix(child.get("content", "")),
            }

    resolved = []
    for spec in specs:
        if spec.get("func_type") == "add_rule" and "node_id" in spec.get("params", {}):
            nid = spec["params"]["node_id"]
            if nid in node_map:
                resolved.append({**spec, "params": node_map[nid]})
            else:
                raise ValueError(f"node_id '{nid}' not found in pool {pool_path}")
        else:
            resolved.append(spec)
    return resolved
