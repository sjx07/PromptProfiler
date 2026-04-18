"""Common utilities — task-agnostic helpers for the unified CubeStore.

Provides generic func seeding and pool.json parsing.

Pool format (new shape, Phase 1b):
  A pool.json is a list of primitive func specs:
    [{"func_type": "insert_node", "params": {...}}, ...]

  Each spec maps directly to a func row. No legacy section/rule nesting.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from core.func_registry import make_func_id
from core.store import CubeStore, OnConflict

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


def seed_pool(
    store: CubeStore,
    pool_path: str | Path,
    on_conflict: OnConflict = OnConflict.SKIP,
) -> int:
    """Parse a new-shape pool.json and seed funcs via seed_funcs.

    Pool format: a JSON array of primitive func specs:
      [{"func_type": "insert_node", "params": {...}}, ...]

    Each spec is passed directly to seed_funcs. func_ids are
    content-addressed via make_func_id if not explicitly provided.

    Returns the number of specs seeded.
    """
    with open(pool_path) as f:
        pool = json.load(f)

    if not isinstance(pool, list):
        raise ValueError(
            f"pool.json must be a JSON array of func specs. Got {type(pool).__name__}. "
            f"File: {pool_path}"
        )

    seed_funcs(store, pool, on_conflict=on_conflict)
    logger.info("Seeded %d funcs from pool %s", len(pool), pool_path)
    return len(pool)
