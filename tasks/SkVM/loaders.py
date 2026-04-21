"""SkVM loaders — generate balanced query grids and seed them into the store."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from core.schema import make_query_id
from core.store import CubeStore, OnConflict
from tasks.SkVM.generators import (
    generate_logic_l2,
    generate_spatial_l3,
    generate_structured_l3,
    iter_logic_l2_grid,
    iter_spatial_l3_grid,
    iter_structured_l3_grid,
)

logger = logging.getLogger(__name__)


def seed_queries_structured_l3(
    store: CubeStore,
    *,
    per_cell: int = 148,
    seed_base: int = 42,
    on_conflict: OnConflict = OnConflict.SKIP,
) -> int:
    """Seed a balanced 27-cell grid of gen.text.structured L3 instances.

    At ``per_cell=148`` this seeds 27 × 148 = 3996 queries.

    Each query row's ``meta._raw`` carries:
      - primitive: "gen.text.structured"
      - level:     "L3"
      - eval_params: dict the evaluator reads (D, T, M)
      - predicates: dict the predicate extractors read (D, T, M)

    Args:
        store: open CubeStore.
        per_cell: instances per (D, T, M) cell. Default 148 → ~4k total.
        seed_base: offset for deterministic per-instance seeds.
        on_conflict: cube conflict policy. Default SKIP — re-seeding is a no-op.

    Returns:
        Number of queries seeded (before conflict resolution).
    """
    dataset = "skvm_structured_l3"
    queries: List[Dict[str, Any]] = []

    for D, T, M, i, seed in iter_structured_l3_grid(
        per_cell=per_cell, seed_base=seed_base,
    ):
        inst = generate_structured_l3(D, T, M, seed=seed)
        qid = make_query_id(
            dataset=dataset,
            content=inst.prompt,
            context=f"cell={D}x{T}x{M}:i={i}:seed={seed}",
        )
        queries.append({
            "query_id": qid,
            "dataset": dataset,
            "content": inst.prompt,
            "meta": {
                "_raw": {
                    "primitive": "gen.text.structured",
                    "level": "L3",
                    "eval_params": inst.eval_params,
                    "predicates": {"D": D, "T": T, "M": M},
                },
            },
        })

    store.upsert_queries(queries, on_conflict=on_conflict)
    logger.info(
        "seed_queries_structured_l3: seeded %d queries to dataset=%s",
        len(queries), dataset,
    )
    return len(queries)


def seed_queries_spatial_l3(
    store: CubeStore,
    *,
    on_conflict: OnConflict = OnConflict.SKIP,
) -> int:
    """Seed reason.spatial L3 instances — 56 ordered city pairs.

    This primitive has NO seeded randomness; the prompt is fully determined
    by the pair. Native dataset size is 56 unique queries. Scaling beyond
    that requires generator enrichment (e.g. coordinate jitter, phrasing
    variants) — not done in this round.
    """
    dataset = "skvm_spatial_l3"
    queries: List[Dict[str, Any]] = []

    for a, b, pair_idx in iter_spatial_l3_grid():
        inst = generate_spatial_l3(a, b)
        qid = make_query_id(
            dataset=dataset,
            content=inst.prompt,
            context=f"pair={a}x{b}:idx={pair_idx}",
        )
        queries.append({
            "query_id": qid,
            "dataset": dataset,
            "content": inst.prompt,
            "meta": {
                "_raw": {
                    "primitive": "reason.spatial",
                    "level": "L3",
                    "eval_params": inst.eval_params,
                    "predicates": inst.predicates,
                },
            },
        })

    store.upsert_queries(queries, on_conflict=on_conflict)
    logger.info(
        "seed_queries_spatial_l3: seeded %d queries to dataset=%s",
        len(queries), dataset,
    )
    return len(queries)


def seed_queries_logic_l2(
    store: CubeStore,
    *,
    per_cell: int = 444,
    seed_base: int = 42,
    on_conflict: OnConflict = OnConflict.SKIP,
) -> int:
    """Seed reason.logic L2 instances — balanced 9-cell × per_cell grid.

    Cells: (K=4, target∈{1..4}) + (K=5, target∈{1..5}).
    per_cell=444 yields 9 × 444 = 3996 queries.
    """
    dataset = "skvm_logic_l2"
    queries: List[Dict[str, Any]] = []

    for K, target_pos, i, seed in iter_logic_l2_grid(
        per_cell=per_cell, seed_base=seed_base,
    ):
        inst = generate_logic_l2(K, target_pos, seed=seed)
        qid = make_query_id(
            dataset=dataset,
            content=inst.prompt,
            context=f"cell=K{K}_pos{target_pos}:i={i}:seed={seed}",
        )
        queries.append({
            "query_id": qid,
            "dataset": dataset,
            "content": inst.prompt,
            "meta": {
                "_raw": {
                    "primitive": "reason.logic",
                    "level": "L2",
                    "eval_params": inst.eval_params,
                    "predicates": inst.predicates,
                },
            },
        })

    store.upsert_queries(queries, on_conflict=on_conflict)
    logger.info(
        "seed_queries_logic_l2: seeded %d queries to dataset=%s",
        len(queries), dataset,
    )
    return len(queries)
