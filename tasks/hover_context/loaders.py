"""HoVer loader.

Reads the local `hover_dev_release_v1.1.json` (downloaded once from
`raw.githubusercontent.com/hover-nlp/hover/main/data/hover/...`) as the
test set (HoVer's test split is unlabeled; langProBe uses validation).
Shuffle with seed=1 and cap to `max_queries` (paper uses 300).
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

from core.schema import make_query_id
from core.store import CubeStore, OnConflict

logger = logging.getLogger(__name__)


def seed_queries_hover(
    store: CubeStore,
    split: str,
    *,
    data_path: str | Path | None = None,
    max_queries: int = 0,
    sample_seed: int = 1,
    on_conflict: OnConflict = OnConflict.SKIP,
) -> int:
    """Seed HoVer claims with their supporting_facts gold titles."""
    if data_path is None:
        raise ValueError(
            "HoVer loader requires cfg.data_path (e.g., "
            "/data/users/jsu323/datasets/hover/hover_dev_release_v1.1.json)"
        )
    rows = _load_rows(Path(data_path))

    if split.lower() == "test":
        # langProBe's convention: validation split, shuffled seed=1
        rng = random.Random(sample_seed)
        rng.shuffle(rows)
    elif split.lower() == "train":
        # 3-hop train subset, shuffled seed=0 (langProBe convention)
        rows = [r for r in rows if r.get("num_hops") == 3]
        rng = random.Random(0)
        rng.shuffle(rows)
    else:
        raise ValueError(f"HoVer supports split ∈ {{train, test}}; got {split!r}")

    queries: List[Dict[str, Any]] = []
    for row in rows:
        claim = row.get("claim") or ""
        if not claim:
            continue
        supporting = row.get("supporting_facts") or []
        source_id = str(row.get("uid") or row.get("hpqa_id") or len(queries))

        meta: Dict[str, Any] = {
            "split": split,
            "source_id": source_id,
            "supporting_facts": supporting,
            "label": row.get("label"),
            "num_hops": row.get("num_hops"),
            "_raw": row,
        }
        query_id = str(
            row.get("uid")
            or make_query_id("hover_context", claim, context=f"{split}:{source_id}")
        )
        queries.append({
            "query_id": query_id,
            "dataset": "hover_context",
            "content": claim,
            "meta": meta,
        })
        if max_queries > 0 and len(queries) >= max_queries:
            break

    store.upsert_queries(queries, on_conflict=on_conflict)
    logger.info(
        "Seeded %d HoVer queries (split=%s, seed=%d)", len(queries), split, sample_seed,
    )
    return len(queries)


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"HoVer data file not found: {path}")
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"HoVer JSON must be a list, got {type(data).__name__}")
    return data
