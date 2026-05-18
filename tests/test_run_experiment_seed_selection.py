from __future__ import annotations

from core.store import CubeStore
from run_experiment import _load_seeded_or_split_queries


def test_load_seeded_queries_ignores_stale_split_rows(tmp_path):
    store = CubeStore(tmp_path / "cube.db")
    store.upsert_queries([
        {
            "query_id": "stale",
            "dataset": "wtq",
            "content": "old sampled row",
            "meta": {"split": "test"},
        },
        {
            "query_id": "fresh",
            "dataset": "wtq",
            "content": "current seeded row",
            "meta": {"split": "test"},
        },
    ])

    rows = _load_seeded_or_split_queries(
        store._get_conn(),
        dataset_key="wtq",
        split="test",
        seeded_query_ids=["fresh"],
    )

    assert [r["query_id"] for r in rows] == ["fresh"]
