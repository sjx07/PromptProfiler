from __future__ import annotations

import random
import sys
from types import SimpleNamespace

from core.store import CubeStore
from tasks.wtq.loaders import seed_queries_wtq


class _FakeWtqDataset:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def select(self, indices: list[int] | range):
        return _FakeWtqDataset([self._rows[i] for i in indices])


def _wtq_row(i: int) -> dict:
    return {
        "id": f"nu-{i}",
        "question": f"question {i}?",
        "answers": [str(i)],
        "table": {
            "name": f"table-{i}.csv",
            "header": ["col"],
            "rows": [[str(i)]],
        },
    }


def test_wtq_seed_ids_are_stable_across_sample_and_full(monkeypatch, tmp_path):
    rows = [_wtq_row(i) for i in range(5)]
    sample_seed = 17
    sampled = sorted(random.Random(sample_seed).sample(range(len(rows)), 2))
    assert sampled != [0, 1]

    def load_dataset(*_args, **_kwargs):
        return _FakeWtqDataset(rows)

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))

    store = CubeStore(tmp_path / "cube.db")
    sampled_ids = seed_queries_wtq(
        store,
        "test",
        max_queries=2,
        sample_seed=sample_seed,
    )
    full_ids = seed_queries_wtq(store, "test")

    conn = store._get_conn()
    n_queries = conn.execute(
        "SELECT COUNT(*) FROM query WHERE dataset = 'wtq'"
    ).fetchone()[0]

    assert n_queries == len(rows)
    assert set(sampled_ids).issubset(set(full_ids))
