from __future__ import annotations

import csv
import random

from core.store import CubeStore
from tasks.sqa.loaders import seed_queries_sqa


def test_sqa_seed_ids_are_stable_across_sample_and_full(tmp_path):
    data_dir = tmp_path / "sqa"
    data_dir.mkdir()

    rows = []
    for i in range(5):
        table_file = f"table-{i}.csv"
        with open(data_dir / table_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["col"])
            writer.writerow([str(i)])
        rows.append({
            "id": f"seq-{i}",
            "annotator": "ann",
            "position": "0",
            "table_file": table_file,
            "question": f"question {i}?",
            "answer_text": f"['{i}']",
            "answer_coordinates": "[]",
        })

    with open(data_dir / "random-split-1-dev.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    sample_seed = 17
    sampled = sorted(random.Random(sample_seed).sample(range(len(rows)), 2))
    assert sampled != [0, 1]

    store = CubeStore(tmp_path / "cube.db")
    sampled_ids = seed_queries_sqa(
        store,
        "test",
        data_dir=str(data_dir),
        max_queries=2,
        sample_seed=sample_seed,
    )
    full_ids = seed_queries_sqa(store, "test", data_dir=str(data_dir))

    conn = store._get_conn()
    n_queries = conn.execute(
        "SELECT COUNT(*) FROM query WHERE dataset = 'sqa'"
    ).fetchone()[0]

    assert n_queries == len(rows)
    assert set(sampled_ids).issubset(set(full_ids))
