"""HotpotQA-context loader.

Rows come from either a local JSONL export (one object per line) or
Hugging Face's `hotpot_qa/fullwiki` split. The `gepa_lite` split mode
reproduces DSPy's canonical `HotPotQA` layout (which the GEPA paper uses):
150 train / 300 dev / 300 test, all `level == "hard"` only, with test drawn
from `fullwiki.validation` and train/dev from `fullwiki.train`, shuffled with
fixed seeds (train_seed=<sample_seed>, eval_seed=2023).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

from core.schema import make_query_id
from core.store import CubeStore, OnConflict

logger = logging.getLogger(__name__)


def seed_queries_hotpotqa_context(
    store: CubeStore,
    split: str,
    *,
    data_path: str | Path | None = None,
    max_queries: int = 0,
    split_mode: str = "dataset",
    sample_seed: int = 1,
    on_conflict: OnConflict = OnConflict.SKIP,
) -> int:
    """Seed HotpotQA questions with their provided context passages."""
    queries: List[Dict[str, Any]] = []
    for row in _iter_rows(split, data_path, split_mode=split_mode, sample_seed=sample_seed):
        if (row.get("split") is not None) and (str(row["split"]) != split):
            continue

        question = row.get("question") or row.get("query") or row.get("input") or ""
        question = str(question).strip()
        if not question:
            raise ValueError("HotpotQA-context row missing question field")

        # Context is optional — BM25 retrieves fresh per query at runtime.
        # Legacy local JSONL rows ship a `context` field; DSPy's HotPotQA test
        # split does not. Keep whichever is present.
        context = row.get("context") or row.get("contexts") or row.get("paragraphs") or []

        source_id = str(row.get("id") or row.get("_id") or len(queries))
        answer = str(row.get("answer") or row.get("gold") or "").strip()

        meta: Dict[str, Any] = {
            "split": split,
            "source_id": source_id,
            "context": _json_safe(context),
            "_raw": _json_safe(row),
        }
        if answer:
            meta["answer"] = answer
        if row.get("supporting_facts") is not None:
            meta["supporting_facts"] = _json_safe(row["supporting_facts"])
        for k in ("level", "type"):
            if row.get(k) is not None:
                meta[k] = row[k]

        query_id = str(
            row.get("query_id")
            or make_query_id("hotpotqa_context", question, context=f"{split}:{source_id}")
        )
        queries.append({
            "query_id": query_id,
            "dataset": "hotpotqa_context",
            "content": question,
            "meta": meta,
        })
        if max_queries > 0 and len(queries) >= max_queries:
            break

    store.upsert_queries(queries, on_conflict=on_conflict)
    logger.info(
        "Seeded %d HotpotQA-context queries (split=%s, split_mode=%s)",
        len(queries), split, split_mode,
    )
    return len(queries)


# ── row iteration ────────────────────────────────────────────────────

def _iter_rows(
    split: str,
    data_path: str | Path | None,
    *,
    split_mode: str = "dataset",
    sample_seed: int = 1,
) -> Iterable[dict]:
    if data_path:
        yield from _iter_local_jsonl(Path(data_path))
        return

    if split_mode in {"gepa", "gepa_lite", "paper", "paper_lite", "dspy_hotpotqa"}:
        yield from _iter_dspy_hotpotqa(split, sample_seed=sample_seed)
        return

    from datasets import load_dataset  # type: ignore[import-not-found]
    for row in load_dataset("hotpot_qa", "fullwiki", split=split, trust_remote_code=True):
        yield dict(row)


def _iter_dspy_hotpotqa(split: str, *, sample_seed: int = 1) -> Iterable[dict]:
    """Yield rows from DSPy's canonical `HotPotQA` class.

    `HotPotQA` filters to `level == "hard"` only and splits:
      - train/dev: `fullwiki.train[hard]`, shuffled seed 0, 75/25 split
      - test: `fullwiki.validation[hard]`, shuffled with eval_seed

    Paper uses 150/300/300. Test is drawn from VALIDATION (held-out), not train.
    Context is empty — BM25 retrieves fresh per query at runtime.
    """
    from dspy.datasets import HotPotQA  # type: ignore[import-not-found]
    ds = HotPotQA(
        train_seed=sample_seed, train_size=150,
        eval_seed=2023, dev_size=300, test_size=300,
    )
    split_key = split.lower()
    if split_key == "test":
        examples = ds.test
    elif split_key in {"validation", "val", "dev"}:
        examples = ds.dev
    elif split_key == "train":
        examples = ds.train
    else:
        raise ValueError(
            f"dspy_hotpotqa split_mode supports split ∈ "
            f"{{train, validation/val/dev, test}}; got {split!r}"
        )
    for ex in examples:
        d = dict(ex)
        d.pop("dspy_uuid", None)
        d.pop("dspy_split", None)
        # DSPy's HotPotQA test examples don't ship a context — BM25 retriever
        # produces passages at runtime. Keep context empty so the seeder is happy.
        d.setdefault("context", [])
        d.setdefault("level", "hard")  # DSPy filters to hard-only
        d["split"] = split
        yield d


def _iter_local_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        raise FileNotFoundError(f"HotpotQA-context data file not found: {path}")
    if path.suffix.lower() != ".jsonl":
        raise ValueError(f"HotpotQA-context local loader only accepts .jsonl, got {path}")
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("HotpotQA-context JSONL rows must be objects")
            yield row


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        return str(value)
