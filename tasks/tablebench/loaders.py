"""TableBench loader — seed TableBench eval rows and TableInstruct train rows.

Filters out the Visualization category (chart-code answers, requires execution
to score; out of FACET single-prompt-single-scorer scope).

Sources:
- https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench
- https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableInstruct
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from core.schema import make_query_id
from core.store import CubeStore, OnConflict
from tasks.tablebench.official_parser import parse_final_answer

logger = logging.getLogger(__name__)

# Categories kept for FACET evaluation (single textual answer).
_KEPT_QTYPES = {"FactChecking", "NumericalReasoning", "DataAnalysis"}
_TABLEBENCH_REPO = "Multilingual-Multimodal-NLP/TableBench"
_TABLEINSTRUCT_REPO = "Multilingual-Multimodal-NLP/TableInstruct"


def seed_queries_tablebench(
    store: CubeStore,
    split: str = "test",
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    train_revision: str | None = None,
    train_instruction_types: Sequence[str] | None = None,
    train_data_path: str | None = None,
    include_visualization: bool = False,
    max_queries: int = 0,
    sample_seed: int = 0,
    on_conflict: OnConflict = OnConflict.SKIP,
) -> int:
    """Seed TableBench-compatible rows into the unified store.

    ``split="test"`` uses the raw TableBench benchmark without baked prompts.
    ``split="train"`` uses the companion TableInstruct training data. TableBench
    itself does not publish a train split in the benchmark repo.
    """
    include_qtypes = _KEPT_QTYPES | ({"Visualization"} if include_visualization else set())
    split_norm = split.lower()
    if split_norm == "train":
        rows_all = _load_tableinstruct_train_rows(
            revision=train_revision,
            cache_dir=cache_dir,
            train_data_path=train_data_path,
            include_qtypes=include_qtypes,
            instruction_types=train_instruction_types,
        )
    else:
        rows_all = _load_tablebench_eval_rows(
            revision=revision,
            cache_dir=cache_dir,
            include_qtypes=include_qtypes,
        )

    class _DS:
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __iter__(self): return iter(self.rows)
        def select(self, idx): return _DS([self.rows[i] for i in idx])
    ds = _DS(rows_all)

    if max_queries > 0 and max_queries < len(ds):
        if sample_seed > 0:
            import random
            rng = random.Random(sample_seed)
            indices = sorted(rng.sample(range(len(ds)), max_queries))
            ds = ds.select(indices)
        else:
            ds = ds.select(range(max_queries))

    queries: List[Dict[str, Any]] = []
    for i, row in enumerate(ds):
        source_line = int(row.get("_source_line", i + 1))
        question = row["question"]
        # TableBench table is a dict {columns: [...], data: [[...], ...]}
        # Normalize to the same {header, rows, name} shape used by WTQ/TabFact.
        tb_table = _parse_table(row.get("table", {}))
        header, rows_list = _normalize_table(tb_table)
        official_table = _official_table_shape(tb_table)

        gold_answer = row.get("answer")
        if gold_answer is None:
            gold_answer = parse_final_answer(row.get("response", ""))

        revision_key = (
            train_revision or "main"
            if split_norm == "train"
            else revision or "main"
        )
        instruction_type = row.get("instruction_type", "")
        query_id = make_query_id(
            "tablebench",
            question,
            context=f"{split}:{revision_key}:{instruction_type}:{source_line}:{row['id']}",
        )
        queries.append({
            "query_id": query_id,
            "dataset": "tablebench",
            "content": question,
            "meta": {
                "gold_answer": str(gold_answer or ""),
                "qtype": row["qtype"],
                "qsubtype": row["qsubtype"],
                "instruction_type": instruction_type,
                "split": split,
                "source_dataset": row.get("_source_dataset", "TableBench"),
                "dataset_revision": revision_key,
                "_raw": {
                    "id": row["id"],
                    "source_line": source_line,
                    "source_dataset": row.get("_source_dataset", "TableBench"),
                    "dataset_revision": revision_key,
                    "question": question,
                    "answer": str(gold_answer or ""),
                    "qtype": row["qtype"],
                    "qsubtype": row["qsubtype"],
                    "instruction": row.get("instruction", ""),
                    "instruction_type": instruction_type,
                    "response": row.get("response", ""),
                    "table": {
                        "header": header,
                        "rows": rows_list,
                        "name": row.get("chart_type") or "",
                    },
                    "official_table": official_table,
                },
            },
        })

    inserted = store.upsert_queries(queries, on_conflict=on_conflict)
    logger.info(
        "Seeded TableBench queries (split=%s, attempted=%d, inserted=%d, kept qtypes=%s)",
        split, len(queries), inserted, sorted(include_qtypes),
    )
    return inserted


def _load_tablebench_eval_rows(
    *,
    revision: str | None,
    cache_dir: str | None,
    include_qtypes: set[str],
) -> List[Dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    fp = hf_hub_download(
        _TABLEBENCH_REPO,
        "TableBench.jsonl",
        repo_type="dataset",
        revision=revision,
        cache_dir=cache_dir,
    )
    rows = []
    for source_line, row in _read_jsonl(fp):
        if row.get("qtype") in include_qtypes:
            row["_source_line"] = source_line
            row["_source_dataset"] = "TableBench"
            rows.append(row)
    return rows


def _load_tableinstruct_train_rows(
    *,
    revision: str | None,
    cache_dir: str | None,
    train_data_path: str | None,
    include_qtypes: set[str],
    instruction_types: Sequence[str] | str | None,
) -> List[Dict[str, Any]]:
    if train_data_path:
        fp = Path(train_data_path)
    else:
        from huggingface_hub import hf_hub_download
        fp = Path(hf_hub_download(
            _TABLEINSTRUCT_REPO,
            "TableInstruct_instructions.jsonl",
            repo_type="dataset",
            revision=revision,
            cache_dir=cache_dir,
        ))

    wanted_instruction_types = _normalize_instruction_type_filter(instruction_types)
    rows = []
    for source_line, row in _read_jsonl(fp):
        if row.get("qtype") not in include_qtypes:
            continue
        if (
            wanted_instruction_types is not None
            and row.get("instruction_type") not in wanted_instruction_types
        ):
            continue
        row["_source_line"] = source_line
        row["_source_dataset"] = "TableInstruct"
        rows.append(row)
    return rows


def _normalize_instruction_type_filter(
    instruction_types: Sequence[str] | str | None,
) -> set[str] | None:
    """Return a filter set, or None when all instruction types are allowed."""
    if instruction_types is None:
        return None
    if isinstance(instruction_types, str):
        values = [instruction_types]
    else:
        values = list(instruction_types)
    normalized = {str(v).strip() for v in values if str(v).strip()}
    return normalized or None


def _read_jsonl(path: str | Path) -> Iterable[tuple[int, Dict[str, Any]]]:
    with open(path) as f:
        for source_line, line in enumerate(f, start=1):
            if line.strip():
                yield source_line, json.loads(line)


def _parse_table(table: Any) -> Dict[str, Any]:
    if isinstance(table, str):
        try:
            parsed = json.loads(table)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return table if isinstance(table, dict) else {}


def _normalize_table(table: Dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    header = list(table.get("columns", table.get("header", [])))
    raw_rows = table.get("data", table.get("rows", []))
    rows = [[str(c) for c in row] for row in raw_rows]
    return header, rows


def _official_table_shape(table: Dict[str, Any]) -> Dict[str, Any]:
    """Preserve TableBench's typed {columns, data} prompt serialization."""
    return {
        "columns": list(table.get("columns", table.get("header", []))),
        "data": [list(row) for row in table.get("data", table.get("rows", []))],
    }
