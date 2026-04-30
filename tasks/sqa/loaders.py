"""SQA loaders — seed sequential table QA queries into the unified CubeStore.

Data source: Microsoft SQA Release 1.0 (TSV + CSV files).
Download:
  https://download.microsoft.com/download/1/D/C/1DC270D2-1B53-4A61-A2E3-88AB3E4E6E1F/SQA%20Release%201.0.zip
"""
from __future__ import annotations

import ast
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List

from core.schema import make_query_id
from core.store import CubeStore, OnConflict

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "/data/users/jsu323/sqa/SQA Release 1.0"

SPLIT_FILES = {
    "train": "random-split-1-train.tsv",
    "validation": "random-split-1-dev.tsv",
    "dev": "random-split-1-dev.tsv",
    "test": "random-split-1-dev.tsv",  # SQA has no public test labels
}


def load_table_csv(data_dir: str, table_file: str) -> Dict[str, Any]:
    """Load a table CSV file and return structured data."""
    path = Path(data_dir) / table_file
    if not path.exists():
        return {"headers": [], "rows": [], "n_rows": 0, "n_cols": 0}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        all_rows = list(reader)

    if not all_rows:
        return {"headers": [], "rows": [], "n_rows": 0, "n_cols": 0}

    headers = all_rows[0]
    rows = all_rows[1:]
    return {
        "headers": headers,
        "rows": rows,
        "n_rows": len(rows),
        "n_cols": len(headers),
    }


def table_to_markdown(table: Dict[str, Any]) -> str:
    """Convert parsed table dict to markdown."""
    headers = table.get("headers", [])
    rows = table.get("rows", [])
    if not headers:
        return ""

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        padded = list(row) + [""] * (len(headers) - len(row))
        lines.append("| " + " | ".join(padded[: len(headers)]) + " |")
    return "\n".join(lines)


def _parse_list_field(val: str) -> List[str]:
    """Parse a string like \"['a', 'b']\" into a list."""
    val = val.strip()
    if not val:
        return []
    try:
        parsed = ast.literal_eval(val)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
        return [str(parsed)]
    except (ValueError, SyntaxError):
        pass
    stripped = val.strip("[]")
    if stripped:
        return [p.strip().strip("'\"") for p in stripped.split(",") if p.strip()]
    return []


def seed_queries_sqa(
    store: CubeStore,
    split: str,
    *,
    data_dir: str = DEFAULT_DATA_DIR,
    max_queries: int = 0,
    sample_seed: int = 0,
    on_conflict: OnConflict = OnConflict.SKIP,
) -> int:
    """Load SQA dataset from local TSV files and seed queries into the store.

    Args:
        split: "train" or "validation"/"dev"/"test" (all map to dev split —
            SQA has no public test labels).
        data_dir: Path to extracted "SQA Release 1.0" directory.
        max_queries: Limit number of queries (0 = all).
        sample_seed: If > 0, shuffle whole conversation sequences with this
            seed and take entire sequences until reaching ~max_queries
            turns. Preserves conversation history. If 0, take the first
            max_queries rows (history may be partial).
    """
    tsv_name = SPLIT_FILES.get(split)
    if not tsv_name:
        raise ValueError(
            f"Unknown split: {split}. Choose from: {list(SPLIT_FILES.keys())}"
        )

    tsv_path = Path(data_dir) / tsv_name
    if not tsv_path.exists():
        raise FileNotFoundError(
            f"SQA TSV not found: {tsv_path}. Download from: "
            "https://download.microsoft.com/download/1/D/C/1DC270D2-1B53-4A61-A2E3-88AB3E4E6E1F/"
            "SQA%20Release%201.0.zip"
        )

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    if max_queries > 0 and max_queries < len(rows):
        if sample_seed > 0:
            import random
            # Group by sequence then shuffle whole sequences to preserve history.
            seqs: Dict[str, List[dict]] = {}
            for r in rows:
                seqs.setdefault(f"{r['id']}-{r['annotator']}", []).append(r)
            rng = random.Random(sample_seed)
            seq_keys = sorted(seqs.keys())
            rng.shuffle(seq_keys)
            chosen: List[dict] = []
            for k in seq_keys:
                if len(chosen) + len(seqs[k]) > max_queries:
                    if not chosen:  # take at least one sequence
                        chosen.extend(seqs[k])
                    break
                chosen.extend(seqs[k])
            rows = chosen
        else:
            rows = rows[:max_queries]

    # Group by sequence to build conversation history
    sequences: Dict[str, List[dict]] = {}
    for row in rows:
        seq_key = f"{row['id']}-{row['annotator']}"
        sequences.setdefault(seq_key, []).append(row)

    for seq in sequences.values():
        seq.sort(key=lambda r: int(r["position"]))

    table_cache: Dict[str, Dict[str, Any]] = {}

    queries: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        question = row["question"]
        position = int(row["position"])
        seq_key = f"{row['id']}-{row['annotator']}"
        table_file = row["table_file"]

        if table_file not in table_cache:
            table_cache[table_file] = load_table_csv(data_dir, table_file)
        table = table_cache[table_file]

        history = []
        seq = sequences[seq_key]
        for prev in seq:
            if int(prev["position"]) >= position:
                break
            history.append({
                "question": prev["question"],
                "answer": _parse_list_field(prev["answer_text"]),
            })

        answer_text = _parse_list_field(row["answer_text"])
        answer_coords = _parse_list_field(row["answer_coordinates"])

        query_id = make_query_id("sqa", question, context=f"{split}:{i}")
        queries.append({
            "query_id": query_id,
            "dataset": "sqa",
            "content": question,
            "meta": {
                "gold_answer": answer_text,
                "split": split,
                "table_file": table_file,
                "position": position,
                "sequence_id": seq_key,
                "_raw": {
                    "id": row["id"],
                    "annotator": row["annotator"],
                    "position": position,
                    "question": question,
                    "table_file": table_file,
                    "answer_coordinates": answer_coords,
                    "answer_text": answer_text,
                    "history": history,
                    "table": table,
                },
            },
        })

    store.upsert_queries(queries, on_conflict=on_conflict)
    logger.info("Seeded %d SQA queries (split=%s)", len(queries), split)
    return len(queries)
