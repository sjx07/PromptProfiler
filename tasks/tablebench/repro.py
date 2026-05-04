"""TableBench reproduction: use the benchmark's literal pre-rendered instruction
prompt (DP / TCoT / SCoT / PoT) and the official scorer.

Bypasses the FACET prompt scaffold — meant for validation against published
numbers, not for feature attribution.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from core.schema import make_query_id
from core.store import CubeStore, OnConflict
from task import BaseTask
from tasks.tablebench.official_parser import (
    parse_chart_code_then_exec,
    parse_final_answer,
    parse_general_code_then_exec,
)
from tasks.tablebench.official_scorer import score_one

logger = logging.getLogger(__name__)

_TEXT_QTYPES = {"FactChecking", "NumericalReasoning", "DataAnalysis"}
_ALL_QTYPES = _TEXT_QTYPES | {"Visualization"}
_VALID_VARIANTS = {"DP", "TCoT", "SCoT", "PoT"}


def _normalize_table(tbl) -> dict:
    """Coerce TableBench's {columns,data} into the {header,rows,name} shape
    the wtq executor expects."""
    if not isinstance(tbl, dict):
        return {"header": [], "rows": [], "name": ""}
    return {
        "header": list(tbl.get("columns", [])),
        "rows": [[str(c) for c in r] for r in tbl.get("data", [])],
        "name": "",
    }


def seed_queries_tablebench_repro(
    store: CubeStore,
    split: str = "test",
    *,
    variant: str = "TCoT",
    revision: str | None = None,
    cache_dir: str | None = None,
    include_visualization: bool = False,
    max_queries: int = 0,
    sample_seed: int = 0,
    on_conflict: OnConflict = OnConflict.SKIP,
) -> int:
    """Load TableBench_<variant>.jsonl with pre-baked instruction prompts."""
    if variant not in _VALID_VARIANTS:
        raise ValueError(f"variant must be one of {_VALID_VARIANTS}, got {variant!r}")

    from huggingface_hub import hf_hub_download
    fp = hf_hub_download(
        "Multilingual-Multimodal-NLP/TableBench",
        f"TableBench_{variant}.jsonl",
        repo_type="dataset",
        revision=revision,
        cache_dir=cache_dir,
    )
    kept_qtypes = _ALL_QTYPES if include_visualization else _TEXT_QTYPES
    rows: List[tuple[int, Dict[str, Any]]] = []
    with open(fp) as f:
        for source_line, line in enumerate(f, start=1):
            r = json.loads(line)
            if r.get("qtype") in kept_qtypes:
                rows.append((source_line, r))

    if max_queries > 0 and max_queries < len(rows):
        if sample_seed > 0:
            import random
            rng = random.Random(sample_seed)
            indices = sorted(rng.sample(range(len(rows)), max_queries))
            rows = [rows[i] for i in indices]
        else:
            rows = rows[:max_queries]

    queries: List[Dict[str, Any]] = []
    for source_line, r in rows:
        question = r["question"]
        # Source line is part of identity because TableBench contains at least
        # one duplicated (id, question) row that the official denominator keeps.
        query_id = make_query_id(
            "tablebench_repro",
            f"{variant}::{question}",
            context=(
                f"{split}:{variant}:{source_line}:{r['id']}"
                if revision is None
                else f"{split}:{revision}:{variant}:{source_line}:{r['id']}"
            ),
        )
        queries.append({
            "query_id": query_id,
            "dataset": "tablebench_repro",
            "content": question,
            "meta": {
                "gold_answer": r["answer"],
                "qtype": r["qtype"],
                "qsubtype": r["qsubtype"],
                "chart_type": r.get("chart_type", ""),
                "instruction_type": r.get("instruction_type", variant),
                "split": split,
                "variant": variant,
                "dataset_revision": revision or "main",
                "instruction": r["instruction"],
                "source_line": source_line,
                "_raw": {
                    "id": r["id"],
                    "source_line": source_line,
                    "dataset_revision": revision or "main",
                    "question": question,
                    "answer": r["answer"],
                    "qtype": r["qtype"],
                    "qsubtype": r["qsubtype"],
                    "chart_type": r.get("chart_type", ""),
                    "instruction_type": r.get("instruction_type", variant),
                    "instruction": r["instruction"],
                    # Keep the official {columns, data} table shape so PoT
                    # and Visualization code can read table.csv exactly as in
                    # the public parser.
                    "table": r.get("table", {}),
                },
            },
        })

    inserted = store.upsert_queries(queries, on_conflict=on_conflict)
    logger.info(
        "Seeded TableBench-repro queries (split=%s, variant=%s, include_visualization=%s, attempted=%d, inserted=%d)",
        split, variant, include_visualization, len(queries), inserted,
    )
    return inserted


# ── parsing: TableBench's "Final Answer: ..." extraction ─────────────

_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.+)", re.IGNORECASE | re.DOTALL)
_PYTHON_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _extract_final_answer(text: str) -> str:
    return parse_final_answer(text)


def _extract_pot_answer(text: str, table_data: dict) -> str:
    """For PoT: extract last python block, execute, search stdout for Final Answer."""
    if not text:
        return ""
    blocks = _PYTHON_BLOCK_RE.findall(text)
    if not blocks:
        return _extract_final_answer(text)
    code = blocks[-1]
    # Reuse wtq's executor (which handles pandas DataFrame setup).
    from tasks.wtq.table_qa import _execute_code
    result = _execute_code(code, table_data)
    if result is None:
        return ""
    text_out = str(result)
    # Their parser: search stdout for "Final Answer:"; otherwise use whole stdout.
    fa = _extract_final_answer(text_out)
    return fa if fa else text_out.strip()


# ── task class ────────────────────────────────────────────────────────


class TableBenchRepro(BaseTask):
    name = "tablebench_repro"
    scorer = "tb_official_acc"
    default_input_fields: Dict[str, str] = {
        "instruction": "Pre-rendered TableBench instruction text",
    }
    default_output_fields: Dict[str, str] = {
        "answer": "Final answer text",
    }

    def build_prompt(self, query: dict) -> tuple[str, str]:
        """Bypass prompt_state — return the literal benchmark instruction."""
        meta = query.get("meta", {})
        if isinstance(meta, str):
            meta = json.loads(meta)
        instruction = meta.get("instruction") or meta.get("_raw", {}).get("instruction", "")
        # System empty, instruction goes verbatim as user content.
        return "", instruction

    def parse_response(self, raw_response: str) -> str:
        # Variant-aware: PoT requires code execution, others use Final Answer regex.
        # We don't have variant on the response, but it's on the query.meta —
        # parse_response signature in BaseTask doesn't take query, so handle in score.
        return raw_response if isinstance(raw_response, str) else str(raw_response)

    def score(self, prediction: str, query_meta: dict) -> tuple[float, dict]:
        if isinstance(query_meta, str):
            query_meta = json.loads(query_meta)
        raw = query_meta.get("_raw", {})
        gold = str(raw.get("answer", query_meta.get("gold_answer", "")))
        qtype = raw.get("qtype", query_meta.get("qtype", "unknown"))
        qsubtype = raw.get("qsubtype", query_meta.get("qsubtype", "unknown"))
        variant = query_meta.get("variant",
                                 raw.get("variant", "TCoT"))

        chart_type = raw.get("chart_type", query_meta.get("chart_type", ""))
        table = raw.get("table", {})

        ecr_1 = None
        if qtype == "Visualization":
            parsed, ecr_1 = parse_chart_code_then_exec(
                prediction,
                table,
                answer=gold,
                chart_type=chart_type,
            )
            score_val = 1.0 if parsed is True else 0.0
            method = "viz_pass_at_1"
            extracted = parsed
        elif variant == "PoT":
            extracted, ecr_1 = parse_general_code_then_exec(prediction, table)
            score_val, method = score_one(qtype, qsubtype, gold, extracted)
        else:
            extracted = _extract_final_answer(prediction)
            score_val, method = score_one(qtype, qsubtype, gold, extracted)

        parse_1 = extracted != ""
        return score_val, {
            "status": "ok",
            "raw_response": prediction[:500] if prediction else "",
            "extracted": extracted,
            "gold": gold,
            "qtype": qtype,
            "qsubtype": qsubtype,
            "chart_type": chart_type,
            "variant": variant,
            "method": method,
            "Parse@1": parse_1,
            "ECR@1": ecr_1,
        }
