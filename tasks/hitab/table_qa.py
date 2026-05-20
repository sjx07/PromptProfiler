"""HiTab table QA task."""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List

from task import BaseTask
from tasks.answer_normalization import normalize_numeric_grouping
from tasks.program_of_thought import ProgramOfThoughtMixin
from tasks.table_python_runtime import execute_python_table_code, runtime_from_rows


_OUTPUT_TO_TABLE_FORMAT = {
    "json": "json_records",
    "markdown": "markdown",
    "plain": "markdown",
    "yaml": "markdown",
    "code_block": "json_records",
}


class HiTabQA(ProgramOfThoughtMixin, BaseTask):
    name = "hitab_qa"
    scorer = "denotation_acc"
    _parser_module_path = "tasks.hitab.parsers"
    default_input_fields: Dict[str, str] = {
        "table": "The hierarchical table data to answer the question about",
        "question": "The question to answer using the table",
    }
    default_output_fields: Dict[str, str] = {
        "answer": "The answer extracted from the table",
    }

    def build_python_table_runtime(self, raw: dict, record: dict | None = None):
        return _build_python_table_runtime(raw)

    def _gold_output(self, meta: dict, raw: dict) -> dict:
        answer = raw.get("answer", meta.get("gold_answer", "[]"))
        values = _parse_answer_string(answer)
        return {"answer": ", ".join(str(v) for v in values)}

    def build_record(self, query: dict, meta: dict, raw: dict) -> dict:
        from tasks.hitab.loaders import table_content_to_markdown, table_content_to_records
        from tasks.wtq.table_formats import get_table_formatter

        table_content = raw.get("table_content", {})
        question = raw.get("question", query.get("content", ""))
        fmt = "markdown"
        has_input_transforms = False
        if self._prompt_state is not None:
            explicit = self._prompt_state.metadata.get("table_format")
            if explicit:
                fmt = explicit
            else:
                style = self._prompt_state.format_style_name
                fmt = _OUTPUT_TO_TABLE_FORMAT.get(style, "markdown")
            has_input_transforms = bool(self._prompt_state.metadata.get("input_transforms"))

        if fmt == "markdown" and not has_input_transforms:
            table_str = table_content_to_markdown(table_content)
        else:
            header, rows = table_content_to_records(table_content)
            header, rows = self._apply_transforms(header, rows, question)
            table_name = raw.get("table_id", "") or raw.get("table_source", "")
            table_str = get_table_formatter(fmt)(header, rows, table_name)
            if self._pending_stats:
                table_str = self._pending_stats + "\n\n" + table_str

        return {
            "table": table_str,
            "question": question,
        }

    def parse_response(self, raw_response: str) -> str:
        return super().parse_response(raw_response)

    def score(self, prediction: str, query_meta: dict) -> tuple[float, dict]:
        if isinstance(query_meta, str):
            query_meta = json.loads(query_meta)

        raw = query_meta.get("_raw", {})
        gold_answer_str = raw.get("answer", query_meta.get("gold_answer", "[]"))

        code_metrics: dict[str, Any] = {}
        code_attempted = False
        if self.is_code_prediction(prediction):
            code_attempted = True
            prediction, code_metrics = self.execute_code_prediction(prediction, raw)

        gold_values = _parse_answer_string(gold_answer_str)
        pred_values = _normalize_answer_list(prediction)

        gold_norm = [_normalize_value(str(v)) for v in gold_values]
        pred_norm = [_normalize_value(v) for v in pred_values]

        score_val = 1.0 if set(pred_norm) == set(gold_norm) else 0.0
        metrics = {
            "status": "ok",
            "prediction": prediction,
            "pred_normalized": pred_norm,
            "gold": gold_values,
            "gold_normalized": gold_norm,
        }
        if code_attempted:
            metrics.update(code_metrics)
        return score_val, metrics


def _build_python_table_runtime(raw: dict):
    from tasks.hitab.loaders import table_content_to_records

    table_content = raw.get("table_content", {})
    header, rows = table_content_to_records(table_content)
    if not header:
        return None
    table_name = raw.get("table_id", "") or raw.get("table_source", "")
    return runtime_from_rows(
        header,
        rows,
        table_name=table_name,
        data_extra={
            "table_content": table_content,
            "question": raw.get("question", ""),
        },
        extras={"table_content": table_content},
    )


def _execute_code(code: str, raw: dict) -> Any:
    """Execute HiTab Python code with flattened and raw table context."""
    runtime = _build_python_table_runtime(raw)
    if runtime is None:
        return None
    outcome = execute_python_table_code(code, runtime)
    return outcome.value


def _parse_answer_string(answer: str) -> List[str]:
    answer = str(answer).strip()
    try:
        parsed = json.loads(answer)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
        return [str(parsed)]
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        parsed = ast.literal_eval(answer)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
        return [str(parsed)]
    except (ValueError, SyntaxError):
        pass
    stripped = answer.strip("[]")
    if "," in stripped:
        return [p.strip().strip("'\"") for p in stripped.split(",") if p.strip()]
    return [stripped] if stripped else []


def _extract_answer(text: str) -> str:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            for key in ("answer", "result", "value", "response"):
                if key in parsed:
                    val = parsed[key]
                    if isinstance(val, list):
                        return ", ".join(str(v) for v in val)
                    return str(val).strip()
        if isinstance(parsed, list):
            return ", ".join(str(v) for v in parsed)
    except (json.JSONDecodeError, TypeError):
        pass

    m = re.search(
        r"(?:answer|result|value)\s*[:\-]\s*(.+?)(?:\n|$)",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().rstrip(".")

    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if lines:
        last = lines[-1]
        for prefix in ["The answer is", "Answer:", "Therefore,", "So,", "Thus,", "The result is"]:
            if last.lower().startswith(prefix.lower()):
                return last[len(prefix):].strip().rstrip(".")
        if len(last) < 200:
            return last

    return text


def _normalize_value(v: str) -> str:
    v = v.strip().lower()
    v = v.strip(".,;:!?\"'")
    v = re.sub(r"\s+", " ", v)
    v = normalize_numeric_grouping(v)
    v = v.replace("%", "")
    try:
        num = float(v)
        if num == int(num):
            v = str(int(num))
        else:
            v = f"{num:.4g}"
    except ValueError:
        pass
    return v


def _normalize_answer_list(prediction: str) -> List[str]:
    prediction = prediction.strip()
    try:
        parsed = json.loads(prediction)
        if isinstance(parsed, list):
            return [_normalize_value(str(v)) for v in parsed]
    except (json.JSONDecodeError, TypeError):
        pass

    if "|" in prediction:
        parts = prediction.split("|")
    elif "," in prediction and "\n" not in prediction:
        parts = prediction.split(",")
    elif "\n" in prediction:
        parts = prediction.split("\n")
    else:
        parts = [prediction]
    return [_normalize_value(p) for p in parts if p.strip()]
