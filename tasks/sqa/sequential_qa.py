"""Sequential Question Answering task — parse answers, score via denotation accuracy."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from task import BaseTask
from tasks.code_result_utils import (
    dataframe_to_records,
    execute_python_code,
    make_string_dataframe,
    make_typed_dataframe,
)


_OUTPUT_TO_TABLE_FORMAT = {
    "json": "json_records",
    "markdown": "markdown",
    "plain": "markdown",
    "yaml": "markdown",
    "code_block": "json_records",
}


def _execute_code(code: str, raw: dict) -> Any:
    result, _error = _execute_code_with_error(code, raw)
    return result


def _execute_code_with_error(code: str, raw: dict) -> tuple[Any, str | None]:
    """Execute SQA Python code with table and conversation context in scope."""
    table_data = raw.get("table", {})
    header = list(table_data.get("headers", []))
    rows = [list(r) for r in table_data.get("rows", [])]
    if not header:
        return None, None

    history = raw.get("history", [])
    typed_df = make_typed_dataframe(header, rows)
    result, error = _execute_code_with_dataframe(code, raw, typed_df, header, history)
    if error:
        string_df = make_string_dataframe(header, rows)
        fallback_result, fallback_error = _execute_code_with_dataframe(
            code, raw, string_df, header, history
        )
        if fallback_error is None and fallback_result is not None:
            return fallback_result, None
    return result, error


def _execute_code_with_dataframe(
    code: str,
    raw: dict,
    df: Any,
    header: list[str],
    history: list[dict[str, Any]],
) -> tuple[Any, str | None]:
    import datetime as _dt
    import math as _math
    import collections as _collections
    import pandas as pd

    table = dataframe_to_records(df)
    data = {"table": raw.get("table_file", ""), "rows": table, "history": history}

    safe_globals = {
        "df": df,
        "pd": pd,
        "data": data,
        "table": table,
        "history": history,
        "header": header,
        "re": re,
        "datetime": _dt.datetime,
        "timedelta": _dt.timedelta,
        "date": _dt.date,
        "collections": _collections,
        "Counter": _collections.Counter,
        "math": _math,
        "len": len, "str": str, "int": int, "float": float, "bool": bool,
        "list": list, "dict": dict, "set": set, "tuple": tuple,
        "sum": sum, "min": min, "max": max, "abs": abs,
        "sorted": sorted, "enumerate": enumerate, "zip": zip,
        "range": range, "map": map, "filter": filter,
        "any": any, "all": all, "round": round,
        "isinstance": isinstance, "type": type,
    }

    outcome = execute_python_code(code, safe_globals)
    return outcome.value, outcome.error


class SequentialQA(BaseTask):
    name = "sequential_qa"
    scorer = "denotation_acc"
    _parser_module_path = "tasks.sqa.parsers"
    default_input_fields: Dict[str, str] = {
        "table": "The table data to answer the question about",
        "conversation_history": "Previous questions and answers in this conversation",
        "question": "The current question to answer using the table",
    }
    default_output_fields: Dict[str, str] = {
        "answer": "The answer extracted from the table",
    }

    def _gold_output(self, meta: dict, raw: dict) -> dict:
        answer_text = raw.get("answer_text", meta.get("gold_answer", []))
        if isinstance(answer_text, str):
            answer_text = [answer_text]
        return {"answer": ", ".join(str(v) for v in answer_text)}

    def build_record(self, query: dict, meta: dict, raw: dict) -> dict:
        from tasks.wtq.table_formats import get_table_formatter

        table = raw.get("table", {})
        question = raw.get("question", query.get("content", ""))
        history = raw.get("history", [])
        table_name = raw.get("table_file", "")

        header = list(table.get("headers", []))
        rows = [list(r) for r in table.get("rows", [])]
        fmt = "markdown"
        if self._prompt_state is not None:
            explicit = self._prompt_state.metadata.get("table_format")
            if explicit:
                fmt = explicit
            else:
                style = self._prompt_state.format_style_name
                fmt = _OUTPUT_TO_TABLE_FORMAT.get(style, "markdown")
        table_str = get_table_formatter(fmt)(header, rows, table_name)

        history_str = ""
        if history:
            parts = []
            for turn in history:
                q = turn.get("question", "")
                a = turn.get("answer", [])
                a_str = ", ".join(str(v) for v in a) if isinstance(a, list) else str(a)
                parts.append(f"Q: {q}\nA: {a_str}")
            history_str = "\n".join(parts)

        return {
            "table": table_str,
            "conversation_history": history_str,
            "question": question,
        }

    def parse_response(self, raw_response: str) -> str:
        return super().parse_response(raw_response)

    def score(self, prediction: str, query_meta: dict) -> tuple[float, dict]:
        if isinstance(query_meta, str):
            query_meta = json.loads(query_meta)

        raw = query_meta.get("_raw", {})
        gold_answer = raw.get("answer_text", query_meta.get("gold_answer", []))
        if isinstance(gold_answer, str):
            gold_answer = [gold_answer]

        code_result = None
        code_error = None
        code_attempted = False
        if prediction.startswith("__CODE__"):
            code_attempted = True
            code_result, code_error = _execute_code_with_error(
                prediction[len("__CODE__"):].strip(),
                raw,
            )
            prediction = "" if code_result is None else str(code_result)

        gold_norm = {_normalize(str(v)) for v in gold_answer}
        pred_values = _parse_prediction(prediction)
        pred_norm = {_normalize(v) for v in pred_values}

        match = pred_norm == gold_norm
        score_val = 1.0 if match else 0.0

        metrics = {
            "status": "ok",
            "prediction": prediction,
            "pred_normalized": sorted(pred_norm),
            "gold": gold_answer,
            "gold_normalized": sorted(gold_norm),
        }
        if code_attempted:
            metrics["code_executed"] = code_error is None and code_result is not None
            if code_result is not None:
                metrics["code_result"] = str(code_result)
            if code_error:
                metrics["code_error"] = code_error

        return score_val, metrics


# ── helpers ──────────────────────────────────────────────────────────


def _extract_answer(text: str) -> str:
    """Extract answer from LLM response.

    Tries JSON, then a labeled `answer:` pattern, then a last-line
    heuristic with common prefixes.
    """
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
        for prefix in ["The answer is", "Answer:", "Therefore,", "So,", "Thus,"]:
            if last.lower().startswith(prefix.lower()):
                return last[len(prefix):].strip().rstrip(".")
        if len(last) < 200:
            return last

    return text


def _normalize(v: str) -> str:
    """Normalize an answer value for set-comparison."""
    v = v.strip().lower()
    v = v.strip(".,;:!?\"'")
    v = re.sub(r"\s+", " ", v)
    v = v.replace("\xa0", " ")
    v = v.replace(",", "")
    try:
        num = float(v)
        if num == int(num):
            v = str(int(num))
        else:
            v = f"{num:.4g}"
    except ValueError:
        pass
    return v


def _parse_prediction(prediction: str) -> List[str]:
    """Parse a prediction string into a list of values."""
    prediction = prediction.strip()

    try:
        parsed = json.loads(prediction)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed]
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

    return [p.strip() for p in parts if p.strip()]
