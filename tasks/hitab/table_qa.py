"""HiTab table QA task."""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List

from task import BaseTask


class HiTabQA(BaseTask):
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

    def _gold_output(self, meta: dict, raw: dict) -> dict:
        answer = raw.get("answer", meta.get("gold_answer", "[]"))
        values = _parse_answer_string(answer)
        return {"answer": ", ".join(str(v) for v in values)}

    def build_record(self, query: dict, meta: dict, raw: dict) -> dict:
        from tasks.hitab.loaders import table_content_to_markdown

        table_content = raw.get("table_content", {})
        question = raw.get("question", query.get("content", ""))
        table_md = table_content_to_markdown(table_content)
        return {
            "table": table_md,
            "question": question,
        }

    def parse_response(self, raw_response: str) -> str:
        return super().parse_response(raw_response)

    def score(self, prediction: str, query_meta: dict) -> tuple[float, dict]:
        if isinstance(query_meta, str):
            query_meta = json.loads(query_meta)

        raw = query_meta.get("_raw", {})
        gold_answer_str = raw.get("answer", query_meta.get("gold_answer", "[]"))

        if prediction.startswith("__CODE__"):
            code_result = _execute_code(prediction[len("__CODE__"):].strip(), raw)
            prediction = "" if code_result is None else str(code_result)

        gold_values = _parse_answer_string(gold_answer_str)
        pred_values = _normalize_answer_list(prediction)

        gold_norm = [_normalize_value(str(v)) for v in gold_values]
        pred_norm = [_normalize_value(v) for v in pred_values]

        score_val = 1.0 if set(pred_norm) == set(gold_norm) else 0.0
        return score_val, {
            "status": "ok",
            "prediction": prediction,
            "pred_normalized": pred_norm,
            "gold": gold_values,
            "gold_normalized": gold_norm,
        }


def _execute_code(code: str, raw: dict) -> Any:
    """Execute HiTab Python code with flattened and raw table context."""
    import contextlib as _contextlib
    import datetime as _dt
    import io as _io
    import math as _math
    import collections as _collections

    import pandas as pd

    from tasks.hitab.loaders import table_content_to_records

    table_content = raw.get("table_content", {})
    header, rows = table_content_to_records(table_content)
    if not header:
        return None

    df = pd.DataFrame(rows, columns=header)
    for col in df.columns:
        cleaned = df[col].astype(str).str.replace(",", "", regex=False)
        cleaned = cleaned.str.replace(r"[\$£€]", "", regex=True)
        cleaned = cleaned.str.replace("%", "", regex=False)
        cleaned = cleaned.str.strip()
        try:
            df[col] = pd.to_numeric(cleaned)
        except (ValueError, TypeError):
            pass

    table = df.to_dict("records")
    data = {"rows": table, "table_content": table_content}
    safe_globals = {
        "df": df,
        "pd": pd,
        "data": data,
        "table": table,
        "table_content": table_content,
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

    def _stringify(result: Any) -> Any:
        if isinstance(result, (list, set, tuple)):
            return ", ".join(str(v) for v in result)
        return result

    try:
        return _stringify(eval(code, safe_globals))
    except Exception:
        pass

    local_vars: dict = {}
    stdout_buf = _io.StringIO()
    try:
        with _contextlib.redirect_stdout(stdout_buf):
            exec(code, safe_globals, local_vars)
    except Exception:
        return None

    if "answer" in local_vars:
        return _stringify(local_vars["answer"])

    captured = stdout_buf.getvalue().strip()
    if captured:
        last_line = captured.splitlines()[-1].strip()
        if last_line:
            return last_line

    return None


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
    v = v.replace("\xa0", " ")
    v = v.replace(",", "")
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
