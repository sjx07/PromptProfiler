"""Table QA task — parse answers from LLM, score via denotation accuracy."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from prompt_profiler.task import BaseTask


def _execute_code(code: str, table_data: dict) -> Any:
    """Execute model-generated Python code with table as a pandas DataFrame.

    Provides `df` (pandas DataFrame with auto-coerced types) and `table`/`header`
    as legacy fallbacks. Returns the result or None on failure.
    """
    import pandas as pd
    import datetime as _dt, math as _math, collections as _collections

    header = table_data.get("header", [])
    rows = table_data.get("rows", [])
    df = pd.DataFrame(rows, columns=header)
    # Clean and coerce: strip commas, %, $, whitespace, then try numeric
    for col in df.columns:
        cleaned = df[col].astype(str).str.replace(",", "", regex=False)
        cleaned = cleaned.str.replace(r"[\$£€]", "", regex=True)
        cleaned = cleaned.str.replace("%", "", regex=False)
        cleaned = cleaned.str.strip()
        try:
            df[col] = pd.to_numeric(cleaned)
        except (ValueError, TypeError):
            df[col] = df[col]  # keep original

    # Legacy: also provide list-of-dicts for backward compat
    table = df.to_dict("records")

    safe_globals = {
        "df": df,
        "pd": pd,
        "table": table,
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
    try:
        result = eval(code, safe_globals)
        if isinstance(result, (list, set, tuple)):
            return ", ".join(str(v) for v in result)
        return result
    except Exception:
        try:
            # Try exec for multi-line code
            local_vars = {}
            exec(code, safe_globals, local_vars)
            if "answer" in local_vars:
                result = local_vars["answer"]
                if isinstance(result, (list, set, tuple)):
                    return ", ".join(str(v) for v in result)
                return result
        except Exception:
            pass
    return None


def _execute_sql(sql: str, table_data: dict) -> Any:
    """Execute model-generated SQL against an in-memory SQLite table.

    Creates table `t` with columns from the header, inserts all rows,
    then runs the query. Returns the result or None on failure.
    """
    import sqlite3

    header = table_data.get("header", [])
    rows = table_data.get("rows", [])
    if not header or not rows:
        return None

    try:
        conn = sqlite3.connect(":memory:")
        # Sanitize column names for SQL (replace spaces/special chars with _)
        safe_cols = []
        col_map = {}
        for h in header:
            safe = re.sub(r"[^a-zA-Z0-9_]", "_", h).strip("_")
            if not safe or safe[0].isdigit():
                safe = "col_" + safe
            col_map[h] = safe
            safe_cols.append(safe)

        # Create table with TEXT columns (SQLite is flexible with types)
        col_defs = ", ".join(f'"{c}" TEXT' for c in safe_cols)
        conn.execute(f"CREATE TABLE t ({col_defs})")

        # Insert rows
        placeholders = ", ".join("?" for _ in safe_cols)
        for row in rows:
            padded = row + [""] * (len(header) - len(row))
            conn.execute(f"INSERT INTO t VALUES ({placeholders})", padded[:len(header)])

        # Rewrite query: replace original column names with safe names
        rewritten = sql
        for orig, safe in col_map.items():
            if orig != safe:
                rewritten = rewritten.replace(f'"{orig}"', f'"{safe}"')
                rewritten = rewritten.replace(f"'{orig}'", f'"{safe}"')

        cursor = conn.execute(rewritten)
        results = cursor.fetchall()
        conn.close()

        if not results:
            return None
        # Single value
        if len(results) == 1 and len(results[0]) == 1:
            return results[0][0]
        # Single column, multiple rows
        if all(len(r) == 1 for r in results):
            return ", ".join(str(r[0]) for r in results)
        # Multiple columns — return first column
        return ", ".join(str(r[0]) for r in results)

    except Exception:
        return None


# Default table format when not explicitly set — follows output format style
_OUTPUT_TO_TABLE_FORMAT = {
    "json": "json_records",
    "markdown": "markdown",
    "plain": "markdown",
    "yaml": "markdown",
}


class TableQA(BaseTask):
    name = "table_qa"
    scorer = "denotation_acc"
    # Parser module for output_field dispatch (code / sql / answer)
    _parser_module_path = "prompt_profiler.tasks.wtq.parsers"
    default_input_fields: Dict[str, str] = {
        "table": "The table data to answer the question about",
        "question": "The question to answer using the table",
    }
    default_output_fields: Dict[str, str] = {
        "answer": "The answer extracted from the table",
    }

    def _gold_output(self, meta: dict, raw: dict) -> dict:
        answers = raw.get("answers", meta.get("gold_answers", []))
        return {"answer": ", ".join(answers)}

    def build_record(self, query: dict, meta: dict, raw: dict) -> dict:
        from prompt_profiler.tasks.wtq.table_formats import get_table_formatter

        table_data = raw.get("table", {})
        header = list(table_data.get("header", []))
        rows = [list(r) for r in table_data.get("rows", [])]
        name = table_data.get("name", "")
        question = raw.get("question", query.get("content", ""))

        # ── Input transforms (generic: legacy flags + transform_input) ──
        header, rows = self._apply_transforms(header, rows, question)

        # ── Table format selection ──────────────────────────────
        fmt = "markdown"
        if self._prompt_state is not None:
            explicit = self._prompt_state.metadata.get("table_format")
            if explicit:
                fmt = explicit
            else:
                style = self._prompt_state.format_style_name
                fmt = _OUTPUT_TO_TABLE_FORMAT.get(style, "markdown")

        formatter = get_table_formatter(fmt)
        table_str = formatter(header, rows, name)

        # ── Column statistics (prepended if computed by transform) ──
        if self._pending_stats:
            table_str = self._pending_stats + "\n\n" + table_str

        return {
            "table": table_str,
            "question": question,
        }

    def parse_response(self, raw_response: str) -> str:
        """Dispatch to the registered parser for the single dispatch output_field."""
        return super().parse_response(raw_response)

    def score(self, prediction: str, query_meta: dict) -> tuple[float, dict]:
        if isinstance(query_meta, str):
            query_meta = json.loads(query_meta)

        raw = query_meta.get("_raw", {})
        gold_answers = raw.get("answers", query_meta.get("gold_answers", []))

        # If prediction contains code or SQL, execute it
        code_result = None
        sql_result = None
        if prediction.startswith("__CODE__"):
            code_str = prediction[len("__CODE__"):].strip()
            code_result = _execute_code(code_str, raw.get("table", {}))
            prediction = str(code_result) if code_result is not None else ""
        elif prediction.startswith("__SQL__"):
            sql_str = prediction[len("__SQL__"):].strip()
            sql_result = _execute_sql(sql_str, raw.get("table", {}))
            prediction = str(sql_result) if sql_result is not None else ""

        gold_values = [_normalize_value(a) for a in gold_answers]

        # Try both split and unsplit — comma is ambiguous (list separator vs part of value)
        pred_values = _normalize_answer_list(prediction)
        match = set(pred_values) == set(gold_values)
        if not match:
            pred_unsplit = [_normalize_value(prediction)]
            match = set(pred_unsplit) == set(gold_values)
            if match:
                pred_values = pred_unsplit

        score_val = 1.0 if match else 0.0

        metrics = {
            "status": "ok",
            "prediction": prediction,
            "pred_normalized": pred_values,
            "gold": gold_answers,
            "gold_normalized": gold_values,
        }
        if code_result is not None:
            metrics["code_executed"] = True
            metrics["code_result"] = str(code_result)
        if sql_result is not None:
            metrics["sql_executed"] = True
            metrics["sql_result"] = str(sql_result)

        return score_val, metrics


def _extract_answer(text: str) -> str:
    """Extract answer from LLM response."""
    text = text.strip()

    # Try JSON parsing
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

    # Try structured pattern: "answer: ..."
    m = re.search(
        r"(?:answer|result|value)\s*[:\-]\s*(.+?)(?:\n|$)",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().rstrip(".")

    # If response has thinking + final answer, take last line
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if lines:
        # Check if last line looks like a clean answer
        last = lines[-1]
        # Remove common prefixes
        for prefix in [
            "The answer is",
            "Answer:",
            "Therefore,",
            "So,",
            "Thus,",
            "The result is",
        ]:
            if last.lower().startswith(prefix.lower()):
                last = last[len(prefix):].strip().rstrip(".")
                return last

        # If short enough, use the last line as-is
        if len(last) < 200:
            return last

    return text


def _normalize_value(v: str) -> str:
    """Normalize a single answer value for comparison."""
    v = v.strip().lower()
    # Remove leading/trailing punctuation
    v = v.strip(".,;:!?\"'")
    # Normalize whitespace
    v = re.sub(r"\s+", " ", v)
    # Remove non-breaking spaces
    v = v.replace("\xa0", " ")
    # Normalize common number formats
    v = v.replace(",", "")  # Remove thousand separators
    # Try to normalize to number if possible
    try:
        num = float(v)
        if num == int(num):
            v = str(int(num))
        else:
            v = str(num)
    except ValueError:
        pass
    return v


def _normalize_answer_list(prediction: str) -> List[str]:
    """Parse a prediction string into a list of normalized values."""
    prediction = prediction.strip()

    # Try JSON list
    try:
        parsed = json.loads(prediction)
        if isinstance(parsed, list):
            return [_normalize_value(str(v)) for v in parsed]
    except (json.JSONDecodeError, TypeError):
        pass

    # Split by common delimiters
    # Check for pipe-separated (WTQ native format)
    if "|" in prediction:
        parts = prediction.split("|")
    elif "," in prediction and "\n" not in prediction:
        parts = prediction.split(",")
    elif "\n" in prediction:
        parts = prediction.split("\n")
    else:
        parts = [prediction]

    return [_normalize_value(p) for p in parts if p.strip()]
