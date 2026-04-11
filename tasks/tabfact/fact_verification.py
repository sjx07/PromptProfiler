"""Fact verification task — parse True/False verdict, score via accuracy."""
from __future__ import annotations

import json
import re
from typing import Dict

from prompt_profiler.task import BaseTask


# Default table format when not explicitly set — follows output format style
_OUTPUT_TO_TABLE_FORMAT = {
    "json": "json_records",
    "markdown": "markdown",
    "plain": "markdown",
    "yaml": "markdown",
}


class FactVerification(BaseTask):
    name = "fact_verification"
    scorer = "fv_acc"
    default_input_fields: Dict[str, str] = {
        "table": "The table data to verify the statement against",
        "statement": "The statement to verify as True or False",
    }
    default_output_fields: Dict[str, str] = {
        "verdict": "True if the statement is supported by the table, False otherwise",
    }
    code_field_description: str = (
        "Python expression using `df` (pandas DataFrame with typed columns) "
        "that verifies the statement against the table data and evaluates to True or False. "
        "You MUST reference `df` in your expression — do NOT output a bare True/False literal. "
        "Example: `df[df['wins'] > 0].shape[0] >= 3`"
    )

    def bind(self, state, **kwargs) -> None:
        super().bind(state, **kwargs)
        # Override generic code field description with task-specific one
        if self._prompt_state and self._prompt_state.code_execution:
            eof = self._prompt_state.effective_output_fields
            if "code" in eof:
                self._prompt_state.semantic.output_fields.pop("verdict", None)
                self._prompt_state.semantic.output_fields["code"] = self.code_field_description

    def _gold_output(self, meta: dict, raw: dict) -> dict:
        gold = raw.get("label", meta.get("gold_label", 0))
        return {"verdict": "True" if int(gold) == 1 else "False"}

    def build_record(self, query: dict, meta: dict, raw: dict) -> dict:
        from prompt_profiler.tasks.tabfact.loaders import parse_table_text
        from prompt_profiler.tasks.wtq.table_formats import get_table_formatter

        table_text = raw.get("table_text", "")
        caption = meta.get("table_caption", "")
        statement = raw.get("statement", query.get("content", ""))

        parsed = parse_table_text(table_text)
        header = list(parsed["headers"])
        rows = [list(r) for r in parsed["rows"]]

        # ── Input transforms (generic: legacy flags + transform_input) ──
        header, rows = self._apply_transforms(header, rows, statement)

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
        table_str = formatter(header, rows, caption)

        # ── Column statistics (prepended if computed by transform) ──
        if self._pending_stats:
            table_str = self._pending_stats + "\n\n" + table_str

        return {
            "table": table_str,
            "statement": statement,
        }

    def parse_response(self, raw_response: str) -> str:
        # Coerce to string — LLM may return bare bool from JSON true/false
        raw_response = str(raw_response) if not isinstance(raw_response, str) else raw_response
        if self._prompt_state is not None:
            parsed = self._prompt_state.parse_output(raw_response)
            if parsed:
                code = str(parsed.get("code", "")).strip()
                if code:
                    return f"__CODE__{code}"
                verdict = str(parsed.get("verdict", "")).strip()
                if verdict:
                    return _extract_verdict(verdict)
        return _extract_verdict(raw_response)

    def score(self, prediction: str, query_meta: dict) -> tuple[float, dict]:
        if isinstance(query_meta, str):
            query_meta = json.loads(query_meta)

        raw = query_meta.get("_raw", {})
        gold_label = raw.get("label", query_meta.get("gold_label", None))

        # Execute code if present
        if prediction.startswith("__CODE__"):
            code_str = prediction[len("__CODE__"):].strip()
            code_result = _execute_verdict_code(code_str, raw)
            if code_result is not None:
                prediction = "True" if code_result else "False"
            else:
                prediction = ""

        pred_binary = _verdict_to_int(prediction)
        gold_binary = int(gold_label) if gold_label is not None else None

        if pred_binary is None or gold_binary is None:
            return 0.0, {
                "status": "parse_error",
                "prediction": prediction,
                "gold": str(gold_label),
            }

        score_val = 1.0 if pred_binary == gold_binary else 0.0
        return score_val, {
            "status": "ok",
            "prediction": prediction,
            "gold": str(gold_label),
            "pred_binary": pred_binary,
            "gold_binary": gold_binary,
        }


def _execute_verdict_code(code: str, raw: dict) -> bool | None:
    """Execute model-generated Python code for fact verification.

    Provides `df` (pandas DataFrame) from the table data.
    Returns True/False or None on failure.
    """
    import pandas as pd

    table_text = raw.get("table_text", "")
    if not table_text:
        return None

    try:
        from prompt_profiler.tasks.tabfact.loaders import parse_table_text
        parsed = parse_table_text(table_text)
        header = list(parsed["headers"])
        rows = [list(r) for r in parsed["rows"]]
    except Exception:
        return None

    if not header or not rows:
        return None

    try:
        df = pd.DataFrame(rows, columns=header)
        # Auto-coerce numeric columns
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

        local_vars: dict = {"df": df, "pd": pd, "len": len, "sum": sum,
                           "min": min, "max": max, "abs": abs, "round": round,
                           "sorted": sorted, "str": str, "int": int, "float": float,
                           "bool": bool, "True": True, "False": False}
        exec(f"__result__ = ({code})", {"__builtins__": {}}, local_vars)
        result = local_vars.get("__result__")
        return bool(result)
    except Exception:
        return None


def _extract_verdict(text: str) -> str:
    """Extract True/False verdict from LLM response."""
    text = text.strip()

    # Try JSON parsing first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            for key in ("verdict", "answer", "result", "label"):
                if key in parsed:
                    val = str(parsed[key]).strip().lower()
                    if val in ("true", "1", "yes", "entailed", "supported"):
                        return "True"
                    if val in ("false", "0", "no", "refuted", "not supported"):
                        return "False"
    except (json.JSONDecodeError, TypeError):
        pass

    # Try structured pattern: "verdict: True/False"
    m = re.search(
        r"(?:verdict|answer|result|conclusion)\s*[:\-]\s*(true|false)\b",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).capitalize()

    # Check start of response
    lower = text.lower()
    if lower.startswith("true"):
        return "True"
    if lower.startswith("false"):
        return "False"

    # Search for last occurrence (conclusion is usually at the end)
    matches = list(re.finditer(r"\b(true|false)\b", lower))
    if matches:
        return matches[-1].group(1).capitalize()

    return text


def _verdict_to_int(text: str) -> int | None:
    """Convert verdict string to binary (1=True, 0=False)."""
    lower = text.strip().lower()
    if lower in ("true", "1", "yes", "entailed", "supported"):
        return 1
    if lower in ("false", "0", "no", "refuted", "not supported"):
        return 0
    return None
