"""Sequential Question Answering task — parse answers, score via denotation accuracy."""
from __future__ import annotations

import ast
import json
import re
from typing import Any, Dict, List

from task import BaseTask
from tasks.answer_normalization import normalize_numeric_grouping
from tasks.code_result_utils import _USE_DEFAULT_NORMALIZER
from tasks.parsing.code_output import CODE_PREFIX
from tasks.program_of_thought import ProgramOfThoughtMixin
from tasks.table_python_runtime import (
    PythonTableExecution,
    execute_python_table_code,
    runtime_from_rows,
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


def _build_python_table_runtime(raw: dict, *, coerce_types: bool = True):
    table_data = raw.get("table", {})
    header = list(table_data.get("headers", []))
    rows = [list(r) for r in table_data.get("rows", [])]
    if not header:
        return None
    history = raw.get("history", [])
    return runtime_from_rows(
        header,
        rows,
        table_name=raw.get("table_file", ""),
        data_extra={
            "history": history,
            "question": raw.get("question", ""),
        },
        extras={"history": history},
        coerce_types=coerce_types,
    )


def _execute_code_with_error(code: str, raw: dict) -> tuple[Any, str | None]:
    """Execute SQA Python code with table and conversation context in scope."""
    runtime = _build_python_table_runtime(raw, coerce_types=True)
    if runtime is None:
        return None, None
    outcome = execute_python_table_code(code, runtime)
    if outcome.error:
        fallback_runtime = _build_python_table_runtime(raw, coerce_types=False)
        if fallback_runtime is not None:
            fallback = execute_python_table_code(code, fallback_runtime)
            if fallback.error is None and fallback.value is not None:
                return fallback.value, None
    return outcome.value, outcome.error


class SequentialQA(ProgramOfThoughtMixin, BaseTask):
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

    def build_python_table_runtime(self, raw: dict, record: dict | None = None):
        return _build_python_table_runtime(raw, coerce_types=True)

    def execute_python_table(
        self,
        code: str,
        raw: dict,
        *,
        record: dict | None = None,
        normalize=_USE_DEFAULT_NORMALIZER,
    ) -> PythonTableExecution:
        runtime = _build_python_table_runtime(raw, coerce_types=True)
        if runtime is None:
            return PythonTableExecution(error="runtime_unavailable", executed=False)
        outcome = execute_python_table_code(code, runtime, normalize=normalize)
        if outcome.error:
            fallback_runtime = _build_python_table_runtime(raw, coerce_types=False)
            if fallback_runtime is not None:
                fallback = execute_python_table_code(
                    code,
                    fallback_runtime,
                    normalize=normalize,
                )
                if fallback.error is None and fallback.value is not None:
                    return fallback
        return outcome

    def execute_code_prediction(
        self,
        prediction: str,
        raw: dict,
        *,
        record: dict | None = None,
        normalize=_USE_DEFAULT_NORMALIZER,
    ) -> tuple[str, dict[str, Any]]:
        code = prediction[len(CODE_PREFIX):].strip()
        outcome = self.execute_python_table(
            code,
            raw,
            record=record,
            normalize=normalize,
        )
        metrics: dict[str, Any] = {
            "code_executed": outcome.executed,
            "runtime_binding": "python_table_scope",
        }
        if outcome.value is not None:
            metrics["code_result"] = str(outcome.value)
        if outcome.error:
            metrics["code_error"] = outcome.error
        prediction_text = "" if outcome.value is None else _format_prediction_value(outcome.value)
        return prediction_text, metrics

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
        history_parts = []
        for turn in history:
            q = turn.get("question", "")
            a = turn.get("answer", [])
            a_str = ", ".join(str(v) for v in a) if isinstance(a, list) else str(a)
            history_parts.append(f"{q} {a_str}".strip())
        transform_text = " ".join([question, *history_parts]).strip()
        header, rows = self._apply_transforms(header, rows, transform_text)
        fmt = "markdown"
        if self._prompt_state is not None:
            explicit = self._prompt_state.metadata.get("table_format")
            if explicit:
                fmt = explicit
            else:
                style = self._prompt_state.format_style_name
                fmt = _OUTPUT_TO_TABLE_FORMAT.get(style, "markdown")
        table_str = get_table_formatter(fmt)(header, rows, table_name)
        if self._pending_stats:
            table_str = self._pending_stats + "\n\n" + table_str

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

        code_metrics: dict[str, Any] = {}
        code_attempted = False
        if self.is_code_prediction(prediction):
            code_attempted = True
            prediction, code_metrics = self.execute_code_prediction(prediction, raw)

        gold_norm = {_normalize(str(v)) for v in gold_answer}
        pred_values = _parse_prediction(prediction)
        pred_norm = {_normalize(v) for v in pred_values}
        parse_strategy = "default"

        match = False
        for strategy, values in _parse_prediction_candidates(prediction):
            norm = {_normalize(v) for v in values}
            if norm == gold_norm:
                pred_values = values
                pred_norm = norm
                parse_strategy = strategy
                match = True
                break
        score_val = 1.0 if match else 0.0

        metrics = {
            "status": "ok",
            "prediction": prediction,
            "pred_normalized": sorted(pred_norm),
            "gold": gold_answer,
            "gold_normalized": sorted(gold_norm),
            "parse_strategy": parse_strategy,
        }
        if code_attempted:
            metrics.update(code_metrics)

        return score_val, metrics


# ── helpers ──────────────────────────────────────────────────────────


def _extract_answer(text: str) -> str:
    """Extract answer from LLM response.

    Tries JSON, then a labeled `answer:`/`answer =` pattern, then a last-line
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
                        return json.dumps([str(v) for v in val], ensure_ascii=False)
                    return str(val).strip()
        if isinstance(parsed, list):
            return json.dumps([str(v) for v in parsed], ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        pass

    m = re.search(
        r"(?:answer|result|value)\s*(?:[:=]|-)\s*(.+?)(?:\n|$)",
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
    v = normalize_numeric_grouping(v)
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
    candidates = _parse_prediction_candidates(prediction)
    return candidates[0][1] if candidates else []


def _parse_prediction_candidates(prediction: str) -> List[tuple[str, List[str]]]:
    """Return plausible SQA answer-list parses.

    SQA gold answers are lists of table-cell strings. Plain comma splitting is
    unsafe because cell values often contain commas, especially thousands
    separators and locations. Keep strict structured formats first, then add
    compatibility fallbacks for older free-form runs.
    """
    prediction = prediction.strip()
    if not prediction:
        return [("empty", [])]

    candidates: List[tuple[str, List[str]]] = []

    try:
        parsed = json.loads(prediction)
        if isinstance(parsed, list):
            _add_candidate(candidates, "json_list", [str(v).strip() for v in parsed])
        elif isinstance(parsed, dict):
            for key in ("answer", "result", "value", "response"):
                if key in parsed:
                    value = parsed[key]
                    if isinstance(value, list):
                        _add_candidate(
                            candidates,
                            "json_dict_list",
                            [str(v).strip() for v in value],
                        )
                    else:
                        _add_candidate(
                            candidates,
                            "json_dict_scalar",
                            [str(value).strip()],
                        )
                    break
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        parsed = ast.literal_eval(prediction)
        if isinstance(parsed, (list, tuple, set)):
            _add_candidate(candidates, "python_list", [str(v).strip() for v in parsed])
    except (ValueError, SyntaxError, TypeError):
        pass

    if "|" in prediction:
        _add_candidate(candidates, "pipe", prediction.split("|"))
    if "\n" in prediction:
        _add_candidate(candidates, "newline", prediction.split("\n"))

    numeric_values = _parse_numeric_thousands_list(prediction)
    if numeric_values:
        _add_candidate(candidates, "numeric_thousands_list", numeric_values)

    if "," in prediction and "\n" not in prediction:
        _add_candidate(candidates, "comma_legacy", prediction.split(","))

    _add_candidate(candidates, "whole", [prediction])
    return candidates


def _format_prediction_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return json.dumps([str(v) for v in value], ensure_ascii=False)
    return str(value)


def _add_candidate(
    candidates: List[tuple[str, List[str]]],
    strategy: str,
    values: List[str],
) -> None:
    cleaned = [str(v).strip() for v in values if str(v).strip()]
    if not any(existing == cleaned for _name, existing in candidates):
        candidates.append((strategy, cleaned))


def _parse_numeric_thousands_list(prediction: str) -> List[str]:
    """Parse lists such as ``20,000, 15,200`` without splitting thousands."""
    text = prediction.strip()
    if not text:
        return []
    grouped_sep = "[, \u00a0\u202f\u2007\u2009]"
    token_re = re.compile(
        rf"(?<![\w.])[$£€]?-?\d{{1,3}}(?:{grouped_sep}\d{{3}})+(?:\.\d+)?%?(?![\w.])"
        r"|(?<![\w.])[$£€]?-?\d+(?:\.\d+)?%?(?![\w.])"
    )
    matches = list(token_re.finditer(text))
    if not matches:
        return []
    remainder = token_re.sub("", text)
    if re.sub(r"[\s,;|]+", "", remainder):
        return []
    return [m.group(0).strip() for m in matches]
