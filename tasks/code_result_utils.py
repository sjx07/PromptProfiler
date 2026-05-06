"""Helpers for normalizing code-executor return values."""
from __future__ import annotations

import contextlib
import io
import ast
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


_USE_DEFAULT_NORMALIZER = object()


@dataclass
class PythonExecutionResult:
    value: Any = None
    error: str | None = None


def make_typed_dataframe(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> Any:
    """Build a pandas DataFrame and coerce numeric-looking columns by position.

    Pandas returns a DataFrame for ``df[col]`` when headers are duplicated, so
    label-based coercion can crash before model code runs. Positional coercion
    keeps duplicate headers usable while preserving the existing typed-DataFrame
    runtime contract.
    """
    import pandas as pd

    header = list(headers)
    df = pd.DataFrame([list(row) for row in rows], columns=header)
    for index in range(len(df.columns)):
        series = df.iloc[:, index]
        cleaned = series.astype(str).str.replace(",", "", regex=False)
        cleaned = cleaned.str.replace(r"[\$£€]", "", regex=True)
        cleaned = cleaned.str.replace("%", "", regex=False)
        cleaned = cleaned.str.strip()
        try:
            df.iloc[:, index] = pd.to_numeric(cleaned)
        except (ValueError, TypeError):
            pass
    return df


def dataframe_to_records(df: Any) -> list[dict[str, Any]]:
    """Convert a DataFrame to records without dropping duplicate columns."""
    columns = _dedupe_columns([str(col) for col in df.columns])
    records = []
    for row in df.itertuples(index=False, name=None):
        records.append(dict(zip(columns, row)))
    return records


def execute_python_code(
    code: str,
    scope: Mapping[str, Any],
    *,
    result_keys: Sequence[str] = ("answer",),
    normalize: Callable[[Any], Any] | object | None = _USE_DEFAULT_NORMALIZER,
) -> PythonExecutionResult:
    """Run model Python with one mutable namespace.

    Using separate globals/locals breaks helper functions referenced from
    comprehensions and generators. A single namespace matches normal script
    execution and preserves assigned variables for result extraction.
    """
    env = dict(scope)

    try:
        value = eval(code, env)
        return PythonExecutionResult(_normalize(value, normalize), None)
    except Exception:
        pass

    stdout_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, env, env)
    except Exception as exc:
        return PythonExecutionResult(None, _format_error(exc))

    for key in result_keys:
        if key in env:
            return PythonExecutionResult(_normalize(env[key], normalize), None)

    captured = stdout_buf.getvalue().strip()
    if captured:
        last_line = captured.splitlines()[-1].strip()
        if last_line:
            return PythonExecutionResult(
                _normalize(_parse_captured_stdout(last_line), normalize),
                None,
            )

    return PythonExecutionResult(None, None)


def stringify_code_result(result: Any) -> Any:
    """Convert common tabular/vector results into scorer-friendly strings.

    Scalar values are left as-is to preserve existing executor behavior. Vector
    and narrow tabular values are comma-joined because denotation scorers expect
    multi-cell answers as a flat textual list.
    """
    if isinstance(result, str) or result is None:
        return result

    try:
        import pandas as pd

        if isinstance(result, pd.Series):
            return _join_values(result.tolist())
        if isinstance(result, pd.Index):
            return _join_values(result.tolist())
        if isinstance(result, pd.DataFrame):
            if result.empty:
                return ""
            if result.shape[1] == 1:
                return _join_values(result.iloc[:, 0].tolist())
            if result.shape[0] == 1:
                return _join_values(result.iloc[0].tolist())
            return result.to_string(index=False)
    except Exception:
        pass

    try:
        import numpy as np

        if isinstance(result, np.ndarray):
            return _join_values(result.ravel().tolist())
        if isinstance(result, np.generic):
            return result.item()
    except Exception:
        pass

    if isinstance(result, set):
        return _join_values(sorted(result, key=str))
    if isinstance(result, (list, tuple)):
        return _join_values(result)

    return result


def _join_values(values: Any) -> str:
    return ", ".join(str(v) for v in values)


def _normalize(value: Any, normalize: Callable[[Any], Any] | object | None) -> Any:
    if normalize is _USE_DEFAULT_NORMALIZER:
        return stringify_code_result(value)
    if normalize is None:
        return value
    return normalize(value)


def _parse_captured_stdout(line: str) -> Any:
    """Recover list-like Python reprs printed by model code."""
    stripped = line.strip()
    if not stripped:
        return line
    if not (
        (stripped.startswith("[") and stripped.endswith("]"))
        or (stripped.startswith("(") and stripped.endswith(")"))
        or (stripped.startswith("{") and stripped.endswith("}"))
    ):
        return line

    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return line
    if isinstance(parsed, (list, tuple, set)):
        return parsed
    return line


def _format_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:500]}"


def _dedupe_columns(columns: Sequence[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for col in columns:
        count = seen.get(col, 0) + 1
        seen[col] = count
        out.append(col if count == 1 else f"{col}.{count}")
    return out
