"""Shared Python table runtime for program-of-thought table tasks."""
from __future__ import annotations

import collections
import datetime as dt
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from tasks.code_result_utils import (
    _USE_DEFAULT_NORMALIZER,
    dataframe_to_records,
    execute_python_code,
    make_string_dataframe,
    make_typed_dataframe,
    stringify_code_result,
)


@dataclass
class PythonTableRuntime:
    """Objects exposed to generated Python code."""

    df: Any
    records: list[dict[str, Any]]
    data: dict[str, Any]
    header: list[str]
    extras: dict[str, Any] = field(default_factory=dict)
    result_keys: tuple[str, ...] = ("answer", "result", "__result__")
    csv_path: str | None = None

    def scope(self) -> dict[str, Any]:
        import pandas as pd

        scope = {
            "df": self.df,
            "pd": pd,
            "records": self.records,
            "table": self.records,
            "data": self.data,
            "header": self.header,
            "re": re,
            "datetime": dt.datetime,
            "timedelta": dt.timedelta,
            "date": dt.date,
            "collections": collections,
            "Counter": collections.Counter,
            "math": math,
            "len": len,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "sorted": sorted,
            "enumerate": enumerate,
            "zip": zip,
            "range": range,
            "map": map,
            "filter": filter,
            "any": any,
            "all": all,
            "round": round,
            "isinstance": isinstance,
            "type": type,
        }
        if self.csv_path:
            scope["csv_path"] = self.csv_path
            scope["table_csv"] = self.csv_path
        scope.update(self.extras)
        return scope


@dataclass
class PythonTableExecution:
    value: Any = None
    error: str | None = None
    executed: bool = False

    @property
    def text(self) -> str:
        return "" if self.value is None else str(self.value)


def runtime_from_rows(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    table_name: str = "",
    data_extra: Mapping[str, Any] | None = None,
    extras: Mapping[str, Any] | None = None,
    result_keys: tuple[str, ...] = ("answer", "result", "__result__"),
    coerce_types: bool = True,
) -> PythonTableRuntime:
    """Build the default in-memory table runtime from tabular rows."""
    header = [str(h) for h in headers]
    df = make_typed_dataframe(header, rows) if coerce_types else make_string_dataframe(header, rows)
    records = dataframe_to_records(df)
    data = {"table": table_name, "rows": records}
    if data_extra:
        data.update(dict(data_extra))
    return PythonTableRuntime(
        df=df,
        records=records,
        data=data,
        header=header,
        extras=dict(extras or {}),
        result_keys=result_keys,
    )


def execute_python_table_code(
    code: str,
    runtime: PythonTableRuntime,
    *,
    normalize: Callable[[Any], Any] | object | None = _USE_DEFAULT_NORMALIZER,
) -> PythonTableExecution:
    """Execute generated Python against a table runtime."""
    outcome = execute_python_code(
        code,
        runtime.scope(),
        result_keys=runtime.result_keys,
        normalize=normalize,
    )
    return PythonTableExecution(
        value=outcome.value,
        error=outcome.error,
        executed=outcome.error is None and outcome.value is not None,
    )


def format_python_table_result(value: Any) -> str:
    """Convert an execution value into scorer-facing prediction text."""
    normalized = stringify_code_result(value)
    return "" if normalized is None else str(normalized)

