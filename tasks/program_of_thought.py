"""Shared program-of-thought execution adapter for table tasks."""
from __future__ import annotations

from typing import Any, Callable

from tasks.code_result_utils import _USE_DEFAULT_NORMALIZER
from tasks.parsing.code_output import CODE_PREFIX
from tasks.table_python_runtime import (
    PythonTableExecution,
    PythonTableRuntime,
    execute_python_table_code,
    format_python_table_result,
)


class ProgramOfThoughtMixin:
    """Mixin for tasks that execute `__CODE__` predictions."""

    def build_python_table_runtime(
        self,
        raw: dict,
        record: dict | None = None,
    ) -> PythonTableRuntime | None:
        raise NotImplementedError

    def execute_python_table(
        self,
        code: str,
        raw: dict,
        *,
        record: dict | None = None,
        normalize: Callable[[Any], Any] | object | None = _USE_DEFAULT_NORMALIZER,
    ) -> PythonTableExecution:
        runtime = self.build_python_table_runtime(raw, record)
        if runtime is None:
            return PythonTableExecution(error="runtime_unavailable", executed=False)
        return execute_python_table_code(code, runtime, normalize=normalize)

    def execute_code_prediction(
        self,
        prediction: str,
        raw: dict,
        *,
        record: dict | None = None,
        normalize: Callable[[Any], Any] | object | None = _USE_DEFAULT_NORMALIZER,
    ) -> tuple[str, dict[str, Any]]:
        """Execute a `__CODE__` prediction and return scorer-facing text + metrics."""
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
        return format_python_table_result(outcome.value), metrics

    @staticmethod
    def is_code_prediction(prediction: str) -> bool:
        return str(prediction or "").startswith(CODE_PREFIX)

