"""NL2SQL parser registry — one registered parser per dispatch output_field.

Both SqlGeneration and SqlRepair use the same dispatch field: "sql_query".
A single parser in this module covers both task classes.

Dispatch fields:
  "sql_query" — model returns a SQL string to answer the question
"""
from __future__ import annotations

from typing import Any, Callable, Dict

PARSER_REGISTRY: Dict[str, Callable[[str, Any], str]] = {}

# Fields that drive dispatch for nl2sql tasks
DISPATCH_FIELDS = frozenset({"sql_query"})


def register_parser(field_name: str) -> Callable:
    """Decorator — register a parser for a single dispatch output_field."""
    def decorator(fn: Callable[[str, Any], str]) -> Callable[[str, Any], str]:
        PARSER_REGISTRY[field_name] = fn
        return fn
    return decorator


@register_parser("sql_query")
def parse_sql_query_field(response_text: str, task: Any) -> str:
    """Extract SQL from LLM response for NL2SQL tasks (generation + repair)."""
    from prompt_profiler.tasks.nl2sql.sql_generation import _extract_sql

    if task._prompt_state is not None:
        parsed = task._prompt_state.parse_output(response_text)
        sql = parsed.get("sql_query", "").strip() if parsed else ""
        if sql:
            return _extract_sql(sql)
    return _extract_sql(response_text)
