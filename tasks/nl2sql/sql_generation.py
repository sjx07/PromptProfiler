"""SQL generation task — parse SQL, score via execution accuracy.

Inherits bind/build_prompt from BaseTask. Defines parse_response and score,
plus SQL extraction helpers that are fully self-contained (json + re only).
"""
from __future__ import annotations

import json
import re
from typing import Optional

from prompt_profiler.task import BaseTask
from prompt_profiler.tasks.nl2sql.evaluate_sql import evaluate_execution_wrapper


class SqlGeneration(BaseTask):
    name = "sql_generation"
    scorer = "ex_acc"
    default_input_fields = {
        "question": "The natural language question to answer",
        "schema": "Database schema (DDL)",
        "evidence": "Hints or domain knowledge for the question",
    }
    default_output_fields = {
        "sql_query": "The SQL query that answers the question",
    }

    # Few-shot: examples show question → SQL only, no schema (too long)
    _example_input_fields = {"question"}

    def _gold_output(self, meta: dict, raw: dict) -> dict:
        return {"sql_query": meta.get("gold_sql", raw.get("gold_sql_query", ""))}

    def build_record(self, query: dict, meta: dict, raw: dict) -> dict:
        record = super().build_record(query, meta, raw)
        # Produce filtered_schema from gold_table_column_map if available
        gold_map = raw.get("gold_table_column_map", {})
        if gold_map:
            record["filtered_schema"] = _gold_map_to_schema_str(gold_map)
        return record

    def parse_response(self, raw_response: str) -> str:
        if self._prompt_state is not None:
            parsed = self._prompt_state.parse_output(raw_response)
            sql = parsed.get("sql_query", "").strip() if parsed else ""
            if sql:
                return _extract_sql(sql)
        return _extract_sql(raw_response)

    def score(self, prediction: str, query_meta: dict) -> tuple[float, dict]:
        if isinstance(query_meta, str):
            query_meta = json.loads(query_meta)
        raw = query_meta.get("_raw", {})
        gold = {
            "db_path": raw.get("db_path", ""),
            "gold_sql_query": query_meta.get("gold_sql", ""),
        }
        result = evaluate_execution_wrapper(gold, prediction)
        return result["score"], {
            "status": result.get("status", ""),
            "error_type": result.get("error_type"),
            "error_message": result.get("error_message"),
        }


# ── gold schema helper ────────────────────────────────────────────────

def _gold_map_to_schema_str(gold_map: dict) -> str:
    """Convert {table: [columns]} to a readable schema string."""
    lines = []
    for table, columns in sorted(gold_map.items()):
        cols = ", ".join(columns) if isinstance(columns, list) else str(columns)
        lines.append(f"Table {table}: {cols}")
    return "\n".join(lines)


# ── SQL extraction ────────────────────────────────────────────────────

def _strip_markdown_block(text: str) -> Optional[str]:
    """Strip a fenced code block (``` ... ```) and return its inner content."""
    lines = text.strip().splitlines()
    if len(lines) < 2 or not lines[0].strip().startswith("```"):
        return None
    for end in range(len(lines) - 1, 0, -1):
        if lines[end].strip() == "```":
            return "\n".join(lines[1:end]).strip()
    return None


def _try_json_sql(text: str) -> Optional[str]:
    """Try to extract sql_query from a JSON blob or a partial JSON pattern."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "sql_query" in parsed:
            return parsed["sql_query"].strip()
    except (json.JSONDecodeError, TypeError, AttributeError):
        m = re.search(r'"sql_query"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
        if m:
            return m.group(1).replace("\\'", "'").replace('\\"', '"').strip()
        return None
    return None


def _extract_sql(raw: str) -> str:
    """Extract a SQL string from raw model output.

    Attempts JSON parsing, then markdown block stripping, then returns
    the input unchanged as a last resort.
    """
    sql = _try_json_sql(raw)
    if sql is not None:
        return sql
    inner = _strip_markdown_block(raw)
    if inner is not None:
        sql = _try_json_sql(inner)
        if sql is not None:
            return sql
        return inner
    return raw
