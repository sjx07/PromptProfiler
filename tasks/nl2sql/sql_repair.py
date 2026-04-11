"""SQL repair task — fix incorrect SQL given error context."""
from __future__ import annotations

import json
from typing import Dict

from prompt_profiler.task import BaseTask


class SqlRepair(BaseTask):
    name = "sql_repair"
    scorer = "ex_acc"
    # Fallback only — new configs should use base_field_specs() explicitly.
    default_input_fields: Dict[str, str] = {
        "question": "The natural language question",
        "schema": "Database schema (DDL)",
        "wrong_sql": "The incorrect SQL query to fix",
    }

    @classmethod
    def base_field_specs(cls) -> list:
        """Explicit input field funcs for the base config.

        Run scripts should seed these and include their func_ids in the
        base config. This makes input fields part of the config identity.
        """
        return [
            {"func_type": "add_input_field", "params": {"name": "question", "description": "The natural language question"}},
            {"func_type": "add_input_field", "params": {"name": "schema", "description": "Database schema (DDL)"}},
            {"func_type": "add_input_field", "params": {"name": "wrong_sql", "description": "The incorrect SQL query to fix"}},
        ]
    default_output_fields: Dict[str, str] = {
        "sql_query": "The corrected SQL query",
    }

    def _gold_output(self, meta: dict, raw: dict) -> dict:
        return {"sql_query": meta.get("gold_sql", raw.get("gold_sql", ""))}

    def build_record(self, query: dict, meta: dict, raw: dict) -> dict:
        return {
            "question": raw.get("question", query.get("content", "")),
            "schema": raw.get("schema", ""),
            "evidence": meta.get("evidence", ""),
            "wrong_sql": meta.get("wrong_sql", raw.get("wrong_sql", "")),
            "error_message": meta.get("error_message", raw.get("error_message", "")),
        }

    def parse_response(self, raw_response: str) -> str:
        from prompt_profiler.tasks.nl2sql.sql_generation import _extract_sql

        if self._prompt_state is not None:
            parsed = self._prompt_state.parse_output(raw_response)
            sql = parsed.get("sql_query", "").strip() if parsed else ""
            if sql:
                return _extract_sql(sql)
        return _extract_sql(raw_response)

    def score(self, prediction: str, query_meta: dict) -> tuple[float, dict]:
        from prompt_profiler.tasks.nl2sql.evaluate_sql import evaluate_execution_wrapper

        if isinstance(query_meta, str):
            query_meta = json.loads(query_meta)
        raw = query_meta.get("_raw", {})
        gold = {
            "db_path": raw.get("db_path", ""),
            "gold_sql_query": query_meta.get("gold_sql", raw.get("gold_sql", "")),
        }
        result = evaluate_execution_wrapper(gold, prediction)
        return result["score"], {
            "status": result.get("status", ""),
            "error_type": result.get("error_type"),
            "error_message": result.get("error_message"),
        }
