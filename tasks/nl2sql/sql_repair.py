"""SQL repair task — fix incorrect SQL given error context."""
from __future__ import annotations

import json
from typing import Dict

from task import BaseTask


class SqlRepair(BaseTask):
    name = "sql_repair"
    scorer = "ex_acc"
    # Parser module for output_field dispatch (sql_query)
    _parser_module_path = "tasks.nl2sql.parsers"
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
            {"func_type": "insert_node", "params": {"node_type": "input_field", "parent_id": "__root__", "payload": {"name": "question", "description": "The natural language question"}}},
            {"func_type": "insert_node", "params": {"node_type": "input_field", "parent_id": "__root__", "payload": {"name": "schema", "description": "Database schema (DDL)"}}},
            {"func_type": "insert_node", "params": {"node_type": "input_field", "parent_id": "__root__", "payload": {"name": "wrong_sql", "description": "The incorrect SQL query to fix"}}},
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
        """Dispatch to the registered parser for the single dispatch output_field."""
        return super().parse_response(raw_response)

    def score(self, prediction: str, query_meta: dict) -> tuple[float, dict]:
        from tasks.nl2sql.evaluate_sql import evaluate_execution_wrapper

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
