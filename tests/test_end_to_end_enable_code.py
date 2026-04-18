"""test_end_to_end_enable_code.py

End-to-end test: materialize enable_code feature → build config → bind TableQA
→ render prompt → parse response → confirm __CODE__ prefix lands in prediction.

Flow:
  1. Build a FeatureRegistry with a minimal enable_code feature (adds output_field=code
     and a rule pointing at a strategy section).
  2. materialize(["enable_code"]) → (specs, f2f)
  3. Seed into a temp CubeStore; create config.
  4. apply_config → PromptBuildState; bind TableQA task.
  5. Render prompt: confirm "code" appears in output_fields.
  6. Simulate model response: parse_response → confirm __CODE__ prefix.
  7. Confirm __CODE__ prediction does NOT raise in score() when raw has no table data
     (graceful None handling expected).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

_TOOL_DIR = str(Path(__file__).parent.parent.parent.parent)
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from prompt_profiler.core.feature_registry import FeatureRegistry
from prompt_profiler.core.func_registry import ROOT_ID, apply_config, make_func_id
from prompt_profiler.core.store import CubeStore, OnConflict
from prompt_profiler.tasks.wtq.table_qa import TableQA


# ── feature definitions ───────────────────────────────────────────────

def _strategy_section_params():
    return {
        "node_type": "section",
        "parent_id": ROOT_ID,
        "payload": {
            "title":     "Strategy",
            "ordinal":   10,
            "is_system": True,
            "min_rules": 0,
            "max_rules": 10,
        },
    }


def _make_registry() -> FeatureRegistry:
    """Build a minimal FeatureRegistry containing _section_strategy + enable_code."""
    sec_params = _strategy_section_params()
    sec_id = make_func_id("insert_node", sec_params)

    features = {
        "_section_strategy": {
            "feature_id": "_section_strategy",
            "task": "table_qa",
            "requires": [],
            "conflicts_with": [],
            "primitive_edits": [
                {"func_type": "insert_node", "params": sec_params},
            ],
        },
        "enable_code": {
            "feature_id": "enable_code",
            "task": "table_qa",
            "requires": ["_section_strategy"],
            "conflicts_with": ["enable_sql"],
            "primitive_edits": [
                {
                    "func_type": "insert_node",
                    "params": {
                        "node_type": "rule",
                        "parent_id": sec_id,
                        "payload": {"content": "Write Python code to compute the answer from the provided table."},
                    },
                },
                {
                    "func_type": "insert_node",
                    "params": {
                        "node_type": "output_field",
                        "parent_id": ROOT_ID,
                        "payload": {"name": "code", "description": "Python code that computes the answer."},
                    },
                },
            ],
        },
    }
    return FeatureRegistry(task="table_qa", features=features)


# ── end-to-end test ───────────────────────────────────────────────────

def test_enable_code_output_field_in_prompt_state():
    """After binding with enable_code feature, 'code' must be in output_fields."""
    reg = _make_registry()
    # _section_strategy is required by enable_code
    specs, f2f = reg.materialize(["_section_strategy", "enable_code"])

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmpf:
        db_path = tmpf.name
    try:
        store = CubeStore(db_path)
        store.upsert_funcs(specs, on_conflict=OnConflict.SKIP)

        func_ids = [s["func_id"] for s in specs]
        build_state = apply_config(func_ids, store)

        task = TableQA()
        task.bind(build_state)

        output_fields = task._prompt_state.semantic.output_fields
        assert "code" in output_fields, f"Expected 'code' in output_fields, got: {output_fields}"

        store.close()
    finally:
        os.unlink(db_path)


def test_enable_code_parse_response_returns_code_prefix():
    """parse_response on a Python expression returns a __CODE__-prefixed string."""
    reg = _make_registry()
    specs, _ = reg.materialize(["_section_strategy", "enable_code"])

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmpf:
        db_path = tmpf.name
    try:
        store = CubeStore(db_path)
        store.upsert_funcs(specs, on_conflict=OnConflict.SKIP)

        func_ids = [s["func_id"] for s in specs]
        build_state = apply_config(func_ids, store)

        task = TableQA()
        task.bind(build_state)

        # Simulate model returning a JSON blob with a code field (json format style)
        model_response = json.dumps({"code": "df['score'].max()"})
        prediction = task.parse_response(model_response)

        assert prediction.startswith("__CODE__"), (
            f"Expected __CODE__ prefix, got: {prediction!r}"
        )
        assert "df" in prediction

        store.close()
    finally:
        os.unlink(db_path)


def test_enable_code_score_graceful_on_empty_table():
    """score() with __CODE__ prediction and empty table data returns 0.0 without raising."""
    reg = _make_registry()
    specs, _ = reg.materialize(["_section_strategy", "enable_code"])

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmpf:
        db_path = tmpf.name
    try:
        store = CubeStore(db_path)
        store.upsert_funcs(specs, on_conflict=OnConflict.SKIP)

        func_ids = [s["func_id"] for s in specs]
        build_state = apply_config(func_ids, store)

        task = TableQA()
        task.bind(build_state)

        prediction = "__CODE__df['score'].max()"
        query_meta = {
            "_raw": {
                "answers": ["42"],
                "table": {"header": [], "rows": []},  # empty table → code execution fails
            },
        }
        score_val, metrics = task.score(prediction, query_meta)

        # Code execution on empty table returns None → prediction becomes "" → score 0.0
        assert score_val == 0.0
        assert "status" in metrics

        store.close()
    finally:
        os.unlink(db_path)


def test_enable_code_dispatch_field_is_code():
    """_dispatch_field() returns 'code' when enable_code feature is active."""
    reg = _make_registry()
    specs, _ = reg.materialize(["_section_strategy", "enable_code"])

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmpf:
        db_path = tmpf.name
    try:
        store = CubeStore(db_path)
        store.upsert_funcs(specs, on_conflict=OnConflict.SKIP)

        func_ids = [s["func_id"] for s in specs]
        build_state = apply_config(func_ids, store)

        task = TableQA()
        task.bind(build_state)

        assert task._dispatch_field() == "code"

        store.close()
    finally:
        os.unlink(db_path)
