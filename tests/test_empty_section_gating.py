"""test_empty_section_gating.py — HARD ABORT GATE.

A config containing a section with no rules must NOT render that section.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

_TOOL_DIR = str(Path(__file__).parent.parent.parent.parent)
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from core.func_registry import ROOT_ID, apply_config, make_func_id
from core.store import CubeStore, OnConflict


def _temp_store():
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    f.close()
    return f.name


def _sec(title, ordinal, is_system=True, min_rules=0, max_rules=10):
    return {
        "node_type": "section",
        "parent_id": ROOT_ID,
        "payload": {
            "title":     title,
            "ordinal":   ordinal,
            "is_system": is_system,
            "min_rules": min_rules,
            "max_rules": max_rules,
        },
    }


def test_empty_section_not_rendered():
    """Section with zero rules must be absent from rendered system prompt."""
    db = _temp_store()
    try:
        store = CubeStore(db)

        sec_role_params = _sec("role", 0, min_rules=1, max_rules=3)
        sec_empty_params = _sec("empty_section", 99, min_rules=0, max_rules=5)

        sec_role_id  = make_func_id("insert_node", sec_role_params)
        sec_empty_id = make_func_id("insert_node", sec_empty_params)

        rule_params = {
            "node_type": "rule",
            "parent_id": sec_role_id,
            "payload":   {"content": "You are an expert analyst."},
        }

        specs = [
            {"func_id": sec_role_id,  "func_type": "insert_node", "params": sec_role_params,  "meta": {}},
            {"func_id": sec_empty_id, "func_type": "insert_node", "params": sec_empty_params, "meta": {}},
            {"func_id": make_func_id("insert_node", rule_params),
             "func_type": "insert_node", "params": rule_params, "meta": {}},
        ]
        store.upsert_funcs(specs, on_conflict=OnConflict.SKIP)

        func_ids = [s["func_id"] for s in specs]
        state = apply_config(func_ids, store)
        rendered = state.to_prompt_state()._build_system_content()
        store.close()

        assert "empty_section" not in rendered, (
            f"empty_section appeared in rendered output:\n{rendered}"
        )
        assert "role" in rendered, (
            f"role section missing from rendered output:\n{rendered}"
        )
    finally:
        os.unlink(db)


def test_section_with_rules_is_rendered():
    """Section with at least one rule must appear in rendered output."""
    db = _temp_store()
    try:
        store = CubeStore(db)

        sec_params = _sec("strategy", 10, min_rules=0, max_rules=10)
        sec_id = make_func_id("insert_node", sec_params)
        rule_params = {
            "node_type": "rule",
            "parent_id": sec_id,
            "payload":   {"content": "Use ORDER BY for sorted results."},
        }
        rule_id = make_func_id("insert_node", rule_params)

        specs = [
            {"func_id": sec_id,  "func_type": "insert_node", "params": sec_params,  "meta": {}},
            {"func_id": rule_id, "func_type": "insert_node", "params": rule_params, "meta": {}},
        ]
        store.upsert_funcs(specs, on_conflict=OnConflict.SKIP)

        state = apply_config([sec_id, rule_id], store)
        rendered = state.to_prompt_state()._build_system_content()
        store.close()

        assert "strategy" in rendered, (
            f"strategy section missing from rendered output:\n{rendered}"
        )
    finally:
        os.unlink(db)


def test_multiple_sections_only_nonempty_rendered():
    """With three sections, only the two with rules appear."""
    db = _temp_store()
    try:
        store = CubeStore(db)

        sec_a_params = _sec("section_a", 0)
        sec_b_params = _sec("section_b", 10)
        sec_c_params = _sec("section_c", 20)

        sec_a_id = make_func_id("insert_node", sec_a_params)
        sec_b_id = make_func_id("insert_node", sec_b_params)
        sec_c_id = make_func_id("insert_node", sec_c_params)

        rule_a_params = {"node_type": "rule", "parent_id": sec_a_id, "payload": {"content": "Rule for A."}}
        rule_c_params = {"node_type": "rule", "parent_id": sec_c_id, "payload": {"content": "Rule for C."}}

        rule_a_id = make_func_id("insert_node", rule_a_params)
        rule_c_id = make_func_id("insert_node", rule_c_params)

        specs = [
            {"func_id": sec_a_id,  "func_type": "insert_node", "params": sec_a_params,  "meta": {}},
            {"func_id": sec_b_id,  "func_type": "insert_node", "params": sec_b_params,  "meta": {}},
            {"func_id": sec_c_id,  "func_type": "insert_node", "params": sec_c_params,  "meta": {}},
            {"func_id": rule_a_id, "func_type": "insert_node", "params": rule_a_params, "meta": {}},
            {"func_id": rule_c_id, "func_type": "insert_node", "params": rule_c_params, "meta": {}},
        ]
        store.upsert_funcs(specs, on_conflict=OnConflict.SKIP)

        func_ids = [s["func_id"] for s in specs]
        state = apply_config(func_ids, store)
        rendered = state.to_prompt_state()._build_system_content()
        store.close()

        assert "section_a" in rendered
        assert "section_b" not in rendered, "section_b has no rules but appeared in output"
        assert "section_c" in rendered
    finally:
        os.unlink(db)
