"""test_render_parity.py — HARD ABORT GATE.

3-5 canonical configs built with new insert_node primitives must render
byte-for-byte identically to baselines captured from the old handlers
on main.

Baselines are in tests/fixtures/phase0_baselines/<name>.txt.

Each canonical config uses:
  - insert_node(section) instead of define_section
  - insert_node(rule)    instead of add_rule
  - insert_node(input_field) instead of add_input_field
  - input_transform      instead of transform_input

The render pipeline (to_prompt_state + _build_system_content) must
produce identical output.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Ensure the tool/ directory is on the path so prompt_profiler imports work
_TOOL_DIR = str(Path(__file__).parent.parent.parent.parent)
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from prompt_profiler.core.func_registry import (
    ROOT_ID,
    PromptBuildState,
    apply_config,
    make_func_id,
)
from prompt_profiler.core.store import CubeStore, OnConflict

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "phase0_baselines"


def _render(state: PromptBuildState) -> str:
    return state.to_prompt_state()._build_system_content()


def _build_store_and_apply(specs: List[Dict[str, Any]]) -> str:
    """Seed specs into a temp CubeStore, apply config, render."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db = f.name
    try:
        store = CubeStore(db)
        func_ids = []
        for spec in specs:
            fid = make_func_id(spec["func_type"], spec["params"])
            store.upsert_funcs([{
                "func_id":   fid,
                "func_type": spec["func_type"],
                "params":    spec["params"],
                "meta":      {},
            }], on_conflict=OnConflict.SKIP)
            func_ids.append(fid)
        state = apply_config(func_ids, store)
        store.close()
        return _render(state)
    finally:
        os.unlink(db)


def _load_baseline(name: str) -> str:
    path = FIXTURE_DIR / f"{name}.txt"
    if not path.exists():
        pytest.skip(f"Baseline fixture not found: {path}")
    return path.read_text()


# ── canonical config 1: role section + 2 rules ────────────────────────

def _sec_params(title: str, ordinal: int, is_system: bool = True,
                min_rules: int = 0, max_rules: int = 10) -> dict:
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


def test_render_parity_c1_role_rules():
    sec_params = _sec_params("role", 0, is_system=True, min_rules=1, max_rules=3)
    sec_id = make_func_id("insert_node", sec_params)

    specs = [
        {"func_type": "insert_node", "params": sec_params},
        {"func_type": "insert_node", "params": {
            "node_type": "rule",
            "parent_id": sec_id,
            "payload":   {"content": "You are a helpful data analyst."},
        }},
        {"func_type": "insert_node", "params": {
            "node_type": "rule",
            "parent_id": sec_id,
            "payload":   {"content": "Answer concisely and precisely."},
        }},
    ]
    rendered = _build_store_and_apply(specs)
    baseline = _load_baseline("c1_role_rules")
    assert rendered == baseline, (
        f"c1_role_rules render drift.\n"
        f"Expected:\n{baseline}\n\nGot:\n{rendered}"
    )


# ── canonical config 2: strategy section + rule + set_format ─────────

def test_render_parity_c2_strategy_format():
    sec_params = _sec_params("strategy", 10, is_system=True, min_rules=0, max_rules=10)
    sec_id = make_func_id("insert_node", sec_params)

    specs = [
        {"func_type": "insert_node", "params": sec_params},
        {"func_type": "insert_node", "params": {
            "node_type": "rule",
            "parent_id": sec_id,
            "payload":   {"content": "Use ORDER BY when counting rows."},
        }},
        {"func_type": "set_format", "params": {"style": "json"}},
    ]
    rendered = _build_store_and_apply(specs)
    baseline = _load_baseline("c2_strategy_format")
    assert rendered == baseline, (
        f"c2_strategy_format render drift.\n"
        f"Expected:\n{baseline}\n\nGot:\n{rendered}"
    )


# ── canonical config 3: reasoning section + rule + input_field + output_field ──
# Previously used enable_cot to inject reasoning; now uses insert_node(output_field)
# directly. Rendered output is byte-for-byte identical — baseline unchanged.

def test_render_parity_c3_reasoning_cot():
    sec_params = _sec_params("reasoning", 30, is_system=True, min_rules=0, max_rules=10)
    sec_id = make_func_id("insert_node", sec_params)

    specs = [
        {"func_type": "insert_node", "params": sec_params},
        {"func_type": "insert_node", "params": {
            "node_type": "rule",
            "parent_id": sec_id,
            "payload":   {"content": "Think step by step before writing the final answer."},
        }},
        {"func_type": "insert_node", "params": {
            "node_type": "input_field",
            "parent_id": ROOT_ID,
            "payload":   {"name": "schema", "description": "Relevant tables and columns"},
        }},
        {"func_type": "insert_node", "params": {
            "node_type": "output_field",
            "parent_id": ROOT_ID,
            "payload":   {"name": "reasoning", "description": "Step-by-step reasoning and thought process"},
        }},
    ]
    rendered = _build_store_and_apply(specs)
    baseline = _load_baseline("c3_reasoning_cot")
    assert rendered == baseline, (
        f"c3_reasoning_cot render drift.\n"
        f"Expected:\n{baseline}\n\nGot:\n{rendered}"
    )


# ── canonical config 4: empty section gating via render parity ────────

def test_render_parity_c4_empty_section_gating():
    """Config with an empty section — baseline captured without that section rendered."""
    sec_role_params = _sec_params("role", 0, is_system=True, min_rules=1, max_rules=3)
    sec_role_id = make_func_id("insert_node", sec_role_params)
    sec_empty_params = _sec_params("empty_section", 99, is_system=False, min_rules=0, max_rules=5)

    specs = [
        {"func_type": "insert_node", "params": sec_role_params},
        {"func_type": "insert_node", "params": {
            "node_type": "rule",
            "parent_id": sec_role_id,
            "payload":   {"content": "You are an expert."},
        }},
        {"func_type": "insert_node", "params": sec_empty_params},  # no rules → not rendered
    ]
    rendered = _build_store_and_apply(specs)
    baseline = _load_baseline("c4_empty_section_gating")
    assert rendered == baseline, (
        f"c4_empty_section_gating render drift.\n"
        f"Expected:\n{baseline}\n\nGot:\n{rendered}"
    )


# ── canonical config 5: input_transform only ─────────────────────────

def test_render_parity_c5_transform_input():
    specs = [
        {"func_type": "input_transform", "params": {"fn": "prune_cols", "kwargs": {"k": 5}}},
        {"func_type": "set_format", "params": {"style": "markdown"}},
    ]
    rendered = _build_store_and_apply(specs)
    baseline = _load_baseline("c5_transform_input")
    assert rendered == baseline, (
        f"c5_transform_input render drift.\n"
        f"Expected:\n{baseline!r}\n\nGot:\n{rendered!r}"
    )
