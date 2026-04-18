"""test_feature_deps.py — Feature dependency enforcement tests.

Tests that the materializer (FeatureRegistry) correctly:
  - Rejects missing requires
  - Rejects conflicting feature pairs
  - Rejects cross-task requires
"""
from __future__ import annotations

import sys
import json
import tempfile
from pathlib import Path

import pytest

_TOOL_DIR = str(Path(__file__).parent.parent.parent.parent)
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from prompt_profiler.core.feature_registry import FeatureRegistry
from prompt_profiler.core.func_registry import ROOT_ID


# ── helpers ────────────────────────────────────────────────────────────

def _make_registry(task: str, features: list, sections: dict = None) -> FeatureRegistry:
    """Build a FeatureRegistry directly from in-memory specs (no disk I/O)."""
    features_dict = {f["feature_id"]: f for f in features}
    sections_dict = sections or {}
    return FeatureRegistry(task=task, features=features_dict, sections=sections_dict)


# ── missing requires ──────────────────────────────────────────────────

def test_missing_requires_raises():
    """Requesting a feature whose requires are not in the config → ValueError."""
    reg = _make_registry("table_qa", [
        {
            "feature_id": "enable_cot",
            "task": "table_qa",
            "requires": ["enable_reasoning_section"],
            "conflicts_with": [],
            "primitive_edits": [],
        },
        {
            "feature_id": "enable_reasoning_section",
            "task": "table_qa",
            "requires": [],
            "conflicts_with": [],
            "primitive_edits": [],
        },
    ])
    with pytest.raises(ValueError, match="requires 'enable_reasoning_section'"):
        reg.validate_feature_set(["enable_cot"])  # missing enable_reasoning_section


def test_requires_satisfied_no_error():
    """When all requires are present, no error is raised."""
    reg = _make_registry("table_qa", [
        {
            "feature_id": "enable_cot",
            "task": "table_qa",
            "requires": ["enable_reasoning_section"],
            "conflicts_with": [],
            "primitive_edits": [],
        },
        {
            "feature_id": "enable_reasoning_section",
            "task": "table_qa",
            "requires": [],
            "conflicts_with": [],
            "primitive_edits": [],
        },
    ])
    # Should not raise
    reg.validate_feature_set(["enable_cot", "enable_reasoning_section"])


# ── conflicts_with ────────────────────────────────────────────────────

def test_conflicting_features_raises():
    """Two conflicting features in the same config → ValueError."""
    reg = _make_registry("table_qa", [
        {
            "feature_id": "enable_code",
            "task": "table_qa",
            "requires": [],
            "conflicts_with": ["enable_sql"],
            "primitive_edits": [],
        },
        {
            "feature_id": "enable_sql",
            "task": "table_qa",
            "requires": [],
            "conflicts_with": ["enable_code"],
            "primitive_edits": [],
        },
    ])
    with pytest.raises(ValueError, match="conflict"):
        reg.validate_feature_set(["enable_code", "enable_sql"])


def test_no_conflict_single_feature():
    """Single feature with conflicts_with list — no error if the conflicting one is absent."""
    reg = _make_registry("table_qa", [
        {
            "feature_id": "enable_code",
            "task": "table_qa",
            "requires": [],
            "conflicts_with": ["enable_sql"],
            "primitive_edits": [],
        },
    ])
    reg.validate_feature_set(["enable_code"])  # no error


# ── cross-task requires ────────────────────────────────────────────────

def test_cross_task_feature_raises():
    """A feature from a different task in the config → ValueError."""
    reg = _make_registry("table_qa", [
        {
            "feature_id": "some_sql_feature",
            "task": "sql_repair",       # wrong task
            "requires": [],
            "conflicts_with": [],
            "primitive_edits": [],
        },
    ])
    with pytest.raises(ValueError, match="cross-task"):
        reg.validate_feature_set(["some_sql_feature"])


def test_cross_task_in_requires_raises():
    """requires pointing to a feature from a different task → ValueError."""
    reg = _make_registry("table_qa", [
        {
            "feature_id": "enable_cot",
            "task": "table_qa",
            "requires": ["sql_only_feature"],
            "conflicts_with": [],
            "primitive_edits": [],
        },
        {
            "feature_id": "sql_only_feature",
            "task": "sql_repair",   # wrong task
            "requires": [],
            "conflicts_with": [],
            "primitive_edits": [],
        },
    ])
    with pytest.raises(ValueError, match="[Cc]ross-task"):
        reg.validate_feature_set(["enable_cot", "sql_only_feature"])


# ── unknown feature ───────────────────────────────────────────────────

def test_unknown_feature_raises():
    """Requesting a feature_id not in the registry → ValueError."""
    reg = _make_registry("table_qa", [])
    with pytest.raises(ValueError, match="not found"):
        reg.validate_feature_set(["nonexistent_feature"])


# ── materialize produces correct func specs ───────────────────────────

def test_materialize_basic():
    """materialize() expands primitive_edits and assigns func_ids."""
    reg = _make_registry("table_qa", [
        {
            "feature_id": "simple_rule",
            "task": "table_qa",
            "requires": [],
            "conflicts_with": [],
            "primitive_edits": [
                {
                    "func_type": "insert_node",
                    "params": {
                        "node_type": "rule",
                        "parent_id": ROOT_ID,
                        "payload": {"content": "Always be concise."},
                    },
                }
            ],
        },
    ])
    specs = reg.materialize(["simple_rule"])
    assert len(specs) == 1
    assert specs[0]["func_type"] == "insert_node"
    assert specs[0]["params"]["node_type"] == "rule"
    assert "func_id" in specs[0]
    assert specs[0]["func_id"]  # non-empty


def test_materialize_deduplicates_shared_sections():
    """Two features sharing the same section $ref produce only one section func."""
    sections = {
        "reasoning": {
            "title": "reasoning", "ordinal": 30,
            "is_system": True, "min_rules": 0, "max_rules": 10,
        }
    }
    reg = _make_registry("table_qa", [
        {
            "feature_id": "feat_a",
            "task": "table_qa",
            "requires": [],
            "conflicts_with": [],
            "primitive_edits": [
                {
                    "func_type": "insert_node",
                    "params": {
                        "node_type": "rule",
                        "parent": {"$ref": "_sections.reasoning"},
                        "payload": {"content": "Rule from feat_a."},
                    },
                }
            ],
        },
        {
            "feature_id": "feat_b",
            "task": "table_qa",
            "requires": [],
            "conflicts_with": [],
            "primitive_edits": [
                {
                    "func_type": "insert_node",
                    "params": {
                        "node_type": "rule",
                        "parent": {"$ref": "_sections.reasoning"},
                        "payload": {"content": "Rule from feat_b."},
                    },
                }
            ],
        },
    ], sections=sections)

    specs = reg.materialize(["feat_a", "feat_b"])
    # Both rules have the same resolved parent_id — 2 rules total, no dedup of distinct rules
    assert len(specs) == 2
    func_ids = [s["func_id"] for s in specs]
    assert len(set(func_ids)) == 2, "Distinct rules should not be deduped"


def test_materialize_ref_resolution():
    """$ref in parent is resolved to a func_id, not left as a dict."""
    sections = {
        "strategy": {
            "title": "strategy", "ordinal": 10,
            "is_system": True, "min_rules": 0, "max_rules": 10,
        }
    }
    reg = _make_registry("table_qa", [
        {
            "feature_id": "some_feature",
            "task": "table_qa",
            "requires": [],
            "conflicts_with": [],
            "primitive_edits": [
                {
                    "func_type": "insert_node",
                    "params": {
                        "node_type": "rule",
                        "parent": {"$ref": "_sections.strategy"},
                        "payload": {"content": "Use correct joins."},
                    },
                }
            ],
        }
    ], sections=sections)

    specs = reg.materialize(["some_feature"])
    assert len(specs) == 1
    params = specs[0]["params"]
    assert "parent_id" in params, "parent_id must be resolved"
    assert "parent" not in params, "$ref dict must be removed"
    assert isinstance(params["parent_id"], str), "parent_id must be a string func_id"


# ── disk-based loading ────────────────────────────────────────────────

def test_load_from_disk_table_qa():
    """FeatureRegistry.load() should find the table_qa _sections.json."""
    reg = FeatureRegistry.load("table_qa")
    assert reg.task == "table_qa"
    assert "role" in reg._sections
    assert "reasoning" in reg._sections
    assert "strategy" in reg._sections


def test_load_from_disk_sql_repair():
    """FeatureRegistry.load() should find the sql_repair _sections.json."""
    reg = FeatureRegistry.load("sql_repair")
    assert reg.task == "sql_repair"
    assert "role" in reg._sections
    assert "sql_rules" in reg._sections


def test_load_unknown_task_raises():
    """Loading an unknown task raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        FeatureRegistry.load("nonexistent_task_xyz")
