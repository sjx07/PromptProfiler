"""test_feature_deps.py — Feature dependency enforcement tests.

Tests that the materializer (FeatureRegistry) correctly:
  - Rejects missing requires
  - Rejects conflicting feature pairs
  - Rejects cross-task requires
  - Materializes and deduplicates correctly (no $ref machinery)
  - Supports features_base injectable path (Phase A)
  - Sections are ordinary features loaded from _section_*.json files
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

_TOOL_DIR = str(Path(__file__).parent.parent.parent.parent)
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from core.feature_registry import FeatureRegistry, _FEATURES_BASE
from core.func_registry import ROOT_ID, make_func_id


# ── helpers ────────────────────────────────────────────────────────────

def _make_registry(task: str, features: list) -> FeatureRegistry:
    """Build a FeatureRegistry directly from in-memory specs (no disk I/O)."""
    features_dict = {f["feature_id"]: f for f in features}
    return FeatureRegistry(task=task, features=features_dict)


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
    """materialize() returns (specs, provenance) and assigns func_ids."""
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
    specs, provenance = reg.materialize(["simple_rule"])
    assert len(specs) == 1
    assert specs[0]["func_type"] == "insert_node"
    assert specs[0]["params"]["node_type"] == "rule"
    assert "func_id" in specs[0]
    assert specs[0]["func_id"]  # non-empty
    # Provenance is keyed by content-hash feature_id (not canonical_id)
    simple_rule_fid = reg.feature_id_for("simple_rule")
    assert simple_rule_fid in provenance
    assert provenance[simple_rule_fid] == [specs[0]["func_id"]]
    # meta.source_feature no longer present -- meta is empty
    assert specs[0]["meta"] == {}


def test_materialize_deduplicates_shared_sections():
    """Two features sharing the same section emit only ONE section func (dedup by content hash)."""
    section_params = {
        "node_type": "section",
        "parent_id": ROOT_ID,
        "payload": {"title": "reasoning", "ordinal": 30,
                    "is_system": True, "min_rules": 0, "max_rules": 10},
    }
    section_fid = make_func_id("insert_node", section_params)

    reg = _make_registry("table_qa", [
        {
            "feature_id": "_section_reasoning",
            "task": "table_qa",
            "requires": [],
            "conflicts_with": [],
            "primitive_edits": [
                {"func_type": "insert_node", "params": section_params},
            ],
        },
        {
            "feature_id": "feat_a",
            "task": "table_qa",
            "requires": ["_section_reasoning"],
            "conflicts_with": [],
            "primitive_edits": [
                {
                    "func_type": "insert_node",
                    "params": {
                        "node_type": "rule",
                        "parent_id": section_fid,
                        "payload": {"content": "Rule from feat_a."},
                    },
                }
            ],
        },
        {
            "feature_id": "feat_b",
            "task": "table_qa",
            "requires": ["_section_reasoning"],
            "conflicts_with": [],
            "primitive_edits": [
                {
                    "func_type": "insert_node",
                    "params": {
                        "node_type": "rule",
                        "parent_id": section_fid,
                        "payload": {"content": "Rule from feat_b."},
                    },
                }
            ],
        },
    ])

    specs, provenance = reg.materialize(["_section_reasoning", "feat_a", "feat_b"])
    # 1 section + 2 distinct rules = 3 specs
    assert len(specs) == 3
    func_ids = [s["func_id"] for s in specs]
    assert len(set(func_ids)) == 3, "All three must be distinct func_ids"
    # Section appears only under _section_reasoning in provenance (keyed by hash)
    sec_feature_id = reg.feature_id_for("_section_reasoning")
    assert provenance[sec_feature_id] == [section_fid]


def test_materialize_shared_primitive_many_to_many():
    """A primitive shared by two features appears in both provenance lists (many-to-many)."""
    shared_params = {
        "node_type": "rule",
        "parent_id": ROOT_ID,
        "payload": {"content": "Shared rule."},
    }
    shared_fid = make_func_id("insert_node", shared_params)

    reg = _make_registry("table_qa", [
        {
            "feature_id": "feat_x",
            "task": "table_qa",
            "requires": [],
            "conflicts_with": [],
            "primitive_edits": [{"func_type": "insert_node", "params": shared_params}],
        },
        {
            "feature_id": "feat_y",
            "task": "table_qa",
            "requires": [],
            "conflicts_with": [],
            "primitive_edits": [{"func_type": "insert_node", "params": shared_params}],
        },
    ])
    specs, provenance = reg.materialize(["feat_x", "feat_y"])
    # Deduped: only one func spec
    assert len(specs) == 1
    assert specs[0]["func_id"] == shared_fid
    # Both features claim it in provenance (many-to-many); keyed by content-hash feature_id
    feat_x_fid = reg.feature_id_for("feat_x")
    feat_y_fid = reg.feature_id_for("feat_y")
    assert shared_fid in provenance[feat_x_fid]
    assert shared_fid in provenance[feat_y_fid]


# ── disk-based loading ────────────────────────────────────────────────

def test_load_from_disk_table_qa():
    """FeatureRegistry.load() finds _section_*.json for table_qa."""
    reg = FeatureRegistry.load("table_qa")
    assert reg.task == "table_qa"
    features = reg.list_features()
    assert "_section_role" in features
    assert "_section_reasoning" in features
    assert "_section_strategy" in features
    assert "_section_table_handling" in features
    assert "_section_format_fix" in features


def test_load_from_disk_sql_repair():
    """FeatureRegistry.load() finds _section_*.json for sql_repair."""
    reg = FeatureRegistry.load("sql_repair")
    assert reg.task == "sql_repair"
    features = reg.list_features()
    assert "_section_role" in features
    assert "_section_sql_rules" in features
    assert "_section_error_analysis" in features


def test_load_unknown_task_raises():
    """Loading an unknown task raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        FeatureRegistry.load("nonexistent_task_xyz")


# ── features_base injectable (Phase A) ───────────────────────────────

def test_features_base_explicit_path():
    """Passing features_base explicitly overrides the package default."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        task_dir = base / "my_task"
        task_dir.mkdir()
        (task_dir / "feat_hello.json").write_text(json.dumps({
            "feature_id": "feat_hello",
            "task": "my_task",
            "requires": [],
            "conflicts_with": [],
            "primitive_edits": [
                {"func_type": "insert_node",
                 "params": {"node_type": "rule", "parent_id": "__root__",
                            "payload": {"content": "Hello from fixture."}}}
            ],
        }))

        reg = FeatureRegistry.load("my_task", features_base=base)
        assert "feat_hello" in reg.list_features()
        # Default package features are NOT loaded
        assert "_section_role" not in reg.list_features()


def test_features_base_env_var(monkeypatch):
    """PROMPTPROFILER_FEATURES_BASE env var overrides package default."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        task_dir = base / "env_task"
        task_dir.mkdir()
        (task_dir / "env_feat.json").write_text(json.dumps({
            "feature_id": "env_feat",
            "task": "env_task",
            "requires": [],
            "conflicts_with": [],
            "primitive_edits": [],
        }))

        monkeypatch.setenv("PROMPTPROFILER_FEATURES_BASE", str(base))
        reg = FeatureRegistry.load("env_task")
        assert "env_feat" in reg.list_features()


def test_features_base_explicit_overrides_env(monkeypatch):
    """Explicit features_base arg takes priority over env var."""
    with tempfile.TemporaryDirectory() as tmpdir_arg, \
         tempfile.TemporaryDirectory() as tmpdir_env:
        arg_base = Path(tmpdir_arg)
        env_base = Path(tmpdir_env)

        (arg_base / "tsk").mkdir()
        (arg_base / "tsk" / "arg_feat.json").write_text(json.dumps({
            "feature_id": "arg_feat", "task": "tsk",
            "requires": [], "conflicts_with": [], "primitive_edits": [],
        }))
        (env_base / "tsk").mkdir()
        (env_base / "tsk" / "env_feat.json").write_text(json.dumps({
            "feature_id": "env_feat", "task": "tsk",
            "requires": [], "conflicts_with": [], "primitive_edits": [],
        }))

        monkeypatch.setenv("PROMPTPROFILER_FEATURES_BASE", str(env_base))
        reg = FeatureRegistry.load("tsk", features_base=arg_base)
        assert "arg_feat" in reg.list_features()
        assert "env_feat" not in reg.list_features()
