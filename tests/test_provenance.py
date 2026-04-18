"""test_provenance.py — Phase C: feature_to_funcs provenance in config.meta.

Demonstrates end-to-end flow:
  materialize() -> (specs, feature_to_funcs) -> store funcs -> get_or_create_config
  with feature_to_funcs stored at config.meta.feature_to_funcs.

Key invariant: a func_id shared by two features appears under both in
feature_to_funcs (many-to-many attribution).
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
from prompt_profiler.core.func_registry import ROOT_ID, make_func_id
from prompt_profiler.core.store import CubeStore, OnConflict


def _make_registry(task: str, features: list) -> FeatureRegistry:
    features_dict = {f["feature_id"]: f for f in features}
    return FeatureRegistry(task=task, features=features_dict)


# ── provenance tuple shape ────────────────────────────────────────────

def test_materialize_returns_tuple():
    """materialize() returns a 2-tuple (specs, feature_to_funcs)."""
    reg = _make_registry("table_qa", [
        {
            "feature_id": "feat_a",
            "task": "table_qa",
            "requires": [],
            "conflicts_with": [],
            "primitive_edits": [
                {"func_type": "insert_node",
                 "params": {"node_type": "rule", "parent_id": ROOT_ID,
                            "payload": {"content": "Rule A."}}}
            ],
        }
    ])
    result = reg.materialize(["feat_a"])
    assert isinstance(result, tuple) and len(result) == 2
    specs, f2f = result
    assert isinstance(specs, list)
    assert isinstance(f2f, dict)


def test_provenance_maps_feature_to_func_ids():
    """feature_to_funcs keys are feature_ids; values are lists of func_ids."""
    rule_params = {"node_type": "rule", "parent_id": ROOT_ID,
                   "payload": {"content": "Think step by step."}}
    expected_fid = make_func_id("insert_node", rule_params)

    reg = _make_registry("table_qa", [
        {
            "feature_id": "enable_cot",
            "task": "table_qa",
            "requires": [],
            "conflicts_with": [],
            "primitive_edits": [
                {"func_type": "insert_node", "params": rule_params}
            ],
        }
    ])
    specs, f2f = reg.materialize(["enable_cot"])
    # f2f is keyed by content-hash feature_id (not canonical_id)
    enable_cot_fid = reg.feature_id_for("enable_cot")
    assert enable_cot_fid in f2f
    assert f2f[enable_cot_fid] == [expected_fid]
    # func_id in provenance matches the emitted spec
    assert specs[0]["func_id"] == expected_fid


def test_no_source_feature_in_func_meta():
    """Emitted func specs must NOT carry meta.source_feature (provenance moved to caller)."""
    reg = _make_registry("table_qa", [
        {
            "feature_id": "some_feat",
            "task": "table_qa",
            "requires": [],
            "conflicts_with": [],
            "primitive_edits": [
                {"func_type": "insert_node",
                 "params": {"node_type": "rule", "parent_id": ROOT_ID,
                            "payload": {"content": "Some rule."}}}
            ],
        }
    ])
    specs, _ = reg.materialize(["some_feat"])
    for spec in specs:
        assert "source_feature" not in spec.get("meta", {}), (
            f"source_feature still present in func meta: {spec['meta']}"
        )


# ── many-to-many attribution ──────────────────────────────────────────

def test_shared_primitive_many_to_many_attribution():
    """A func_id produced by two features appears in both provenance entries."""
    shared_params = {"node_type": "rule", "parent_id": ROOT_ID,
                     "payload": {"content": "Always use JSON output."}}
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
    specs, f2f = reg.materialize(["feat_x", "feat_y"])

    # Only one unique func emitted (deduped)
    assert len(specs) == 1
    assert specs[0]["func_id"] == shared_fid

    # But provenance records both (f2f keyed by content-hash feature_id)
    feat_x_fid = reg.feature_id_for("feat_x")
    feat_y_fid = reg.feature_id_for("feat_y")
    assert shared_fid in f2f[feat_x_fid]
    assert shared_fid in f2f[feat_y_fid]


# ── end-to-end: store provenance in config.meta ───────────────────────

def test_end_to_end_provenance_in_config_meta():
    """Full flow: materialize -> seed funcs -> create config with feature_to_funcs in meta."""
    rule_params = {"node_type": "rule", "parent_id": ROOT_ID,
                   "payload": {"content": "Be concise."}}
    rule_fid = make_func_id("insert_node", rule_params)

    reg = _make_registry("table_qa", [
        {
            "feature_id": "enable_cot",
            "task": "table_qa",
            "requires": [],
            "conflicts_with": [],
            "primitive_edits": [
                {"func_type": "insert_node", "params": rule_params}
            ],
        }
    ])
    specs, feature_to_funcs = reg.materialize(["enable_cot"])
    enable_cot_fid = reg.feature_id_for("enable_cot")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        store = CubeStore(db_path)
        store.upsert_funcs(specs, on_conflict=OnConflict.SKIP)

        func_ids = [s["func_id"] for s in specs]
        meta = {
            "canonical_ids": ["enable_cot"],
            "feature_ids": [enable_cot_fid],
            "feature_to_funcs": feature_to_funcs,
        }
        config_id = store.get_or_create_config(func_ids, meta=meta)

        # Read back
        configs = store.list_configs()
        assert len(configs) == 1
        stored_meta = json.loads(configs[0]["meta"])
        assert "feature_to_funcs" in stored_meta
        # feature_to_funcs is keyed by content-hash feature_id
        assert stored_meta["feature_to_funcs"][enable_cot_fid] == [rule_fid]
        assert stored_meta["feature_ids"] == [enable_cot_fid]
        assert stored_meta["canonical_ids"] == ["enable_cot"]

        store.close()
    finally:
        os.unlink(db_path)


def test_end_to_end_many_to_many_in_config_meta():
    """Many-to-many provenance is correctly stored and retrieved from config.meta."""
    shared_params = {"node_type": "rule", "parent_id": ROOT_ID,
                     "payload": {"content": "Output JSON."}}
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
    specs, feature_to_funcs = reg.materialize(["feat_x", "feat_y"])
    feat_x_fid = reg.feature_id_for("feat_x")
    feat_y_fid = reg.feature_id_for("feat_y")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    try:
        store = CubeStore(db_path)
        store.upsert_funcs(specs, on_conflict=OnConflict.SKIP)
        func_ids = [s["func_id"] for s in specs]
        meta = {
            "canonical_ids": ["feat_x", "feat_y"],
            "feature_ids": [feat_x_fid, feat_y_fid],
            "feature_to_funcs": feature_to_funcs,
        }
        store.get_or_create_config(func_ids, meta=meta)

        configs = store.list_configs()
        stored_meta = json.loads(configs[0]["meta"])
        f2f = stored_meta["feature_to_funcs"]

        # Both features attribute the shared func (keyed by content-hash feature_id)
        assert shared_fid in f2f[feat_x_fid]
        assert shared_fid in f2f[feat_y_fid]

        store.close()
    finally:
        os.unlink(db_path)
