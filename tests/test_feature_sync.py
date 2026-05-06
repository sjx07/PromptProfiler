"""test_feature_sync.py

Tests for FeatureRegistry.sync_to_cube() and CubeStore.sync_features().

Invariants:
  1. After sync, feature table count matches registry size.
  2. Re-syncing is idempotent: same count, same rows.
  3. Fields are correctly stored (feature_id, canonical_id, task, etc.).
  4. In-memory registries (no _source_path) store NULL source_path.
  5. Disk-loaded registries store the absolute path in source_path.
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

from core.feature_registry import FeatureRegistry, compute_feature_id
from core.func_registry import ROOT_ID, make_func_id
from core.store import CubeStore, OnConflict


# ── helpers ───────────────────────────────────────────────────────────

def _make_registry(task: str, features: list) -> FeatureRegistry:
    features_dict = {f["feature_id"]: f for f in features}
    return FeatureRegistry(task=task, features=features_dict)


def _open_fresh_store() -> tuple[CubeStore, str]:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    return CubeStore(db_path), db_path


SIMPLE_FEATURES = [
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
        "rationale": "Test rule A.",
    },
    {
        "feature_id": "feat_b",
        "task": "table_qa",
        "requires": [],
        "conflicts_with": ["feat_a"],
        "primitive_edits": [
            {"func_type": "insert_node",
             "params": {"node_type": "rule", "parent_id": ROOT_ID,
                        "payload": {"content": "Rule B."}}}
        ],
    },
]


# ── count matches registry size ───────────────────────────────────────

def test_sync_count_matches_registry_size():
    """After sync, feature table row count equals registry.list_features() length."""
    reg = _make_registry("table_qa", SIMPLE_FEATURES)
    store, db_path = _open_fresh_store()
    try:
        result = reg.sync_to_cube(store)
        assert result["synced"] == len(SIMPLE_FEATURES)
        assert store.stats()["feature"] == len(SIMPLE_FEATURES)
    finally:
        store.close()
        os.unlink(db_path)


def test_sync_empty_registry():
    """Syncing an empty registry results in zero feature rows."""
    reg = FeatureRegistry(task="table_qa", features={})
    store, db_path = _open_fresh_store()
    try:
        result = reg.sync_to_cube(store)
        assert result["synced"] == 0
        assert store.stats()["feature"] == 0
    finally:
        store.close()
        os.unlink(db_path)


# ── idempotency ───────────────────────────────────────────────────────

def test_sync_idempotent_same_count():
    """Re-syncing the same registry does not add duplicate rows."""
    reg = _make_registry("table_qa", SIMPLE_FEATURES)
    store, db_path = _open_fresh_store()
    try:
        reg.sync_to_cube(store)
        reg.sync_to_cube(store)  # second sync
        assert store.stats()["feature"] == len(SIMPLE_FEATURES)
    finally:
        store.close()
        os.unlink(db_path)


def test_sync_idempotent_same_rows():
    """Re-syncing does not change stored field values for unchanged features."""
    reg = _make_registry("table_qa", SIMPLE_FEATURES[:1])
    store, db_path = _open_fresh_store()
    try:
        reg.sync_to_cube(store)
        first_row = store._get_conn().execute(
            "SELECT * FROM feature"
        ).fetchone()

        reg.sync_to_cube(store)  # second sync
        second_row = store._get_conn().execute(
            "SELECT feature_id, canonical_id, task, primitive_spec FROM feature"
        ).fetchone()

        assert first_row["feature_id"] == second_row["feature_id"]
        assert first_row["canonical_id"] == second_row["canonical_id"]
    finally:
        store.close()
        os.unlink(db_path)


# ── field correctness ─────────────────────────────────────────────────

def test_sync_feature_id_is_content_hash():
    """Stored feature_id matches compute_feature_id(primitive_edits)."""
    feat = SIMPLE_FEATURES[0]
    expected_fid = compute_feature_id(feat["primitive_edits"])
    reg = _make_registry("table_qa", [feat])
    store, db_path = _open_fresh_store()
    try:
        reg.sync_to_cube(store)
        row = store._get_conn().execute(
            "SELECT * FROM feature WHERE canonical_id = 'feat_a'"
        ).fetchone()
        assert row is not None
        assert row["feature_id"] == expected_fid
    finally:
        store.close()
        os.unlink(db_path)


def test_sync_canonical_id_stored():
    """canonical_id column matches the human-readable name."""
    reg = _make_registry("table_qa", SIMPLE_FEATURES)
    store, db_path = _open_fresh_store()
    try:
        reg.sync_to_cube(store)
        rows = store._get_conn().execute(
            "SELECT canonical_id FROM feature ORDER BY canonical_id"
        ).fetchall()
        canonical_ids = [r["canonical_id"] for r in rows]
        assert canonical_ids == sorted(["feat_a", "feat_b"])
    finally:
        store.close()
        os.unlink(db_path)


def test_sync_requires_and_conflicts_stored_as_json():
    """requires_json and conflicts_json are valid JSON arrays."""
    reg = _make_registry("table_qa", SIMPLE_FEATURES)
    store, db_path = _open_fresh_store()
    try:
        reg.sync_to_cube(store)
        row = store._get_conn().execute(
            "SELECT requires_json, conflicts_json FROM feature WHERE canonical_id = 'feat_b'"
        ).fetchone()
        requires = json.loads(row["requires_json"])
        conflicts = json.loads(row["conflicts_json"])
        assert requires == []
        assert conflicts == ["feat_a"]
    finally:
        store.close()
        os.unlink(db_path)


def test_sync_rationale_stored():
    """rationale field is persisted when present in spec."""
    reg = _make_registry("table_qa", SIMPLE_FEATURES[:1])
    store, db_path = _open_fresh_store()
    try:
        reg.sync_to_cube(store)
        row = store._get_conn().execute(
            "SELECT rationale FROM feature WHERE canonical_id = 'feat_a'"
        ).fetchone()
        assert row["rationale"] == "Test rule A."
    finally:
        store.close()
        os.unlink(db_path)


def test_sync_source_path_null_for_in_memory():
    """In-memory registry (no _source_path) stores NULL source_path."""
    reg = _make_registry("table_qa", SIMPLE_FEATURES[:1])
    store, db_path = _open_fresh_store()
    try:
        reg.sync_to_cube(store)
        row = store._get_conn().execute(
            "SELECT source_path FROM feature"
        ).fetchone()
        assert row["source_path"] is None
    finally:
        store.close()
        os.unlink(db_path)


def test_sync_source_path_set_for_disk_loaded(tmp_path):
    """Disk-loaded registry stores absolute source_path in feature table."""
    task_dir = tmp_path / "my_task"
    task_dir.mkdir()
    feat_path = task_dir / "feat_hello.json"
    feat_path.write_text(json.dumps({
        "canonical_id": "feat_hello",
        "task": "my_task",
        "requires": [],
        "conflicts_with": [],
        "primitive_edits": [
            {"func_type": "insert_node",
             "params": {"node_type": "rule", "parent_id": ROOT_ID,
                        "payload": {"content": "Hello."}}}
        ],
    }))

    reg = FeatureRegistry.load("my_task", features_base=tmp_path)
    store, db_path = _open_fresh_store()
    try:
        reg.sync_to_cube(store)
        row = store._get_conn().execute(
            "SELECT source_path FROM feature WHERE canonical_id = 'feat_hello'"
        ).fetchone()
        assert row["source_path"] == str(feat_path)
    finally:
        store.close()
        os.unlink(db_path)


def test_sync_semantic_labels_and_component_scope():
    """Feature semantic_labels sync to labels; scope stays on component row."""
    feature = {
        "feature_id": "tcot",
        "task": "table_qa",
        "semantic_labels": [
            "reasoning.text_chain_of_thought",
            {
                "label": "style.explain_steps",
                "role": "style_rule",
                "description": "Ask the model to explain intermediate steps.",
            },
        ],
        "scope": {
            "dataset": "table_qa",
            "predicates": {"answer_type": "numeric"},
        },
        "primitive_edits": [
            {"func_type": "insert_node",
             "params": {"node_type": "rule", "parent_id": ROOT_ID,
                        "payload": {"content": "Think step by step."}}}
        ],
    }
    reg = _make_registry("table_qa", [feature])
    store, db_path = _open_fresh_store()
    try:
        reg.sync_to_cube(store)
        conn = store._get_conn()
        row = conn.execute(
            "SELECT feature_id, semantic_labels_json, scope_json FROM feature"
        ).fetchone()
        feature_id = row["feature_id"]
        assert json.loads(row["semantic_labels_json"]) == feature["semantic_labels"]
        assert json.loads(row["scope_json"]) == feature["scope"]

        labels = conn.execute(
            "SELECT label_id, description FROM feature_label ORDER BY label_id"
        ).fetchall()
        assert [r["label_id"] for r in labels] == [
            "reasoning.text_chain_of_thought",
            "style.explain_steps",
        ]
        assert labels[1]["description"] == "Ask the model to explain intermediate steps."

        memberships = conn.execute(
            """SELECT label_id, role
               FROM feature_label_membership
               WHERE feature_id = ?
               ORDER BY label_id""",
            (feature_id,),
        ).fetchall()
        assert [(r["label_id"], r["role"]) for r in memberships] == [
            ("reasoning.text_chain_of_thought", "implements"),
            ("style.explain_steps", "style_rule"),
        ]
    finally:
        store.close()
        os.unlink(db_path)


def test_sync_semantic_label_removal_clears_memberships():
    """Re-syncing a registry-owned feature with no labels removes old memberships."""
    primitive_edits = [
        {"func_type": "insert_node",
         "params": {"node_type": "rule", "parent_id": ROOT_ID,
                    "payload": {"content": "Think step by step."}}}
    ]
    labeled = {
        "feature_id": "tcot",
        "task": "table_qa",
        "semantic_labels": ["reasoning.text_chain_of_thought"],
        "primitive_edits": primitive_edits,
    }
    unlabeled = {
        "feature_id": "tcot",
        "task": "table_qa",
        "primitive_edits": primitive_edits,
    }
    store, db_path = _open_fresh_store()
    try:
        _make_registry("table_qa", [labeled]).sync_to_cube(store)
        conn = store._get_conn()
        assert conn.execute(
            "SELECT COUNT(*) FROM feature_label_membership"
        ).fetchone()[0] == 1

        _make_registry("table_qa", [unlabeled]).sync_to_cube(store)
        assert conn.execute(
            "SELECT COUNT(*) FROM feature_label_membership"
        ).fetchone()[0] == 0
    finally:
        store.close()
        os.unlink(db_path)
