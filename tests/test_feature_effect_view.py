"""test_feature_effect_view.py

Tests for the feature_effect view and CubeStore.feature_effect_df().

The view joins: evaluation -> execution -> config -> json_each(meta.feature_ids)
-> feature.  It only works for v7+ configs where meta.feature_ids stores
content-hash feature_ids (not canonical strings).

Tests:
  1. Empty cube → empty DataFrame (no crash).
  2. Full join: seed feature + config + query + execution + evaluation;
     query feature_effect view; verify score and canonical_id are present.
  3. Two features, each with one evaluation: GROUP BY canonical_id aggregation
     produces correct per-feature average scores.
  4. Cross-task: feature rows from different tasks both appear when joined.
  5. Pre-v7 config (canonical_id strings in meta.feature_ids) → NOT joinable.
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
from core.schema import make_query_id
from core.store import CubeStore, OnConflict


# ── helpers ───────────────────────────────────────────────────────────

def _open_fresh_store() -> tuple[CubeStore, str]:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    return CubeStore(db_path), db_path


def _make_registry(task: str, canonical_id: str, rule_content: str) -> FeatureRegistry:
    features = {
        canonical_id: {
            "canonical_id": canonical_id,
            "task": task,
            "requires": [],
            "conflicts_with": [],
            "primitive_edits": [
                {"func_type": "insert_node",
                 "params": {"node_type": "rule", "parent_id": ROOT_ID,
                            "payload": {"content": rule_content}}}
            ],
        }
    }
    return FeatureRegistry(task=task, features=features)


def _seed_full_pipeline(
    store: CubeStore,
    reg: FeatureRegistry,
    canonical_id: str,
    score: float,
    model: str = "test_model",
    scorer: str = "acc",
) -> dict:
    """Seed feature + func + config + query + execution + evaluation.

    Returns dict with all created IDs for assertion.
    """
    # Sync feature
    reg.sync_to_cube(store)
    feature_id = reg.feature_id_for(canonical_id)

    # Materialize and upsert funcs
    specs, f2f = reg.materialize([canonical_id])
    store.upsert_funcs(specs, on_conflict=OnConflict.SKIP)

    func_ids = [s["func_id"] for s in specs]
    meta = {
        "canonical_ids": [canonical_id],
        "feature_ids": [feature_id],
        "feature_to_funcs": f2f,
    }
    config_id = store.get_or_create_config(func_ids, meta=meta)

    # Seed query
    query_id = make_query_id("test_ds", f"What is {canonical_id}?")
    store.upsert_queries([{
        "query_id": query_id,
        "dataset": "test_ds",
        "content": f"What is {canonical_id}?",
        "meta": {},
    }], on_conflict=OnConflict.SKIP)

    # Seed execution
    exec_id = store.insert_execution(
        config_id, query_id, model,
        system_prompt="sys", user_content="usr",
        raw_response="42", prediction="42",
    )

    # Seed evaluation
    store.upsert_evaluation(exec_id, scorer, score, on_conflict=OnConflict.SKIP)

    return {
        "feature_id": feature_id,
        "config_id": config_id,
        "query_id": query_id,
        "exec_id": exec_id,
    }


# ── empty cube ────────────────────────────────────────────────────────

def test_feature_effect_df_empty_cube():
    """Empty cube returns empty DataFrame without crashing."""
    pytest.importorskip("pandas")
    store, db_path = _open_fresh_store()
    try:
        df = store.feature_effect_df()
        assert len(df) == 0
        assert "canonical_id" in df.columns
        assert "score" in df.columns
    finally:
        store.close()
        os.unlink(db_path)


# ── full join ─────────────────────────────────────────────────────────

def test_feature_effect_view_basic_join():
    """A fully seeded pipeline produces one row in feature_effect view."""
    pytest.importorskip("pandas")
    reg = _make_registry("table_qa", "enable_cot", "Think step by step.")
    store, db_path = _open_fresh_store()
    try:
        ids = _seed_full_pipeline(store, reg, "enable_cot", score=0.8)
        df = store.feature_effect_df()
        assert len(df) == 1
        row = df.iloc[0]
        assert row["canonical_id"] == "enable_cot"
        assert row["task"] == "table_qa"
        assert row["feature_id"] == ids["feature_id"]
        assert abs(row["score"] - 0.8) < 1e-9
    finally:
        store.close()
        os.unlink(db_path)


def test_feature_effect_view_query_id_correct():
    """query_id in feature_effect matches what was seeded."""
    pytest.importorskip("pandas")
    reg = _make_registry("table_qa", "enable_code", "Write Python code.")
    store, db_path = _open_fresh_store()
    try:
        ids = _seed_full_pipeline(store, reg, "enable_code", score=0.5)
        df = store.feature_effect_df()
        assert len(df) == 1
        assert df.iloc[0]["query_id"] == ids["query_id"]
    finally:
        store.close()
        os.unlink(db_path)


# ── two features ──────────────────────────────────────────────────────

def test_feature_effect_two_features_group_by():
    """Two features each with one score: GROUP BY canonical_id gives correct averages."""
    pytest.importorskip("pandas")
    store, db_path = _open_fresh_store()
    try:
        reg_a = _make_registry("table_qa", "feat_alpha", "Rule alpha.")
        reg_b = _make_registry("table_qa", "feat_beta", "Rule beta.")

        _seed_full_pipeline(store, reg_a, "feat_alpha", score=1.0, model="m1")
        _seed_full_pipeline(store, reg_b, "feat_beta", score=0.0, model="m2")

        df = store.feature_effect_df()
        assert len(df) == 2

        by_canonical = df.set_index("canonical_id")["score"].to_dict()
        assert abs(by_canonical["feat_alpha"] - 1.0) < 1e-9
        assert abs(by_canonical["feat_beta"] - 0.0) < 1e-9
    finally:
        store.close()
        os.unlink(db_path)


# ── cross-task ────────────────────────────────────────────────────────

def test_feature_effect_cross_task_both_appear():
    """Features from two different tasks both appear in the view."""
    pytest.importorskip("pandas")
    store, db_path = _open_fresh_store()
    try:
        reg_wtq = _make_registry("table_qa", "enable_cot", "Think step by step.")
        reg_nl2sql = _make_registry("nl2sql", "enable_repair", "Fix the SQL error.")

        _seed_full_pipeline(store, reg_wtq, "enable_cot", score=0.9, model="m1")
        _seed_full_pipeline(store, reg_nl2sql, "enable_repair", score=0.6, model="m2")

        df = store.feature_effect_df()
        assert len(df) == 2
        tasks = set(df["task"])
        assert "table_qa" in tasks
        assert "nl2sql" in tasks
    finally:
        store.close()
        os.unlink(db_path)


# ── pre-v7 config not joinable ────────────────────────────────────────

def test_pre_v7_config_not_joinable():
    """Config with canonical_id strings in feature_ids (pre-v7) does not join."""
    pytest.importorskip("pandas")
    reg = _make_registry("table_qa", "enable_cot", "Think step by step.")
    store, db_path = _open_fresh_store()
    try:
        # Sync feature so the feature table has the row
        reg.sync_to_cube(store)

        # Seed funcs
        specs, _ = reg.materialize(["enable_cot"])
        store.upsert_funcs(specs, on_conflict=OnConflict.SKIP)
        func_ids = [s["func_id"] for s in specs]

        # Config uses canonical string "enable_cot" (not hash) — pre-v7 style
        meta = {
            "feature_ids": ["enable_cot"],   # canonical string, not hash!
        }
        config_id = store.get_or_create_config(func_ids, meta=meta)

        query_id = make_query_id("test_ds", "Pre-v7 query")
        store.upsert_queries([{
            "query_id": query_id, "dataset": "test_ds",
            "content": "Pre-v7 query", "meta": {},
        }], on_conflict=OnConflict.SKIP)
        exec_id = store.insert_execution(config_id, query_id, "m1")
        store.upsert_evaluation(exec_id, "acc", 1.0, on_conflict=OnConflict.SKIP)

        df = store.feature_effect_df()
        # The canonical string "enable_cot" != feature_id hash → no join
        assert len(df) == 0
    finally:
        store.close()
        os.unlink(db_path)
