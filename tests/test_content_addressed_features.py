"""test_content_addressed_features.py

Tests for the content-addressed feature_id = sha256(primitive_edits)[:12].

Invariants:
  1. Same primitive_edits → same feature_id (stability).
  2. Different primitive_edits → different feature_id.
  3. Changing requires/conflicts_with/rationale/task does NOT change feature_id.
  4. canonical_id stays the same even when primitive_edits change (different feature_id).
  5. Two feature files with identical primitive_edits produce the same feature_id
     regardless of canonical_id (cross-task hash identity).
  6. compute_feature_id([]) is stable (empty primitive_edits).
  7. configs pinning the old feature_id hash are still analyzable (they reference
     the exact content version used at run time).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TOOL_DIR = str(Path(__file__).parent.parent.parent.parent)
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from core.feature_registry import FeatureRegistry, compute_feature_id
from core.func_registry import ROOT_ID


# ── helpers ───────────────────────────────────────────────────────────

def _make_registry(canonical_id: str, primitive_edits: list, **kwargs) -> FeatureRegistry:
    spec = {
        "canonical_id": canonical_id,
        "task": kwargs.get("task", "table_qa"),
        "requires": kwargs.get("requires", []),
        "conflicts_with": kwargs.get("conflicts_with", []),
        "primitive_edits": primitive_edits,
        "rationale": kwargs.get("rationale"),
    }
    return FeatureRegistry(task=spec["task"], features={canonical_id: spec})


RULE_A_EDITS = [
    {"func_type": "insert_node",
     "params": {"node_type": "rule", "parent_id": ROOT_ID,
                "payload": {"content": "Rule A."}}}
]

RULE_B_EDITS = [
    {"func_type": "insert_node",
     "params": {"node_type": "rule", "parent_id": ROOT_ID,
                "payload": {"content": "Rule B."}}}
]


# ── stability ─────────────────────────────────────────────────────────

def test_same_edits_same_feature_id():
    """Same primitive_edits → same feature_id, even in separate registries."""
    fid1 = compute_feature_id(RULE_A_EDITS)
    fid2 = compute_feature_id(RULE_A_EDITS)
    assert fid1 == fid2


def test_different_edits_different_feature_id():
    """Different primitive_edits → different feature_id."""
    fid_a = compute_feature_id(RULE_A_EDITS)
    fid_b = compute_feature_id(RULE_B_EDITS)
    assert fid_a != fid_b


def test_feature_id_is_12_chars():
    """feature_id is exactly 12 hex characters."""
    fid = compute_feature_id(RULE_A_EDITS)
    assert len(fid) == 12
    assert all(c in "0123456789abcdef" for c in fid)


def test_empty_primitive_edits_stable():
    """compute_feature_id([]) returns the same value each time."""
    fid1 = compute_feature_id([])
    fid2 = compute_feature_id([])
    assert fid1 == fid2
    assert len(fid1) == 12


# ── metadata changes do NOT change feature_id ─────────────────────────

def test_requires_change_does_not_change_feature_id():
    """Changing requires does NOT change feature_id."""
    reg1 = _make_registry("feat", RULE_A_EDITS, requires=[])
    reg2 = _make_registry("feat", RULE_A_EDITS, requires=["some_dep"])
    assert reg1.feature_id_for("feat") == reg2.feature_id_for("feat")


def test_conflicts_change_does_not_change_feature_id():
    """Changing conflicts_with does NOT change feature_id."""
    reg1 = _make_registry("feat", RULE_A_EDITS, conflicts_with=[])
    reg2 = _make_registry("feat", RULE_A_EDITS, conflicts_with=["other_feat"])
    assert reg1.feature_id_for("feat") == reg2.feature_id_for("feat")


def test_rationale_change_does_not_change_feature_id():
    """Changing rationale does NOT change feature_id."""
    reg1 = _make_registry("feat", RULE_A_EDITS, rationale=None)
    reg2 = _make_registry("feat", RULE_A_EDITS, rationale="Updated rationale.")
    assert reg1.feature_id_for("feat") == reg2.feature_id_for("feat")


def test_canonical_id_change_does_not_change_feature_id():
    """Two features with the same primitive_edits but different canonical_ids
    share the same feature_id (content identity)."""
    fid_a = compute_feature_id(RULE_A_EDITS)
    reg = _make_registry("enable_cot_v2", RULE_A_EDITS)
    assert reg.feature_id_for("enable_cot_v2") == fid_a


# ── canonical_id stability ────────────────────────────────────────────

def test_canonical_id_stable_when_edits_change():
    """canonical_id is always the human-readable name regardless of feature_id."""
    reg1 = _make_registry("enable_cot", RULE_A_EDITS)
    reg2 = _make_registry("enable_cot", RULE_B_EDITS)
    assert reg1.canonical_id_for(reg1.feature_id_for("enable_cot")) == "enable_cot"
    assert reg2.canonical_id_for(reg2.feature_id_for("enable_cot")) == "enable_cot"
    # But their feature_ids differ
    assert reg1.feature_id_for("enable_cot") != reg2.feature_id_for("enable_cot")


# ── cross-task hash identity ──────────────────────────────────────────

def test_cross_task_identical_edits_same_feature_id():
    """Two features from different tasks with identical primitive_edits
    produce the same feature_id (content identity is task-agnostic)."""
    reg_wtq = _make_registry("enable_cot", RULE_A_EDITS, task="table_qa")
    reg_nl2sql = _make_registry("enable_cot", RULE_A_EDITS, task="nl2sql")
    assert reg_wtq.feature_id_for("enable_cot") == reg_nl2sql.feature_id_for("enable_cot")


def test_cross_task_different_edits_different_feature_id():
    """Two 'enable_cot' features pointing at different section parent_ids
    produce different feature_ids (different content)."""
    edits_wtq = [
        {"func_type": "insert_node",
         "params": {"node_type": "rule", "parent_id": "aabbcc112233",
                    "payload": {"content": "Think step by step."}}}
    ]
    edits_nl2sql = [
        {"func_type": "insert_node",
         "params": {"node_type": "rule", "parent_id": "ddeeff445566",
                    "payload": {"content": "Think step by step."}}}
    ]
    fid_wtq = compute_feature_id(edits_wtq)
    fid_nl2sql = compute_feature_id(edits_nl2sql)
    assert fid_wtq != fid_nl2sql


# ── old feature_id pins still analyzable ─────────────────────────────

def test_old_feature_id_pin_still_resolved():
    """A config that pinned an old feature_id hash can still be resolved
    via feature_id_for() on the registry that produced it."""
    reg = _make_registry("enable_cot", RULE_A_EDITS)
    old_fid = reg.feature_id_for("enable_cot")

    # Simulate a "new" version with different primitive_edits
    new_reg = _make_registry("enable_cot", RULE_B_EDITS)
    new_fid = new_reg.feature_id_for("enable_cot")

    assert old_fid != new_fid
    # The old registry can still resolve the old pinned hash
    assert reg.canonical_id_for(old_fid) == "enable_cot"
    # The new registry cannot resolve the old hash (different content)
    with pytest.raises(KeyError):
        new_reg.canonical_id_for(old_fid)


# ── lookup helpers ────────────────────────────────────────────────────

def test_feature_id_for_unknown_raises_key_error():
    """feature_id_for with unknown canonical_id raises KeyError."""
    reg = _make_registry("feat_a", RULE_A_EDITS)
    with pytest.raises(KeyError):
        reg.feature_id_for("nonexistent")


def test_canonical_id_for_unknown_raises_key_error():
    """canonical_id_for with unknown feature_id raises KeyError."""
    reg = _make_registry("feat_a", RULE_A_EDITS)
    with pytest.raises(KeyError):
        reg.canonical_id_for("000000000000")
