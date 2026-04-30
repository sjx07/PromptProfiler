"""End-to-end checks for round-1 authored features.

Every round-1 feature (provenance ∈ {creative, vista_synthesis, gepa_decompose})
must:
  1. parse as JSON
  2. validate against task module's BASE + SECTION feature set
  3. materialize cleanly
  4. render its rule body into the target module's system prompt
  5. NOT leak into other modules

Round-1 features are discovered dynamically by scanning features/<task>/*.json
and filtering by provenance, so future authoring batches are auto-tested.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from core.feature_registry import FeatureRegistry
from core.func_registry import apply_config_modules
from core.store import CubeStore

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = REPO_ROOT / "features"

ROUND_1_PROVENANCES = {"creative", "vista_synthesis", "gepa_decompose"}

# This test targets multi-module HoVer/HotpotQA features. Single-prompt tasks
# (sql_generation) are validated separately by tests/test_factory_validators.py.
ROUND_1_TASKS = {"hover_context", "hotpotqa_context"}


def _round_1_cases() -> List[Tuple[str, str]]:
    cases: List[Tuple[str, str]] = []
    if not FEATURES_DIR.is_dir():
        return cases
    for task_dir in sorted(FEATURES_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        if task_dir.name not in ROUND_1_TASKS:
            continue
        for path in sorted(task_dir.glob("[!_]*.json")):
            if path.name.startswith(("base_", "gepa_")):
                continue
            try:
                spec = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if spec.get("provenance") in ROUND_1_PROVENANCES:
                cases.append((task_dir.name, path.stem))
    return cases


def _modules_for_task(task: str) -> List[str]:
    if task == "hover_context":
        from tasks.hover_context.hover_context import HoverContextTask as T
    elif task == "hotpotqa_context":
        from tasks.hotpotqa_context.hotpotqa_context import HotpotQAContextTask as T
    else:
        raise ValueError(f"unknown task: {task}")
    return list(T.module_specs.keys())


def _record_for_render() -> Dict[str, str]:
    # Superset of input fields used by hover_context / hotpotqa_context modules,
    # so build_messages succeeds for any of them.
    return {
        "claim": "X", "question": "X",
        "summary_1": "Y", "summary_2": "Z",
        "context": "C", "passages": "P",
    }


_CASES = _round_1_cases()


@pytest.mark.skipif(not _CASES, reason="no round-1 features authored yet")
@pytest.mark.parametrize("task,canonical_id", _CASES)
def test_round1_feature_e2e(task: str, canonical_id: str) -> None:
    reg = FeatureRegistry.load(task=task)
    spec = reg._by_canonical[canonical_id]
    target_module = spec["target_module"]
    edits = spec["primitive_edits"]
    assert edits, f"{canonical_id}: no primitive_edits"

    # Use a stable rule-body prefix for matching (avoids quoting/escaping mismatches
    # when the rendered prompt JSON-encodes special chars).
    # Coalition features may have a section node first (no "content"); skip to the
    # first rule edit that carries a content payload.
    rule_edit = next(
        (e for e in edits if "content" in e["params"]["payload"]),
        None,
    )
    assert rule_edit is not None, f"{canonical_id}: no rule edit with content found"
    rule_body = rule_edit["params"]["payload"]["content"]
    needle = rule_body[:60]

    # 1. Build BASE + this feature
    modules = _modules_for_task(task)
    base_cids: List[str] = []
    for m in modules:
        base_cids += [f"_section_{m}", f"base_{m}"]
    cids = base_cids + [canonical_id]

    # 2. Validate (catches missing required sections, conflicts, cross-task ref)
    reg.validate_feature_set(cids)

    # 3. Materialize and ingest into an in-memory cube
    store = CubeStore(":memory:")
    reg.sync_to_cube(store)
    func_specs, _ = reg.materialize(cids)
    store.upsert_funcs(func_specs)
    func_ids = [s["func_id"] for s in func_specs]

    # 4. Render every module's system prompt
    states = apply_config_modules(func_ids, store, module_names=modules)
    rendered = {m: states[m].to_prompt_state()._build_system_content() for m in modules}

    # 5. Rule body lands in the target module
    assert needle in rendered[target_module], (
        f"{canonical_id}: rule body prefix not found in {target_module} system "
        f"prompt — likely wrong parent_id or stale section reference"
    )

    # 6. No leak into other modules
    leaks = [m for m in modules if m != target_module and needle in rendered[m]]
    assert not leaks, f"{canonical_id}: rule leaked into modules {leaks}"


def test_round1_features_present() -> None:
    """Belt-and-suspenders: at least one round-1 feature exists; if zero, the
    parametrized suite above silently no-ops, which would be a worse signal."""
    assert _CASES, "no round-1 features discovered — check provenance tags"
