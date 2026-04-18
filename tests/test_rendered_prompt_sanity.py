"""test_rendered_prompt_sanity.py — render generator outputs to disk for human review.

For each feature-level generator (add_one_feature / leave_one_out_feature /
coalition_feature) this test:

  1. Loads the real table_qa FeatureRegistry.
  2. Seeds func specs into a temp CubeStore.
  3. Generates configs using the MVP base + experiment feature sets.
  4. For every generated config, builds a PromptBuildState, binds TableQA,
     and renders (system_prompt, user_prompt) against a canned query.
  5. Writes the rendered prompts to ``tests/renders/<generator>/<label>.txt``
     so both the assistant and the user can eyeball them.

The assertions are deliberately light (non-empty, feature-specific markers).
The primary artifact is the rendered text files.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

_TOOL_DIR = str(Path(__file__).parent.parent.parent.parent)
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from core.feature_registry import FeatureRegistry
from core.func_registry import apply_config
from core.store import CubeStore, OnConflict
from experiment.config_generators import generate
from tasks.wtq.table_qa import TableQA


# ── MVP feature set (matches runs/example_add_one.json) ───────────────

BASE_FEATURES = [
    "_section_role",
    "_section_strategy",
    "_section_table_handling",
    "_section_reasoning",
    "_section_format_fix",
]
EXPERIMENT_FEATURES = [
    "enable_cot",
    "enable_code",
    "enable_sql",
    "enable_column_pruning",
    "enable_type_annotation",
    "enable_column_stats",
    "exact_cell_format",
    "extract_then_compute",
    "answer_type_check",
    "no_external_knowledge",
]


# ── canned query ──────────────────────────────────────────────────────

CANDIDATE_QUERY = {
    "query_id": "sanity_0",
    "content": "Who won the 2022 Monaco Grand Prix?",
    "meta": {
        "split": "dev",
        "_raw": {
            "question": "Who won the 2022 Monaco Grand Prix?",
            "answers": ["Sergio Perez"],
            "table": {
                "name": "2022 Monaco Grand Prix Results",
                "header": ["Position", "Driver", "Team", "Time", "Points"],
                "rows": [
                    ["1", "Sergio Perez",    "Red Bull",  "1:56:30", "25"],
                    ["2", "Carlos Sainz",    "Ferrari",   "+1.1s",   "18"],
                    ["3", "Max Verstappen",  "Red Bull",  "+1.5s",   "15"],
                    ["4", "Charles Leclerc", "Ferrari",   "+2.7s",   "12"],
                    ["5", "George Russell",  "Mercedes",  "+11.9s",  "10"],
                ],
            },
        },
    },
}


# ── output directory ──────────────────────────────────────────────────

RENDER_DIR = Path(__file__).parent / "renders"


# ── helpers ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def registry_and_bundles():
    """Load registry, materialize per-feature, seed store, build bundles + conflicts."""
    reg = FeatureRegistry.load(task="table_qa")

    base_specs, _ = reg.materialize(BASE_FEATURES)
    base_ids = [s["func_id"] for s in base_specs]
    base_set = set(base_ids)

    all_specs_by_fid = {s["func_id"]: s for s in base_specs}
    bundles: Dict[str, Tuple[str, List[str]]] = {}
    conflicts: Dict[str, frozenset] = {}
    for cid in EXPERIMENT_FEATURES:
        feat_specs, feat_f2f = reg.materialize(BASE_FEATURES + [cid])
        for s in feat_specs:
            all_specs_by_fid[s["func_id"]] = s
        fid_hash = reg.feature_id_for(cid)
        feature_fids = feat_f2f[fid_hash]
        add_funcs = [f for f in feature_fids if f not in base_set]
        bundles[cid] = (fid_hash, add_funcs)
        conflicts[cid] = frozenset(reg._by_canonical[cid].get("conflicts_with", []))

    full_specs = list(all_specs_by_fid.values())

    db_path = tempfile.mktemp(suffix=".db")
    store = CubeStore(db_path)
    store.upsert_funcs(full_specs, on_conflict=OnConflict.SKIP)
    yield reg, store, base_ids, bundles, conflicts
    store.close()
    Path(db_path).unlink(missing_ok=True)


def _render(store: CubeStore, func_ids: List[str]) -> Tuple[bool, str, str, str]:
    """apply_config → TableQA.bind → build_prompt.

    Returns (ok, system_prompt, user_prompt, error_msg).
    Never raises — errors are captured so the caller can write them to disk
    alongside successful renders for inspection.
    """
    try:
        state = apply_config(func_ids, store)
        task = TableQA()
        task.bind(state)
        system_prompt, user_prompt = task.build_prompt(CANDIDATE_QUERY)
        return True, system_prompt, user_prompt, ""
    except Exception as e:
        return False, "", "", f"{type(e).__name__}: {e}"


def _write_render(
    out_dir: Path,
    label: str,
    meta: dict,
    system: str,
    user: str,
    error: str = "",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".ERROR.txt" if error else ".txt"
    path = out_dir / f"{label}{suffix}"
    banner = "=" * 72
    body = (
        f"{banner}\n"
        f"label:    {label}\n"
        f"meta:     {meta}\n"
    )
    if error:
        body += (
            f"status:   FAILED\n"
            f"error:    {error}\n"
            f"{banner}\n"
        )
    else:
        body += (
            f"status:   OK\n"
            f"{banner}\n"
            f"\n"
            f"{'─' * 30} SYSTEM PROMPT {'─' * 30}\n"
            f"{system}\n"
            f"\n"
            f"{'─' * 31} USER PROMPT {'─' * 32}\n"
            f"{user}\n"
        )
    path.write_text(body)
    return path


def _render_and_write(store, out_dir: Path, label: str, meta: dict, func_ids: List[str]) -> bool:
    """Render one config, write artifact, return True if render succeeded."""
    ok, system, user, err = _render(store, func_ids)
    _write_render(out_dir, label, meta, system, user, error=err)
    return ok


def _slug(text: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")


# ── setup / teardown ──────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def _reset_render_dir():
    """Wipe tests/renders/ at the start of the module so runs don't leak."""
    if RENDER_DIR.exists():
        shutil.rmtree(RENDER_DIR)
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    yield


# ── base render (reference) ───────────────────────────────────────────

def test_render_base_config(registry_and_bundles):
    """Render the base-only config. Human check: section titles present, no feature rules."""
    _reg, store, base_ids, _bundles, _conflicts = registry_and_bundles
    ok = _render_and_write(
        store, RENDER_DIR / "base", "base",
        {"kind": "base", "features": BASE_FEATURES},
        base_ids,
    )
    assert ok, "base config must render cleanly; see tests/renders/base/"
    print(f"\nBase render: {RENDER_DIR / 'base' / 'base.txt'}")


# ── add_one_feature ───────────────────────────────────────────────────

def test_render_add_one_feature(registry_and_bundles):
    """Render every add_one_feature config; all must succeed."""
    _reg, store, base_ids, bundles, conflicts = registry_and_bundles
    configs = generate("add_one_feature", store, base_ids=base_ids, bundles=bundles)

    out_dir = RENDER_DIR / "add_one_feature"
    n_ok = 0
    n_err = 0
    errors: List[str] = []
    for _cid, func_ids, meta in configs:
        canonical = meta["canonical_id"]
        ok, _sys, _usr, err = _render(store, func_ids)
        _write_render(out_dir, canonical, meta, _sys, _usr, error=err)
        if ok:
            n_ok += 1
        else:
            n_err += 1
            errors.append(f"  {canonical}: {err}")

    print(f"\nadd_one_feature renders: {out_dir}  ({n_ok} ok, {n_err} err)")
    assert n_err == 0, "add_one_feature render failures:\n" + "\n".join(errors)


# ── leave_one_out_feature ─────────────────────────────────────────────

def test_render_leave_one_out_feature(registry_and_bundles):
    """Render every leave_one_out_feature config with conflict resolution.

    With conflicts passed in, the generator drops enable_sql when both sides
    would be live (lex tiebreak). All configs should render cleanly.
    """
    _reg, store, base_ids, bundles, conflicts = registry_and_bundles
    configs = generate(
        "leave_one_out_feature", store,
        base_ids=base_ids, bundles=bundles, conflicts=conflicts,
    )

    out_dir = RENDER_DIR / "leave_one_out_feature"
    n_ok = 0
    n_err = 0
    errors: List[str] = []
    for _cid, func_ids, meta in configs:
        removed = meta["removed_canonical_id"]
        ok, _sys, _usr, err = _render(store, func_ids)
        _write_render(out_dir, f"minus_{removed}", meta, _sys, _usr, error=err)
        if ok:
            n_ok += 1
        else:
            n_err += 1
            errors.append(f"  minus_{removed}: {err}")

    print(f"\nleave_one_out_feature renders: {out_dir}  ({n_ok} ok, {n_err} err)")
    assert n_err == 0, "leave_one_out_feature render failures:\n" + "\n".join(errors)


# ── coalition_feature ─────────────────────────────────────────────────

def test_render_coalition_feature_samples(registry_and_bundles):
    """Render a deterministic sample of conflict-aware coalition configs."""
    _reg, store, base_ids, bundles, conflicts = registry_and_bundles
    configs = generate(
        "coalition_feature", store,
        base_ids=base_ids, bundles=bundles, conflicts=conflicts,
        n_samples=12, seed=42, min_features=2, max_features=4,
    )

    out_dir = RENDER_DIR / "coalition_feature"
    n_ok = 0
    n_err = 0
    errors: List[str] = []
    for i, (_cid, func_ids, meta) in enumerate(configs):
        label = f"coalition_{i:02d}__" + "+".join(_slug(c) for c in meta["canonical_ids"])
        ok, _sys, _usr, err = _render(store, func_ids)
        _write_render(out_dir, label, meta, _sys, _usr, error=err)
        if ok:
            n_ok += 1
        else:
            n_err += 1
            errors.append(f"  {label}: {err}")

    print(f"\ncoalition_feature renders: {out_dir}  ({n_ok} ok, {n_err} err)")
    assert n_err == 0, "coalition_feature render failures:\n" + "\n".join(errors)


# ══════════════════════════════════════════════════════════════════════
# Positive tests for the two gaps closed in round 3.
# ══════════════════════════════════════════════════════════════════════


def test_enable_cot_alone_renders_successfully(registry_and_bundles):
    """Round 3 Finding A fix: enable_cot.json now declares BOTH reasoning
    and answer output_fields, so the add_one_feature config with only
    enable_cot renders cleanly.
    """
    _reg, store, base_ids, bundles, conflicts = registry_and_bundles
    configs = generate("add_one_feature", store, base_ids=base_ids, bundles=bundles)
    cot_config = next((c for c in configs if c[2]["canonical_id"] == "enable_cot"), None)
    assert cot_config is not None

    _cid, func_ids, _meta = cot_config
    ok, system, user, err = _render(store, func_ids)
    assert ok, f"enable_cot should render cleanly; got: {err}"
    # The rendered system prompt must declare both output fields.
    assert '"reasoning"' in system and '"answer"' in system, (
        f"expected both 'reasoning' and 'answer' in output_fields; "
        f"system={system!r}"
    )


def test_leave_one_out_feature_no_dispatch_ambiguity(registry_and_bundles):
    """Round 3 Finding B fix: LOO with conflicts resolves the code/sql pair,
    so no LOO config should fail dispatch validation.
    """
    _reg, store, base_ids, bundles, conflicts = registry_and_bundles
    configs = generate(
        "leave_one_out_feature", store,
        base_ids=base_ids, bundles=bundles, conflicts=conflicts,
    )
    ambiguous = []
    for _cid, func_ids, meta in configs:
        ok, _sys, _usr, err = _render(store, func_ids)
        if not ok and "ambiguous" in err.lower():
            ambiguous.append((meta["removed_canonical_id"], err))
    assert not ambiguous, f"LOO configs still ambiguous: {ambiguous}"


def test_render_coalition_with_conflicting_pair(registry_and_bundles):
    """Direct hand-built enable_code + enable_sql config — bypasses the
    generator's conflict filter. Still fails at bind() as the runtime
    safety net (TableQA._validate_dispatch_field). Render the error
    artifact for reference.
    """
    _reg, store, base_ids, bundles, conflicts = registry_and_bundles
    code_bundle = bundles["enable_code"]
    sql_bundle = bundles["enable_sql"]
    seen = set()
    func_ids: List[str] = []
    for f in list(base_ids) + code_bundle[1] + sql_bundle[1]:
        if f not in seen:
            seen.add(f)
            func_ids.append(f)

    ok = _render_and_write(
        store,
        RENDER_DIR / "coalition_feature_conflict",
        "enable_code+enable_sql",
        {"canonical_ids": ["enable_code", "enable_sql"]},
        func_ids,
    )
    # Either outcome is informative; write the artifact regardless.
    print(f"\nconflict render: "
          f"{RENDER_DIR / 'coalition_feature_conflict'}  "
          f"({'OK' if ok else 'ERROR artifact written'})")
