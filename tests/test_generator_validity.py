"""test_generator_validity.py — structural correctness of feature-level generators.

Uses the real ``features/table_qa/*.json`` registry. Validates:

  * add_one_feature:        count + per-config composition
  * leave_one_out_feature:  count + removed-feature isolation
  * coalition_feature:      n_samples respected, no invalid samples

Plus two gap-documenting tests that surface current limitations:

  * coalition_feature does NOT consult ``conflicts_with``
  * coalition_feature does NOT consult ``requires`` (but base currently
    provides all sections, so it's currently safe — we check that
    invariant, not the absence of a filter).
"""
from __future__ import annotations

import sys
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

_TOOL_DIR = str(Path(__file__).parent.parent.parent.parent)
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from prompt_profiler.core.feature_registry import FeatureRegistry
from prompt_profiler.core.store import CubeStore, OnConflict
from prompt_profiler.experiment.config_generators import generate


# ── fixture ────────────────────────────────────────────────────────────

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


@pytest.fixture
def store_and_bundles():
    """Load real feature registry, seed funcs into a temp store, build bundles
    and the conflicts map (mirrors run_experiment._build_feature_bundles).
    """
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


# ── add_one_feature ───────────────────────────────────────────────────

def test_add_one_feature_count_matches_experiment_features(store_and_bundles):
    reg, store, base_ids, bundles, conflicts = store_and_bundles
    configs = generate("add_one_feature", store, base_ids=base_ids, bundles=bundles)
    # Every feature has ≥1 incremental func over base (checked in shape test).
    assert len(configs) == len(bundles), (
        f"expected {len(bundles)} configs, got {len(configs)}"
    )


def test_add_one_feature_composition(store_and_bundles):
    """Each config = base + exactly one feature's add_funcs (no extras, no drops)."""
    reg, store, base_ids, bundles, conflicts = store_and_bundles
    configs = generate("add_one_feature", store, base_ids=base_ids, bundles=bundles)
    base_set = set(base_ids)

    seen_feature_cids = set()
    for cid, func_ids, meta in configs:
        canonical_id = meta["canonical_id"]
        seen_feature_cids.add(canonical_id)
        expected_add = set(bundles[canonical_id][1])
        got_add = set(func_ids) - base_set
        assert got_add == expected_add, (
            f"feature {canonical_id}: expected add_funcs {expected_add}, got {got_add}"
        )
        # Base must be wholly present.
        assert base_set.issubset(set(func_ids)), (
            f"feature {canonical_id}: base ids missing: "
            f"{base_set - set(func_ids)}"
        )
    assert seen_feature_cids == set(bundles.keys())


def test_add_one_feature_meta_carries_feature_hash(store_and_bundles):
    reg, store, base_ids, bundles, conflicts = store_and_bundles
    configs = generate("add_one_feature", store, base_ids=base_ids, bundles=bundles)
    for _cid, _func_ids, meta in configs:
        assert meta["feature_id"] == bundles[meta["canonical_id"]][0]


# ── leave_one_out_feature ─────────────────────────────────────────────

def test_leave_one_out_feature_count(store_and_bundles):
    reg, store, base_ids, bundles, conflicts = store_and_bundles
    configs = generate("leave_one_out_feature", store, base_ids=base_ids, bundles=bundles)
    assert len(configs) == len(bundles)


def test_leave_one_out_feature_removed_feature_funcs_absent(store_and_bundles):
    reg, store, base_ids, bundles, conflicts = store_and_bundles
    configs = generate("leave_one_out_feature", store, base_ids=base_ids, bundles=bundles)
    for _cid, func_ids, meta in configs:
        removed = meta["removed_canonical_id"]
        removed_funcs = set(bundles[removed][1])
        leakage = removed_funcs & set(func_ids)
        assert not leakage, (
            f"leave-one-out of {removed}: removed-feature funcs leaked: {leakage}"
        )


def test_leave_one_out_feature_all_other_funcs_present(store_and_bundles):
    reg, store, base_ids, bundles, conflicts = store_and_bundles
    configs = generate("leave_one_out_feature", store, base_ids=base_ids, bundles=bundles)
    base_set = set(base_ids)
    for _cid, func_ids, meta in configs:
        removed = meta["removed_canonical_id"]
        removed_funcs = set(bundles[removed][1])
        for other_cid, (_h, other_funcs) in bundles.items():
            if other_cid == removed:
                continue
            # other_funcs may overlap with removed_funcs when two features
            # contribute the same primitive (dedup via content hash).
            # Only primitives NOT shared with the removed feature are
            # guaranteed to be present.
            unique_other = set(other_funcs) - removed_funcs
            missing = unique_other - set(func_ids)
            assert not missing, (
                f"leave-one-out of {removed}: other feature {other_cid} "
                f"missing funcs: {missing}"
            )
        assert base_set.issubset(set(func_ids)), (
            f"leave-one-out of {removed}: base ids missing"
        )


# ── coalition_feature ─────────────────────────────────────────────────

def test_coalition_feature_respects_n_samples_small(store_and_bundles):
    reg, store, base_ids, bundles, conflicts = store_and_bundles
    configs = generate(
        "coalition_feature", store,
        base_ids=base_ids, bundles=bundles, conflicts=conflicts,
        n_samples=20, seed=7, min_features=1, max_features=4,
    )
    assert len(configs) == 20


def test_coalition_feature_enumerates_fully_when_n_samples_huge(store_and_bundles):
    """With n_samples >= total_possible AND conflict filtering, every valid
    (non-conflicting) non-empty subset should appear exactly once.

    10 features, one conflict pair (enable_code ↔ enable_sql):
      * total nonempty subsets = 2^10 - 1 = 1023
      * subsets containing BOTH code and sql = 2^8 = 256
      * valid subsets = 1023 - 256 = 767
    """
    reg, store, base_ids, bundles, conflicts = store_and_bundles
    n = len(bundles)
    total_nonempty = (1 << n) - 1
    invalid = 1 << (n - 2)  # subsets that fix both sides of the one conflict pair
    expected = total_nonempty - invalid

    configs = generate(
        "coalition_feature", store,
        base_ids=base_ids, bundles=bundles, conflicts=conflicts,
        n_samples=total_nonempty + 100, seed=3,
        min_features=1, max_features=n,
    )
    assert len(configs) == expected, (
        f"expected {expected} valid coalitions (1023 total - 256 with conflict), "
        f"got {len(configs)}"
    )
    subsets = {tuple(sorted(meta["canonical_ids"])) for _, _, meta in configs}
    assert len(subsets) == expected


def test_coalition_feature_configs_are_supersets_of_base(store_and_bundles):
    reg, store, base_ids, bundles, conflicts = store_and_bundles
    configs = generate(
        "coalition_feature", store,
        base_ids=base_ids, bundles=bundles, conflicts=conflicts,
        n_samples=50, seed=11,
    )
    base_set = set(base_ids)
    for _cid, func_ids, _meta in configs:
        assert base_set.issubset(set(func_ids))


# ── conflict-awareness ────────────────────────────────────────────────

def test_coalition_feature_filters_conflicts(store_and_bundles):
    """coalition_feature (with conflicts kwarg) must NOT emit any config
    containing both enable_code and enable_sql."""
    reg, store, base_ids, bundles, conflicts = store_and_bundles
    configs = generate(
        "coalition_feature", store,
        base_ids=base_ids, bundles=bundles, conflicts=conflicts,
        n_samples=1000, seed=17,
    )
    offenders = [
        meta for _cid, _fids, meta in configs
        if "enable_code" in meta["canonical_ids"] and "enable_sql" in meta["canonical_ids"]
    ]
    assert not offenders, (
        f"coalition_feature leaked {len(offenders)} configs with both "
        f"enable_code and enable_sql: {offenders[:3]}"
    )


def test_leave_one_out_feature_resolves_conflicts(store_and_bundles):
    """When removing a non-conflict feature, LOO must drop the
    lexicographically-larger side of any remaining conflict pair.
    For enable_code ↔ enable_sql, that means enable_sql is dropped.
    """
    reg, store, base_ids, bundles, conflicts = store_and_bundles
    configs = generate(
        "leave_one_out_feature", store,
        base_ids=base_ids, bundles=bundles, conflicts=conflicts,
    )
    for _cid, _func_ids, meta in configs:
        removed = meta["removed_canonical_id"]
        active = set(meta["active_canonical_ids"])
        if removed in ("enable_code", "enable_sql"):
            # One side of the conflict pair already gone; no resolution needed.
            assert not ({"enable_code", "enable_sql"} <= active)
            continue
        # Remaining set must not have both sides live.
        assert not ({"enable_code", "enable_sql"} <= active), (
            f"LOO minus {removed}: active set {active} still has both "
            f"enable_code and enable_sql"
        )
        # Specifically: enable_sql should be dropped (lex > enable_code).
        assert "enable_sql" not in active, (
            f"LOO minus {removed}: expected enable_sql to be dropped by "
            f"conflict resolution, got active={active}"
        )
        # conflict_resolutions meta should record it.
        resolutions = meta.get("conflict_resolutions", [])
        assert any(
            r["dropped"] == "enable_sql" and r["kept"] == "enable_code"
            for r in resolutions
        ), f"expected conflict_resolutions entry, got: {resolutions}"


def test_coalition_requires_currently_satisfied_by_base(store_and_bundles):
    """All experiment-feature ``requires`` entries resolve to a canonical_id
    in ``BASE_FEATURES``. This means the generator can skip a ``requires``
    filter without breaking correctness TODAY, but will need one the moment
    a feature requires another non-section feature.

    This test pins that invariant so a future feature with non-section
    ``requires`` breaks loudly here, not silently in production.
    """
    reg, _store, _base_ids, _bundles, _conflicts = store_and_bundles
    base_set = set(BASE_FEATURES)
    for cid in EXPERIMENT_FEATURES:
        spec = reg._by_canonical[cid]
        for req in spec.get("requires", []):
            assert req in base_set, (
                f"feature {cid} requires {req!r}, which is not in BASE_FEATURES. "
                f"coalition_feature will need a requires-aware filter."
            )
