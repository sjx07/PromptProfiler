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

from core.feature_registry import FeatureRegistry
from core.store import CubeStore, OnConflict
from experiment.config_generators import generate


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


def test_add_one_feature_meta_feature_ids_includes_base(store_and_bundles):
    """config.meta.feature_ids must be the FULL active feature set
    (base + added), so downstream `has_feature` / `feature_effect`
    queries resolve correctly.
    """
    import json as _json
    reg, store, base_ids, bundles, conflicts = store_and_bundles
    base_feature_ids = [reg.feature_id_for(c) for c in BASE_FEATURES]
    configs = generate(
        "add_one_feature", store,
        base_ids=base_ids, bundles=bundles,
        base_canonical_ids=BASE_FEATURES,
        base_feature_ids=base_feature_ids,
    )
    conn = store._get_conn()
    for config_id, _func_ids, meta in configs:
        # 1. feature_ids in the generator-returned meta is base+added.
        assert set(meta["feature_ids"]) == set(base_feature_ids) | {meta["feature_id"]}
        assert set(meta["canonical_ids"]) == set(BASE_FEATURES) | {meta["canonical_id"]}
        # 2. The same arrays made it to config.meta in the store.
        row = conn.execute(
            "SELECT meta FROM config WHERE config_id = ?", (config_id,)
        ).fetchone()
        stored = _json.loads(row["meta"])
        assert set(stored["feature_ids"]) == set(base_feature_ids) | {meta["feature_id"]}


def test_leave_one_out_feature_meta_feature_ids_includes_base(store_and_bundles):
    """LOO config.meta.feature_ids must be base + all-non-removed features
    (minus any conflict-resolved drops).
    """
    reg, store, base_ids, bundles, conflicts = store_and_bundles
    base_feature_ids = [reg.feature_id_for(c) for c in BASE_FEATURES]
    configs = generate(
        "leave_one_out_feature", store,
        base_ids=base_ids, bundles=bundles, conflicts=conflicts,
        base_canonical_ids=BASE_FEATURES,
        base_feature_ids=base_feature_ids,
    )
    for _cid, _func_ids, meta in configs:
        active_hashes = {bundles[c][0] for c in meta["active_canonical_ids"]}
        assert set(meta["feature_ids"]) == set(base_feature_ids) | active_hashes


# ── leave_one_out_feature ─────────────────────────────────────────────

def test_leave_one_out_feature_count(store_and_bundles):
    reg, store, base_ids, bundles, conflicts = store_and_bundles
    configs = generate("leave_one_out_feature", store, base_ids=base_ids, bundles=bundles)
    assert len(configs) == len(bundles)


def test_leave_one_out_feature_removed_feature_funcs_absent(store_and_bundles):
    """Func_ids uniquely contributed by the removed feature must not appear.

    Shared primitives (content-addressed dedup across features — e.g.
    enable_code and enable_sql both emit set_format(code_block) → same
    func_id) may still be present, because another active feature is
    supplying them. Only the REMOVED-only primitives must vanish.
    """
    reg, store, base_ids, bundles, conflicts = store_and_bundles
    configs = generate("leave_one_out_feature", store, base_ids=base_ids, bundles=bundles)
    for _cid, func_ids, meta in configs:
        removed = meta["removed_canonical_id"]
        removed_funcs = set(bundles[removed][1])
        # Union of everyone else's funcs → funcs that could be contributed
        # by another active feature.
        shared_with_others: set = set()
        for other, (_h, other_funcs) in bundles.items():
            if other != removed:
                shared_with_others.update(other_funcs)
        unique_to_removed = removed_funcs - shared_with_others
        leakage = unique_to_removed & set(func_ids)
        assert not leakage, (
            f"leave-one-out of {removed}: unique removed-feature funcs leaked: {leakage}"
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

    Expected count is computed from the conflicts map via brute-force
    enumeration rather than a closed-form formula, so the test stays
    correct as more conflict pairs get declared.
    """
    from itertools import combinations
    reg, store, base_ids, bundles, conflicts = store_and_bundles
    n = len(bundles)
    names = list(bundles.keys())

    def _has_live_conflict(subset):
        s = set(subset)
        for cid in s:
            if s & conflicts.get(cid, frozenset()):
                return True
        return False

    all_subsets = [
        tuple(sorted(c))
        for k in range(1, n + 1)
        for c in combinations(names, k)
    ]
    expected = sum(1 for s in all_subsets if not _has_live_conflict(s))

    configs = generate(
        "coalition_feature", store,
        base_ids=base_ids, bundles=bundles, conflicts=conflicts,
        n_samples=len(all_subsets) + 100, seed=3,
        min_features=1, max_features=n,
    )
    assert len(configs) == expected, (
        f"expected {expected} valid coalitions given declared conflicts, "
        f"got {len(configs)}"
    )
    # subset_canonical_ids holds the varying part; canonical_ids holds full set.
    subsets = {
        tuple(sorted(meta["subset_canonical_ids"]))
        for _, _, meta in configs
    }
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
