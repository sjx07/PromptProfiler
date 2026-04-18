"""test_analyze.py — analyze.* primitives on a seeded fixture cube.

Exercises every layer:
  * meta.list_* + summary
  * ExecutionQuery filters: scope, structural, predicate, score, error
  * ExecutionQuery terminals: count, rows, df, agg, projection, order/limit
  * compare.score_diff, add_one_deltas, feature_effect_ranking, predicate_slice
  * ProgressMonitor.overall / by_config / errors / recent
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

from prompt_profiler.core.store import CubeStore, OnConflict
from prompt_profiler.core.func_registry import make_func_id
from prompt_profiler.analyze import (
    ExecutionQuery,
    ProgressMonitor,
    add_one_deltas,
    feature_effect_ranking,
    list_configs,
    list_configs_with_features,
    list_datasets,
    list_features_in_cube,
    list_models,
    list_phases,
    list_predicates,
    list_scorers,
    predicate_slice,
    score_diff,
    summary,
)


# ── fixture ────────────────────────────────────────────────────────────

MODEL = "test-model"
SCORER = "exact_match"


@pytest.fixture
def seeded_store():
    """Build a small but representative cube:

      * 2 features (fA, fB) with canonical_ids "feat_a", "feat_b".
      * 3 configs: base (no features), +feat_a, +feat_b.
      * 4 queries, 2 tagged with predicate has_agg=true and has_agg=false.
      * Executions for all 3 configs × 4 queries, under 2 models, tagged phase="p1".
      * Evaluations with scorer "exact_match" scoring 0 or 1.
      * 1 deliberate error row in the second model, config_b.
    """
    db_path = tempfile.mktemp(suffix=".db")
    store = CubeStore(db_path)

    # ── funcs (one per feature; make_func_id keyed by (func_type, params)) ──
    fa_params = {
        "node_type": "rule", "parent_id": "__root__",
        "payload": {"content": "rule A"},
    }
    fb_params = {
        "node_type": "rule", "parent_id": "__root__",
        "payload": {"content": "rule B"},
    }
    fa_fid = make_func_id("insert_node", fa_params)
    fb_fid = make_func_id("insert_node", fb_params)
    store.upsert_funcs(
        [
            {"func_id": fa_fid, "func_type": "insert_node", "params": fa_params, "meta": {}},
            {"func_id": fb_fid, "func_type": "insert_node", "params": fb_params, "meta": {}},
        ],
        on_conflict=OnConflict.SKIP,
    )

    # ── features (content-hashed feature_ids) ─────────────────────────
    import hashlib
    def _hash(edits):
        canon = json.dumps(
            sorted(f"insert_node:{json.dumps(e['params'], sort_keys=True)}" for e in edits),
            sort_keys=True,
        )
        return hashlib.sha256(canon.encode()).hexdigest()[:12]

    fa_edits = [{"func_type": "insert_node", "params": fa_params}]
    fb_edits = [{"func_type": "insert_node", "params": fb_params}]
    fa_hash = _hash(fa_edits)
    fb_hash = _hash(fb_edits)
    store.sync_features([
        {"feature_id": fa_hash, "canonical_id": "feat_a", "task": "table_qa",
         "requires_json": "[]", "conflicts_json": "[]",
         "primitive_spec": json.dumps(fa_edits), "rationale": "a"},
        {"feature_id": fb_hash, "canonical_id": "feat_b", "task": "table_qa",
         "requires_json": "[]", "conflicts_json": "[]",
         "primitive_spec": json.dumps(fb_edits), "rationale": "b"},
    ])

    # ── configs ──────────────────────────────────────────────────────
    base_cid = store.get_or_create_config([], meta={"kind": "base",
                                                     "canonical_ids": [], "feature_ids": []})
    a_cid = store.get_or_create_config(
        [fa_fid],
        meta={"kind": "add_one_feature", "canonical_id": "feat_a",
              "feature_id": fa_hash, "feature_ids": [fa_hash]},
    )
    b_cid = store.get_or_create_config(
        [fb_fid],
        meta={"kind": "add_one_feature", "canonical_id": "feat_b",
              "feature_id": fb_hash, "feature_ids": [fb_hash]},
    )

    # ── queries ──────────────────────────────────────────────────────
    queries = [
        {"query_id": "q1", "dataset": "wtq",
         "content": "Q1?", "meta": {"split": "dev"}},
        {"query_id": "q2", "dataset": "wtq",
         "content": "Q2?", "meta": {"split": "dev"}},
        {"query_id": "q3", "dataset": "wtq",
         "content": "Q3?", "meta": {"split": "dev"}},
        {"query_id": "q4", "dataset": "wtq",
         "content": "Q4?", "meta": {"split": "dev"}},
    ]
    store.upsert_queries(queries, on_conflict=OnConflict.SKIP)

    # ── predicates ───────────────────────────────────────────────────
    conn = store._get_conn()
    with store._cursor() as cur:
        for qid, val in [("q1", "true"), ("q2", "true"),
                         ("q3", "false"), ("q4", "false")]:
            cur.execute(
                "INSERT OR IGNORE INTO predicate (query_id, name, value) VALUES (?, ?, ?)",
                (qid, "has_agg", val),
            )

    # ── executions + evaluations ─────────────────────────────────────
    # Scoring pattern:
    #   base: 0,0,1,1  (avg=0.5)
    #   +feat_a: 1,1,1,1 (avg=1.0 — helps)
    #   +feat_b: 0,0,0,0 (avg=0.0 — hurts)
    scores = {
        base_cid: {"q1": 0.0, "q2": 0.0, "q3": 1.0, "q4": 1.0},
        a_cid:    {"q1": 1.0, "q2": 1.0, "q3": 1.0, "q4": 1.0},
        b_cid:    {"q1": 0.0, "q2": 0.0, "q3": 0.0, "q4": 0.0},
    }
    for cfg_id, per_q in scores.items():
        for qid, s in per_q.items():
            exec_id = store.insert_execution(
                cfg_id, qid, MODEL,
                system_prompt=f"sys_cfg_{cfg_id}",
                user_content=f"user_{qid}",
                raw_response=f"resp_{cfg_id}_{qid}",
                prediction=f"pred_{cfg_id}_{qid}",
                latency_ms=100.0, prompt_tokens=10, completion_tokens=5,
                phase="p1",
                on_conflict=OnConflict.SKIP,
            )
            store.upsert_evaluation(exec_id, SCORER, s, on_conflict=OnConflict.SKIP)

    # Add a second model with one error for error-inspection tests.
    MODEL2 = "second-model"
    exec_err = store.insert_execution(
        b_cid, "q1", MODEL2,
        error="ValueError: synthetic test error",
        phase="p2",
        on_conflict=OnConflict.SKIP,
    )
    # Also one ok execution under MODEL2 so n_executions > 1.
    ok2 = store.insert_execution(
        base_cid, "q1", MODEL2,
        prediction="pred_b2",
        phase="p2",
        on_conflict=OnConflict.SKIP,
    )
    store.upsert_evaluation(ok2, SCORER, 0.5, on_conflict=OnConflict.SKIP)

    yield store, {
        "base_cid": base_cid, "a_cid": a_cid, "b_cid": b_cid,
        "fa_fid": fa_fid, "fb_fid": fb_fid,
        "fa_hash": fa_hash, "fb_hash": fb_hash,
        "MODEL": MODEL, "MODEL2": MODEL2, "SCORER": SCORER,
    }
    store.close()
    Path(db_path).unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════
# meta.py
# ══════════════════════════════════════════════════════════════════════

def test_list_configs(seeded_store):
    store, ctx = seeded_store
    cfgs = list_configs(store)
    assert len(cfgs) == 3
    by_id = {c["config_id"]: c for c in cfgs}
    assert by_id[ctx["a_cid"]]["kind"] == "add_one_feature"
    assert by_id[ctx["a_cid"]]["n_funcs"] == 1
    assert ctx["fa_hash"] in by_id[ctx["a_cid"]]["feature_ids"]


def test_list_configs_with_features_resolves_canonical(seeded_store):
    store, ctx = seeded_store
    cfgs = list_configs_with_features(store)
    by_id = {c["config_id"]: c for c in cfgs}
    assert by_id[ctx["a_cid"]]["resolved_canonical_ids"] == ["feat_a"]
    assert by_id[ctx["b_cid"]]["resolved_canonical_ids"] == ["feat_b"]
    assert by_id[ctx["base_cid"]]["resolved_canonical_ids"] == []


def test_list_models(seeded_store):
    store, _ = seeded_store
    rows = list_models(store)
    names = {r["model"] for r in rows}
    assert {MODEL, "second-model"} <= names


def test_list_scorers(seeded_store):
    store, _ = seeded_store
    rows = list_scorers(store)
    scorers = {r["scorer"] for r in rows}
    assert SCORER in scorers


def test_list_phases(seeded_store):
    store, _ = seeded_store
    rows = list_phases(store)
    names = {r["phase"] for r in rows}
    # phases got timestamped; startswith suffices
    assert any(p.startswith("p1") for p in names)
    assert any(p.startswith("p2") for p in names)


def test_list_datasets(seeded_store):
    store, _ = seeded_store
    rows = list_datasets(store)
    wtq = next(r for r in rows if r["dataset"] == "wtq")
    assert wtq["n_queries"] == 4
    assert "dev" in wtq["splits"]


def test_list_predicates(seeded_store):
    store, _ = seeded_store
    rows = list_predicates(store)
    names = {r["name"] for r in rows}
    assert "has_agg" in names


def test_list_features_in_cube(seeded_store):
    store, ctx = seeded_store
    rows = list_features_in_cube(store, task="table_qa")
    cids = {r["canonical_id"] for r in rows}
    assert cids == {"feat_a", "feat_b"}


def test_summary_shape(seeded_store):
    store, _ = seeded_store
    s = summary(store)
    assert {"counts", "models", "scorers", "phases", "datasets", "tasks"} <= set(s.keys())
    assert s["counts"]["execution"] >= 12


# ══════════════════════════════════════════════════════════════════════
# query.py — ExecutionQuery
# ══════════════════════════════════════════════════════════════════════

def test_query_count_unfiltered(seeded_store):
    store, _ = seeded_store
    # 3 configs × 4 queries × MODEL + 2 under MODEL2 = 14
    assert ExecutionQuery(store).count() == 14


def test_query_model_filter(seeded_store):
    store, _ = seeded_store
    assert ExecutionQuery(store).model(MODEL).count() == 12


def test_query_config_filter(seeded_store):
    store, ctx = seeded_store
    assert ExecutionQuery(store).config(ctx["a_cid"]).count() == 4


def test_query_has_func_filter(seeded_store):
    store, ctx = seeded_store
    # Only the a_cid config contains fa_fid.
    assert ExecutionQuery(store).has_func(ctx["fa_fid"]).count() == 4


def test_query_has_feature_canonical_resolution(seeded_store):
    store, ctx = seeded_store
    # feat_a is only in a_cid, run under MODEL only → 4 executions.
    assert ExecutionQuery(store).has_feature("feat_a").count() == 4
    # feat_b is in b_cid, run under MODEL (4) + MODEL2 error row (1) → 5.
    assert ExecutionQuery(store).has_feature("feat_b").count() == 5
    # Narrowing by model should recover the canonical 4.
    assert ExecutionQuery(store).has_feature("feat_b").model(MODEL).count() == 4


def test_query_predicate_filter(seeded_store):
    store, _ = seeded_store
    # has_agg=true covers q1, q2 → 3 configs × 2 = 6 under MODEL (+1 error
    # under MODEL2, q1 has_agg=true). So 6 executions under MODEL.
    assert ExecutionQuery(store).model(MODEL).predicate("has_agg", "true").count() == 6


def test_query_scorer_filter_and_score_filter(seeded_store):
    store, ctx = seeded_store
    # score == 1 under feat_a should be 4 rows.
    q = (ExecutionQuery(store)
         .model(MODEL)
         .scorer(SCORER)
         .config(ctx["a_cid"])
         .where_score("=", 1.0))
    assert q.count() == 4


def test_query_error_filter(seeded_store):
    store, _ = seeded_store
    assert ExecutionQuery(store).with_error().count() == 1
    assert ExecutionQuery(store).without_error().count() == 13


def test_query_rows_parses_json(seeded_store):
    store, _ = seeded_store
    rows = ExecutionQuery(store).model(MODEL).limit(1).rows()
    assert rows
    # phase_ids and meta should be parsed from JSON strings.
    assert isinstance(rows[0]["phase_ids"], list)
    assert isinstance(rows[0]["meta"], dict)


def test_query_columns_projection(seeded_store):
    store, ctx = seeded_store
    rows = (ExecutionQuery(store)
            .config(ctx["a_cid"])
            .model(MODEL)
            .columns(["config_id", "query_id", "prediction"])
            .rows())
    assert set(rows[0].keys()) == {"config_id", "query_id", "prediction"}


def test_query_agg_by_config_avg_score(seeded_store):
    store, ctx = seeded_store
    result = (ExecutionQuery(store)
              .model(MODEL).scorer(SCORER)
              .agg(by=["config_id"], fn="avg", metric="score"))
    by_cid = {r["config_id"]: r["avg_score"] for r in result}
    assert by_cid[ctx["a_cid"]] == 1.0
    assert by_cid[ctx["b_cid"]] == 0.0
    assert by_cid[ctx["base_cid"]] == 0.5


def test_query_order_by_and_limit(seeded_store):
    store, _ = seeded_store
    rows = (ExecutionQuery(store)
            .model(MODEL)
            .order_by("e.execution_id DESC")
            .limit(3)
            .rows())
    assert len(rows) == 3


# ══════════════════════════════════════════════════════════════════════
# compare.py
# ══════════════════════════════════════════════════════════════════════

def test_score_diff_positive(seeded_store):
    store, ctx = seeded_store
    d = score_diff(store, config_a=ctx["base_cid"], config_b=ctx["a_cid"],
                   model=MODEL, scorer=SCORER)
    assert d["n_shared"] == 4
    assert d["avg_delta"] == pytest.approx(0.5)
    # q1, q2 flip up (0 → 1); q3, q4 stay 1.
    assert set(d["flipped_up"]) == {"q1", "q2"}
    assert d["flipped_down"] == []


def test_score_diff_negative(seeded_store):
    store, ctx = seeded_store
    d = score_diff(store, config_a=ctx["base_cid"], config_b=ctx["b_cid"],
                   model=MODEL, scorer=SCORER)
    # base=0.5, b=0.0 → delta = -0.5; q3, q4 flip down.
    assert d["avg_delta"] == pytest.approx(-0.5)
    assert set(d["flipped_down"]) == {"q3", "q4"}


def test_add_one_deltas_sorted(seeded_store):
    store, ctx = seeded_store
    df = add_one_deltas(store, base_config_id=ctx["base_cid"],
                        model=MODEL, scorer=SCORER)
    assert list(df["canonical_id"]) == ["feat_a", "feat_b"]
    assert df.iloc[0]["avg_delta"] > df.iloc[1]["avg_delta"]


def test_feature_effect_ranking(seeded_store):
    store, _ = seeded_store
    df = feature_effect_ranking(store, model=MODEL, scorer=SCORER, task="table_qa")
    by_cid = {r["canonical_id"]: r["mean_score"] for _, r in df.iterrows()}
    assert by_cid["feat_a"] == 1.0
    assert by_cid["feat_b"] == 0.0


def test_predicate_slice(seeded_store):
    store, ctx = seeded_store
    df = predicate_slice(
        store, model=MODEL, scorer=SCORER, predicate_name="has_agg",
    )
    # 3 configs × 2 predicate_values = 6 rows
    assert len(df) == 6
    # feat_a scores 1.0 on both slices, feat_b 0.0.
    a_rows = df[df["config_id"] == ctx["a_cid"]]
    assert set(a_rows["mean_score"]) == {1.0}


# ══════════════════════════════════════════════════════════════════════
# monitor.py
# ══════════════════════════════════════════════════════════════════════

def test_monitor_overall(seeded_store):
    store, _ = seeded_store
    mon = ProgressMonitor(store, model=MODEL, scorer=SCORER)
    o = mon.overall()
    assert o["n_executions"] == 12
    assert o["n_configs"] == 3
    assert o["n_queries"] == 4
    assert o["mean_score"] == pytest.approx(0.5)


def test_monitor_by_config(seeded_store):
    store, ctx = seeded_store
    mon = ProgressMonitor(store, model=MODEL, scorer=SCORER)
    rows = mon.by_config(total_queries_expected=4)
    by_cid = {r["config_id"]: r for r in rows}
    assert by_cid[ctx["a_cid"]]["n_done"] == 4
    assert by_cid[ctx["a_cid"]]["pct"] == 100.0
    assert by_cid[ctx["a_cid"]]["mean_score"] == 1.0
    assert by_cid[ctx["a_cid"]]["canonical_id"] == "feat_a"


def test_monitor_errors(seeded_store):
    store, _ = seeded_store
    mon = ProgressMonitor(store, model="second-model")
    errs = mon.errors(limit=5)
    assert len(errs) == 1
    assert "synthetic" in errs[0]["error"]


def test_monitor_recent(seeded_store):
    store, _ = seeded_store
    mon = ProgressMonitor(store, model=MODEL)
    r = mon.recent(5)
    assert len(r) == 5
    # most-recent first
    assert r[0]["execution_id"] >= r[-1]["execution_id"]
