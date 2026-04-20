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

from core.store import CubeStore, OnConflict
from core.func_registry import make_func_id
from analyze import (
    ExecutionQuery,
    ProgressMonitor,
    add_one_deltas,
    feature_effect_ranking,
    feature_predicate_table,
    flip_rows,
    harm_cases,
    help_cases,
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


# ══════════════════════════════════════════════════════════════════════
# flip_rows / harm_cases / help_cases
# ══════════════════════════════════════════════════════════════════════

def test_flip_rows_both_directions(seeded_store):
    """Fixture scoring:
      base:    q1=0, q2=0, q3=1, q4=1
      feat_a:  q1=1, q2=1, q3=1, q4=1  (helped on q1,q2; same on q3,q4)
      feat_b:  q1=0, q2=0, q3=0, q4=0  (same on q1,q2; hurt on q3,q4)
    """
    store, ctx = seeded_store
    # Compare base vs feat_a → 2 up-flips, 0 down-flips.
    rows_a = flip_rows(store, base_config=ctx["base_cid"],
                       target_config=ctx["a_cid"],
                       model=MODEL, scorer=SCORER)
    assert len(rows_a) == 2
    assert all(r["direction"] == "up" for r in rows_a)
    assert {r["query_id"] for r in rows_a} == {"q1", "q2"}

    # Compare base vs feat_b → 0 up-flips, 2 down-flips.
    rows_b = flip_rows(store, base_config=ctx["base_cid"],
                       target_config=ctx["b_cid"],
                       model=MODEL, scorer=SCORER)
    assert len(rows_b) == 2
    assert all(r["direction"] == "down" for r in rows_b)
    assert {r["query_id"] for r in rows_b} == {"q3", "q4"}


def test_flip_rows_direction_filter(seeded_store):
    store, ctx = seeded_store
    # feat_b has 2 down-flips (q3, q4) and 0 up-flips.
    down = flip_rows(store, base_config=ctx["base_cid"],
                     target_config=ctx["b_cid"],
                     model=MODEL, scorer=SCORER, direction="down")
    up = flip_rows(store, base_config=ctx["base_cid"],
                   target_config=ctx["b_cid"],
                   model=MODEL, scorer=SCORER, direction="up")
    assert len(down) == 2
    assert len(up) == 0


def test_flip_rows_includes_predictions_and_question(seeded_store):
    store, ctx = seeded_store
    rows = flip_rows(store, base_config=ctx["base_cid"],
                     target_config=ctx["a_cid"],
                     model=MODEL, scorer=SCORER)
    r = rows[0]
    assert set(r.keys()) >= {
        "query_id", "question", "direction",
        "base_score", "target_score",
        "base_prediction", "target_prediction",
        "base_raw", "target_raw",
        "gold",
    }
    # Predictions/raw were seeded to distinct strings per config_id × query.
    assert r["base_prediction"] != r["target_prediction"]
    assert r["base_raw"] != r["target_raw"]


def test_harm_cases_alias(seeded_store):
    store, ctx = seeded_store
    harms = harm_cases(store, base_config=ctx["base_cid"],
                       target_config=ctx["b_cid"],
                       model=MODEL, scorer=SCORER)
    assert len(harms) == 2
    assert all(h["direction"] == "down" for h in harms)
    # 'harm': base was right, feature wrong.
    assert all(h["base_score"] == 1.0 and h["target_score"] == 0.0 for h in harms)


def test_help_cases_alias(seeded_store):
    store, ctx = seeded_store
    helps = help_cases(store, base_config=ctx["base_cid"],
                       target_config=ctx["a_cid"],
                       model=MODEL, scorer=SCORER)
    assert len(helps) == 2
    assert all(h["direction"] == "up" for h in helps)
    assert all(h["base_score"] == 0.0 and h["target_score"] == 1.0 for h in helps)


def test_flip_rows_bad_direction_raises(seeded_store):
    store, ctx = seeded_store
    with pytest.raises(ValueError, match="direction must be"):
        flip_rows(store, base_config=ctx["base_cid"],
                  target_config=ctx["a_cid"],
                  model=MODEL, scorer=SCORER, direction="sideways")


# ══════════════════════════════════════════════════════════════════════
# feature_predicate_table
# ══════════════════════════════════════════════════════════════════════
#
# Fixture recap (has_agg predicate):
#   q1=true  q2=true  q3=false q4=false
#
# Per-config × slice mean scores:
#                     has_agg=true    has_agg=false
#   base_cid            0.0             1.0
#   a_cid (feat_a)      1.0             1.0
#   b_cid (feat_b)      0.0             0.0


def test_fpt_simple_lift(seeded_store):
    store, ctx = seeded_store
    df = feature_predicate_table(
        store,
        model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
    )
    # Rows: 2 features × 2 predicate values = 4 (for has_agg).
    idx = {(r["canonical_id"], r["predicate_value"]): r
           for _, r in df.iterrows()}

    # feat_a at has_agg=true:  lift = 1.0 - 0.0 = 1.0
    # feat_a at has_agg=false: lift = 1.0 - 1.0 = 0.0
    assert idx[("feat_a", "true")]["lift"] == pytest.approx(1.0)
    assert idx[("feat_a", "false")]["lift"] == pytest.approx(0.0)

    # feat_b at has_agg=true:  lift = 0.0 - 0.0 =  0.0
    # feat_b at has_agg=false: lift = 0.0 - 1.0 = -1.0
    assert idx[("feat_b", "true")]["lift"] == pytest.approx(0.0)
    assert idx[("feat_b", "false")]["lift"] == pytest.approx(-1.0)


def test_fpt_simple_did_binary_auto_reference(seeded_store):
    """For a binary predicate, DiD's reference defaults to alphabetically
    first value ('false'), so did = lift[value] - lift[false]."""
    store, ctx = seeded_store
    df = feature_predicate_table(
        store,
        model=MODEL, scorer=SCORER,
        method="simple", metric="did",
        base_config_id=ctx["base_cid"],
    )
    idx = {(r["canonical_id"], r["predicate_value"]): r
           for _, r in df.iterrows()}

    # Reference is "false"; did at ref is 0.0 by convention.
    assert idx[("feat_a", "false")]["did"] == pytest.approx(0.0)
    assert idx[("feat_b", "false")]["did"] == pytest.approx(0.0)
    # did_true = lift_true - lift_false
    assert idx[("feat_a", "true")]["did"] == pytest.approx(1.0 - 0.0)
    assert idx[("feat_b", "true")]["did"] == pytest.approx(0.0 - (-1.0))


def test_fpt_marginal_lift_per_config_pooling(seeded_store):
    """Marginal pooling = average per-config means.

    configs_containing(feat_a): {a_cid}
    configs_NOT_containing(feat_a): {base_cid, b_cid}

    has_agg=true:
      with:    mean_over_configs({a_cid→1.0}) = 1.0
      without: mean_over_configs({base→0.0, b_cid→0.0}) = 0.0
      lift = 1.0

    has_agg=false:
      with:    mean_over_configs({a_cid→1.0}) = 1.0
      without: mean_over_configs({base→1.0, b_cid→0.0}) = 0.5
      lift = 0.5
    """
    store, ctx = seeded_store
    df = feature_predicate_table(
        store,
        model=MODEL, scorer=SCORER,
        method="marginal", metric="lift",
    )
    idx = {(r["canonical_id"], r["predicate_value"]): r
           for _, r in df.iterrows()}

    assert idx[("feat_a", "true")]["lift"] == pytest.approx(1.0)
    assert idx[("feat_a", "false")]["lift"] == pytest.approx(0.5)
    # feat_b: with={b_cid→0.0}, without={base→..., a→...}
    # has_agg=true:  without_mean = mean(0.0, 1.0) = 0.5  → lift = 0.0 - 0.5 = -0.5
    # has_agg=false: without_mean = mean(1.0, 1.0) = 1.0  → lift = 0.0 - 1.0 = -1.0
    assert idx[("feat_b", "true")]["lift"] == pytest.approx(-0.5)
    assert idx[("feat_b", "false")]["lift"] == pytest.approx(-1.0)


def test_fpt_marginal_did(seeded_store):
    store, ctx = seeded_store
    df = feature_predicate_table(
        store,
        model=MODEL, scorer=SCORER,
        method="marginal", metric="did",
    )
    idx = {(r["canonical_id"], r["predicate_value"]): r
           for _, r in df.iterrows()}
    # DiD for feat_a: lift_true (1.0) - lift_false (0.5) = 0.5 at value=true.
    assert idx[("feat_a", "true")]["did"] == pytest.approx(0.5)
    assert idx[("feat_b", "true")]["did"] == pytest.approx(-0.5 - (-1.0))


def test_fpt_simple_requires_base_config(seeded_store):
    store, _ = seeded_store
    with pytest.raises(ValueError, match="base_config_id"):
        feature_predicate_table(
            store, model=MODEL, scorer=SCORER,
            method="simple", metric="lift",
        )


def test_fpt_bad_method_raises(seeded_store):
    store, ctx = seeded_store
    with pytest.raises(ValueError, match="method"):
        feature_predicate_table(
            store, model=MODEL, scorer=SCORER,
            method="fancy", metric="lift",
            base_config_id=ctx["base_cid"],
        )


def test_fpt_predicate_names_filter(seeded_store):
    store, ctx = seeded_store
    # Only one predicate seeded — filter to an empty list means empty output.
    df = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
        predicate_names=[],
    )
    assert df.empty

    # Filter to the present predicate — non-empty.
    df2 = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
        predicate_names=["has_agg"],
    )
    assert not df2.empty
    assert set(df2["predicate_name"].unique()) == {"has_agg"}


def test_fpt_multi_value_predicate_did_needs_reference(seeded_store):
    """Add a multi-value predicate (3 values); without reference_values
    the ``did`` column is all NaN. With it supplied, DiD computed.
    """
    import pandas as pd
    store, ctx = seeded_store

    # Seed a 3-value predicate.
    with store._cursor() as cur:
        for qid, val in [("q1", "easy"), ("q2", "med"),
                         ("q3", "hard"), ("q4", "hard")]:
            cur.execute(
                "INSERT OR IGNORE INTO predicate (query_id, name, value) VALUES (?, ?, ?)",
                (qid, "difficulty", val),
            )

    # Without reference_values: `did` column exists but all NaN for difficulty.
    df = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="did",
        base_config_id=ctx["base_cid"],
        predicate_names=["difficulty"],
    )
    assert "did" in df.columns
    assert df["did"].isna().all()

    # With reference_values={"difficulty": "easy"}: did populated relative
    # to the "easy" slice.
    df2 = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="did",
        base_config_id=ctx["base_cid"],
        predicate_names=["difficulty"],
        reference_values={"difficulty": "easy"},
    )
    # At the reference value, did = 0.0 by convention.
    easy_rows = df2[df2["predicate_value"] == "easy"]
    assert (easy_rows["did"] == 0.0).all()
    # Non-reference rows have a non-NaN did (or NaN where lift itself is NaN).
    non_easy = df2[df2["predicate_value"] != "easy"]
    assert not non_easy.empty


def test_fpt_output_schema(seeded_store):
    store, ctx = seeded_store
    df = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
    )
    expected_cols = {
        "canonical_id", "predicate_name", "predicate_value",
        "n_with", "mean_with", "n_without", "mean_without", "lift",
    }
    assert expected_cols <= set(df.columns)
    # No `did` column when metric="lift".
    assert "did" not in df.columns
    # `_ref` is an internal helper column and must not leak.
    assert "_ref" not in df.columns


def test_fpt_drops_unmatched_features_by_default(seeded_store):
    """Features declared in the registry but with no matching config
    (n_with == 0) should not appear in the default output.

    Inject a "phantom" feature into the feature table — no config in the
    cube contains its primitive — and assert it's dropped.
    """
    import hashlib
    store, ctx = seeded_store

    phantom_params = {
        "node_type": "rule", "parent_id": "__root__",
        "payload": {"content": "phantom rule never used in any config"},
    }
    phantom_edits = [{"func_type": "insert_node", "params": phantom_params}]
    def _hash(edits):
        canon = json.dumps(
            sorted(f"insert_node:{json.dumps(e['params'], sort_keys=True)}" for e in edits),
            sort_keys=True,
        )
        return hashlib.sha256(canon.encode()).hexdigest()[:12]
    phantom_fid = _hash(phantom_edits)
    store.sync_features([{
        "feature_id": phantom_fid,
        "canonical_id": "phantom",
        "task": "table_qa",
        "requires_json": "[]", "conflicts_json": "[]",
        "primitive_spec": json.dumps(phantom_edits),
        "rationale": "never in any config",
    }])

    # Default: unmatched features dropped.
    df = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
    )
    assert "phantom" not in set(df["canonical_id"])

    # Audit mode: include them.
    df_all = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
        include_unmatched=True,
    )
    assert "phantom" in set(df_all["canonical_id"])
    phantom_rows = df_all[df_all["canonical_id"] == "phantom"]
    # The phantom has zero matches → n_with==0 and lift is NaN.
    assert (phantom_rows["n_with"] == 0).all()
    assert phantom_rows["lift"].isna().all()


def test_fpt_skips_base_features(seeded_store):
    """Features whose primitives are entirely in base (e.g. section features
    under add-one design) must NOT appear in the output. They have no
    meaningful 'simple effect' (nothing to turn off) and no meaningful
    marginal effect (they're in every config).

    Setup: register a new feature ``section_like`` whose primitive_spec
    maps to a func_id that is already present in the base config — so
    it's a "base-like" feature by definition and should be filtered.
    """
    import json as _json
    import hashlib
    from core.func_registry import make_func_id

    store, ctx = seeded_store

    # Define a new primitive that we'll put into BOTH the base config and
    # as the sole primitive of a new feature.
    section_params = {
        "node_type": "section", "parent_id": "__root__",
        "payload": {"title": "test_section", "ordinal": 0,
                    "is_system": True, "min_rules": 0, "max_rules": 10},
    }
    section_fid = make_func_id("insert_node", section_params)

    # Register the func.
    store.upsert_funcs(
        [{"func_id": section_fid, "func_type": "insert_node",
          "params": section_params, "meta": {}}],
        on_conflict=OnConflict.SKIP,
    )

    # Rebuild base config to INCLUDE this func (was empty list).
    with store._cursor() as cur:
        cur.execute(
            "UPDATE config SET func_ids = ? WHERE config_id = ?",
            (_json.dumps(sorted([section_fid])), ctx["base_cid"]),
        )

    # Register a feature whose sole primitive is section_params — so its
    # complete primitive set == {section_fid} ⊆ base.
    section_edits = [{"func_type": "insert_node", "params": section_params}]
    def _hash(edits):
        canon = json.dumps(
            sorted(f"insert_node:{json.dumps(e['params'], sort_keys=True)}" for e in edits),
            sort_keys=True,
        )
        return hashlib.sha256(canon.encode()).hexdigest()[:12]
    section_feature_id = _hash(section_edits)
    store.sync_features([{
        "feature_id": section_feature_id,
        "canonical_id": "section_like",
        "task": "table_qa",
        "requires_json": "[]", "conflicts_json": "[]",
        "primitive_spec": json.dumps(section_edits),
        "rationale": "base-like feature for test",
    }])

    df = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
    )
    # The new "base-like" feature must NOT appear — all its primitives
    # live in base, so its simple effect is undefined.
    assert "section_like" not in set(df["canonical_id"])
    # feat_a and feat_b still appear (their funcs aren't in base).
    assert {"feat_a", "feat_b"} <= set(df["canonical_id"])


# ══════════════════════════════════════════════════════════════════════
# Round 6 — confidence / ranking / report
# ══════════════════════════════════════════════════════════════════════
#
# The seeded fixture has perfect binary scores, so bootstrap output is
# deterministic:
#
#   feat_a @ has_agg=true:  every pair diff = +1   → CI=[1,1], P(>0)=1.0
#   feat_a @ has_agg=false: every pair diff =  0   → CI=[0,0], P(>0)=0.0
#   feat_b @ has_agg=true:  every pair diff =  0   → CI=[0,0], P(>0)=0.0
#   feat_b @ has_agg=false: every pair diff = -1   → CI=[-1,-1], P(>0)=0.0
#
# Tests below lean on these deterministic values rather than probabilistic
# assertions that would flake.


def test_fpt_backwards_compat_no_confidence(seeded_store):
    """Default call (confidence=False) returns the same schema as R5."""
    store, ctx = seeded_store
    df = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
    )
    for col in ("ci_lo", "ci_hi", "p_gt_zero", "effect_lb"):
        assert col not in df.columns


def test_fpt_confidence_adds_columns(seeded_store):
    store, ctx = seeded_store
    df = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
        confidence=True, n_bootstrap=500,
    )
    for col in ("ci_lo", "ci_hi", "p_gt_zero", "effect_lb"):
        assert col in df.columns
    # effect_lb == ci_lo (alias).
    assert (df["effect_lb"] == df["ci_lo"]).all()


def test_fpt_confidence_values_deterministic_paired(seeded_store):
    """Perfect-score fixture → bootstrap collapses to exact values."""
    store, ctx = seeded_store
    df = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
        confidence=True, n_bootstrap=500,
    )
    idx = {(r["canonical_id"], r["predicate_value"]): r
           for _, r in df.iterrows()}

    # feat_a @ has_agg=true: every diff=+1
    r = idx[("feat_a", "true")]
    assert r["ci_lo"] == pytest.approx(1.0)
    assert r["ci_hi"] == pytest.approx(1.0)
    assert r["p_gt_zero"] == pytest.approx(1.0)

    # feat_a @ has_agg=false: every diff=0 → p_gt_zero=0 (strictly >)
    r = idx[("feat_a", "false")]
    assert r["ci_lo"] == pytest.approx(0.0)
    assert r["ci_hi"] == pytest.approx(0.0)
    assert r["p_gt_zero"] == pytest.approx(0.0)

    # feat_b @ has_agg=false: every diff=-1
    r = idx[("feat_b", "false")]
    assert r["ci_lo"] == pytest.approx(-1.0)
    assert r["ci_hi"] == pytest.approx(-1.0)
    assert r["p_gt_zero"] == pytest.approx(0.0)


def test_fpt_sort_by_lift(seeded_store):
    store, ctx = seeded_store
    df = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
        sort_by="lift",
    )
    # Top row should be the strongest positive lift.
    assert df.iloc[0]["canonical_id"] == "feat_a"
    assert df.iloc[0]["predicate_value"] == "true"
    assert df.iloc[0]["lift"] == pytest.approx(1.0)
    # Sorted descending.
    lifts = df["lift"].tolist()
    assert lifts == sorted(lifts, reverse=True)


def test_fpt_sort_by_effect_lb_requires_confidence(seeded_store):
    store, ctx = seeded_store
    with pytest.raises(ValueError, match="requires confidence"):
        feature_predicate_table(
            store, model=MODEL, scorer=SCORER,
            method="simple", metric="lift",
            base_config_id=ctx["base_cid"],
            sort_by="effect_lb",
        )


def test_fpt_sort_by_p_gt_zero(seeded_store):
    store, ctx = seeded_store
    df = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
        confidence=True, n_bootstrap=500,
        sort_by="p_gt_zero",
    )
    # feat_a @ true has p_gt_zero=1.0; everything else is 0.0.
    assert df.iloc[0]["canonical_id"] == "feat_a"
    assert df.iloc[0]["predicate_value"] == "true"


def test_fpt_top_k(seeded_store):
    store, ctx = seeded_store
    df = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
        sort_by="lift", top_k=2,
    )
    assert len(df) == 2


def test_fpt_top_k_negative_raises(seeded_store):
    store, ctx = seeded_store
    with pytest.raises(ValueError, match="top_k must be"):
        feature_predicate_table(
            store, model=MODEL, scorer=SCORER,
            method="simple", metric="lift",
            base_config_id=ctx["base_cid"],
            top_k=-1,
        )


def test_fpt_confidence_min_filters(seeded_store):
    """confidence_min=0.5 drops everything except feat_a@true (P=1.0)."""
    store, ctx = seeded_store
    df = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
        confidence=True, n_bootstrap=500,
        confidence_min=0.5,
    )
    assert len(df) == 1
    assert df.iloc[0]["canonical_id"] == "feat_a"
    assert df.iloc[0]["predicate_value"] == "true"


def test_fpt_confidence_min_requires_confidence(seeded_store):
    store, ctx = seeded_store
    with pytest.raises(ValueError, match="requires confidence"):
        feature_predicate_table(
            store, model=MODEL, scorer=SCORER,
            method="simple", metric="lift",
            base_config_id=ctx["base_cid"],
            confidence_min=0.9,
        )


def test_fpt_confidence_min_out_of_range(seeded_store):
    store, ctx = seeded_store
    with pytest.raises(ValueError, match="must be in"):
        feature_predicate_table(
            store, model=MODEL, scorer=SCORER,
            method="simple", metric="lift",
            base_config_id=ctx["base_cid"],
            confidence=True,
            confidence_min=1.5,
        )


def test_fpt_report_markdown(seeded_store):
    store, ctx = seeded_store
    df, md = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
        confidence=True, n_bootstrap=500,
        sort_by="lift", top_k=3,
        report="markdown",
    )
    assert isinstance(md, str)
    # Expected sections.
    assert "# Feature × Predicate effect analysis" in md
    assert "## Ranking" in md
    assert "## Interpretation" in md
    assert "## Caveats" in md
    # Run metadata.
    assert MODEL in md
    assert SCORER in md
    # Top feature mentioned.
    assert "feat_a" in md
    # Bootstrap description present when confidence=True.
    assert "bootstrap" in md.lower()


def test_fpt_report_text(seeded_store):
    store, ctx = seeded_store
    df, txt = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
        sort_by="lift", top_k=2,
        report="text",
    )
    assert isinstance(txt, str)
    assert "FEATURE x PREDICATE" in txt
    assert "Caveats" in txt


def test_fpt_report_both(seeded_store):
    store, ctx = seeded_store
    df, reports = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
        report="both",
    )
    assert isinstance(reports, dict)
    assert "markdown" in reports and "text" in reports
    assert isinstance(reports["markdown"], str)
    assert isinstance(reports["text"], str)


def test_fpt_report_bad_value_raises(seeded_store):
    store, ctx = seeded_store
    with pytest.raises(ValueError, match="report must be"):
        feature_predicate_table(
            store, model=MODEL, scorer=SCORER,
            method="simple", metric="lift",
            base_config_id=ctx["base_cid"],
            report="pdf",
        )


def test_fpt_marginal_confidence_returns_nan_on_small_n(seeded_store):
    """Fixture has only 1 config 'with' feature → unpaired bootstrap needs
    ≥ 2 per side. Confidence columns should be NaN for all rows."""
    import pandas as pd
    store, ctx = seeded_store
    df = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="marginal", metric="lift",
        confidence=True, n_bootstrap=500,
    )
    # All confidence values NaN.
    assert df["p_gt_zero"].isna().all()
    assert df["ci_lo"].isna().all()
    assert df["ci_hi"].isna().all()


# ── R6.5: progress + workers smoke tests ─────────────────────────────

def test_fpt_progress_flag_runs_without_tqdm_crash(seeded_store):
    """progress=True must not crash — tqdm is a soft dep. Output must
    be identical to progress=False."""
    store, ctx = seeded_store
    df_noprog = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
        confidence=True, n_bootstrap=200,
        progress=False,
    )
    df_prog = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
        confidence=True, n_bootstrap=200,
        progress=True,
    )
    # Same shape, same numbers.
    import pandas as pd
    pd.testing.assert_frame_equal(
        df_noprog.reset_index(drop=True),
        df_prog.reset_index(drop=True),
        check_like=True,
    )


# ── R6.5: numeric-predicate guard ────────────────────────────────────

def test_predicate_kinds_classifies_categorical_and_numeric(seeded_store):
    """has_agg = {'true','false'} → categorical; row_count = ints → numeric."""
    from analyze import predicate_kinds
    store, ctx = seeded_store
    # Inject a numeric predicate.
    with store._cursor() as cur:
        for qid, v in [("q1", "5"), ("q2", "12"), ("q3", "3"), ("q4", "47")]:
            cur.execute(
                "INSERT OR IGNORE INTO predicate (query_id, name, value) VALUES (?, ?, ?)",
                (qid, "row_count", v),
            )
    kinds = predicate_kinds(store)
    assert kinds["has_agg"] == "categorical"
    assert kinds["row_count"] == "numeric"


def test_fpt_skips_numeric_predicate_with_warning(seeded_store):
    """Default skip_numeric=True → numeric predicate dropped + UserWarning emitted."""
    import warnings
    store, ctx = seeded_store
    with store._cursor() as cur:
        for qid, v in [("q1", "5"), ("q2", "12"), ("q3", "3"), ("q4", "47")]:
            cur.execute(
                "INSERT OR IGNORE INTO predicate (query_id, name, value) VALUES (?, ?, ?)",
                (qid, "row_count", v),
            )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df = feature_predicate_table(
            store, model=MODEL, scorer=SCORER,
            method="simple", metric="lift",
            base_config_id=ctx["base_cid"],
        )
    # The numeric one is dropped; only has_agg rows remain.
    assert "row_count" not in set(df["predicate_name"])
    assert "has_agg" in set(df["predicate_name"])
    # Warning was emitted.
    msgs = [str(x.message) for x in w]
    assert any("row_count" in m and "numeric" in m for m in msgs)


def test_fpt_skip_numeric_false_includes_numeric(seeded_store):
    """skip_numeric=False forces numeric predicates back into scope."""
    import warnings
    store, ctx = seeded_store
    with store._cursor() as cur:
        for qid, v in [("q1", "5"), ("q2", "12"), ("q3", "3"), ("q4", "47")]:
            cur.execute(
                "INSERT OR IGNORE INTO predicate (query_id, name, value) VALUES (?, ?, ?)",
                (qid, "row_count", v),
            )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = feature_predicate_table(
            store, model=MODEL, scorer=SCORER,
            method="simple", metric="lift",
            base_config_id=ctx["base_cid"],
            skip_numeric=False,
        )
    # row_count rows are present (one per unique value, useless but visible).
    assert "row_count" in set(df["predicate_name"])


def test_fpt_workers_gt_1_matches_serial(seeded_store):
    """workers=2 should produce the same numbers as workers=1 for the
    same seed."""
    store, ctx = seeded_store
    df_serial = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
        confidence=True, n_bootstrap=200,
        random_seed=42, workers=1,
    )
    df_parallel = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
        confidence=True, n_bootstrap=200,
        random_seed=42, workers=2,
    )
    import pandas as pd
    pd.testing.assert_frame_equal(
        df_serial.reset_index(drop=True),
        df_parallel.reset_index(drop=True),
        check_like=True,
    )


# ── R7: flipped_responses export ─────────────────────────────────────

def test_flipped_responses_basic_shape(seeded_store):
    """All flips against base, tagged by feature canonical_id, predicates joined."""
    from analyze import flipped_responses
    store, ctx = seeded_store
    df = flipped_responses(
        store,
        base_config_id=ctx["base_cid"],
        model=MODEL, scorer=SCORER,
    )
    # feat_a: q1, q2 (base 0 → target 1) = up flips on has_agg=true.
    # feat_b: q3, q4 (base 1 → target 0) = down flips on has_agg=false.
    assert len(df) == 4
    assert set(df["feature_canonical_id"]) == {"feat_a", "feat_b"}
    assert set(df["direction"]) == {"up", "down"}
    # Predicates joined.
    assert all("has_agg" in p for p in df["predicates"])
    # Error columns present.
    assert {"error_base", "error_target"} <= set(df.columns)


def test_flipped_responses_direction_filter(seeded_store):
    from analyze import flipped_responses
    store, ctx = seeded_store
    df_up = flipped_responses(
        store, base_config_id=ctx["base_cid"],
        model=MODEL, scorer=SCORER, direction="up",
    )
    assert set(df_up["direction"]) == {"up"}
    df_down = flipped_responses(
        store, base_config_id=ctx["base_cid"],
        model=MODEL, scorer=SCORER, direction="down",
    )
    assert set(df_down["direction"]) == {"down"}


def test_flipped_responses_feature_filter(seeded_store):
    from analyze import flipped_responses
    store, ctx = seeded_store
    df = flipped_responses(
        store, base_config_id=ctx["base_cid"],
        model=MODEL, scorer=SCORER,
        feature_filter=["feat_a"],
    )
    assert set(df["feature_canonical_id"]) == {"feat_a"}


def test_flipped_responses_writes_jsonl(seeded_store, tmp_path):
    from analyze import flipped_responses
    store, ctx = seeded_store
    out = tmp_path / "flips.jsonl"
    df = flipped_responses(
        store, base_config_id=ctx["base_cid"],
        model=MODEL, scorer=SCORER,
        out_path=out, fmt="jsonl",
    )
    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(df)
    rec = json.loads(lines[0])
    # Required keys round-tripped.
    for k in ("query_id", "feature_canonical_id", "direction",
              "base_score", "target_score", "predicates"):
        assert k in rec


def test_flipped_responses_writes_csv(seeded_store, tmp_path):
    from analyze import flipped_responses
    import csv
    store, ctx = seeded_store
    out = tmp_path / "flips.csv"
    flipped_responses(
        store, base_config_id=ctx["base_cid"],
        model=MODEL, scorer=SCORER,
        out_path=out, fmt="csv",
    )
    assert out.exists()
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    # gold + predicates JSON-serialized for spreadsheet safety.
    assert rows[0]["predicates"].startswith("{")
    assert rows[0]["gold"].startswith("[")


def test_flipped_responses_bad_direction_raises(seeded_store):
    from analyze import flipped_responses
    store, ctx = seeded_store
    with pytest.raises(ValueError, match="direction"):
        flipped_responses(
            store, base_config_id=ctx["base_cid"],
            model=MODEL, scorer=SCORER,
            direction="sideways",
        )


def test_flipped_responses_bad_fmt_raises(seeded_store):
    from analyze import flipped_responses
    store, ctx = seeded_store
    with pytest.raises(ValueError, match="fmt"):
        flipped_responses(
            store, base_config_id=ctx["base_cid"],
            model=MODEL, scorer=SCORER,
            fmt="parquet",
        )


def test_flipped_responses_git_add_requires_out_path(seeded_store):
    """git_add=True without out_path is a configuration error."""
    from analyze import flipped_responses
    store, ctx = seeded_store
    with pytest.raises(ValueError, match="git_add=True requires out_path"):
        flipped_responses(
            store, base_config_id=ctx["base_cid"],
            model=MODEL, scorer=SCORER,
            git_add=True,
        )


def test_flipped_responses_git_add_outside_repo_warns(seeded_store, tmp_path):
    """git_add=True in a non-git directory warns + skips, doesn't raise.

    tmp_path is a fresh isolated dir — definitely not in any git repo
    above it (pytest's tmp_path is typically /tmp/pytest-of-...).
    """
    import warnings
    from analyze import flipped_responses
    store, ctx = seeded_store
    out = tmp_path / "flips.jsonl"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df = flipped_responses(
            store, base_config_id=ctx["base_cid"],
            model=MODEL, scorer=SCORER,
            out_path=out, fmt="jsonl",
            git_add=True,
        )
    # File still got written.
    assert out.exists()
    assert len(df) > 0
    # Warning emitted (could be "not in work tree" or "git not found").
    msgs = [str(x.message) for x in w]
    assert any("git_add" in m or "git " in m for m in msgs)


def test_flipped_responses_git_add_inside_repo_stages_file(seeded_store, tmp_path):
    """git_add=True inside an init'd git repo actually runs `git add`."""
    import shutil
    import subprocess
    if shutil.which("git") is None:
        pytest.skip("git not available")

    # Spin up a throwaway repo.
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)

    out = repo / "flips.jsonl"
    from analyze import flipped_responses
    store, ctx = seeded_store
    flipped_responses(
        store, base_config_id=ctx["base_cid"],
        model=MODEL, scorer=SCORER,
        out_path=out, fmt="jsonl",
        git_add=True,
    )
    # File is staged → `git diff --cached` shows it.
    res = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        cwd=repo, capture_output=True, text=True, check=True,
    )
    assert "flips.jsonl" in res.stdout


# ══════════════════════════════════════════════════════════════════════
# analysis/R1 — feature_profile + predicate_overlap
# ══════════════════════════════════════════════════════════════════════

def test_predicate_overlap_empty_when_no_redundancy(seeded_store):
    """Fixture has only has_agg — no other predicate to overlap with."""
    from analyze import predicate_overlap
    store, _ = seeded_store
    df = predicate_overlap(store)
    assert df.empty


def test_predicate_overlap_detects_redundant_pair(seeded_store):
    """Inject a redundant predicate tagging the same queries as has_agg."""
    from analyze import predicate_overlap
    store, _ = seeded_store
    # Copy has_agg values under a new name — perfect overlap.
    with store._cursor() as cur:
        rows = cur.execute(
            "SELECT query_id, value FROM predicate WHERE name = 'has_agg'"
        ).fetchall()
        for r in rows:
            cur.execute(
                "INSERT OR IGNORE INTO predicate (query_id, name, value) VALUES (?, ?, ?)",
                (r["query_id"], "is_aggregation", r["value"]),
            )
    df = predicate_overlap(store)
    assert not df.empty
    # At least one pair has has_agg ↔ is_aggregation at jaccard=1.0.
    matches = df[
        ((df["pred_a_name"] == "has_agg") & (df["pred_b_name"] == "is_aggregation")) |
        ((df["pred_b_name"] == "has_agg") & (df["pred_a_name"] == "is_aggregation"))
    ]
    assert len(matches) >= 1
    assert (matches["jaccard"] == 1.0).any()


def test_predicate_overlap_ignores_same_name_complementary(seeded_store):
    """has_agg=yes vs has_agg=no are disjoint but both live under the same
    predicate name — they must NOT be flagged as redundant."""
    from analyze import predicate_overlap
    store, _ = seeded_store
    df = predicate_overlap(store, threshold=0.0)  # flag ALL overlaps
    # No rows where both sides share a predicate name (we skip those).
    if not df.empty:
        assert not (df["pred_a_name"] == df["pred_b_name"]).any()


def test_predicate_overlap_bad_threshold_raises(seeded_store):
    from analyze import predicate_overlap
    store, _ = seeded_store
    with pytest.raises(ValueError, match="threshold"):
        predicate_overlap(store, threshold=1.5)


def test_feature_profile_basic_schema(seeded_store):
    """Profile DF has one row per feature and expected summary cols."""
    import warnings
    from analyze import feature_profile
    store, ctx = seeded_store
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # may warn on empty cubes etc.
        df = feature_profile(
            store, model=MODEL, scorer=SCORER,
            base_config_id=ctx["base_cid"],
            n_bootstrap=100,
        )
    # One row per non-base feature (feat_a + feat_b; base-like filtered out).
    assert len(df) >= 1
    for col in ("canonical_id", "feature_id",
                "n_sig_slices", "max_abs_did",
                "breadth_mean", "classification"):
        assert col in df.columns
    # did_has_agg column is present (the fixture's one predicate).
    did_cols = [c for c in df.columns if c.startswith("did_")]
    assert len(did_cols) >= 1


def test_feature_profile_classifies_null_when_all_ties(seeded_store):
    """Both feat_a and feat_b in the fixture have DiD within threshold for
    some predicates — verify classification field is one of the allowed
    labels."""
    import warnings
    from analyze import feature_profile
    store, ctx = seeded_store
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = feature_profile(
            store, model=MODEL, scorer=SCORER,
            base_config_id=ctx["base_cid"],
            n_bootstrap=100,
        )
    assert df["classification"].isin(
        ["null", "selective", "broad", "mixed"]
    ).all()


def test_feature_profile_selective_with_strong_interaction(seeded_store):
    """Fixture feat_a has lift=+1.0 @has_agg=yes and 0.0 @has_agg=no → DiD=+1.0.
    That's one big slice above the magnitude threshold → 'selective'."""
    import warnings
    from analyze import feature_profile
    store, ctx = seeded_store
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = feature_profile(
            store, model=MODEL, scorer=SCORER,
            base_config_id=ctx["base_cid"],
            n_bootstrap=100,
            magnitude_threshold=0.05,  # easy threshold for this fixture
        )
    row_a = df[df["canonical_id"] == "feat_a"].iloc[0]
    # feat_a's has_agg DiD should be large and positive.
    assert row_a["did_has_agg"] > 0.5
    assert row_a["max_abs_did"] >= 0.5
    # Classification should land on selective (only one predicate in scope).
    assert row_a["classification"] == "selective"


def test_feature_profile_warns_on_redundant_predicates(seeded_store):
    """Inject a redundant predicate; feature_profile must emit a warning."""
    import warnings
    from analyze import feature_profile
    store, ctx = seeded_store
    with store._cursor() as cur:
        rows = cur.execute(
            "SELECT query_id, value FROM predicate WHERE name = 'has_agg'"
        ).fetchall()
        for r in rows:
            cur.execute(
                "INSERT OR IGNORE INTO predicate (query_id, name, value) VALUES (?, ?, ?)",
                (r["query_id"], "is_aggregation", r["value"]),
            )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        feature_profile(
            store, model=MODEL, scorer=SCORER,
            base_config_id=ctx["base_cid"],
            n_bootstrap=50,
        )
    msgs = [str(x.message) for x in w]
    assert any("redundant predicate" in m.lower() for m in msgs)


def test_fpt_min_effect_filters_on_lift(seeded_store):
    """min_effect=0.5 drops rows where |lift| < 0.5."""
    store, ctx = seeded_store
    df = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
        min_effect=0.5,
    )
    # Every surviving row has |lift| >= 0.5.
    assert (df["lift"].abs() >= 0.5).all()
    # At least one row survives (feat_a @ has_agg=true has lift=+1.0).
    assert len(df) >= 1


def test_fpt_min_effect_targets_did_when_metric_is_did(seeded_store):
    """metric='did' + min_effect → filters on |did|, not |lift|."""
    store, ctx = seeded_store
    df = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="did",
        base_config_id=ctx["base_cid"],
        min_effect=0.5,
    )
    import pandas as pd
    # Every surviving row has |did| >= 0.5 (NaN rows drop via fillna(-1)).
    survivors = df["did"].dropna()
    if len(survivors):
        assert (survivors.abs() >= 0.5).all()


def test_fpt_min_effect_negative_raises(seeded_store):
    store, ctx = seeded_store
    with pytest.raises(ValueError, match="min_effect must be"):
        feature_predicate_table(
            store, model=MODEL, scorer=SCORER,
            method="simple", metric="lift",
            base_config_id=ctx["base_cid"],
            min_effect=-0.1,
        )


def test_fpt_min_lift_in_pair_drops_universal_losers(seeded_store):
    """feat_b has lift=0 @has_agg=yes and lift=-1.0 @has_agg=false — no
    side helps, so min_lift_in_pair=0.0 drops the whole (feat_b, has_agg)
    pair. feat_a keeps both rows (lift=+1.0 @yes passes the threshold)."""
    store, ctx = seeded_store
    df = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
        min_lift_in_pair=0.0,
    )
    present = set(df["canonical_id"])
    # feat_a survives (helps on yes), feat_b dropped (helps nowhere).
    assert "feat_a" in present
    assert "feat_b" not in present
    # Both of feat_a's rows are kept (it's a pair-level filter).
    assert len(df[df["canonical_id"] == "feat_a"]) == 2


def test_fpt_min_lift_in_pair_with_did_mode(seeded_store):
    """Works in DiD mode too — pair-level lift check is metric-agnostic.

    Fixture: feat_a has max(lift)=1.0 (helps on has_agg=yes);
    feat_b has max(lift)=0.0 (lifts are 0.0 and -1.0). With strict
    `> 0.0`, feat_a passes and feat_b is dropped.
    """
    store, ctx = seeded_store
    df = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="did",
        base_config_id=ctx["base_cid"],
        min_lift_in_pair=0.0,
    )
    assert "feat_a" in set(df["canonical_id"])
    assert "feat_b" not in set(df["canonical_id"])


def test_fpt_min_lift_in_pair_runs_before_min_effect(seeded_store):
    """Pair-level filter must run BEFORE per-row min_effect — otherwise
    min_effect could trim rows away and distort the pair-max decision.

    Setup: feat_a pair has rows with lift=[1.0, 0.0]. With
    min_effect=0.5, the 0.0 row would drop first if ordering is wrong.
    The pair-max is still 1.0 in any ordering, but we verify that with
    strict combo the feat_a pair survives (feat_b gets dropped by both).
    """
    store, ctx = seeded_store
    df = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
        min_lift_in_pair=0.0,
        min_effect=0.5,
    )
    # feat_a survives pair-filter. Then min_effect drops the 0.0 row
    # from feat_a's pair, leaving only the 1.0 row. feat_b dropped entirely.
    assert set(df["canonical_id"]) == {"feat_a"}
    assert (df["lift"].abs() >= 0.5).all()


def test_fpt_min_lift_in_pair_missing_lift_column_raises():
    """If somehow called on a df without 'lift' column — explicit error."""
    import pandas as pd
    from analyze.rank import rank
    empty = pd.DataFrame({"canonical_id": [], "predicate_name": []})
    with pytest.raises(ValueError, match="min_lift_in_pair requires a 'lift' column"):
        rank(empty, min_lift_in_pair=0.0)


def test_fpt_min_effect_combined_with_confidence(seeded_store):
    """min_effect and confidence_min compose: both gates applied."""
    store, ctx = seeded_store
    df = feature_predicate_table(
        store, model=MODEL, scorer=SCORER,
        method="simple", metric="lift",
        base_config_id=ctx["base_cid"],
        confidence=True, n_bootstrap=100,
        min_effect=0.5,
        confidence_min=0.9,
    )
    # Both conditions hold on every surviving row.
    assert (df["lift"].abs() >= 0.5).all()
    assert (df["p_gt_zero"] >= 0.9).all()


def test_feature_profile_warn_redundant_false_is_silent(seeded_store):
    """warn_redundant_predicates=False suppresses the diagnostic."""
    import warnings
    from analyze import feature_profile
    store, ctx = seeded_store
    with store._cursor() as cur:
        rows = cur.execute(
            "SELECT query_id, value FROM predicate WHERE name = 'has_agg'"
        ).fetchall()
        for r in rows:
            cur.execute(
                "INSERT OR IGNORE INTO predicate (query_id, name, value) VALUES (?, ?, ?)",
                (r["query_id"], "is_aggregation", r["value"]),
            )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        feature_profile(
            store, model=MODEL, scorer=SCORER,
            base_config_id=ctx["base_cid"],
            n_bootstrap=50,
            warn_redundant_predicates=False,
        )
    # None of the warnings mention "redundant".
    msgs = [str(x.message) for x in w]
    assert not any("redundant predicate" in m.lower() for m in msgs)


# ══════════════════════════════════════════════════════════════════════
# R8 — vectorized attach_ci bootstrap (confidence.py)
# ══════════════════════════════════════════════════════════════════════

def test_r8_vec_matches_serial_simple(seeded_store):
    """Vectorized attach_ci output is close to a serial loop of
    paired_bootstrap. Bootstrap is stochastic and the two paths call the
    RNG in different orders (batched vs per-row), so we use a loose
    tolerance on p_gt_zero / ci_lo rather than exact equality."""
    import numpy as np
    import pandas as pd

    from analyze.confidence import attach_ci, paired_bootstrap
    from analyze.data import per_query_scores, scores_df

    store, ctx = seeded_store
    sdf = scores_df(store, model=MODEL, scorer=SCORER)

    # Build a minimal lift_df covering every (feature × predicate_value)
    # pair present in the fixture.
    rows = []
    for canon in ("feat_a", "feat_b"):
        for pv in ("true", "false"):
            rows.append({
                "canonical_id": canon,
                "predicate_name": "has_agg",
                "predicate_value": pv,
            })
    lift_df = pd.DataFrame(rows)

    canonical_to_cid = {"feat_a": ctx["a_cid"], "feat_b": ctx["b_cid"]}

    got = attach_ci(
        lift_df,
        sdf,
        method="simple",
        base_config_id=ctx["base_cid"],
        canonical_to_cid=canonical_to_cid,
        n_boot=1000,
        seed=42,
    )

    # Independent reference: call paired_bootstrap per row.
    ref_lo, ref_hi, ref_p = [], [], []
    for _, r in lift_df.iterrows():
        target = canonical_to_cid[r["canonical_id"]]
        w = per_query_scores(sdf, config_id=target,
                             predicate_name=r["predicate_name"]).get(r["predicate_value"], {})
        o = per_query_scores(sdf, config_id=ctx["base_cid"],
                             predicate_name=r["predicate_name"]).get(r["predicate_value"], {})
        lo, hi, p = paired_bootstrap(w, o, n_boot=1000, seed=42)
        ref_lo.append(lo)
        ref_hi.append(hi)
        ref_p.append(p)

    for i in range(len(lift_df)):
        # NaN rows must agree on NaN-ness.
        if np.isnan(ref_p[i]):
            assert np.isnan(got["p_gt_zero"].iloc[i])
            continue
        assert abs(got["p_gt_zero"].iloc[i] - ref_p[i]) <= 0.05, (
            f"row {i}: p_gt_zero {got['p_gt_zero'].iloc[i]} vs {ref_p[i]}"
        )
        assert abs(got["ci_lo"].iloc[i] - ref_lo[i]) <= 0.05, (
            f"row {i}: ci_lo {got['ci_lo'].iloc[i]} vs {ref_lo[i]}"
        )


def test_r8_vec_nan_on_small_n(seeded_store):
    """A row whose paired intersection has only one shared query_id must
    produce NaN×3 in all confidence columns."""
    import numpy as np
    import pandas as pd

    from analyze.confidence import attach_ci
    from analyze.data import scores_df

    store, ctx = seeded_store
    sdf = scores_df(store, model=MODEL, scorer=SCORER)

    # Construct a lift_df row that targets a predicate_value with no
    # shared queries in the fixture. Every (feature × has_agg=true) in
    # the fixture has 2+ shared queries, so we fake a narrow slice by
    # filtering sdf to a single query_id. This drops the paired share
    # count to 1 → NaN.
    sdf_narrow = sdf[sdf["query_id"] == "q1"].copy()
    lift_df = pd.DataFrame([{
        "canonical_id": "feat_a",
        "predicate_name": "has_agg",
        "predicate_value": "true",
    }])
    out = attach_ci(
        lift_df,
        sdf_narrow,
        method="simple",
        base_config_id=ctx["base_cid"],
        canonical_to_cid={"feat_a": ctx["a_cid"]},
        n_boot=200,
        seed=42,
    )
    assert np.isnan(out["ci_lo"].iloc[0])
    assert np.isnan(out["ci_hi"].iloc[0])
    assert np.isnan(out["p_gt_zero"].iloc[0])
    # effect_lb is aliased to ci_lo → NaN too.
    assert np.isnan(out["effect_lb"].iloc[0])


def test_r8_vec_chunked_matches_single_batch(seeded_store, monkeypatch):
    """Monkeypatching _CHUNK_SIZE down to 50 on a 150-row synthetic
    lift_df must produce the same output as the single-batch path."""
    import numpy as np
    import pandas as pd

    from analyze import confidence as _conf
    from analyze.confidence import attach_ci
    from analyze.data import scores_df

    store, ctx = seeded_store
    sdf = scores_df(store, model=MODEL, scorer=SCORER)

    # Synthetic lift_df: repeat the 4 real (feature × predicate_value)
    # rows enough times to exceed the forced chunk boundary several
    # times over.
    base_rows = []
    for canon in ("feat_a", "feat_b"):
        for pv in ("true", "false"):
            base_rows.append({
                "canonical_id": canon,
                "predicate_name": "has_agg",
                "predicate_value": pv,
            })
    lift_df = pd.DataFrame(base_rows * 40)  # 160 rows
    assert len(lift_df) >= 150
    canonical_to_cid = {"feat_a": ctx["a_cid"], "feat_b": ctx["b_cid"]}

    # Single-batch: force chunk ≥ n_rows.
    monkeypatch.setattr(_conf, "_CHUNK_SIZE", 10_000)
    single = attach_ci(
        lift_df,
        sdf,
        method="simple",
        base_config_id=ctx["base_cid"],
        canonical_to_cid=canonical_to_cid,
        n_boot=300,
        seed=42,
    )

    # Chunked: force chunk_size=50 → 160 rows ⇒ 4 chunks.
    monkeypatch.setattr(_conf, "_CHUNK_SIZE", 50)
    chunked = attach_ci(
        lift_df,
        sdf,
        method="simple",
        base_config_id=ctx["base_cid"],
        canonical_to_cid=canonical_to_cid,
        n_boot=300,
        seed=42,
    )

    # Same chunk layout ⇒ identical numbers row-for-row. The first
    # chunk of the "chunked" run (rows 0..49) uses seed=42+0, which is
    # the same seed as the single-batch path for rows 0..49 — so those
    # first-chunk rows match exactly. Later chunks in the chunked run
    # use different seeds than the single-batch run, so we can only
    # assert loose agreement (≤ 0.05) on them.
    n = len(lift_df)
    for i in range(50):
        if np.isnan(single["p_gt_zero"].iloc[i]):
            assert np.isnan(chunked["p_gt_zero"].iloc[i])
            continue
        assert chunked["p_gt_zero"].iloc[i] == single["p_gt_zero"].iloc[i]
        assert chunked["ci_lo"].iloc[i] == single["ci_lo"].iloc[i]
    for i in range(50, n):
        if np.isnan(single["p_gt_zero"].iloc[i]):
            assert np.isnan(chunked["p_gt_zero"].iloc[i])
            continue
        assert abs(chunked["p_gt_zero"].iloc[i] - single["p_gt_zero"].iloc[i]) <= 0.05
        assert abs(chunked["ci_lo"].iloc[i] - single["ci_lo"].iloc[i]) <= 0.10
