from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from core.func_registry import make_func_id
from core.store import CubeStore, OnConflict
from analyze.facet_matrix import (
    compare_binary_rules_across_benchmarks,
    config_feature_sets,
    context_itemsets_from_context,
    discover_benchmark_subgroups,
    feature_itemsets_from_delta,
    infer_baseline_config_id,
    paired_delta_rows,
    query_context_sets,
    score_binary_contextual_rules,
    score_contextual_feature_itemsets,
    score_feature_itemsets,
)
from analyze.facet_matrix_ops import FacetOperatorChain


MODEL = "test-model"
SCORER = "exact"


def _feature_row(canonical_id: str):
    params = {
        "node_type": "rule",
        "parent_id": "__root__",
        "payload": {"content": canonical_id},
    }
    func_id = make_func_id("insert_node", params)
    primitive = [{"func_type": "insert_node", "params": params}]
    blob = json.dumps(primitive, sort_keys=True)
    feature_id = hashlib.sha256(blob.encode()).hexdigest()[:12]
    return {
        "func_id": func_id,
        "feature_id": feature_id,
        "feature": {
            "feature_id": feature_id,
            "canonical_id": canonical_id,
            "task": "wtq",
            "requires_json": "[]",
            "conflicts_json": "[]",
            "primitive_spec": json.dumps(primitive),
            "semantic_labels_json": "[]",
            "scope_json": "{}",
            "rationale": canonical_id,
        },
        "func": {
            "func_id": func_id,
            "func_type": "insert_node",
            "params": params,
            "meta": {},
        },
    }


@pytest.fixture
def facet_store(tmp_path):
    db_path = tmp_path / "facet.db"
    store = CubeStore(db_path)

    names = [
        "output_contract_json_answer_list",
        "facet_dp_scaffold",
        "table_serialization_html",
        "input_context_column_statistics",
    ]
    features = {name: _feature_row(name) for name in names}
    store.upsert_funcs([row["func"] for row in features.values()], on_conflict=OnConflict.SKIP)
    store.sync_features([row["feature"] for row in features.values()])

    def cfg(label, active_names):
        func_ids = [features[name]["func_id"] for name in active_names]
        feature_ids = [features[name]["feature_id"] for name in active_names]
        return store.get_or_create_config(
            func_ids,
            meta={
                "clean_label": label,
                "canonical_ids": active_names,
                "feature_ids": feature_ids,
                "dataset": "wtq",
                "datasets": ["wtq"],
            },
        )

    base = cfg("base", ["output_contract_json_answer_list", "facet_dp_scaffold"])
    ser = cfg("ser.html", [
        "output_contract_json_answer_list",
        "facet_dp_scaffold",
        "table_serialization_html",
    ])
    ctx = cfg("ctx.stats", [
        "output_contract_json_answer_list",
        "facet_dp_scaffold",
        "input_context_column_statistics",
    ])
    coal = cfg("ser.html__ctx.stats", [
        "output_contract_json_answer_list",
        "facet_dp_scaffold",
        "table_serialization_html",
        "input_context_column_statistics",
    ])

    queries = [
        ("q1", "count hard 1", "count", "hard", 0.0, 1.0, 0.0, 1.0),
        ("q2", "count hard 2", "count", "hard", 0.0, 1.0, 0.0, 1.0),
        ("q3", "count easy", "count", "easy", 1.0, 1.0, 1.0, 1.0),
        ("q4", "lookup hard", "lookup", "hard", 1.0, 1.0, 1.0, 1.0),
        ("q5", "lookup easy", "lookup", "easy", 1.0, 1.0, 1.0, 1.0),
        ("q6", "count hard 3", "count", "hard", 0.0, 1.0, 0.0, 1.0),
    ]
    store.upsert_queries(
        [
            {"query_id": qid, "dataset": "wtq", "content": text, "meta": {"split": "test"}}
            for qid, text, *_ in queries
        ],
        on_conflict=OnConflict.SKIP,
    )

    with store._cursor() as cur:
        for qid, _text, op, diff, *_scores in queries:
            for name, value in (
                ("operation_type", op),
                ("difficulty", diff),
                ("unique_query_marker", qid),
            ):
                cur.execute(
                    "INSERT OR IGNORE INTO predicate (query_id, name, value) VALUES (?, ?, ?)",
                    (qid, name, value),
                )

    cfg_scores = {
        base: [row[4] for row in queries],
        ser: [row[5] for row in queries],
        ctx: [row[6] for row in queries],
        coal: [row[7] for row in queries],
    }
    qids = [row[0] for row in queries]
    for config_id, scores in cfg_scores.items():
        for qid, score in zip(qids, scores):
            execution_id = store.insert_execution(
                config_id,
                qid,
                MODEL,
                prediction=str(score),
                on_conflict=OnConflict.SKIP,
            )
            store.upsert_evaluation(
                execution_id,
                SCORER,
                score,
                on_conflict=OnConflict.SKIP,
            )

    yield store, {"base": base, "ser": ser, "ctx": ctx, "coal": coal}
    store.close()
    Path(db_path).unlink(missing_ok=True)


def test_config_feature_sets_drops_structural_and_constant_contract(facet_store):
    store, ctx = facet_store
    rows = config_feature_sets(store, config_ids=[ctx["base"], ctx["ser"], ctx["ctx"], ctx["coal"]])
    by_label = {row["label"]: row["active_features"] for _, row in rows.iterrows()}

    assert by_label["base"] == frozenset()
    assert by_label["ser.html"] == frozenset({"table_serialization_html"})
    assert by_label["ctx.stats"] == frozenset({"input_context_column_statistics"})
    assert by_label["ser.html__ctx.stats"] == frozenset({
        "table_serialization_html",
        "input_context_column_statistics",
    })


def test_paired_delta_rows_and_baseline_inference(facet_store):
    store, ctx = facet_store
    assert infer_baseline_config_id(store, model=MODEL, scorer=SCORER, dataset="wtq", split="test") == ctx["base"]

    rows = paired_delta_rows(store, model=MODEL, scorer=SCORER, dataset="wtq", split="test")
    assert set(rows["config_id"]) == {ctx["ser"], ctx["ctx"], ctx["coal"]}
    ser = rows[rows["config_id"] == ctx["ser"]]
    assert ser["delta"].sum() == pytest.approx(3.0)
    assert ser["active_features"].iloc[0] == frozenset({"table_serialization_html"})


def test_query_context_sets_filters_high_cardinality_predicate(facet_store):
    store, _ = facet_store
    rows = query_context_sets(
        store,
        dataset="wtq",
        split="test",
        max_values_per_predicate=3,
    )
    atoms = set().union(*rows["context_atoms"].tolist())
    assert "operation_type=count" in atoms
    assert "difficulty=hard" in atoms
    assert all(not atom.startswith("unique_query_marker=") for atom in atoms)


def test_itemsets_and_contextual_effects(facet_store):
    store, _ = facet_store
    delta = paired_delta_rows(store, model=MODEL, scorer=SCORER, dataset="wtq", split="test")
    context = query_context_sets(store, dataset="wtq", split="test", max_values_per_predicate=3)
    feature_sets = feature_itemsets_from_delta(delta, max_order=2, min_config_support=1)
    context_sets = context_itemsets_from_context(context, max_order=2, min_query_support=2)

    assert ("table_serialization_html",) in feature_sets
    assert ("input_context_column_statistics", "table_serialization_html") in feature_sets
    assert ("operation_type=count",) in context_sets

    global_effects = score_feature_itemsets(delta, feature_sets, n_bootstrap=50, seed=1)
    html_global = global_effects[global_effects["feature_rule"] == "table_serialization_html"].iloc[0]
    assert html_global["mean_delta"] == pytest.approx(0.5)
    assert html_global["n_queries"] == 6

    conditional = score_contextual_feature_itemsets(
        delta,
        context,
        feature_sets,
        context_sets,
        min_query_support=2,
        min_observation_support=2,
        n_bootstrap=50,
        seed=1,
    )
    count_html = conditional[
        (conditional["feature_rule"] == "table_serialization_html")
        & (conditional["context_rule"] == "operation_type=count")
    ].iloc[0]
    assert count_html["n_queries"] == 4
    assert count_html["mean_delta"] == pytest.approx(0.75)
    assert count_html["conditional_lift"] == pytest.approx(0.25)


def test_binary_rules_and_cross_benchmark_comparison(facet_store):
    store, _ = facet_store
    result = discover_benchmark_subgroups(
        store,
        dataset="wtq",
        model=MODEL,
        scorer=SCORER,
        split="test",
        max_feature_order=2,
        max_context_order=2,
        min_query_support=2,
        n_bootstrap=50,
        seed=1,
    )
    binary = score_binary_contextual_rules(
        result["delta_rows"],
        result["context_rows"],
        result["feature_itemsets"],
        result["context_itemsets"],
        min_query_support=2,
        min_observation_support=2,
    )
    assert not binary.empty
    count_html = binary[
        (binary["feature_rule"] == "table_serialization_html")
        & (binary["context_rule"] == "operation_type=count")
    ].iloc[0]
    assert 0.0 <= count_html["wilson_ci_lo"] <= count_html["wilson_ci_hi"] <= 1.0
    assert count_html["useful_rate"] == pytest.approx(0.75)

    cross = compare_binary_rules_across_benchmarks({"wtq": binary, "sqa": binary})
    shared = cross[cross["rule_key"].str.contains("table_serialization_html :: operation_type=count", regex=False)]
    assert not shared.empty
    assert shared.iloc[0]["n_datasets"] == 2


def test_operator_chain_matches_one_shot_outputs(facet_store):
    store, ctx = facet_store

    chain = (
        FacetOperatorChain.for_scope(
            store,
            dataset="wtq",
            model=MODEL,
            scorer=SCORER,
            split="test",
        )
        .with_paired_deltas()
        .with_context(max_values_per_predicate=3)
        .with_itemsets(max_feature_order=2, max_context_order=2, min_query_support=2)
        .with_global_effects(n_bootstrap=0)
        .with_conditional_effects(min_query_support=2, n_bootstrap=50, seed=1)
        .with_binary_rules(min_query_support=2)
    )

    assert chain.state.base_config_id == ctx["base"]
    assert set(chain.state.delta_rows["config_id"]) == {ctx["ser"], ctx["ctx"], ctx["coal"]}
    assert ("table_serialization_html",) in chain.state.feature_itemsets
    assert ("operation_type=count",) in chain.state.context_itemsets
    assert not chain.state.global_effects.empty
    assert not chain.state.conditional_effects.empty
    assert not chain.state.binary_rules.empty
