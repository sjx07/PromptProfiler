from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analyze import cube_ops
from core.store import CubeStore, OnConflict


MODEL = "test-model"
SCORER = "exact"


@pytest.fixture
def mini_cube(tmp_path):
    db_path = tmp_path / "cube_ops.db"
    store = CubeStore(db_path)
    base = store.get_or_create_config([], meta={"kind": "base", "canonical_ids": []})
    feature_id = "feat_target_hash"
    store.sync_features([
        {
            "feature_id": feature_id,
            "canonical_id": "target_feature",
            "task": "tablebench",
            "requires_json": "[]",
            "conflicts_json": "[]",
            "primitive_spec": "[]",
            "semantic_labels_json": (
                '[{"label_id":"style.test","role":"style_rule"}]'
            ),
            "scope_json": "{}",
            "label_rows": [
                {
                    "label_id": "style.test",
                    "description": "Test style label",
                }
            ],
            "label_memberships": [
                {
                    "feature_id": feature_id,
                    "label_id": "style.test",
                    "role": "style_rule",
                }
            ],
        }
    ])
    target = store.get_or_create_config(
        ["func_target"],
        meta={
            "kind": "add_one_feature",
            "canonical_id": "target_feature",
            "canonical_ids": ["target_feature"],
            "feature_ids": [feature_id],
        },
    )
    coalition = store.get_or_create_config(
        ["func_extra", "func_target"],
        meta={
            "kind": "explicit_coalition",
            "label": "pot_all",
            "canonical_ids": [
                "_section_role",
                "tb_pot_fixed_scaffold",
                "tb_pot_reasoning_field",
            ],
            "feature_ids": [feature_id],
        },
    )
    store.upsert_queries(
        [
            {
                "query_id": "q1",
                "dataset": "tablebench",
                "content": "Q1?",
                "meta": {
                    "qtype": "NumericalReasoning",
                    "qsubtype": "Counting",
                    "gold_answer": "one",
                },
            },
            {
                "query_id": "q2",
                "dataset": "tablebench",
                "content": "Q2?",
                "meta": {
                    "qtype": "NumericalReasoning",
                    "qsubtype": "Counting",
                    "gold_answer": "two",
                },
            },
            {
                "query_id": "q3",
                "dataset": "tablebench",
                "content": "Q3?",
                "meta": {
                    "qtype": "DataAnalysis",
                    "qsubtype": "CausalAnalysis",
                    "gold_answer": "three",
                },
            },
        ],
        on_conflict=OnConflict.SKIP,
    )
    with store._cursor() as cur:
        cur.executemany(
            "INSERT OR IGNORE INTO predicate (query_id, name, value) VALUES (?, ?, ?)",
            [
                ("q1", "needs_math", "true"),
                ("q2", "needs_math", "true"),
                ("q3", "needs_math", "false"),
            ],
        )

    scores = {
        base: {"q1": 0.0, "q2": 1.0, "q3": 1.0},
        target: {"q1": 1.0, "q2": 0.0, "q3": 0.0},
    }
    for config_id, per_query in scores.items():
        for query_id, score in per_query.items():
            raw = f"raw {config_id} {query_id}"
            metrics = {"ECR@1": True, "output_mode": "plain"}
            if config_id == target and query_id == "q2":
                raw = "Final Answer: wrong"
                metrics = {"ECR@1": False, "output_mode": "python_exec"}
            if config_id == target and query_id == "q3":
                raw = "print('Final Answer: wrong')"
                metrics = {"ECR@1": False, "output_mode": "python_exec"}
            raw_reasoning = ""
            if config_id == target and query_id == "q3":
                raw_reasoning = "First inspect the table, then compute the final answer."
            execution_id = store.insert_execution(
                config_id,
                query_id,
                MODEL,
                system_prompt=f"system {config_id}",
                user_content=f"user {query_id}",
                raw_response=raw,
                raw_reasoning=raw_reasoning,
                prediction=f"pred {config_id} {query_id}",
                finish_reason="stop",
                phase="p1",
                on_conflict=OnConflict.SKIP,
            )
            store.upsert_evaluation(
                execution_id,
                SCORER,
                score,
                metrics=metrics,
                on_conflict=OnConflict.SKIP,
            )

    yield store, {"base": base, "target": target, "coalition": coalition}
    store.close()


def test_config_inventory_and_meta_fields(mini_cube):
    store, ctx = mini_cube
    configs = cube_ops.list_configs_detailed(store, model=MODEL, scorer=SCORER)
    by_id = {row["configId"]: row for row in configs}

    assert by_id[ctx["base"]]["canonicalId"] == "base"
    assert by_id[ctx["target"]]["canonicalId"] == "target_feature"
    assert by_id[ctx["target"]]["nExecutions"] == 3
    assert by_id[ctx["target"]]["nEvaluations"] == 3
    assert by_id[ctx["target"]]["avgScore"] == pytest.approx(1 / 3)
    assert by_id[ctx["coalition"]]["canonicalId"] == "pot_all"
    assert by_id[ctx["coalition"]]["configGroup"] == "90 explicit_coalition"
    assert by_id[ctx["coalition"]]["surfaceAtoms"] == ["pot_all"]
    assert by_id[ctx["coalition"]]["canonicalIds"] == [
        "_section_role",
        "tb_pot_fixed_scaffold",
        "tb_pot_reasoning_field",
    ]

    fields = {row["field"] for row in cube_ops.list_query_meta_fields(store)}
    assert "query.meta.qtype" in fields
    assert "query.meta.qsubtype" in fields


def test_config_feature_shape_filters(mini_cube):
    store, ctx = mini_cube
    surface = store.get_or_create_config(
        ["func_fmt", "func_ser", "func_ctx"],
        meta={
            "kind": "explicit_coalition",
            "canonical_id": "fmt.json__ser.records__ctx.stats",
            "canonical_ids": [
                "prompt_format_json",
                "table_serialization_json_records",
                "input_context_column_statistics",
            ],
        },
    )
    rows = cube_ops.list_configs_detailed(store, model=MODEL, scorer=SCORER)
    by_id = {row["configId"]: row for row in rows}

    assert by_id[surface]["configGroup"] == "30 surface matrix"
    assert by_id[surface]["surfaceAtoms"] == ["fmt.json", "ser.records", "ctx.stats"]
    assert by_id[surface]["configAxes"] == ["fmt", "ser", "ctx"]

    contains_atom = cube_ops.filter_config_rows_by_features(
        rows,
        mode="contains_atoms",
        atoms=["fmt.json"],
    )
    assert {row["configId"] for row in contains_atom} == {surface}

    exact_atom = cube_ops.filter_config_rows_by_features(
        rows,
        mode="exact_atoms",
        atoms=["fmt.json", "ser.records", "ctx.stats"],
    )
    assert {row["configId"] for row in exact_atom} == {surface}

    contains_axes = cube_ops.filter_config_rows_by_features(
        rows,
        mode="contains_axes",
        axes=["fmt", "ctx"],
    )
    assert {row["configId"] for row in contains_axes} == {surface}

    exact_axes = cube_ops.filter_config_rows_by_features(
        rows,
        mode="exact_axes",
        axes=["fmt", "ctx"],
    )
    assert surface not in {row["configId"] for row in exact_axes}

    exact_axes = cube_ops.filter_config_rows_by_features(
        rows,
        mode="exact_axes",
        axes=["fmt", "ser", "ctx"],
    )
    assert {row["configId"] for row in exact_axes} == {surface}

    group_rows = cube_ops.filter_config_rows_by_features(
        rows,
        mode="groups",
        groups=["30 surface matrix"],
    )
    assert {row["configId"] for row in group_rows} == {surface}
    target_rows = cube_ops.filter_config_rows_by_features(
        rows,
        mode="feature_exists",
        atoms=["target_feature"],
    )
    assert {row["configId"] for row in target_rows} == {ctx["target"]}


def test_reasoning_protocol_canonical_atom_aliases():
    assert cube_ops._canonical_atom("reasoning_scaffold_visible_cot") == "scaffold.trace"
    assert cube_ops._canonical_atom("reasoning_scaffold_visible_operation") == "scaffold.operation"
    assert cube_ops._canonical_atom("reasoning_scaffold_visible_workpad") == "scaffold.workpad"
    assert cube_ops._canonical_atom("reasoning_evidence_localization") == "reason.localize"
    assert cube_ops._canonical_atom("reasoning_candidate_enumeration") == "reason.enumerate"
    assert cube_ops._canonical_atom("reasoning_symbolic_operation") == "reason.symbolic_op"
    assert cube_ops._canonical_atom("reasoning_plan_then_answer") == "reason.plan_execute"


def test_predicate_fields_are_available_for_slicing(mini_cube):
    store, _ctx = mini_cube
    fields = cube_ops.list_predicate_fields(store)
    by_name = {row["name"]: row for row in fields}

    assert by_name["needs_math"]["field"] == "predicate.needs_math"
    assert by_name["needs_math"]["nRows"] == 3
    assert by_name["needs_math"]["nQueries"] == 3
    assert by_name["needs_math"]["nValues"] == 2
    assert set(by_name["needs_math"]["samples"]) == {"true", "false"}


def test_benchmark_filter_limits_inventory_slices_and_labels(mini_cube):
    store, ctx = mini_cube
    store.upsert_queries(
        [
            {
                "query_id": "wq1",
                "dataset": "wtq",
                "content": "WTQ test?",
                "meta": {
                    "split": "test",
                    "qtype": "Lookup",
                    "qsubtype": "Argmax",
                    "gold_answer": "lee",
                },
            },
            {
                "query_id": "wq2",
                "dataset": "wtq",
                "content": "WTQ validation?",
                "meta": {
                    "split": "validation",
                    "qtype": "Lookup",
                    "qsubtype": "Counting",
                    "gold_answer": "kim",
                },
            },
            {
                "query_id": "wq3",
                "dataset": "wtq",
                "content": "WTQ unevaluated?",
                "meta": {
                    "split": "test",
                    "qtype": "Lookup",
                    "qsubtype": "Unused",
                    "unused_only": "yes",
                    "gold_answer": "park",
                },
            },
        ],
        on_conflict=OnConflict.SKIP,
    )
    with store._cursor() as cur:
        cur.execute(
            "INSERT OR IGNORE INTO predicate (query_id, name, value) VALUES (?, ?, ?)",
            ("wq3", "unused_predicate", "true"),
        )
    extra_scores = {
        ctx["base"]: {"wq1": 0.0, "wq2": 1.0},
        ctx["target"]: {"wq1": 1.0, "wq2": 0.0},
    }
    for config_id, per_query in extra_scores.items():
        for query_id, score in per_query.items():
            execution_id = store.insert_execution(
                config_id,
                query_id,
                MODEL,
                prediction=f"bench pred {config_id} {query_id}",
                on_conflict=OnConflict.SKIP,
            )
            store.upsert_evaluation(
                execution_id,
                SCORER,
                score,
                metrics={"output_mode": "plain"},
                on_conflict=OnConflict.SKIP,
            )

    configs = cube_ops.list_configs_detailed(
        store,
        model=MODEL,
        scorer=SCORER,
        dataset="wtq",
        split="test",
        only_with_results=True,
    )
    by_id = {row["configId"]: row for row in configs}
    assert ctx["coalition"] not in by_id
    assert by_id[ctx["base"]]["nExecutions"] == 1
    assert by_id[ctx["base"]]["avgScore"] == pytest.approx(0.0)
    assert by_id[ctx["target"]]["nExecutions"] == 1
    assert by_id[ctx["target"]]["avgScore"] == pytest.approx(1.0)

    wtq_all = cube_ops.list_configs_detailed(
        store,
        model=MODEL,
        scorer=SCORER,
        dataset="wtq",
    )
    by_id = {row["configId"]: row for row in wtq_all}
    assert by_id[ctx["target"]]["nExecutions"] == 2
    assert by_id[ctx["target"]]["avgScore"] == pytest.approx(0.5)

    no_split = cube_ops.list_configs_detailed(
        store,
        model=MODEL,
        scorer=SCORER,
        dataset="tablebench",
        split="(no split)",
    )
    by_id = {row["configId"]: row for row in no_split}
    assert by_id[ctx["target"]]["nExecutions"] == 3

    wtq_fields = {
        row["field"]: row
        for row in cube_ops.list_query_meta_fields(
            store,
            dataset="wtq",
            split="test",
            model=MODEL,
            scorer=SCORER,
            only_with_results=True,
        )
    }
    assert wtq_fields["query.meta.qtype"]["nQueries"] == 1
    assert wtq_fields["query.meta.qsubtype"]["samples"] == ["Argmax"]
    assert "query.meta.unused_only" not in wtq_fields
    all_wtq_fields = {
        row["field"]
        for row in cube_ops.list_query_meta_fields(
            store,
            dataset="wtq",
            split="test",
        )
    }
    assert "query.meta.unused_only" in all_wtq_fields

    assert cube_ops.list_predicate_fields(
        store,
        dataset="wtq",
        split="test",
        model=MODEL,
        scorer=SCORER,
        only_with_results=True,
    ) == []
    assert cube_ops.list_predicate_fields(
        store,
        dataset="wtq",
        split="test",
    )[0]["name"] == "unused_predicate"
    tablebench_predicates = cube_ops.list_predicate_fields(
        store,
        dataset="tablebench",
        split="(no split)",
    )
    assert tablebench_predicates[0]["name"] == "needs_math"
    assert tablebench_predicates[0]["nQueries"] == 3

    bench_filters = [
        {"field": "dataset", "op": "=", "value": "wtq"},
        {"field": "split", "op": "=", "value": "test"},
    ]
    rows = cube_ops.slice_scores(
        store,
        model=MODEL,
        scorer=SCORER,
        config_ids=[ctx["base"], ctx["target"]],
        group_by=["dataset", "split"],
        filters=bench_filters,
        base_config_id=ctx["base"],
    )
    idx = {
        (row["configId"], row["group"]["dataset"], row["group"]["split"]): row
        for row in rows
    }
    target = idx[(ctx["target"], "wtq", "test")]
    assert target["n"] == 1
    assert target["avgScore"] == pytest.approx(1.0)
    assert target["deltaVsBase"] == pytest.approx(1.0)

    labels = cube_ops.feature_label_analysis(
        store,
        model=MODEL,
        scorer=SCORER,
        config_ids=[ctx["target"]],
        base_config_id=ctx["base"],
        filters=bench_filters,
    )
    assert len(labels) == 1
    assert labels[0]["n"] == 1
    assert labels[0]["avgScore"] == pytest.approx(1.0)
    assert labels[0]["baseScore"] == pytest.approx(0.0)
    assert labels[0]["deltaVsBase"] == pytest.approx(1.0)


def test_slice_scores_by_query_meta_with_delta(mini_cube):
    store, ctx = mini_cube
    rows = cube_ops.slice_scores(
        store,
        model=MODEL,
        scorer=SCORER,
        config_ids=[ctx["base"], ctx["target"]],
        group_by=["query.meta.qtype", "query.meta.qsubtype"],
        base_config_id=ctx["base"],
    )

    idx = {
        (row["configId"], row["group"]["query.meta.qtype"], row["group"]["query.meta.qsubtype"]): row
        for row in rows
    }
    counting_target = idx[(ctx["target"], "NumericalReasoning", "Counting")]
    causal_target = idx[(ctx["target"], "DataAnalysis", "CausalAnalysis")]

    assert counting_target["n"] == 2
    assert counting_target["avgScore"] == pytest.approx(0.5)
    assert counting_target["deltaVsBase"] == pytest.approx(0.0)
    assert causal_target["avgScore"] == pytest.approx(0.0)
    assert causal_target["deltaVsBase"] == pytest.approx(-1.0)


def test_slice_scores_by_predicate_with_delta(mini_cube):
    store, ctx = mini_cube
    rows = cube_ops.slice_scores(
        store,
        model=MODEL,
        scorer=SCORER,
        config_ids=[ctx["base"], ctx["target"]],
        group_by=["predicate.needs_math"],
        base_config_id=ctx["base"],
    )

    idx = {
        (row["configId"], row["group"]["predicate.needs_math"]): row
        for row in rows
    }

    target_true = idx[(ctx["target"], "true")]
    target_false = idx[(ctx["target"], "false")]
    assert target_true["n"] == 2
    assert target_true["avgScore"] == pytest.approx(0.5)
    assert target_true["deltaVsBase"] == pytest.approx(0.0)
    assert target_false["n"] == 1
    assert target_false["avgScore"] == pytest.approx(0.0)
    assert target_false["deltaVsBase"] == pytest.approx(-1.0)


def test_feature_label_analysis_overall_and_by_predicate(mini_cube):
    store, ctx = mini_cube
    overall = cube_ops.feature_label_analysis(
        store,
        model=MODEL,
        scorer=SCORER,
        config_ids=[ctx["target"]],
        base_config_id=ctx["base"],
    )
    assert len(overall) == 1
    assert overall[0]["labelId"] == "style.test"
    assert overall[0]["role"] == "style_rule"
    assert overall[0]["predicateValue"] is None
    assert overall[0]["nConfigs"] == 1
    assert overall[0]["nQueries"] == 3
    assert overall[0]["n"] == 3
    assert overall[0]["avgScore"] == pytest.approx(1 / 3)
    assert overall[0]["baseScore"] == pytest.approx(2 / 3)
    assert overall[0]["deltaVsBase"] == pytest.approx(-1 / 3)
    assert overall[0]["components"] == ["target_feature"]

    by_predicate = cube_ops.feature_label_analysis(
        store,
        model=MODEL,
        scorer=SCORER,
        config_ids=[ctx["target"]],
        predicate_name="needs_math",
        base_config_id=ctx["base"],
    )
    idx = {row["predicateValue"]: row for row in by_predicate}

    assert idx["true"]["n"] == 2
    assert idx["true"]["avgScore"] == pytest.approx(0.5)
    assert idx["true"]["baseScore"] == pytest.approx(0.5)
    assert idx["true"]["deltaVsBase"] == pytest.approx(0.0)
    assert idx["false"]["n"] == 1
    assert idx["false"]["avgScore"] == pytest.approx(0.0)
    assert idx["false"]["baseScore"] == pytest.approx(1.0)
    assert idx["false"]["deltaVsBase"] == pytest.approx(-1.0)


def test_feature_summary_rolls_up_canonical_features(mini_cube):
    store, ctx = mini_cube
    rows = cube_ops.feature_summary(
        store,
        model=MODEL,
        scorer=SCORER,
        config_ids=[ctx["target"], ctx["coalition"]],
        base_config_id=ctx["base"],
    )
    by_feature = {row["canonicalId"]: row for row in rows}

    assert by_feature["target_feature"]["family"] == "other"
    assert by_feature["target_feature"]["nConfigs"] == 1
    assert by_feature["target_feature"]["nQueries"] == 3
    assert by_feature["target_feature"]["n"] == 3
    assert by_feature["target_feature"]["avgScore"] == pytest.approx(1 / 3)
    assert by_feature["target_feature"]["baseScore"] == pytest.approx(2 / 3)
    assert by_feature["target_feature"]["deltaVsBase"] == pytest.approx(-1 / 3)
    assert "_section_role" not in by_feature


def test_examples_artifact_and_compare(mini_cube):
    store, ctx = mini_cube
    examples = cube_ops.examples(
        store,
        model=MODEL,
        scorer=SCORER,
        config_ids=[ctx["target"]],
        filters=[{"field": "query.meta.qsubtype", "op": "=", "value": "CausalAnalysis"}],
    )
    assert [row["queryId"] for row in examples] == ["q3"]
    assert examples[0]["gold"] == "three"
    assert examples[0]["reasoningChars"] > 0
    assert examples[0]["rawReasoningPreview"].startswith("First inspect")
    assert examples[0]["finishReason"] == "stop"

    artifact = cube_ops.execution_artifact(store, execution_id=examples[0]["executionId"])
    assert artifact["systemPrompt"] == f"system {ctx['target']}"
    assert artifact["userContent"] == "user q3"
    assert artifact["rawReasoning"].startswith("First inspect")
    assert artifact["reasoningChars"] == len(artifact["rawReasoning"])
    assert artifact["finishReason"] == "stop"
    assert artifact["metrics"]["output_mode"] == "python_exec"

    compare = cube_ops.compare_configs(
        store,
        model=MODEL,
        scorer=SCORER,
        base_config_id=ctx["base"],
        target_config_id=ctx["target"],
    )
    assert compare["nShared"] == 3
    assert compare["flippedUp"] == 1
    assert compare["flippedDown"] == 2
    assert compare["avgDelta"] == pytest.approx(-1 / 3)

    down = cube_ops.comparison_examples(
        store,
        model=MODEL,
        scorer=SCORER,
        base_config_id=ctx["base"],
        target_config_id=ctx["target"],
        direction="down",
    )
    assert [row["queryId"] for row in down] == ["q2", "q3"]


def test_diagnostics_and_delete_plan_are_read_only(mini_cube):
    store, ctx = mini_cube
    diag = cube_ops.diagnostics(
        store,
        model=MODEL,
        scorer=SCORER,
        config_ids=[ctx["target"]],
    )
    ecr = {row["value"]: row["n"] for row in diag["buckets"]["ecr"]}
    patterns = {row["value"]: row["n"] for row in diag["buckets"]["responsePattern"]}
    reasoning = {row["value"]: row["n"] for row in diag["buckets"]["reasoningState"]}
    finish = {row["value"]: row["n"] for row in diag["buckets"]["finishReason"]}

    assert ecr["true"] == 1
    assert ecr["false"] == 2
    assert patterns["bare_final_answer_line"] == 1
    assert patterns["prints_final_answer"] == 1
    assert reasoning["has_internal_reasoning"] == 1
    assert reasoning["no_internal_reasoning"] == 2
    assert finish["stop"] == 3

    plan = cube_ops.plan_delete(
        store,
        model=MODEL,
        config_ids=[ctx["target"]],
    )
    assert plan["nExecutions"] == 3
    assert plan["nEvaluations"] == 3
    assert "DELETE FROM evaluation" in plan["sqlPreview"][0]

    after = cube_ops.list_configs_detailed(store, model=MODEL, scorer=SCORER)
    by_id = {row["configId"]: row for row in after}
    assert by_id[ctx["target"]]["nExecutions"] == 3
