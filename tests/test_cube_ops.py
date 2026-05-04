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
    target = store.get_or_create_config(
        ["func_target"],
        meta={"kind": "add_one_feature", "canonical_id": "target_feature"},
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
            execution_id = store.insert_execution(
                config_id,
                query_id,
                MODEL,
                system_prompt=f"system {config_id}",
                user_content=f"user {query_id}",
                raw_response=raw,
                prediction=f"pred {config_id} {query_id}",
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

    yield store, {"base": base, "target": target}
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

    fields = {row["field"] for row in cube_ops.list_query_meta_fields(store)}
    assert "query.meta.qtype" in fields
    assert "query.meta.qsubtype" in fields


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

    artifact = cube_ops.execution_artifact(store, execution_id=examples[0]["executionId"])
    assert artifact["systemPrompt"] == f"system {ctx['target']}"
    assert artifact["userContent"] == "user q3"
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

    assert ecr["true"] == 1
    assert ecr["false"] == 2
    assert patterns["bare_final_answer_line"] == 1
    assert patterns["prints_final_answer"] == 1

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
