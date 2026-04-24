"""Compound task and module-targeted feature behavior."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_TOOL_DIR = str(Path(__file__).parent.parent.parent.parent)
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from core.feature_registry import FeatureRegistry, compute_feature_id
from core.func_registry import ROOT_ID, apply_config_modules
from core.store import CubeStore, OnConflict
from execution.runner import run_config
from task import CompoundTask, ModuleRuntime, ModuleSpec


def _targeted_rule(content: str, target_module: str) -> dict:
    return {
        "func_type": "insert_node",
        "target_module": target_module,
        "params": {
            "node_type": "rule",
            "parent_id": ROOT_ID,
            "payload": {"content": content},
        },
    }


def test_target_module_participates_in_feature_hash():
    """The same primitive edit aimed at two modules is a different feature."""
    retrieve = [_targeted_rule("Be concise.", "retrieve")]
    answer = [_targeted_rule("Be concise.", "answer")]

    assert compute_feature_id(retrieve) != compute_feature_id(answer)


def test_materialize_records_target_module_and_distinct_func_ids():
    """Targeted edits carry module metadata and do not collide as funcs."""
    reg = FeatureRegistry(
        task="compound",
        features={
            "retrieve_rule": {
                "canonical_id": "retrieve_rule",
                "task": "compound",
                "requires": [],
                "conflicts_with": [],
                "primitive_edits": [_targeted_rule("Be concise.", "retrieve")],
            },
            "answer_rule": {
                "canonical_id": "answer_rule",
                "task": "compound",
                "requires": [],
                "conflicts_with": [],
                "primitive_edits": [_targeted_rule("Be concise.", "answer")],
            },
        },
    )

    specs, _ = reg.materialize(["retrieve_rule", "answer_rule"])

    assert len(specs) == 2
    assert specs[0]["func_id"] != specs[1]["func_id"]
    assert {s["meta"]["target_module"] for s in specs} == {"retrieve", "answer"}


def test_apply_config_modules_routes_funcs_by_target_module():
    """Module-targeted funcs only mutate their named module state."""
    reg = FeatureRegistry(
        task="compound",
        features={
            "answer_rule": {
                "canonical_id": "answer_rule",
                "task": "compound",
                "requires": [],
                "conflicts_with": [],
                "primitive_edits": [_targeted_rule("Answer carefully.", "answer")],
            },
        },
    )
    specs, _ = reg.materialize(["answer_rule"])

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmpf:
        db_path = tmpf.name
    try:
        store = CubeStore(db_path)
        store.upsert_funcs(specs, on_conflict=OnConflict.SKIP)
        states = apply_config_modules(
            [s["func_id"] for s in specs],
            store,
            module_names=["retrieve", "answer"],
        )

        assert ROOT_ID not in states["retrieve"].rules
        assert states["answer"].rules[ROOT_ID][0]["content"] == "Answer carefully."
        store.close()
    finally:
        os.unlink(db_path)


class TwoStageTask(CompoundTask):
    name = "two_stage"
    module_specs = {
        "extract": ModuleSpec(
            input_fields={"question": "Question to inspect."},
            output_fields={"extracted": "Intermediate extracted fact."},
        ),
        "answer": ModuleSpec(
            input_fields={
                "question": "Original question.",
                "extracted": "Intermediate extracted fact.",
            },
            output_fields={"answer": "Final answer."},
        ),
    }

    def run(self, query: dict, runtime: ModuleRuntime) -> str:
        system_prompt, user_content = self.build_module_prompt(
            "extract",
            {"question": query["content"]},
        )
        extracted = runtime.call(
            "extract",
            system_prompt,
            user_content,
            parse=lambda raw: raw.strip(),
        ).parsed_output

        system_prompt, user_content = self.build_module_prompt(
            "answer",
            {"question": query["content"], "extracted": extracted},
        )
        return runtime.call(
            "answer",
            system_prompt,
            user_content,
            parse=lambda raw: raw.strip(),
        ).parsed_output


def test_compound_runner_stores_final_prediction_and_module_traces():
    """A compound task still produces one e2e execution row with trace JSON."""
    calls = []

    def fake_llm(system_prompt: str, user_content: str) -> dict:
        calls.append((system_prompt, user_content))
        if len(calls) == 1:
            return {
                "raw_response": "intermediate fact",
                "prompt_tokens": 3,
                "completion_tokens": 2,
            }
        return {
            "raw_response": "final answer",
            "prompt_tokens": 5,
            "completion_tokens": 4,
        }

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmpf:
        db_path = tmpf.name
    try:
        store = CubeStore(db_path)
        store.upsert_queries(
            [{"query_id": "q1", "dataset": "synthetic", "content": "What?", "meta": {}}],
            on_conflict=OnConflict.ERROR,
        )
        config_id = store.get_or_create_config([])

        task = TwoStageTask()
        task.bind_modules({})
        run_config(
            store,
            config_id,
            [{"query_id": "q1", "dataset": "synthetic", "content": "What?", "meta": "{}"}],
            task,
            "fake-model",
            fake_llm,
        )

        row = store.get_cached_execution(config_id, "q1", "fake-model")
        meta = json.loads(row["meta"])

        assert row["prediction"] == "final answer"
        assert row["raw_response"] == "final answer"
        assert row["prompt_tokens"] == 8
        assert row["completion_tokens"] == 6
        assert [t["module_name"] for t in meta["module_traces"]] == ["extract", "answer"]
        assert meta["module_traces"][0]["raw_response"] == "intermediate fact"
        assert meta["module_traces"][1]["parsed_output"] == "final answer"
        assert len(calls) == 2
        store.close()
    finally:
        os.unlink(db_path)
