"""HotpotQA-context compound task integration tests."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_TOOL_DIR = str(Path(__file__).parent.parent.parent.parent)
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from common import seed_funcs
from core.feature_registry import FeatureRegistry
from core.func_registry import apply_config_modules
from core.store import CubeStore, OnConflict
from execution.runner import run_config
from tasks.hotpotqa_context.hotpotqa_context import (
    HotpotQAContextTask,
    parse_answer,
    parse_hop_query,
    parse_summary,
    retrieve_context_passages,
)
from tasks.hotpotqa_context.loaders import seed_queries_hotpotqa_context


def _hotpot_row() -> dict:
    return {
        "id": "hp-1",
        "split": "test",
        "question": "What city is the headquarters of the organization that created Python?",
        "answer": "Wilmington",
        "context": [
            {
                "title": "Python Software Foundation",
                "sentences": [
                    "The Python Software Foundation manages the open source Python language.",
                    "Its headquarters are in Wilmington, Delaware.",
                ],
            },
            {
                "title": "Monty Python",
                "sentences": ["Monty Python was a British comedy group."],
            },
        ],
        "supporting_facts": [["Python Software Foundation", 1]],
    }


def test_hotpotqa_context_loader_seeds_local_jsonl():
    rows = [
        {**_hotpot_row(), "id": "train-1", "split": "train"},
        _hotpot_row(),
    ]

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as data_f:
        for row in rows:
            data_f.write(json.dumps(row) + "\n")
        data_path = data_f.name
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_f:
        db_path = db_f.name

    try:
        store = CubeStore(db_path)
        n = seed_queries_hotpotqa_context(
            store,
            split="test",
            data_path=data_path,
            on_conflict=OnConflict.ERROR,
        )
        seeded = store._get_conn().execute("SELECT * FROM query").fetchall()

        assert n == 1
        assert len(seeded) == 1
        assert seeded[0]["dataset"] == "hotpotqa_context"
        assert seeded[0]["content"] == rows[1]["question"]

        meta = json.loads(seeded[0]["meta"])
        assert meta["split"] == "test"
        assert meta["answer"] == "Wilmington"
        assert meta["_raw"]["supporting_facts"] == [["Python Software Foundation", 1]]
        store.close()
    finally:
        os.unlink(data_path)
        os.unlink(db_path)


def test_hotpotqa_context_retriever_prefers_matching_passages():
    row = _hotpot_row()
    passages = retrieve_context_passages(
        "Python Software Foundation headquarters",
        row["context"],
        k=1,
    )

    assert len(passages) == 1
    assert passages[0]["title"] == "Python Software Foundation"
    assert "Wilmington" in passages[0]["text"]


def test_hotpotqa_context_base_features_route_to_modules():
    reg = FeatureRegistry.load("hotpotqa_context")
    specs, _ = reg.materialize([
        "_section_summarize1",
        "base_summarize1",
        "_section_create_query_hop2",
        "base_create_query_hop2",
        "_section_summarize2",
        "base_summarize2",
        "_section_final_answer",
        "base_final_answer",
    ])

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_f:
        db_path = db_f.name
    try:
        store = CubeStore(db_path)
        seed_funcs(store, specs, on_conflict=OnConflict.ERROR)
        states = apply_config_modules(
            [spec["func_id"] for spec in specs],
            store,
            module_names=HotpotQAContextTask.module_names(),
        )

        assert states["summarize1"].sections
        assert states["create_query_hop2"].sections
        assert states["summarize2"].sections
        assert states["final_answer"].sections
        assert not states["__main__"].sections
        store.close()
    finally:
        os.unlink(db_path)


def test_hotpotqa_context_defaults_include_gepa_reasoning_fields():
    task = HotpotQAContextTask()
    task.bind_modules({})

    cases = {
        "summarize1": (
            {
                "question": "What city is the headquarters of the organization that created Python?",
                "passages": "Passage 1 (Python Software Foundation): Its headquarters are in Wilmington.",
            },
            "summary",
        ),
        "create_query_hop2": (
            {
                "question": "What city is the headquarters of the organization that created Python?",
                "summary_1": "Python is managed by the Python Software Foundation.",
            },
            "query",
        ),
        "summarize2": (
            {
                "question": "What city is the headquarters of the organization that created Python?",
                "summary_1": "Python is managed by the Python Software Foundation.",
                "passages": "Passage 1 (Python Software Foundation): Its headquarters are in Wilmington.",
            },
            "summary",
        ),
        "final_answer": (
            {
                "question": "What city is the headquarters of the organization that created Python?",
                "summary_1": "Python is managed by the Python Software Foundation.",
                "summary_2": "The Python Software Foundation headquarters are in Wilmington.",
            },
            "answer",
        ),
    }

    for module_name, (record, task_output_field) in cases.items():
        system_prompt, _ = task.build_module_prompt(module_name, record)
        assert '"reasoning"' in system_prompt
        assert f'"{task_output_field}"' in system_prompt


def test_hotpotqa_context_run_uses_context_retrieval_and_records_traces():
    calls = []

    def fake_llm(system_prompt: str, user_content: str) -> dict:
        calls.append({"system": system_prompt, "user": user_content})
        if len(calls) == 1:
            assert "Python Software Foundation" in user_content
            return {
                "raw_response": json.dumps({
                    "summary": "Python was created by an organization called the Python Software Foundation.",
                }),
                "prompt_tokens": 11,
                "completion_tokens": 5,
            }
        if len(calls) == 2:
            return {
                "raw_response": json.dumps({
                    "query": "Python Software Foundation headquarters city",
                }),
                "prompt_tokens": 7,
                "completion_tokens": 4,
            }
        if len(calls) == 3:
            assert "Wilmington" in user_content
            return {
                "raw_response": json.dumps({
                    "summary": "The Python Software Foundation headquarters are in Wilmington, Delaware.",
                }),
                "prompt_tokens": 13,
                "completion_tokens": 6,
            }
        return {
            "raw_response": json.dumps({"answer": "Wilmington"}),
            "prompt_tokens": 9,
            "completion_tokens": 3,
        }

    row = _hotpot_row()
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_f:
        db_path = db_f.name
    try:
        store = CubeStore(db_path)
        store.upsert_queries(
            [{
                "query_id": "hp-q1",
                "dataset": "hotpotqa_context",
                "content": row["question"],
                "meta": {"split": "test", "answer": row["answer"], "_raw": row},
            }],
            on_conflict=OnConflict.ERROR,
        )
        config_id = store.get_or_create_config([])

        task = HotpotQAContextTask()
        task.bind_modules({})
        run_config(
            store,
            config_id,
            [{
                "query_id": "hp-q1",
                "dataset": "hotpotqa_context",
                "content": row["question"],
                "meta": json.dumps({"split": "test", "answer": row["answer"], "_raw": row}),
            }],
            task,
            "fake-model",
            fake_llm,
        )

        stored = store.get_cached_execution(config_id, "hp-q1", "fake-model")
        meta = json.loads(stored["meta"])
        traces = meta["module_traces"]

        assert stored["prediction"] == "Wilmington"
        assert stored["prompt_tokens"] == 40
        assert stored["completion_tokens"] == 18
        assert [t["module_name"] for t in traces] == [
            "summarize1",
            "create_query_hop2",
            "summarize2",
            "final_answer",
        ]
        assert "Wilmington" in traces[2]["user_content"]
        assert len(calls) == 4
        store.close()
    finally:
        os.unlink(db_path)


def test_hotpotqa_context_score_reports_em_and_f1():
    task = HotpotQAContextTask()
    score, metrics = task.score("The answer is Wilmington.", {"answer": "Wilmington"})

    assert score == 0.5
    assert metrics["exact_match"] == 0.0
    assert metrics["f1"] == 0.5

    score, metrics = task.score("Wilmington", {"answer": "Wilmington"})

    assert score == 1.0
    assert metrics["exact_match"] == 1.0
    assert metrics["f1"] == 1.0


def test_hotpotqa_context_parsers_strip_qwen_thinking_blocks():
    assert parse_summary(
        '<think>\nFind first-hop evidence.\n</think>\n{"summary":"First fact."}'
    ) == "First fact."
    assert parse_hop_query(
        '<think>\nNeed second hop.\n</think>\n{"query":"headquarters city"}'
    ) == "headquarters city"
    assert parse_answer(
        '<think>\nNow answer.\n</think>\n{"answer":"Wilmington"}'
    ) == "Wilmington"
    assert parse_answer("<think>\nTruncated reasoning") == ""
