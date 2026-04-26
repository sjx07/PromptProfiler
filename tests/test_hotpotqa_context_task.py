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


def test_hotpotqa_context_defaults_expose_primary_output_fields():
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
                "context": "Python is managed by the Python Software Foundation.",
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
        assert f'"{task_output_field}"' in system_prompt
        assert '"reasoning"' in system_prompt


def test_hotpotqa_context_gepa_qwen3_merge_features_target_modules():
    reg = FeatureRegistry.load("hotpotqa_context")
    specs, _ = reg.materialize([
        "_section_summarize1",
        "gepa_qwen3_merge_summarize1",
        "_section_create_query_hop2",
        "gepa_qwen3_merge_create_query_hop2",
        "_section_summarize2",
        "gepa_qwen3_merge_summarize2",
        "_section_final_answer",
        "gepa_qwen3_merge_final_answer",
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

        task = HotpotQAContextTask()
        task.bind_modules(states)
        records = {
            "summarize1": (
                {
                    "question": "Which album includes the song mentioned in the first summary?",
                    "passages": "Passage 1 (Song): The passage identifies the song.",
                },
                "summary",
            ),
            "create_query_hop2": (
                {
                    "question": "Which album includes the song mentioned in the first summary?",
                    "summary_1": "The first passage identifies the song but not its album.",
                },
                "query",
            ),
            "summarize2": (
                {
                    "question": "Which album includes the song mentioned in the first summary?",
                    "context": "The first passage identifies the song but not its album.",
                    "passages": "Passage 1 (Album): The album includes the song.",
                },
                "summary",
            ),
            "final_answer": (
                {
                    "question": "Which album includes the song mentioned in the first summary?",
                    "summary_1": "The first passage identifies the song but not its album.",
                    "summary_2": "The album is identified explicitly.",
                },
                "answer",
            ),
        }

        # Verbatim-text markers taken from the paper's GEPA+Merge Qwen3-8B prompts.
        verbatim_markers = {
            "summarize1": "first-hop **summarization module**",
            "create_query_hop2": "Your query must target documents not retrieved in the first hop",
            "summarize2": "Your task is to synthesize information from the question, context, and newly retrieved passages",
            "final_answer": "Extracting precise terminology",
        }
        for module_name, (record, output_field) in records.items():
            system_prompt, _ = task.build_module_prompt(module_name, record)
            assert verbatim_markers[module_name] in system_prompt
            assert '"reasoning"' in system_prompt
            assert f'"{output_field}"' in system_prompt
        store.close()
    finally:
        os.unlink(db_path)


def test_hotpotqa_context_run_uses_configured_retriever_and_records_traces():
    llm_calls = []
    retrieval_calls = []

    class FakeRetriever:
        def search(self, query: str, *, context=None, k: int = 7):
            retrieval_calls.append({"query": query, "context": context, "k": k})
            if len(retrieval_calls) == 1:
                return [{
                    "title": "Python Software Foundation",
                    "text": "The Python Software Foundation manages the open source Python language.",
                }]
            return [{
                "title": "Python Software Foundation",
                "text": "Its headquarters are in Wilmington, Delaware.",
            }]

    def fake_llm(system_prompt: str, user_content: str) -> dict:
        llm_calls.append({"system": system_prompt, "user": user_content})
        if len(llm_calls) == 1:
            assert "Python Software Foundation" in user_content
            return {
                "raw_response": json.dumps({
                    "summary": "Python was created by an organization called the Python Software Foundation.",
                }),
                "prompt_tokens": 11,
                "completion_tokens": 5,
            }
        if len(llm_calls) == 2:
            return {
                "raw_response": json.dumps({
                    "query": "Python Software Foundation headquarters city",
                }),
                "prompt_tokens": 7,
                "completion_tokens": 4,
            }
        if len(llm_calls) == 3:
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

        task = HotpotQAContextTask(retriever=FakeRetriever())
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
        assert '"context"' in traces[2]["user_content"]
        assert '"summary_1"' not in traces[2]["user_content"]
        assert [call["query"] for call in retrieval_calls] == [
            row["question"],
            "Python Software Foundation headquarters city",
        ]
        assert len(llm_calls) == 4
        store.close()
    finally:
        os.unlink(db_path)


def test_hotpotqa_context_score_reports_em_and_f1():
    task = HotpotQAContextTask()
    score, metrics = task.score("The answer is Wilmington.", {"answer": "Wilmington"})

    assert score == 0.0
    assert metrics["exact_match"] == 0.0
    assert metrics["f1"] == 0.5

    score, metrics = task.score("Wilmington", {"answer": "Wilmington"})

    assert score == 1.0
    assert metrics["exact_match"] == 1.0
    assert metrics["f1"] == 1.0


def test_hotpotqa_context_parser_handles_truncated_json():
    # Model hit max_tokens mid-answer: closing `}` missing.
    truncated = '{\n  "reasoning": "some reasoning text",\n  "answer": "Oakland County"\n'
    assert parse_answer(truncated) == "Oakland County"
    truncated_summary = '{\n  "reasoning": "think",\n  "summary": "key facts"\n'
    assert parse_summary(truncated_summary) == "key facts"


def test_hotpotqa_context_parser_serializes_nested_summary_as_json():
    # GEPA+Merge's summarize1 prompt instructs structured output, so the model
    # returns a dict for `summary`. Must be emitted as JSON (readable downstream)
    # rather than Python repr (garbled single-quotes dict-literal).
    raw = (
        '{\n'
        '  "summary": {\n'
        '    "Entity/Person Mention": "J. Searle Dawley, Ken Annakin",\n'
        '    "Direct Answer": {\n'
        '      "J. Searle Dawley": "Directed 149 films."\n'
        '    }\n'
        '  }\n'
        '}'
    )
    parsed = parse_summary(raw)
    # Must be valid JSON that round-trips.
    assert parsed.startswith("{")
    roundtrip = json.loads(parsed)
    assert roundtrip["Entity/Person Mention"] == "J. Searle Dawley, Ken Annakin"
    assert "Direct Answer" in roundtrip
    # Must NOT be Python repr (single quotes).
    assert "'Entity/Person Mention'" not in parsed


def test_hotpotqa_context_parser_preserves_empty_answer():
    # Model said `"answer": ""` — return empty string, not the raw JSON blob.
    raw = '{\n  "reasoning": "cannot determine",\n  "answer": ""\n}'
    assert parse_answer(raw) == ""


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
