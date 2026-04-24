"""PUPA/PAPILLON compound task integration tests."""
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
from tasks.pupa.loaders import seed_queries_pupa
from tasks.pupa.pupa import (
    PupaPrivacyDelegationTask,
    parse_final_response,
    parse_redacted_request,
)


def test_pupa_loader_seeds_split_from_jsonl():
    rows = [
        {
            "id": "train-1",
            "split": "train",
            "user_query": "Private training query",
            "reference": "Training reference",
        },
        {
            "id": "test-1",
            "split": "test",
            "user_query": "Private test query",
            "reference": "Test reference",
            "forbidden_terms": ["Private test"],
        },
    ]

    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as data_f:
        for row in rows:
            data_f.write(json.dumps(row) + "\n")
        data_path = data_f.name
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_f:
        db_path = db_f.name

    try:
        store = CubeStore(db_path)
        n = seed_queries_pupa(store, data_path, split="test", on_conflict=OnConflict.ERROR)
        seeded = store._get_conn().execute("SELECT * FROM query").fetchall()

        assert n == 1
        assert len(seeded) == 1
        assert seeded[0]["dataset"] == "pupa"
        assert seeded[0]["content"] == "Private test query"

        meta = json.loads(seeded[0]["meta"])
        assert meta["split"] == "test"
        assert meta["reference"] == "Test reference"
        assert meta["forbidden_terms"] == ["Private test"]
        assert meta["_raw"]["id"] == "test-1"
        store.close()
    finally:
        os.unlink(data_path)
        os.unlink(db_path)


def test_pupa_base_features_route_to_trusted_modules():
    reg = FeatureRegistry.load("pupa")
    specs, _ = reg.materialize([
        "_section_privacy_rewrite",
        "base_craft_redacted_request",
        "_section_response_synthesis",
        "base_respond_to_query",
    ])

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_f:
        db_path = db_f.name
    try:
        store = CubeStore(db_path)
        seed_funcs(store, specs, on_conflict=OnConflict.ERROR)
        states = apply_config_modules(
            [spec["func_id"] for spec in specs],
            store,
            module_names=PupaPrivacyDelegationTask.module_names(),
        )

        assert states["craft_redacted_request"].sections
        assert states["respond_to_query"].sections
        assert not states["__main__"].sections
        store.close()
    finally:
        os.unlink(db_path)


def test_pupa_run_traces_external_call_without_private_query_leakage():
    calls = []

    def fake_llm(system_prompt: str, user_content: str) -> dict:
        calls.append({"system": system_prompt, "user": user_content})
        if len(calls) == 1:
            assert "Alice" in user_content
            return {
                "raw_response": json.dumps({
                    "reasoning": "Removed the client's name.",
                    "llm_request": "Give general advice for a client communication.",
                }),
                "prompt_tokens": 10,
                "completion_tokens": 4,
            }
        if len(calls) == 2:
            assert "Alice" not in user_content
            assert "general advice" in user_content
            return {
                "raw_response": "Use a concise and professional tone.",
                "prompt_tokens": 5,
                "completion_tokens": 6,
            }
        assert "Alice" in user_content
        assert "Use a concise" in user_content
        return {
            "raw_response": json.dumps({
                "response": "Alice should receive a concise professional reply.",
            }),
            "prompt_tokens": 9,
            "completion_tokens": 7,
        }

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as db_f:
        db_path = db_f.name
    try:
        store = CubeStore(db_path)
        store.upsert_queries(
            [{
                "query_id": "pupa-q1",
                "dataset": "pupa",
                "content": "Draft a reply to my client Alice about the renewal.",
                "meta": {"split": "test"},
            }],
            on_conflict=OnConflict.ERROR,
        )
        config_id = store.get_or_create_config([])

        task = PupaPrivacyDelegationTask()
        task.bind_modules({})
        run_config(
            store,
            config_id,
            [{
                "query_id": "pupa-q1",
                "dataset": "pupa",
                "content": "Draft a reply to my client Alice about the renewal.",
                "meta": json.dumps({"split": "test"}),
            }],
            task,
            "fake-model",
            fake_llm,
        )

        row = store.get_cached_execution(config_id, "pupa-q1", "fake-model")
        meta = json.loads(row["meta"])
        traces = meta["module_traces"]

        assert row["prediction"] == "Alice should receive a concise professional reply."
        assert row["prompt_tokens"] == 24
        assert row["completion_tokens"] == 17
        assert [t["module_name"] for t in traces] == [
            "craft_redacted_request",
            "external_llm",
            "respond_to_query",
        ]
        assert "Alice" not in traces[1]["user_content"]
        assert len(calls) == 3
        store.close()
    finally:
        os.unlink(db_path)


def test_pupa_score_uses_reference_and_forbidden_terms_when_available():
    task = PupaPrivacyDelegationTask()
    score, metrics = task.score(
        "Send a concise professional reply.",
        {
            "reference": "Send a concise professional reply.",
            "forbidden_terms": ["Alice"],
        },
    )

    assert score == 1.0
    assert metrics["status"] == "ok"
    assert metrics["exact_match"] == 1.0
    assert metrics["leak_count"] == 0

    score, metrics = task.score(
        "Alice should get a reply.",
        {
            "reference": "Send a concise professional reply.",
            "forbidden_terms": ["Alice"],
        },
    )

    assert score == 0.0
    assert metrics["leak_count"] == 1


def test_pupa_parsers_strip_qwen_thinking_blocks():
    redacted = parse_redacted_request(
        '<think>\nNeed to remove names.\n</think>\n'
        '{"reasoning":"removed name","llm_request":"Give general renewal advice."}'
    )
    final = parse_final_response(
        '<think>\nUse the external answer.\n</think>\n'
        '{"response":"Send a concise renewal reply."}'
    )

    assert redacted == {
        "reasoning": "removed name",
        "llm_request": "Give general renewal advice.",
    }
    assert final == "Send a concise renewal reply."

    assert parse_final_response("<think>\nTruncated reasoning") == ""
