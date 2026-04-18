"""test_parser_dispatch_single_identifier.py

Validates BaseTask._validate_dispatch_field() enforcement:
  - Zero registered output_fields → bind() merges in task defaults as a
    fallback. If the merged set still has zero dispatch fields → ValueError.
    (The merge is the "fallback-unless-replaced" output-field contract.)
  - Two+ registered output_fields → ValueError (ambiguous dispatch)
  - Exactly one registered output_field → bind() succeeds
  - No _parser_module_path → bind() succeeds (no registry, no validation)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

_TOOL_DIR = str(Path(__file__).parent.parent.parent.parent)
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from core.func_registry import ROOT_ID, PromptBuildState, make_func_id
from core.store import CubeStore, OnConflict
from task import BaseTask


# ── helpers ───────────────────────────────────────────────────────────

def _make_state_with_output_fields(*field_names: str) -> PromptBuildState:
    """Build a PromptBuildState with the given output_field names."""
    state = PromptBuildState()
    for name in field_names:
        state.output_fields[name] = f"Description of {name}"
    return state


class _TableQALike(BaseTask):
    """Stub that points at the WTQ parsers (dispatch fields: code, sql, answer)."""
    name = "table_qa_stub"
    scorer = "denotation_acc"
    _parser_module_path = "tasks.wtq.parsers"
    default_input_fields = {"question": "The question", "table": "The table"}
    default_output_fields = {"answer": "The answer"}

    def score(self, prediction, query_meta):
        raise NotImplementedError


class _NoRegistryTask(BaseTask):
    """Stub with no _parser_module_path — validation should be skipped."""
    name = "no_registry"
    scorer = "acc"
    default_input_fields = {"question": "The question"}
    default_output_fields = {"answer": "The answer"}

    def parse_response(self, raw_response):
        return raw_response

    def score(self, prediction, query_meta):
        raise NotImplementedError


# ── zero registered fields ────────────────────────────────────────────

def test_zero_registered_declared_fields_fallback_injects_default_dispatch():
    """When a config declares only non-registered output_fields, bind() merges
    the task's default_output_fields in as a fallback. If the defaults include
    a registered field (`answer` for TableQA), bind succeeds with that field
    as the dispatch target and the declared field kept as auxiliary.
    """
    task = _TableQALike()
    # "summary" is not in WTQ DISPATCH_FIELDS; only `answer` from defaults is.
    state = _make_state_with_output_fields("summary")
    task.bind(state)  # should not raise
    assert task._dispatch_field() == "answer"
    assert "summary" in task._prompt_state.semantic.output_fields, (
        "auxiliary feature-added field should be preserved after fallback merge"
    )


class _NoAnswerDefaults(BaseTask):
    """Stub whose defaults ALSO have no registered dispatch field."""
    name = "no_answer_defaults"
    scorer = "acc"
    _parser_module_path = "tasks.wtq.parsers"
    default_input_fields = {"question": "q"}
    default_output_fields = {"summary": "Some summary"}  # not registered

    def score(self, prediction, query_meta):
        raise NotImplementedError


def test_zero_registered_fields_even_after_fallback_raises():
    """When NEITHER the declared fields NOR the task defaults have a
    registered dispatch field, bind() raises `no dispatch parser`.
    """
    task = _NoAnswerDefaults()
    state = _make_state_with_output_fields("summary")
    with pytest.raises(ValueError, match="no dispatch parser"):
        task.bind(state)


# ── two registered fields ─────────────────────────────────────────────

def test_two_registered_fields_raises():
    """bind() raises ValueError when two output_fields both have registered parsers."""
    task = _TableQALike()
    # Both "code" and "answer" are in WTQ DISPATCH_FIELDS
    state = _make_state_with_output_fields("code", "answer")
    with pytest.raises(ValueError, match="ambiguous dispatch"):
        task.bind(state)


def test_three_registered_fields_raises():
    """Three dispatch fields also raise ValueError."""
    task = _TableQALike()
    state = _make_state_with_output_fields("code", "sql", "answer")
    with pytest.raises(ValueError, match="ambiguous dispatch"):
        task.bind(state)


# ── exactly one registered field ──────────────────────────────────────

def test_one_registered_field_succeeds():
    """bind() succeeds when exactly one output_field has a registered parser."""
    task = _TableQALike()
    state = _make_state_with_output_fields("answer")
    task.bind(state)  # no exception
    assert task._prompt_state is not None


def test_one_registered_field_with_auxiliary_succeeds():
    """Auxiliary output fields (not in registry) do not trigger dispatch validation."""
    task = _TableQALike()
    # "reasoning" is not in DISPATCH_FIELDS, so only "answer" counts
    state = _make_state_with_output_fields("reasoning", "answer")
    task.bind(state)  # no exception
    assert task._dispatch_field() == "answer"


def test_code_field_single_dispatch():
    task = _TableQALike()
    state = _make_state_with_output_fields("code")
    task.bind(state)
    assert task._dispatch_field() == "code"


def test_sql_field_single_dispatch():
    task = _TableQALike()
    state = _make_state_with_output_fields("sql")
    task.bind(state)
    assert task._dispatch_field() == "sql"


# ── no registry → skip validation ────────────────────────────────────

def test_no_registry_bind_always_succeeds():
    """Tasks without _parser_module_path skip dispatch validation entirely."""
    task = _NoRegistryTask()
    # Even zero or multiple fields — no error since there's no registry
    state = _make_state_with_output_fields("whatever", "fields")
    task.bind(state)  # no exception
    assert task._get_parser_registry() is None
    assert task._dispatch_field() is None
