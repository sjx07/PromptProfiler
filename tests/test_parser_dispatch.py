"""test_parser_dispatch.py — unit tests for per-task PARSER_REGISTRY + @register_parser.

Verifies:
  1. @register_parser decorator populates PARSER_REGISTRY correctly
  2. Each registered parser is callable and returns the expected prefix/value
  3. DISPATCH_FIELDS matches the registry keys
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TOOL_DIR = str(Path(__file__).parent.parent.parent.parent)
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)


# ── WTQ parsers ───────────────────────────────────────────────────────

def test_wtq_registry_keys():
    from tasks.wtq.parsers import PARSER_REGISTRY, DISPATCH_FIELDS
    assert set(PARSER_REGISTRY.keys()) == DISPATCH_FIELDS
    assert DISPATCH_FIELDS == frozenset({"code", "sql", "answer"})


def test_wtq_parse_code_returns_prefix(tmp_mock_task):
    from tasks.wtq.parsers import PARSER_REGISTRY
    parser = PARSER_REGISTRY["code"]
    result = parser("df['wins'].sum()", tmp_mock_task)
    assert result.startswith("__CODE__")
    assert "df" in result


def test_wtq_parse_sql_returns_prefix(tmp_mock_task):
    from tasks.wtq.parsers import PARSER_REGISTRY
    parser = PARSER_REGISTRY["sql"]
    result = parser("SELECT MAX(score) FROM t", tmp_mock_task)
    assert result.startswith("__SQL__")
    assert "SELECT" in result


def test_wtq_parse_answer_returns_plain(tmp_mock_task):
    from tasks.wtq.parsers import PARSER_REGISTRY
    parser = PARSER_REGISTRY["answer"]
    result = parser("42", tmp_mock_task)
    assert not result.startswith("__CODE__")
    assert not result.startswith("__SQL__")
    assert "42" in result


def test_wtq_parse_code_markdown_block(tmp_mock_task):
    from tasks.wtq.parsers import PARSER_REGISTRY
    parser = PARSER_REGISTRY["code"]
    response = "```python\ndf['wins'].sum()\n```"
    result = parser(response, tmp_mock_task)
    assert result.startswith("__CODE__")
    assert "df" in result


# ── nl2sql parsers ────────────────────────────────────────────────────

def test_nl2sql_registry_keys():
    from tasks.nl2sql.parsers import PARSER_REGISTRY, DISPATCH_FIELDS
    assert set(PARSER_REGISTRY.keys()) == DISPATCH_FIELDS
    assert DISPATCH_FIELDS == frozenset({"sql_query"})


def test_nl2sql_parse_sql_query(tmp_mock_task):
    from tasks.nl2sql.parsers import PARSER_REGISTRY
    parser = PARSER_REGISTRY["sql_query"]
    result = parser("SELECT id FROM users WHERE age > 30", tmp_mock_task)
    assert "SELECT" in result


def test_nl2sql_parse_sql_query_from_json(tmp_mock_task):
    from tasks.nl2sql.parsers import PARSER_REGISTRY
    import json
    parser = PARSER_REGISTRY["sql_query"]
    payload = json.dumps({"sql_query": "SELECT COUNT(*) FROM t"})
    result = parser(payload, tmp_mock_task)
    assert "SELECT COUNT" in result


# ── tabfact parsers ───────────────────────────────────────────────────

def test_tabfact_registry_keys():
    from tasks.tabfact.parsers import PARSER_REGISTRY, DISPATCH_FIELDS
    assert set(PARSER_REGISTRY.keys()) == DISPATCH_FIELDS
    assert DISPATCH_FIELDS == frozenset({"code", "verdict"})


def test_tabfact_parse_verdict_true(tmp_mock_task):
    from tasks.tabfact.parsers import PARSER_REGISTRY
    parser = PARSER_REGISTRY["verdict"]
    result = parser("True", tmp_mock_task)
    assert result == "True"


def test_tabfact_parse_verdict_false(tmp_mock_task):
    from tasks.tabfact.parsers import PARSER_REGISTRY
    parser = PARSER_REGISTRY["verdict"]
    result = parser("The statement is False.", tmp_mock_task)
    assert result == "False"


def test_tabfact_parse_code_returns_prefix(tmp_mock_task):
    from tasks.tabfact.parsers import PARSER_REGISTRY
    parser = PARSER_REGISTRY["code"]
    result = parser("df['wins'].sum() > 3", tmp_mock_task)
    assert result.startswith("__CODE__")


# ── fixture ───────────────────────────────────────────────────────────

class _MockTask:
    """Minimal task stub — _prompt_state=None so parsers use their fallback paths."""
    _prompt_state = None


@pytest.fixture
def tmp_mock_task():
    return _MockTask()
