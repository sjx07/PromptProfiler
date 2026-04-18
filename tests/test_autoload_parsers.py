"""test_autoload_parsers.py

Tests for pkgutil autoload_parsers() and GLOBAL_PARSER_REGISTRY.

Invariants:
  1. After `import prompt_profiler`, GLOBAL_PARSER_REGISTRY contains all 3
     task parser modules (wtq, nl2sql, tabfact).
  2. Registered parsers for each module match the expected dispatch fields.
  3. autoload_parsers() is idempotent (safe to call multiple times).
  4. get_parser_registry(module_path) returns the correct dict.
  5. get_parser_registry(None) returns None.
  6. get_parser_registry("nonexistent") returns None (no crash).
  7. BaseTask._get_parser_registry() returns correct dict without any
     per-call importlib.import_module (uses global registry).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TOOL_DIR = str(Path(__file__).parent.parent.parent.parent)
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)


# ── autoload on package import ────────────────────────────────────────

def test_global_registry_populated_after_import():
    """GLOBAL_PARSER_REGISTRY has entries for all 3 task parser modules after import."""
    import prompt_profiler  # triggers autoload  # noqa: F401
    from core.parser_registry import GLOBAL_PARSER_REGISTRY

    expected_modules = {
        "tasks.wtq.parsers",
        "tasks.nl2sql.parsers",
        "tasks.tabfact.parsers",
    }
    for mod_path in expected_modules:
        assert mod_path in GLOBAL_PARSER_REGISTRY, (
            f"Expected {mod_path} in GLOBAL_PARSER_REGISTRY after autoload"
        )


def test_wtq_dispatch_fields_registered():
    """WTQ parsers module registers exactly {code, sql, answer}."""
    import prompt_profiler  # noqa: F401
    from core.parser_registry import GLOBAL_PARSER_REGISTRY

    reg = GLOBAL_PARSER_REGISTRY["tasks.wtq.parsers"]
    assert set(reg.keys()) == {"code", "sql", "answer"}


def test_nl2sql_dispatch_fields_registered():
    """nl2sql parsers module registers exactly {sql_query}."""
    import prompt_profiler  # noqa: F401
    from core.parser_registry import GLOBAL_PARSER_REGISTRY

    reg = GLOBAL_PARSER_REGISTRY["tasks.nl2sql.parsers"]
    assert set(reg.keys()) == {"sql_query"}


def test_tabfact_dispatch_fields_registered():
    """tabfact parsers module registers exactly {code, verdict}."""
    import prompt_profiler  # noqa: F401
    from core.parser_registry import GLOBAL_PARSER_REGISTRY

    reg = GLOBAL_PARSER_REGISTRY["tasks.tabfact.parsers"]
    assert set(reg.keys()) == {"code", "verdict"}


# ── idempotency ───────────────────────────────────────────────────────

def test_autoload_parsers_idempotent():
    """Calling autoload_parsers() multiple times does not raise or duplicate entries."""
    from core.parser_registry import autoload_parsers, GLOBAL_PARSER_REGISTRY

    count_before = len(GLOBAL_PARSER_REGISTRY)
    autoload_parsers()
    autoload_parsers()
    count_after = len(GLOBAL_PARSER_REGISTRY)
    # Count must be the same (no duplicates or removals)
    assert count_after == count_before


# ── get_parser_registry helper ────────────────────────────────────────

def test_get_parser_registry_returns_correct_dict():
    """get_parser_registry returns PARSER_REGISTRY dict for a known module."""
    import prompt_profiler  # noqa: F401
    from core.parser_registry import get_parser_registry

    reg = get_parser_registry("tasks.wtq.parsers")
    assert reg is not None
    assert callable(reg.get("code"))
    assert callable(reg.get("answer"))


def test_get_parser_registry_none_returns_none():
    """get_parser_registry(None) returns None."""
    from core.parser_registry import get_parser_registry
    assert get_parser_registry(None) is None


def test_get_parser_registry_nonexistent_returns_none():
    """get_parser_registry with a nonexistent module path returns None (no crash)."""
    from core.parser_registry import get_parser_registry
    result = get_parser_registry("tasks.nonexistent_task_xyz.parsers")
    assert result is None


# ── BaseTask._get_parser_registry integration ─────────────────────────

def test_base_task_get_parser_registry_uses_global():
    """BaseTask._get_parser_registry() returns correct dict via global registry."""
    import prompt_profiler  # noqa: F401
    from tasks.wtq.table_qa import TableQA

    task = TableQA()
    reg = task._get_parser_registry()
    assert reg is not None
    assert "code" in reg
    assert "sql" in reg
    assert "answer" in reg


def test_base_task_no_module_path_returns_none():
    """Task without _parser_module_path returns None from _get_parser_registry()."""
    import prompt_profiler  # noqa: F401
    from task import BaseTask

    class _NoParserTask(BaseTask):
        name = "no_parser"
        scorer = "acc"
        def parse_response(self, r): return r
        def score(self, p, m): raise NotImplementedError

    task = _NoParserTask()
    assert task._get_parser_registry() is None


def test_all_task_parser_modules_are_callable():
    """Every registered parser in every module is callable."""
    import prompt_profiler  # noqa: F401
    from core.parser_registry import GLOBAL_PARSER_REGISTRY

    for mod_path, reg in GLOBAL_PARSER_REGISTRY.items():
        for field_name, parser_fn in reg.items():
            assert callable(parser_fn), (
                f"Parser for {mod_path}[{field_name!r}] is not callable"
            )
