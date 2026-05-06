"""test_code_block_format.py — fenced-code-block format style.

The `code_block` format style exists to avoid JSON's escape tax for
features that emit a single code blob (enable_code, enable_sql). Parser
extracts the last fenced block; renderer instructs the model to reply
with one fence. No escape juggling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TOOL_DIR = str(Path(__file__).parent.parent.parent.parent)
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from prompt_profiler.prompt.format_styles import CodeBlockStyle, FORMAT_STYLES


CODE_FIELDS = {"code": "Python code that computes the answer."}
SQL_FIELDS = {"sql": "SQL query that answers the question."}


# ── registry wiring ────────────────────────────────────────────────────

def test_code_block_registered():
    assert "code_block" in FORMAT_STYLES
    assert isinstance(FORMAT_STYLES["code_block"], CodeBlockStyle)


# ── system message template ────────────────────────────────────────────

def test_structure_template_python_fence_for_code():
    style = CodeBlockStyle()
    out = style.format_structure_template({}, CODE_FIELDS)
    assert "```python" in out
    assert "```" in out


def test_structure_template_sql_fence_for_sql():
    style = CodeBlockStyle()
    out = style.format_structure_template({}, SQL_FIELDS)
    assert "```sql" in out


def test_structure_template_rejects_multiple_output_fields():
    style = CodeBlockStyle()
    with pytest.raises(ValueError, match="exactly one output_field"):
        style.format_structure_template({}, {"code": "d", "reasoning": "r"})


def test_system_message_skips_empty_rule_sections():
    from prompt.rules import RuleItem, RuleSection, RuleTree
    from prompt.semantic_content import SemanticContent

    empty = RuleSection(title="rules", node_id="empty")
    nonempty = RuleSection(
        title="reasoning",
        children=[RuleItem(text="Write executable code.", node_id="r1")],
        node_id="reasoning",
    )
    tree = RuleTree(roots=[empty, nonempty])
    tree._rebuild_index()
    semantic = SemanticContent(
        rule_sections=[empty, nonempty],
        tree=tree,
        input_fields={"table": "The table."},
        output_fields=CODE_FIELDS,
    )

    out = CodeBlockStyle().format_system_message(semantic)
    assert "## rules" not in out
    assert "## reasoning" in out
    assert "- Write executable code." in out


# ── parse_output ───────────────────────────────────────────────────────

def test_parse_extracts_simple_fenced_block():
    style = CodeBlockStyle()
    resp = "```python\nprint(42)\n```"
    out = style.parse_output(resp, CODE_FIELDS)
    assert out == {"code": "print(42)"}


def test_parse_extracts_multiline_block():
    style = CodeBlockStyle()
    resp = (
        "```python\n"
        "import pandas as pd\n"
        "df = pd.DataFrame(table['rows'])\n"
        "print(df.iloc[0]['Team'])\n"
        "```\n"
    )
    out = style.parse_output(resp, CODE_FIELDS)
    assert "pandas" in out["code"]
    assert "print(df" in out["code"]
    # The code must contain multiple lines — that's the whole point.
    assert out["code"].count("\n") >= 2


def test_parse_ignores_preamble_and_uses_last_block():
    """Many models emit prose before the code. We keep the last fence."""
    style = CodeBlockStyle()
    resp = (
        "Let me think about this.\n\n"
        "```python\n"
        "# wrong first attempt\n"
        "print('no')\n"
        "```\n\n"
        "Actually let me reconsider.\n\n"
        "```python\n"
        "print('yes')\n"
        "```\n"
    )
    out = style.parse_output(resp, CODE_FIELDS)
    assert out["code"] == "print('yes')"


def test_parse_strips_think_block():
    style = CodeBlockStyle()
    resp = (
        "<think>I should use pandas here</think>\n"
        "```python\n"
        "print('ok')\n"
        "```\n"
    )
    out = style.parse_output(resp, CODE_FIELDS)
    assert out["code"] == "print('ok')"


def test_parse_handles_fence_without_language_tag():
    style = CodeBlockStyle()
    resp = "```\nprint(1)\n```"
    out = style.parse_output(resp, CODE_FIELDS)
    assert out["code"] == "print(1)"


def test_parse_charitable_fallback_when_fence_missing():
    """If the model forgets the fence entirely, use the whole response."""
    style = CodeBlockStyle()
    resp = "print('no fence')"
    out = style.parse_output(resp, CODE_FIELDS)
    assert out["code"] == "print('no fence')"


def test_parse_rejects_multi_field_request():
    style = CodeBlockStyle()
    with pytest.raises(ValueError, match="exactly one output_field"):
        style.parse_output("```\nx\n```", {"code": "c", "reasoning": "r"})


# ── render_output (for few-shot examples) ──────────────────────────────

def test_render_output_emits_language_tagged_fence():
    style = CodeBlockStyle()
    rendered = style.render_output({"code": "print(42)"})
    assert rendered == "```python\nprint(42)\n```"


def test_render_output_for_sql():
    style = CodeBlockStyle()
    rendered = style.render_output({"sql": "SELECT 1"})
    assert rendered == "```sql\nSELECT 1\n```"


# ── end-to-end through set_format ──────────────────────────────────────

def test_set_format_code_block_produces_code_block_style():
    """The set_format primitive 'code_block' routes through FORMAT_STYLES
    and into PromptState.format_style — a smoke check that it composes
    end-to-end.
    """
    from prompt_profiler.core.func_registry import (
        ROOT_ID, PromptBuildState, apply_config, make_func_id,
    )
    from prompt_profiler.core.store import CubeStore, OnConflict
    import tempfile

    # Build a tiny config: one rule + one output_field(code) + set_format(code_block)
    rule_params = {
        "node_type": "rule", "parent_id": ROOT_ID,
        "payload": {"content": "Write Python code."},
    }
    of_params = {
        "node_type": "output_field", "parent_id": ROOT_ID,
        "payload": {"name": "code", "description": "Python code."},
    }
    fmt_params = {"style": "code_block"}

    specs = [
        {"func_id": make_func_id("insert_node", rule_params),
         "func_type": "insert_node", "params": rule_params, "meta": {}},
        {"func_id": make_func_id("insert_node", of_params),
         "func_type": "insert_node", "params": of_params,  "meta": {}},
        {"func_id": make_func_id("set_format", fmt_params),
         "func_type": "set_format", "params": fmt_params, "meta": {}},
    ]

    db = tempfile.mktemp(suffix=".db")
    try:
        store = CubeStore(db)
        store.upsert_funcs(specs, on_conflict=OnConflict.SKIP)
        state = apply_config([s["func_id"] for s in specs], store)
        assert state.format_style == "code_block"

        ps = state.to_prompt_state()
        assert ps.format_style_name == "code_block"

        # Round-trip through parse_output with a real LLM-shaped response.
        parsed = ps.parse_output("```python\nprint(1+1)\n```")
        assert parsed == {"code": "print(1+1)"}
        store.close()
    finally:
        Path(db).unlink(missing_ok=True)
