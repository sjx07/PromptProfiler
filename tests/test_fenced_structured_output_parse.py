"""Regression tests for structured outputs wrapped in Markdown fences."""
from __future__ import annotations

import sys
from pathlib import Path

_TOOL_DIR = str(Path(__file__).parent.parent.parent.parent)
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from prompt_profiler.prompt.format_styles import (  # noqa: E402
    PlainStyle,
    YAMLStyle,
    fallback_parse_output,
)


OUTPUT_FIELDS = {"answer": "final answer"}


def test_yaml_style_strips_enclosing_yaml_fence():
    parsed = YAMLStyle().parse_output("```yaml\nanswer: Tom Adams\n```", OUTPUT_FIELDS)
    assert parsed == {"answer": "Tom Adams"}


def test_plain_style_strips_enclosing_fence():
    parsed = PlainStyle().parse_output("```\nanswer: Tom Adams\n```", OUTPUT_FIELDS)
    assert parsed == {"answer": "Tom Adams"}


def test_fallback_parse_strips_enclosing_yaml_fence():
    parsed = fallback_parse_output("```yaml\nanswer: Tom Adams\n```", OUTPUT_FIELDS)
    assert parsed == {"answer": "Tom Adams"}


def test_inner_fence_inside_value_is_preserved():
    response = "answer: use ```x``` as the literal token"
    parsed = YAMLStyle().parse_output(response, OUTPUT_FIELDS)
    assert parsed == {"answer": "use ```x``` as the literal token"}
