"""test_multiline_code_parse.py — regression tests for the `code`-field parser.

Bug: LLM emits `{"code": "import json\\nimport pandas\\n..."}` with LITERAL
newlines inside the JSON string value. `json.loads` rejects it, and the
fallback key-value regex (.+? terminated at \\n) captured only the first
line — so the executor ran `import json` alone and returned nothing.

Fix: `JSONStyle.parse_output` and `fallback_parse_output` now try a
newline-repair step (raw \\n inside string values → escaped \\\\n) before
falling through to the line-terminated regex; a second strategy extracts
quoted fields directly and tolerates multi-line values.

These tests use real response samples observed during a WTQ pilot run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TOOL_DIR = str(Path(__file__).parent.parent.parent.parent)
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from prompt_profiler.prompt.format_styles import (
    JSONStyle,
    fallback_parse_output,
    _repair_unescaped_newlines_in_json,
    _extract_quoted_field,
)


OUTPUT_FIELDS = {"code": "Python code that computes the answer."}


# ── repair primitive ───────────────────────────────────────────────────

def test_repair_preserves_valid_json():
    s = '{"code": "print(\\"hi\\")"}'
    assert _repair_unescaped_newlines_in_json(s) == s


def test_repair_escapes_raw_newlines_inside_string():
    raw = '{"code": "a\nb\nc"}'
    repaired = _repair_unescaped_newlines_in_json(raw)
    assert repaired == '{"code": "a\\nb\\nc"}'


def test_repair_does_not_touch_newlines_outside_strings():
    raw = '{\n  "code": "a\nb"\n}'
    repaired = _repair_unescaped_newlines_in_json(raw)
    # Newlines between tokens (outside strings) stay.
    # Newlines inside the string become \n.
    assert '"a\\nb"' in repaired
    assert "\n  " in repaired  # the indentation newline is preserved


# ── extract_quoted_field primitive ─────────────────────────────────────

def test_extract_single_line_value():
    s = '{"code": "print(42)"}'
    assert _extract_quoted_field(s, "code") == "print(42)"


def test_extract_multiline_value():
    s = '{"code": "import json\nimport pandas\nprint(next_team)"}'
    val = _extract_quoted_field(s, "code")
    assert val is not None
    assert val.startswith("import json")
    assert "pandas" in val
    assert val.endswith("print(next_team)")


def test_extract_handles_escaped_quotes():
    s = '{"code": "print(\\"hi\\")"}'
    assert _extract_quoted_field(s, "code") == 'print("hi")'


def test_extract_returns_none_when_field_absent():
    assert _extract_quoted_field('{"other": 1}', "code") is None


# ── JSONStyle.parse_output end-to-end (real LLM samples) ───────────────

SAMPLE_1 = '''{
  "code": "import json
import pandas as pd

# Load the table from the provided JSON
table = json.loads(data['table'])

# Convert the table to a pandas DataFrame
df = pd.DataFrame(table['rows'])

# Find the index of the team 'Widnes Vikings'
index = df[df['Team'] == 'Widnes Vikings'].index[0]

# Get the next team
next_team = df.iloc[index + 1]['Team']

print(next_team)"
}'''


SAMPLE_2_OUTPUTFIELDS_WRAPPER = '''{
  "OutputFields": {
    "code": "import pandas as pd
import json

# Load the table data
table_data = json.loads(table['rows'])
print(first_official['Olympics'].values[0].split(' ')[2])"
  }
}'''


SAMPLE_3_SHORT_CODE = '{"code": "print(\\"The rider who came in first is Doriano Romboni.\\")"}'


def test_json_style_parses_multiline_code_sample_1():
    js = JSONStyle()
    parsed = js.parse_output(SAMPLE_1, OUTPUT_FIELDS)
    code = parsed.get("code")
    assert code is not None, f"code field missing from parse output: {parsed}"
    # Must include the FULL body, not just the first line.
    assert "import json" in code
    assert "import pandas" in code
    assert "print(next_team)" in code
    # Critically: more than the first line.
    assert code.count("\n") >= 5, (
        f"expected multi-line code, got single line: {code!r}"
    )


def test_json_style_parses_single_line_code():
    js = JSONStyle()
    parsed = js.parse_output(SAMPLE_3_SHORT_CODE, OUTPUT_FIELDS)
    assert parsed.get("code") == 'print("The rider who came in first is Doriano Romboni.")'


def test_json_style_handles_outputfields_wrapper_by_falling_through():
    """Sample 2 wraps the payload in {"OutputFields": {"code": "..."}}.

    Since `code` is nested and unescaped newlines break naive json.loads,
    fallback_parse_output's quoted-field extractor should still recover it.
    """
    js = JSONStyle()
    parsed = js.parse_output(SAMPLE_2_OUTPUTFIELDS_WRAPPER, OUTPUT_FIELDS)
    code = parsed.get("code")
    assert code is not None
    assert "import pandas" in code


# ── parse_code_field end-to-end ────────────────────────────────────────

def test_parse_code_field_returns_full_multiline_code():
    """Simulate the WTQ parser's full path with a Task stub."""
    from prompt_profiler.tasks.wtq.parsers import parse_code_field

    class _StubPromptState:
        def parse_output(self, response_text):
            return JSONStyle().parse_output(response_text, OUTPUT_FIELDS)

    class _StubTask:
        _prompt_state = _StubPromptState()

    tagged = parse_code_field(SAMPLE_1, _StubTask())
    assert tagged.startswith("__CODE__")
    body = tagged[len("__CODE__"):]
    # The big regression: body used to be just "import json". Must be full.
    assert "pandas" in body
    assert "print(next_team)" in body
