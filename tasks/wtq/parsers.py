"""WTQ / TableQA parser registry — one registered parser per dispatch output_field.

Each parser is registered for exactly one output_field name.  The task dispatcher
(BaseTask.parse_response) finds the single registered field present in the config's
output_fields and routes the raw LLM response to the matching parser.

Dispatch fields (drive task execution path):
  "code"   — model returns Python; executor runs it against the DataFrame
  "sql"    — model returns SQL; executor runs it against an in-memory SQLite table
  "answer" — model returns a direct answer string

Auxiliary fields (parsed/recorded but do NOT trigger dispatch):
  "reasoning" — CoT trace; extracted but not dispatched
"""
from __future__ import annotations

import re
import json
import textwrap
from typing import Any, Callable, Dict

PARSER_REGISTRY: Dict[str, Callable[[str, Any], str]] = {}

# Fields that drive dispatch (exactly one must be present per config)
DISPATCH_FIELDS = frozenset({"code", "sql", "answer"})


def register_parser(field_name: str) -> Callable:
    """Decorator — register a parser for a single dispatch output_field."""
    def decorator(fn: Callable[[str, Any], str]) -> Callable[[str, Any], str]:
        PARSER_REGISTRY[field_name] = fn
        return fn
    return decorator


@register_parser("code")
def parse_code_field(response_text: str, task: Any) -> str:
    """Extract Python code from LLM response; tag with __CODE__ prefix for executor."""
    fenced = _last_markdown_block(response_text)
    if fenced:
        return f"__CODE__{_clean_code(fenced)}"
    if task._prompt_state is not None:
        parsed = task._prompt_state.parse_output(response_text)
        if parsed:
            code = str(parsed.get("code", "")).strip()
            if code:
                return f"__CODE__{_clean_code(code)}"
    # Fallback: strip markdown code block
    inner = _strip_markdown_block(response_text)
    if inner:
        return f"__CODE__{_clean_code(inner)}"
    return f"__CODE__{_clean_code(response_text)}"


@register_parser("sql")
def parse_sql_field(response_text: str, task: Any) -> str:
    """Extract SQL query from LLM response; tag with __SQL__ prefix for executor."""
    if task._prompt_state is not None:
        parsed = task._prompt_state.parse_output(response_text)
        if parsed:
            sql = str(parsed.get("sql", "")).strip()
            if sql:
                return f"__SQL__{sql}"
    inner = _strip_markdown_block(response_text)
    if inner:
        return f"__SQL__{inner}"
    return f"__SQL__{response_text.strip()}"


@register_parser("answer")
def parse_answer_field(response_text: str, task: Any) -> str:
    """Extract direct answer from LLM response."""
    if task._prompt_state is not None:
        parsed = task._prompt_state.parse_output(response_text)
        if parsed:
            answer = str(parsed.get("answer", "")).strip()
            if answer:
                return answer
    return _extract_answer_fallback(response_text)


# ── private helpers ───────────────────────────────────────────────────

def _strip_markdown_block(text: str) -> str:
    """Strip fenced code block markers and return inner content, or ''."""
    lines = text.strip().splitlines()
    if len(lines) < 2 or not lines[0].strip().startswith("```"):
        return ""
    for end in range(len(lines) - 1, 0, -1):
        if lines[end].strip() == "```":
            return "\n".join(lines[1:end]).strip()
    return ""


def _clean_code(text: str) -> str:
    """Normalize generated code without flattening real block indentation."""
    code = textwrap.dedent((text or "").strip())
    lines = code.splitlines()
    if len(lines) <= 1:
        return code

    # JSON-string outputs often preserve a single accidental indent on every
    # top-level line after the first. Remove the common post-first indent while
    # preserving relative indentation inside blocks.
    rest = [line for line in lines[1:] if line.strip()]
    first_code_line = lines[0].rstrip()
    if rest and not first_code_line.endswith(":") and all(line[:1].isspace() for line in rest):
        min_indent = min(len(line) - len(line.lstrip()) for line in rest)
        if min_indent > 0:
            lines = [lines[0]] + [
                line[min_indent:] if line.strip() else line
                for line in lines[1:]
            ]
    return "\n".join(lines).strip()


def _last_markdown_block(text: str) -> str:
    """Return the inner text of the last fenced block anywhere in the response."""
    matches = list(re.finditer(
        r"```(?:[a-zA-Z0-9_+-]*)\s*\n?(.*?)```",
        text or "",
        re.DOTALL,
    ))
    if not matches:
        return ""
    return matches[-1].group(1).strip()


def _extract_answer_fallback(text: str) -> str:
    """Fallback answer extractor when structured parsing fails."""
    text = text.strip()
    # Try JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            for key in ("answer", "result", "value", "response"):
                if key in parsed:
                    val = parsed[key]
                    if isinstance(val, list):
                        return ", ".join(str(v) for v in val)
                    return str(val).strip()
        if isinstance(parsed, list):
            return ", ".join(str(v) for v in parsed)
    except (json.JSONDecodeError, TypeError):
        pass
    # Structured pattern
    m = re.search(
        r"(?:answer|result|value)\s*[:\-]\s*(.+?)(?:\n|$)",
        text, re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().rstrip(".")
    # Last line heuristic
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    if lines:
        last = lines[-1]
        for prefix in ["The answer is", "Answer:", "Therefore,", "So,", "Thus,", "The result is"]:
            if last.lower().startswith(prefix.lower()):
                return last[len(prefix):].strip().rstrip(".")
        if len(last) < 200:
            return last
    return text
