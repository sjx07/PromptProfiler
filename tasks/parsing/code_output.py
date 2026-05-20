"""Shared extraction for Python-code output fields."""
from __future__ import annotations

import re
import textwrap
from typing import Any


CODE_PREFIX = "__CODE__"


def parse_code_output(response_text: str, task: Any, *, field_name: str = "code") -> str:
    """Extract Python code and tag it for task scorer execution.

    The task parser registries still own dispatch; this helper keeps the code
    extraction semantics aligned across table tasks.
    """
    fenced = last_markdown_block(response_text)
    if fenced:
        return f"{CODE_PREFIX}{clean_code(fenced)}"

    prompt_state = getattr(task, "_prompt_state", None)
    if prompt_state is not None:
        parsed = prompt_state.parse_output(response_text)
        if parsed:
            code = str(parsed.get(field_name, "")).strip()
            if code:
                return f"{CODE_PREFIX}{clean_code(code)}"

    inner = strip_markdown_block(response_text)
    if inner:
        return f"{CODE_PREFIX}{clean_code(inner)}"
    return f"{CODE_PREFIX}{clean_code(response_text)}"


def last_markdown_block(text: str) -> str:
    """Return the inner text of the last fenced block anywhere in the response."""
    matches = list(re.finditer(
        r"```(?:[a-zA-Z0-9_+-]*)\s*\n?(.*?)```",
        text or "",
        re.DOTALL,
    ))
    if not matches:
        return ""
    return matches[-1].group(1).strip()


def strip_markdown_block(text: str) -> str:
    """Strip a full-response fenced block, if present."""
    lines = str(text or "").strip().splitlines()
    if len(lines) < 2 or not lines[0].strip().startswith("```"):
        return ""
    for end in range(len(lines) - 1, 0, -1):
        if lines[end].strip() == "```":
            return "\n".join(lines[1:end]).strip()
    return ""


def clean_code(text: str) -> str:
    """Normalize generated code without flattening real block indentation."""
    code = textwrap.dedent(text or "").strip()
    lines = code.splitlines()
    if len(lines) <= 1:
        return code

    # JSON-string outputs often preserve one accidental indent on every
    # top-level line after the first. Remove that common post-first indent
    # while preserving relative indentation inside blocks.
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

