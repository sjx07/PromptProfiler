"""SkVM parser registry — one dispatch parser per task output_field.

All SkVM tasks use a single dispatch output_field called ``answer``. The
LLM response is passed through with minimal normalization (whitespace
strip, fenced-code-block strip). Structural evaluation happens later in
the task's ``score()`` method, not here.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict

PARSER_REGISTRY: Dict[str, Callable[[str, Any], str]] = {}

# Fields that drive BaseTask dispatch. Must match exactly one entry in the
# active config's output_fields at bind() time.
DISPATCH_FIELDS = frozenset({"answer"})


def register_parser(field_name: str) -> Callable:
    """Decorator — register a parser for a single dispatch output_field."""
    def decorator(fn: Callable[[str, Any], str]) -> Callable[[str, Any], str]:
        PARSER_REGISTRY[field_name] = fn
        return fn
    return decorator


@register_parser("answer")
def parse_answer(response_text: str, task: Any) -> str:
    """Minimal parser — strip whitespace and a single fenced code block."""
    # Prefer structured parse when the prompt state knows the format.
    if getattr(task, "_prompt_state", None) is not None:
        try:
            parsed = task._prompt_state.parse_output(response_text)
        except Exception:
            parsed = None
        if parsed and isinstance(parsed, dict):
            val = parsed.get("answer")
            if val is not None:
                return str(val).strip()

    text = response_text.strip()
    m = re.match(r"^```(?:json)?\s*\n(.+?)\n```\s*$", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text
