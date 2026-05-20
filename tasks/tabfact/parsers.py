"""TabFact / FactVerification parser registry.

Dispatch fields:
  "code"    — model returns Python boolean expression; executor evaluates it
  "verdict" — model returns True/False verdict string directly
"""
from __future__ import annotations

from typing import Any, Callable, Dict

from tasks.parsing.code_output import parse_code_output

PARSER_REGISTRY: Dict[str, Callable[[str, Any], str]] = {}

DISPATCH_FIELDS = frozenset({"code", "verdict"})


def register_parser(field_name: str) -> Callable:
    """Decorator — register a parser for a single dispatch output_field."""
    def decorator(fn: Callable[[str, Any], str]) -> Callable[[str, Any], str]:
        PARSER_REGISTRY[field_name] = fn
        return fn
    return decorator


@register_parser("code")
def parse_code_field(response_text: str, task: Any) -> str:
    return parse_code_output(response_text, task)


@register_parser("verdict")
def parse_verdict_field(response_text: str, task: Any) -> str:
    """Extract True/False verdict from LLM response."""
    from tasks.tabfact.fact_verification import _extract_verdict
    if task._prompt_state is not None:
        parsed = task._prompt_state.parse_output(response_text)
        if parsed:
            verdict = str(parsed.get("verdict", "")).strip()
            if verdict:
                return _extract_verdict(verdict)
    return _extract_verdict(response_text)
