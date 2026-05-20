"""HiTab parser registry."""
from __future__ import annotations

from typing import Any, Callable, Dict

from tasks.parsing.code_output import parse_code_output

PARSER_REGISTRY: Dict[str, Callable[[str, Any], str]] = {}
DISPATCH_FIELDS = frozenset({"code", "answer"})


def register_parser(field_name: str) -> Callable:
    def decorator(fn: Callable[[str, Any], str]) -> Callable[[str, Any], str]:
        PARSER_REGISTRY[field_name] = fn
        return fn
    return decorator


@register_parser("code")
def parse_code_field(response_text: str, task: Any) -> str:
    return parse_code_output(response_text, task)


@register_parser("answer")
def parse_answer_field(response_text: str, task: Any) -> str:
    from tasks.hitab.table_qa import _extract_answer

    if task._prompt_state is not None:
        parsed = task._prompt_state.parse_output(response_text)
        if parsed:
            val = parsed.get("answer", "")
            if isinstance(val, list):
                answer = ", ".join(str(v) for v in val).strip()
            else:
                answer = str(val).strip()
            if answer:
                return answer
    return _extract_answer(response_text)
