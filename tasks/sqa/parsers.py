"""SQA parser registry.

Dispatch fields:
  "code"   - model returns Python code; executor runs it against the table
  "answer" - model returns a direct answer string
"""
from __future__ import annotations

import json
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
    from tasks.sqa.sequential_qa import _extract_answer

    if task._prompt_state is not None:
        parsed = task._prompt_state.parse_output(response_text)
        if parsed:
            val = parsed.get("answer", "")
            if isinstance(val, list):
                answer = json.dumps([str(v) for v in val], ensure_ascii=False)
            else:
                answer = str(val).strip()
            if answer:
                return answer
    return _extract_answer(response_text)
