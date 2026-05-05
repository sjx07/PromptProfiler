"""HiTab parser registry."""
from __future__ import annotations

import re
import textwrap
from typing import Any, Callable, Dict

PARSER_REGISTRY: Dict[str, Callable[[str, Any], str]] = {}
DISPATCH_FIELDS = frozenset({"code", "answer"})


def register_parser(field_name: str) -> Callable:
    def decorator(fn: Callable[[str, Any], str]) -> Callable[[str, Any], str]:
        PARSER_REGISTRY[field_name] = fn
        return fn
    return decorator


@register_parser("code")
def parse_code_field(response_text: str, task: Any) -> str:
    fenced = _last_markdown_block(response_text)
    if fenced:
        return f"__CODE__{_clean_code(fenced)}"
    if task._prompt_state is not None:
        parsed = task._prompt_state.parse_output(response_text)
        if parsed:
            code = str(parsed.get("code", "")).strip()
            if code:
                return f"__CODE__{_clean_code(code)}"
    return f"__CODE__{_clean_code(response_text)}"


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


def _last_markdown_block(text: str) -> str:
    matches = list(re.finditer(
        r"```(?:[a-zA-Z0-9_+-]*)\s*\n?(.*?)```",
        text or "",
        re.DOTALL,
    ))
    if not matches:
        return ""
    return matches[-1].group(1).strip()


def _clean_code(text: str) -> str:
    return textwrap.dedent((text or "").strip())
