"""Shared answer normalization helpers for denotation-style scorers."""
from __future__ import annotations

import re


_GROUPED_DIGIT_SPACE_RE = re.compile(r"(?<=\d) (?=\d{3}(?:\D|$))")


def normalize_numeric_grouping(value: str) -> str:
    """Collapse common thousands separators inside numeric tokens.

    This handles comma grouping (``4,000``) and space grouping
    (``4 000``, ``4\u00a0000``, ``4\u202f000``) without removing ordinary
    spaces from non-numeric answers such as names or phrases.
    """
    text = value
    for ch in ("\u00a0", "\u202f", "\u2007", "\u2009"):
        text = text.replace(ch, " ")
    text = _GROUPED_DIGIT_SPACE_RE.sub("", text)
    return text.replace(",", "")
