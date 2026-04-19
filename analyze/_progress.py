"""Tiny progress-bar helper.

Wraps ``tqdm.tqdm`` when available + enabled; otherwise returns the
original iterable. Keeps tqdm a soft dependency and keeps the call
sites readable.
"""
from __future__ import annotations

from typing import Iterable, Optional


def progress_iter(
    iterable: Iterable,
    *,
    enable: bool = False,
    total: Optional[int] = None,
    desc: str = "",
):
    """Yield from ``iterable``, optionally wrapped in a tqdm bar.

    Args:
        iterable: the thing to iterate.
        enable:   set True to turn on the bar. Defaults to False (no-op).
        total:    hint for length (tqdm shows ETA when known).
        desc:     short label shown to the left of the bar.
    """
    if not enable:
        return iterable
    try:
        from tqdm import tqdm  # type: ignore
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)
