"""Layer 4 — sort, filter, top_k.

Input: an effect DataFrame (from ``effect.lift_*`` possibly via
``effect.did`` and/or ``confidence.attach_ci``).
Output: the same DataFrame, sorted/filtered/truncated per user kwargs.

Validation rules are enforced here (e.g. sort_by='did' requires a did
column present).
"""
from __future__ import annotations

from typing import Optional

ALLOWED_SORTS = {"lift", "did", "effect_lb", "p_gt_zero"}


def rank(
    df_in,
    *,
    sort_by: Optional[str] = None,
    top_k: Optional[int] = None,
    confidence_min: Optional[float] = None,
    metric: str = "lift",
    confidence: bool = False,
):
    """Filter + sort + top_k.

    Order of operations:
      1. ``confidence_min`` filter (drops rows with ``p_gt_zero`` below
         threshold — requires ``confidence=True``).
      2. ``sort_by`` (descending, NaN last).
      3. ``top_k`` head cut.

    Default sort (``sort_by=None``) is alphabetical by
    ``(canonical_id, predicate_name, predicate_value)``.
    """
    df = df_in

    # ── confidence_min filter ──
    if confidence_min is not None:
        if not confidence:
            raise ValueError("confidence_min requires confidence=True")
        if not (0.0 <= confidence_min <= 1.0):
            raise ValueError(f"confidence_min must be in [0,1]; got {confidence_min!r}")
        df = df[df["p_gt_zero"].fillna(-1.0) >= confidence_min].reset_index(drop=True)

    # ── sort ──
    if sort_by is not None:
        if sort_by not in ALLOWED_SORTS:
            raise ValueError(
                f"sort_by must be one of {sorted(ALLOWED_SORTS)}; got {sort_by!r}"
            )
        if sort_by == "did" and metric != "did":
            raise ValueError("sort_by='did' requires metric='did'")
        if sort_by in ("effect_lb", "p_gt_zero") and not confidence:
            raise ValueError(f"sort_by={sort_by!r} requires confidence=True")
        df = df.sort_values(sort_by, ascending=False, na_position="last").reset_index(drop=True)
    else:
        df = df.sort_values(
            ["canonical_id", "predicate_name", "predicate_value"]
        ).reset_index(drop=True)

    # ── top_k ──
    if top_k is not None:
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0; got {top_k!r}")
        df = df.head(top_k).reset_index(drop=True)

    return df
