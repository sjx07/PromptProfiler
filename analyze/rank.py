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
    min_effect: Optional[float] = None,
    min_lift_in_pair: Optional[float] = None,
    metric: str = "lift",
    confidence: bool = False,
):
    """Filter + sort + top_k.

    Order of operations:
      1. ``confidence_min`` filter (per-row probability gate — drops
         rows with ``p_gt_zero`` below threshold; requires
         ``confidence=True``).
      2. ``min_lift_in_pair`` filter (pair-level — groups by
         ``(canonical_id, predicate_name)`` and drops whole groups
         whose MAX lift across predicate values is **≤** threshold;
         comparison is strict ``>``). Useful for saying "only show
         interaction cells where the feature helps *somewhere* on this
         predicate" (threshold=0.0 keeps pairs with at least one
         strictly-positive lift) or "helps noticeably somewhere"
         (threshold=0.05). Runs BEFORE ``min_effect`` so pair
         decisions use all rows.
      3. ``min_effect`` filter (per-row magnitude gate — drops rows
         with ``|metric_col| < min_effect``, where ``metric_col`` is
         ``"lift"`` or ``"did"`` per the ``metric`` kwarg).
      4. ``sort_by`` (descending, NaN last).
      5. ``top_k`` head cut.

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

    # ── min_lift_in_pair (pair-level) filter ──
    # Groups are (canonical_id, predicate_name). We keep a group iff
    # max(lift across its rows) >= threshold. Runs BEFORE min_effect
    # so the "does this feature help SOMEWHERE on this predicate?"
    # decision sees the full pair, not a version already trimmed by
    # the per-row magnitude gate.
    if min_lift_in_pair is not None:
        if "lift" not in df.columns:
            raise ValueError("min_lift_in_pair requires a 'lift' column")
        group_max = df.groupby(
            ["canonical_id", "predicate_name"], dropna=False,
        )["lift"].transform("max")
        # Strict > for consistency with the "lift > 0" intent: threshold=0.0
        # keeps only pairs where at least one side genuinely helps.
        df = df[group_max.fillna(float("-inf")) > min_lift_in_pair].reset_index(drop=True)

    # ── min_effect (magnitude) filter ──
    # Named "min_effect" because it targets whichever effect column is
    # active — lift or did — so users don't have to swap kwarg names
    # when they switch metrics.
    if min_effect is not None:
        if min_effect < 0:
            raise ValueError(f"min_effect must be >= 0; got {min_effect!r}")
        col = "did" if metric == "did" else "lift"
        if col not in df.columns:
            raise ValueError(
                f"min_effect targets '{col}' column but it's not in the DataFrame"
            )
        df = df[df[col].abs().fillna(-1.0) >= min_effect].reset_index(drop=True)

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
