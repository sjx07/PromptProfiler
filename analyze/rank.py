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
    require_sign: Optional[str] = None,      # "positive" | "negative" | "any" | None
    sort_secondary: Optional[object] = None, # str or list[str] — tie-breaker
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
    effect_col = "did" if metric == "did" else "lift"

    # ── filter pipeline ──
    # Each entry: (user-supplied value, mask_fn(df, value) → boolean Series).
    # mask_fn validates its inputs and raises ValueError on bad config.
    # Pair-level filter runs BEFORE magnitude so the "does this feature
    # help SOMEWHERE on this predicate?" decision sees the full pair.
    def _by_p_gt_zero(df, t):
        if not confidence:
            raise ValueError("confidence_min requires confidence=True")
        if not (0.0 <= t <= 1.0):
            raise ValueError(f"confidence_min must be in [0,1]; got {t!r}")
        return df["p_gt_zero"].fillna(-1.0) >= t

    def _by_pair_max_lift(df, t):
        if "lift" not in df.columns:
            raise ValueError("min_lift_in_pair requires a 'lift' column")
        gm = df.groupby(
            ["canonical_id", "predicate_name"], dropna=False,
        )["lift"].transform("max")
        # Strict > so threshold=0.0 keeps pairs with at least one row
        # genuinely > 0, not just ties at zero.
        return gm.fillna(float("-inf")) > t

    def _by_magnitude(df, t):
        if t < 0:
            raise ValueError(f"min_effect must be >= 0; got {t!r}")
        if effect_col not in df.columns:
            raise ValueError(
                f"min_effect targets '{effect_col}' column but it's not in the DataFrame"
            )
        return df[effect_col].abs().fillna(-1.0) >= t

    def _by_sign(df, sign):
        if sign not in ("positive", "negative", "any"):
            raise ValueError(
                f"require_sign must be 'positive' | 'negative' | 'any'; got {sign!r}"
            )
        if sign == "any":
            return df[effect_col].notna() | df[effect_col].isna()  # all True
        if effect_col not in df.columns:
            raise ValueError(
                f"require_sign targets '{effect_col}' column but it's not in the DataFrame"
            )
        # Drop NaN rows when a sign is required (ambiguous sign).
        if sign == "positive":
            return df[effect_col].fillna(float("-inf")) > 0
        return df[effect_col].fillna(float("+inf")) < 0

    for value, mask_fn in (
        (confidence_min, _by_p_gt_zero),
        (min_lift_in_pair, _by_pair_max_lift),
        (min_effect, _by_magnitude),
        (require_sign, _by_sign),
    ):
        if value is None:
            continue
        df = df[mask_fn(df, value)].reset_index(drop=True)

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

        # Assemble sort key list. Primary is sort_by (desc). Secondary
        # breaks ties — caller passes a string or a list of strings; each
        # must be a column present in the DataFrame. Default ascending
        # for secondary (stable, predictable), but caller can pass tuples
        # (col, ascending_bool) for explicit control.
        keys: list = [sort_by]
        ascending: list = [False]
        if sort_secondary is not None:
            sec_list = [sort_secondary] if isinstance(sort_secondary, (str, tuple)) else list(sort_secondary)
            for item in sec_list:
                if isinstance(item, tuple):
                    col, asc = item
                else:
                    col, asc = item, True
                if col not in df.columns:
                    raise ValueError(
                        f"sort_secondary column '{col}' not in DataFrame"
                    )
                keys.append(col)
                ascending.append(asc)
        df = df.sort_values(keys, ascending=ascending, na_position="last").reset_index(drop=True)
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
