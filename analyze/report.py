"""Layer 5 — markdown / text rendering for effect tables.

Pure presentation. Given an effect DataFrame + run-metadata, render
human-readable summaries. No SQL, no arithmetic, no filtering.

Public entry point:

  * ``render(df, *, fmt, run_meta)`` → str (or dict when fmt="both")
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union


def render(
    df,
    *,
    fmt: str,                       # "markdown" | "text" | "both"
    run_meta: Dict[str, Any],
) -> Union[str, Dict[str, str]]:
    """Render an effect DataFrame in the requested format.

    run_meta keys consumed: model, scorer, method, metric, confidence,
    n_bootstrap, sort_by, top_k, confidence_min.
    """
    if fmt not in ("markdown", "text", "both"):
        raise ValueError(f"fmt must be 'markdown' | 'text' | 'both'; got {fmt!r}")

    md = _render_markdown(df, **run_meta)
    if fmt == "markdown":
        return md
    txt = _render_text(df, **run_meta)
    if fmt == "text":
        return txt
    return {"markdown": md, "text": txt}


# ── formatting helpers ────────────────────────────────────────────────

def _fmt_num(x, ndigits: int = 3, *, signed: bool = True) -> str:
    try:
        import pandas as pd
        if pd.isna(x):
            return "—"
    except Exception:
        pass
    if x is None:
        return "—"
    try:
        spec = f"+.{ndigits}f" if signed else f".{ndigits}f"
        return format(float(x), spec)
    except (TypeError, ValueError):
        return str(x)


def _interpret_row(r, *, confidence: bool) -> str:
    """One-sentence interpretation of a ranking row."""
    try:
        import pandas as pd
        def _nan(x): return x is None or (isinstance(x, float) and pd.isna(x))
    except Exception:  # pragma: no cover
        def _nan(x): return x is None

    canon = r["canonical_id"]
    pred  = f"{r['predicate_name']}={r['predicate_value']}"
    lift  = r["lift"]
    magnitude = abs(lift) if not _nan(lift) else 0.0

    if _nan(lift) or magnitude < 0.01:
        phrase = "shows no measurable effect on"
    else:
        strength = (
            "strongly" if magnitude >= 0.15 else
            "notably"  if magnitude >= 0.05 else
            "slightly"
        )
        direction = "helps" if lift > 0 else "hurts"
        phrase = f"{strength} {direction} on"

    if confidence and not _nan(r.get("p_gt_zero")):
        p = r["p_gt_zero"]
        if _nan(lift) or magnitude < 0.01:
            verdict = "no effect detected"
        elif lift > 0:
            verdict = (
                "high confidence this is real"  if p >= 0.95 else
                "moderate confidence"           if p >= 0.80 else
                "inconclusive — wide CI"
            )
        else:
            verdict = (
                "high confidence this hurts"    if p <= 0.05 else
                "moderate confidence of harm"   if p <= 0.20 else
                "inconclusive — wide CI"
            )
        ci = f"CI [{_fmt_num(r.get('ci_lo'))}, {_fmt_num(r.get('ci_hi'))}]"
        return (f"**`{canon}`** {phrase} `{pred}` "
                f"(lift {_fmt_num(lift)}, {ci}, "
                f"P(lift>0)={_fmt_num(r['p_gt_zero'], ndigits=2, signed=False)}) — "
                f"_{verdict}_.")
    return (f"**`{canon}`** {phrase} `{pred}` "
            f"(lift {_fmt_num(lift)}, n={int(r['n_with'])}/{int(r['n_without'])}).")


def _caveat_bullets(*, method: str, confidence: bool) -> List[str]:
    out = [
        "- Effects are within-cube and conditional on `(model, scorer)`; "
        "they do not generalize to other models without re-running.",
        "- No multiple-testing correction is applied. When sweeping many "
        "(feature × predicate) cells, expect ~5% of uncorrelated nulls to "
        "cross a 95% threshold by chance.",
    ]
    if method == "simple":
        out.append(
            "- `method=simple` measures a **controlled** add-one effect. "
            "It holds the rest of the prompt at baseline, so the number "
            "is a feature-in-isolation estimate, not 'how this feature "
            "behaves inside a real stack of features'."
        )
    else:
        out.append(
            "- `method=marginal` pools across all configs containing the "
            "feature. Useful as an average uplift, but confounded by "
            "which other features those configs happened to contain."
        )
    if confidence:
        out.append(
            "- `p_gt_zero` is the fraction of bootstrap resamples where "
            "`mean_with > mean_without`. It is NOT a p-value. A value "
            "of 0.90 means 'in 90% of resamples the feature looked "
            "better' — treat 0.90 as weak-positive, 0.95+ as solid."
        )
        if method == "simple":
            out.append(
                "- Paired bootstrap assumes each query_id appears under "
                "both the base and the feature-only config. Confirmed "
                "via shared-id intersection."
            )
        else:
            out.append(
                "- Unpaired bootstrap resamples per-config means; it "
                "needs ≥ 2 configs per side. Rows failing this show NaN."
            )
    return out


# ── markdown renderer ────────────────────────────────────────────────

def _render_markdown(
    df, *, model, scorer, method, metric,
    confidence=False, n_bootstrap=1000,
    sort_by=None, top_k=None, confidence_min=None,
) -> str:
    lines: List[str] = []
    lines.append("# Feature × Predicate effect analysis")
    lines.append("")
    lines.append(f"- **model:** `{model}`")
    lines.append(f"- **scorer:** `{scorer}`")
    lines.append(f"- **method:** `{method}` "
                 f"({'paired add-one' if method == 'simple' else 'pooled-per-config'})")
    lines.append(f"- **metric:** `{metric}`")
    if confidence:
        pair_kind = "paired" if method == "simple" else "unpaired"
        lines.append(f"- **confidence:** 95% {pair_kind} bootstrap CI "
                     f"(n_boot={n_bootstrap}); `p_gt_zero` = fraction of "
                     f"resamples where lift > 0.")
    if sort_by:
        lines.append(f"- **sort_by:** `{sort_by}` (descending)")
    if top_k is not None:
        lines.append(f"- **top_k:** {top_k}")
    if confidence_min is not None:
        lines.append(f"- **confidence_min:** p_gt_zero ≥ {confidence_min:.2f}")
    lines.append(f"- **rows:** {len(df)}")
    lines.append("")

    if df.empty:
        lines.append("_No rows after filtering._")
        return "\n".join(lines)

    lines.append("## Ranking")
    lines.append("")
    has_did = "did" in df.columns
    header = ["#", "feature", "predicate", "value", "lift"]
    if has_did:
        header.append("did")
    if confidence:
        header += ["95% CI", "P(lift>0)"]
    header += ["n_with", "n_without"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        row = [
            str(i),
            f"`{r['canonical_id']}`",
            f"`{r['predicate_name']}`",
            f"`{r['predicate_value']}`",
            _fmt_num(r["lift"]),
        ]
        if has_did:
            row.append(_fmt_num(r.get("did")))
        if confidence:
            row.append(f"[{_fmt_num(r.get('ci_lo'))}, {_fmt_num(r.get('ci_hi'))}]")
            row.append(_fmt_num(r.get("p_gt_zero"), ndigits=2, signed=False))
        row.append(str(int(r["n_with"])))
        row.append(str(int(r["n_without"])))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    for _, r in df.iterrows():
        lines.append("- " + _interpret_row(r, confidence=confidence))
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.extend(_caveat_bullets(method=method, confidence=confidence))
    return "\n".join(lines)


# ── text renderer ─────────────────────────────────────────────────────

def _render_text(
    df, *, model, scorer, method, metric,
    confidence=False, n_bootstrap=1000,
    sort_by=None, top_k=None, confidence_min=None,
) -> str:
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("FEATURE x PREDICATE effect analysis")
    lines.append("=" * 72)
    lines.append(f"model:   {model}")
    lines.append(f"scorer:  {scorer}")
    lines.append(f"method:  {method} "
                 f"({'paired add-one' if method == 'simple' else 'pooled-per-config'})")
    lines.append(f"metric:  {metric}")
    if confidence:
        pair_kind = "paired" if method == "simple" else "unpaired"
        lines.append(f"CI:      95% {pair_kind} bootstrap (n_boot={n_bootstrap}); "
                     f"p_gt_zero = P(lift > 0) over resamples")
    if sort_by:
        lines.append(f"sort_by: {sort_by} (desc)")
    if top_k is not None:
        lines.append(f"top_k:   {top_k}")
    if confidence_min is not None:
        lines.append(f"conf>=:  {confidence_min:.2f}")
    lines.append(f"rows:    {len(df)}")
    lines.append("-" * 72)

    if df.empty:
        lines.append("No rows after filtering.")
        return "\n".join(lines)

    has_did = "did" in df.columns
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        head = (
            f"[{i}] {r['canonical_id']}  "
            f"{r['predicate_name']}={r['predicate_value']}  "
            f"lift={_fmt_num(r['lift'])}  "
            f"n_with={int(r['n_with'])}  n_without={int(r['n_without'])}"
        )
        if has_did:
            head += f"  did={_fmt_num(r.get('did'))}"
        if confidence:
            head += (f"  CI=[{_fmt_num(r.get('ci_lo'))},"
                     f"{_fmt_num(r.get('ci_hi'))}]"
                     f"  P(>0)={_fmt_num(r.get('p_gt_zero'), ndigits=2, signed=False)}")
        lines.append(head)
        lines.append("    " + _interpret_row(r, confidence=confidence))

    lines.append("-" * 72)
    lines.append("Caveats:")
    for c in _caveat_bullets(method=method, confidence=confidence):
        lines.append("  " + c.lstrip("- "))
    return "\n".join(lines)
