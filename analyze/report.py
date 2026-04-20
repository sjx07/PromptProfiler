"""Layer 5 — markdown / text / html rendering for effect tables.

Pure presentation. Given an effect DataFrame + run-metadata, render
human-readable summaries. No SQL, no arithmetic, no filtering.

Public entry point:

  * ``render(df, *, fmt, run_meta, verbose)`` → str (or dict when fmt="both")

Supported ``fmt``:
  * "markdown" — Obsidian / GitHub style
  * "text"     — plain ASCII
  * "html"     — tier-1 self-contained HTML file with client-side
                 filter widgets (sliders for min_effect, confidence_min,
                 min_lift_in_pair; multi-select predicates; sort dropdown).
                 Uses DataTables CDN; no Python server needed.
  * "both"     — {"markdown": str, "text": str}
"""
from __future__ import annotations

import html as _htmlmod
import json as _json
from typing import Any, Dict, List, Optional, Union


def render(
    df,
    *,
    fmt: str,                       # "markdown" | "text" | "html" | "both"
    run_meta: Dict[str, Any],
    verbose: bool = True,
) -> Union[str, Dict[str, str]]:
    """Render an effect DataFrame in the requested format.

    run_meta keys consumed: model, scorer, method, metric, confidence,
    n_bootstrap, sort_by, top_k, confidence_min. Optional:
    ci_level, require_sign, sort_secondary.

    Args:
        verbose: when True (default) include the per-row Interpretation
            section. When False, emit table + caveats only.
    """
    if fmt not in ("markdown", "text", "html", "both"):
        raise ValueError(
            f"fmt must be 'markdown' | 'text' | 'html' | 'both'; got {fmt!r}"
        )

    if fmt == "html":
        return _render_html(df, verbose=verbose, **run_meta)

    md = _render_markdown(df, verbose=verbose, **run_meta)
    if fmt == "markdown":
        return md
    txt = _render_text(df, verbose=verbose, **run_meta)
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

    n_str = f"n={int(r['n_with'])}/{int(r['n_without'])}"
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
                f"P(lift>0)={_fmt_num(r['p_gt_zero'], ndigits=2, signed=False)}, "
                f"{n_str}) — "
                f"_{verdict}_.")
    return (f"**`{canon}`** {phrase} `{pred}` "
            f"(lift {_fmt_num(lift)}, {n_str}).")


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
    verbose: bool = True,
    # Unused-in-markdown extras (accepted to avoid TypeErrors from Pipeline run_meta):
    ci_level=None, require_sign=None, sort_secondary=None,
    min_effect=None, min_lift_in_pair=None,
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

    if verbose:
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
    verbose: bool = True,
    ci_level=None, require_sign=None, sort_secondary=None,
    min_effect=None, min_lift_in_pair=None,
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
        if verbose:
            lines.append("    " + _interpret_row(r, confidence=confidence))

    lines.append("-" * 72)
    lines.append("Caveats:")
    for c in _caveat_bullets(method=method, confidence=confidence):
        lines.append("  " + c.lstrip("- "))
    return "\n".join(lines)


# ── HTML renderer (tier 1: self-contained, client-side filter widgets) ─

_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
<style>
  body {{ font-family: -apple-system, Segoe UI, sans-serif; margin: 2em auto; max-width: 1200px; color: #222; padding: 0 1em; }}
  h1 {{ font-size: 1.4em; margin-bottom: 0.2em; }}
  .meta {{ color: #555; font-size: 0.9em; margin-bottom: 1em; }}
  .meta code {{ background: #f2f2f2; padding: 1px 5px; border-radius: 3px; }}
  .controls {{ background: #fafafa; border: 1px solid #ddd; padding: 1em; border-radius: 4px; margin-bottom: 1em; display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8em; }}
  .ctrl label {{ display: block; font-size: 0.85em; color: #555; margin-bottom: 2px; }}
  .ctrl input[type=range] {{ width: 100%; }}
  .ctrl .val {{ float: right; font-variant-numeric: tabular-nums; color: #333; font-weight: 600; }}
  .ctrl select {{ width: 100%; padding: 4px; }}
  .ctrl.full {{ grid-column: 1 / -1; }}
  table.dataTable {{ font-size: 0.9em; font-variant-numeric: tabular-nums; }}
  table.dataTable td.feat {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  table.dataTable td.pos {{ color: #0a7f3d; }}
  table.dataTable td.neg {{ color: #c32a2a; }}
  .caveats {{ margin-top: 1.5em; font-size: 0.88em; color: #444; }}
  .caveats li {{ margin-bottom: 0.4em; }}
  .interp {{ margin-top: 1em; font-size: 0.88em; color: #333; display: {interp_display}; }}
  .interp li {{ margin-bottom: 0.3em; }}
  .interp code {{ background: #f2f2f2; padding: 1px 4px; border-radius: 3px; }}
</style>
</head>
<body>

<h1>{h1_title}</h1>
<div class="meta">
  <div>model: <code>{model}</code> · scorer: <code>{scorer}</code> · method: <code>{method}</code> · metric: <code>{metric}</code></div>
  <div>{meta_line}</div>
</div>

<div class="controls">
  <div class="ctrl">
    <label>min |effect| <span class="val" id="v_eff">0.000</span></label>
    <input type="range" id="s_eff" min="0" max="0.2" step="0.005" value="0">
  </div>
  <div class="ctrl">
    <label>min P(lift&gt;0) <span class="val" id="v_pgt">0.00</span></label>
    <input type="range" id="s_pgt" min="0" max="1" step="0.05" value="0">
  </div>
  <div class="ctrl">
    <label>min_lift_in_pair (strict &gt;) <span class="val" id="v_pair">—</span></label>
    <input type="range" id="s_pair" min="-0.01" max="0.15" step="0.005" value="-0.01">
  </div>
  <div class="ctrl">
    <label>sort by</label>
    <select id="s_sort">
      <option value="__default__">(default: canonical order)</option>
      <option value="lift">lift</option>
      {did_sort_option}
      {ci_sort_options}
    </select>
  </div>
  <div class="ctrl">
    <label>predicate (filter)</label>
    <select id="s_pred" multiple size="1">
      <option value="__all__" selected>all</option>
      {predicate_options}
    </select>
  </div>
  <div class="ctrl">
    <label>&nbsp;</label>
    <label style="font-size:0.85em;"><input type="checkbox" id="c_verbose" {verbose_checked}> show per-row interpretation</label>
  </div>
</div>

<table id="main" class="display" style="width:100%"></table>

<div class="interp" id="interp">
  <h3 style="font-size:1em;">Interpretation</h3>
  <ul id="interp_list"></ul>
</div>

<div class="caveats">
  <h3 style="font-size:1em;">Caveats</h3>
  <ul>
    {caveat_html}
  </ul>
</div>

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
<script>
const DATA = {data_json};
const HAS_DID = {has_did_js};
const HAS_CI  = {has_ci_js};

function fmt(x, d, signed) {{
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const v = Number(x);
  const s = v.toFixed(d === undefined ? 3 : d);
  if (signed === false) return s;
  return v >= 0 ? "+" + s : s;
}}
function signClass(x) {{
  if (x === null || x === undefined || Number.isNaN(x)) return "";
  return x > 0 ? "pos" : (x < 0 ? "neg" : "");
}}

function buildColumns() {{
  const cols = [
    {{ title: "#", data: null, render: (d, t, r, m) => m.row + 1 }},
    {{ title: "feature", data: "canonical_id", className: "feat" }},
    {{ title: "predicate", data: "predicate_name" }},
    {{ title: "value", data: "predicate_value" }},
    {{ title: "lift", data: "lift", className: "num",
      render: (d, t) => t === "display" ? `<span class="${{signClass(d)}}">${{fmt(d)}}</span>` : d }},
  ];
  if (HAS_DID) cols.push({{ title: "did", data: "did",
      render: (d, t) => t === "display" ? `<span class="${{signClass(d)}}">${{fmt(d)}}</span>` : d }});
  if (HAS_CI) {{
    cols.push({{ title: "95% CI", data: null,
        render: (d) => `[${{fmt(d.ci_lo)}}, ${{fmt(d.ci_hi)}}]` }});
    cols.push({{ title: "P(>0)", data: "p_gt_zero",
        render: (d, t) => t === "display" ? fmt(d, 2, false) : d }});
  }}
  cols.push({{ title: "n_with", data: "n_with", render: d => d ?? "—" }});
  cols.push({{ title: "n_without", data: "n_without", render: d => d ?? "—" }});
  return cols;
}}

// Filter: apply all active controls to DATA, return surviving rows.
function applyFilters() {{
  const effMin  = parseFloat(document.getElementById("s_eff").value);
  const pgtMin  = parseFloat(document.getElementById("s_pgt").value);
  const pairMin = parseFloat(document.getElementById("s_pair").value);
  const sortBy  = document.getElementById("s_sort").value;
  const predSel = [...document.getElementById("s_pred").selectedOptions].map(o => o.value);

  document.getElementById("v_eff").textContent = effMin.toFixed(3);
  document.getElementById("v_pgt").textContent = pgtMin.toFixed(2);
  document.getElementById("v_pair").textContent = pairMin < 0 ? "off" : pairMin.toFixed(3);

  let rows = DATA.slice();

  // Predicate filter.
  if (!predSel.includes("__all__") && predSel.length > 0) {{
    rows = rows.filter(r => predSel.includes(r.predicate_name));
  }}

  // Pair-level filter: group by (canonical_id, predicate_name) and
  // require max(lift) STRICTLY > threshold. Skipped when threshold < 0.
  if (pairMin >= 0) {{
    const groupMax = new Map();
    for (const r of rows) {{
      const k = r.canonical_id + "\\u0001" + r.predicate_name;
      const lift = r.lift;
      if (lift === null || lift === undefined || Number.isNaN(lift)) continue;
      if (!groupMax.has(k) || lift > groupMax.get(k)) groupMax.set(k, lift);
    }}
    rows = rows.filter(r => {{
      const k = r.canonical_id + "\\u0001" + r.predicate_name;
      return groupMax.has(k) && groupMax.get(k) > pairMin;
    }});
  }}

  // Row-level: |effect| >= threshold, P(lift>0) >= threshold.
  const effectCol = HAS_DID ? "did" : "lift";
  rows = rows.filter(r => {{
    const v = r[effectCol];
    if (v === null || v === undefined || Number.isNaN(v)) return effMin === 0;
    return Math.abs(v) >= effMin;
  }});
  if (HAS_CI) {{
    rows = rows.filter(r => {{
      const p = r.p_gt_zero;
      if (p === null || p === undefined || Number.isNaN(p)) return pgtMin === 0;
      return p >= pgtMin;
    }});
  }}

  // Sort.
  if (sortBy !== "__default__") {{
    rows.sort((a, b) => {{
      const va = a[sortBy], vb = b[sortBy];
      if (va === null || va === undefined || Number.isNaN(va)) return 1;
      if (vb === null || vb === undefined || Number.isNaN(vb)) return -1;
      return vb - va;  // descending
    }});
  }}
  return rows;
}}

function refreshInterp(rows) {{
  const ul = document.getElementById("interp_list");
  ul.innerHTML = "";
  for (const r of rows) {{
    const lift = r.lift;
    const mag  = (lift == null || Number.isNaN(lift)) ? 0 : Math.abs(lift);
    let phrase;
    if (mag < 0.01) phrase = "shows no measurable effect on";
    else {{
      const strength = mag >= 0.15 ? "strongly" : mag >= 0.05 ? "notably" : "slightly";
      phrase = strength + " " + (lift > 0 ? "helps" : "hurts") + " on";
    }}
    let verdict = "";
    if (HAS_CI && r.p_gt_zero !== null && r.p_gt_zero !== undefined) {{
      const p = r.p_gt_zero;
      if (mag < 0.01) verdict = "no effect detected";
      else if (lift > 0) verdict = p >= 0.95 ? "high confidence this is real"
                                 : p >= 0.80 ? "moderate confidence"
                                 : "inconclusive — wide CI";
      else verdict = p <= 0.05 ? "high confidence this hurts"
                   : p <= 0.20 ? "moderate confidence of harm"
                               : "inconclusive — wide CI";
    }}
    const ci = HAS_CI ? `, CI [${{fmt(r.ci_lo)}}, ${{fmt(r.ci_hi)}}]` : "";
    const p = HAS_CI ? `, P(>0)=${{fmt(r.p_gt_zero, 2, false)}}` : "";
    const v = verdict ? ` — <i>${{verdict}}</i>` : "";
    const li = document.createElement("li");
    li.innerHTML = `<b><code>${{r.canonical_id}}</code></b> ${{phrase}} <code>${{r.predicate_name}}=${{r.predicate_value}}</code> `
                 + `(lift ${{fmt(lift)}}${{ci}}${{p}}, n=${{r.n_with ?? "—"}}/${{r.n_without ?? "—"}})${{v}}.`;
    ul.appendChild(li);
  }}
}}

let table = null;
function refresh() {{
  const rows = applyFilters();
  if (table) {{ table.clear().rows.add(rows).draw(); }}
  refreshInterp(rows);
}}

$(document).ready(function() {{
  table = $("#main").DataTable({{
    data: DATA,
    columns: buildColumns(),
    pageLength: 25,
    order: []
  }});
  document.querySelectorAll("#s_eff,#s_pgt,#s_pair,#s_sort,#s_pred").forEach(el => {{
    el.addEventListener("input", refresh);
    el.addEventListener("change", refresh);
  }});
  document.getElementById("c_verbose").addEventListener("change", e => {{
    document.getElementById("interp").style.display = e.target.checked ? "block" : "none";
  }});
  refresh();
}});
</script>

</body>
</html>"""


def _render_html(
    df, *, model, scorer, method, metric,
    confidence=False, n_bootstrap=1000,
    sort_by=None, top_k=None, confidence_min=None,
    verbose: bool = True,
    ci_level=None, require_sign=None, sort_secondary=None,
    min_effect=None, min_lift_in_pair=None,
) -> str:
    """Tier-1 self-contained HTML: DataTables + client-side filter widgets.

    The full DataFrame is embedded as JSON; filter/sort controls at the
    top let the reader tune thresholds without re-running Python. Pair-
    level filter grouped by (canonical_id, predicate_name).
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas required for html render") from e

    # Serialize rows. We hand-build each dict so NaN → null in JSON
    # (pandas.to_json sometimes emits "NaN" which is not valid JSON).
    def _clean(v):
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        if hasattr(v, "item"):  # numpy scalar
            try:
                v = v.item()
            except Exception:
                pass
        return v

    records: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        records.append({k: _clean(r[k]) for k in df.columns})

    has_did = "did" in df.columns
    has_ci  = "p_gt_zero" in df.columns
    pair_kind = "paired" if method == "simple" else "unpaired"

    # Meta line echoes the non-default run params.
    meta_bits = [f"confidence: {'on' if confidence else 'off'} "
                 f"({pair_kind} bootstrap, n_boot={n_bootstrap})" if confidence else "confidence: off"]
    if sort_by:          meta_bits.append(f"sort_by: <code>{sort_by}</code>")
    if top_k is not None: meta_bits.append(f"top_k: {top_k}")
    if confidence_min is not None: meta_bits.append(f"confidence_min ≥ {confidence_min:.2f}")
    if min_effect is not None: meta_bits.append(f"min |effect| ≥ {min_effect:.3f}")
    if min_lift_in_pair is not None: meta_bits.append(f"min_lift_in_pair &gt; {min_lift_in_pair:.3f}")
    if require_sign: meta_bits.append(f"require_sign: <code>{require_sign}</code>")
    if ci_level is not None: meta_bits.append(f"ci_level: {ci_level:.2f}")
    meta_line = " · ".join(meta_bits) + f" · rows: {len(df)}"

    # Predicate multi-select options.
    preds = sorted(df["predicate_name"].unique().tolist()) if not df.empty else []
    predicate_options = "\n        ".join(
        f'<option value="{_htmlmod.escape(p)}">{_htmlmod.escape(p)}</option>'
        for p in preds
    )

    did_sort_option = '<option value="did">did</option>' if has_did else ""
    ci_sort_options = (
        '<option value="effect_lb">effect_lb</option>\n      '
        '<option value="p_gt_zero">P(lift&gt;0)</option>'
    ) if has_ci else ""

    caveat_html = "\n    ".join(
        f"<li>{_htmlmod.escape(c.lstrip('- '))}</li>"
        for c in _caveat_bullets(method=method, confidence=confidence)
    )

    title = f"{model} × {scorer}"
    return _HTML_TEMPLATE.format(
        title=_htmlmod.escape(title),
        h1_title=f"Feature × Predicate effect analysis — <code>{_htmlmod.escape(model)}</code>",
        model=_htmlmod.escape(model),
        scorer=_htmlmod.escape(scorer),
        method=_htmlmod.escape(method),
        metric=_htmlmod.escape(metric),
        meta_line=meta_line,
        data_json=_json.dumps(records, default=str),
        has_did_js="true" if has_did else "false",
        has_ci_js="true" if has_ci else "false",
        did_sort_option=did_sort_option,
        ci_sort_options=ci_sort_options,
        predicate_options=predicate_options,
        caveat_html=caveat_html,
        interp_display="block" if verbose else "none",
        verbose_checked="checked" if verbose else "",
    )
