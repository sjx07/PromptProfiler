#!/usr/bin/env python3
"""Probe a primitive canonical context-attribute v1 over WikiTable tasks.

This is an exploratory analysis script, not a runtime extractor. It checks
whether a shared primitive vocabulary is representable from input-only fields
for WTQ, SQA, TableBench, TabFact, and HiTab, then reports prevalence and
near-duplicate atom overlaps.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATASETS = ("wtq", "sqa", "tablebench", "tab_fact", "hitab")

STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "its",
    "of", "on", "or", "that", "the", "their", "there", "this", "to", "was",
    "were", "what", "when", "where", "which", "who", "with",
}


PATTERNS = {
    "intent.count": re.compile(r"\b(how many|number of|count(?:s|ed|ing)?|total number)\b", re.I),
    "intent.superlative_rank": re.compile(
        r"\b(most|least|highest|lowest|best|worst|largest|smallest|greatest|fewest|"
        r"maximum|minimum|top|bottom|longest|shortest|oldest|newest|first|last)\b",
        re.I,
    ),
    "intent.comparison": re.compile(
        r"\b(more|less|greater|fewer|higher|lower|better|worse|bigger|smaller|"
        r"longer|shorter|older|newer|earlier|later)\s+than\b",
        re.I,
    ),
    "intent.arithmetic": re.compile(
        r"\b(difference|subtract|add|multiply|divide|ratio|percent|percentage|"
        r"how much more|how much less|how many more|how many fewer)\b",
        re.I,
    ),
    "intent.reduction_noncount": re.compile(
        r"\b(total|sum|average|mean|combined|altogether)\b",
        re.I,
    ),
    "intent.temporal": re.compile(
        r"\b(before|after|during|between|year|month|day|date|season|century|decade|when|"
        r"january|february|march|april|may|june|july|august|september|october|november|december|"
        r"\d{4})\b",
        re.I,
    ),
    "intent.negation_marker": re.compile(
        r"\b(not|no|never|neither|none|nobody|nothing|nowhere|nor|without|"
        r"didn't|doesn't|don't|wasn't|weren't|isn't|aren't|hasn't|haven't|"
        r"hadn't|won't|wouldn't|couldn't|shouldn't)\b",
        re.I,
    ),
    "text.has_number": re.compile(r"\b\d+(?:\.\d+)?\b"),
    "text.has_year": re.compile(r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b"),
    "text.has_percent": re.compile(r"\d+(?:\.\d+)?\s*%|\bpercent(?:age)?\b", re.I),
    "text.has_currency": re.compile(r"[$£€¥]|\b(?:dollars?|pounds?|euros?)\b", re.I),
    "text.has_ordinal": re.compile(r"\b(?:first|second|third|fourth|fifth|last|\d+(?:st|nd|rd|th))\b", re.I),
    "text.has_quoted_span": re.compile(r"(['\"]).+?\1"),
    "dialog.has_reference_marker": re.compile(
        r"\b(those|that|these|them|they|it|its|their|the same|which one|which ones|above|previous|mentioned)\b",
        re.I,
    ),
}


HEADER_PATTERNS = {
    "schema.has_date_col": re.compile(r"\b(date|year|month|day|season|week|time)\b", re.I),
    "schema.has_rank_col": re.compile(r"^(#|rank|place|position|seed|round)$|\brank\b", re.I),
    "schema.has_score_col": re.compile(r"\b(score|points?|goals?|result|won|lost|drawn|difference|bonus)\b", re.I),
    "schema.has_unit_col": re.compile(r"[%$£€¥]|\b(km|kg|miles?|area|population|density|rate|percent|percentage)\b", re.I),
    "schema.has_entity_col": re.compile(
        r"\b(name|player|team|country|city|club|person|film|title|municipality|province|state|school|company)\b",
        re.I,
    ),
}


CELL_PATTERNS = {
    "cell.has_empty": re.compile(r"^$"),
    "cell.has_comma_number": re.compile(r"\b\d{1,3}(?:,\d{3})+\b"),
    "cell.has_percent": re.compile(r"\d+(?:\.\d+)?\s*%"),
    "cell.has_currency": re.compile(r"[$£€¥]"),
    "cell.has_range_or_score": re.compile(r"\b\d+\s*[-–]\s*\d+\b|\b\d+\.\d+\s*\(\d+\)"),
    "cell.has_date_like": re.compile(
        r"\b(?:1[5-9]\d{2}|20\d{2}|21\d{2})\b|"
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b",
        re.I,
    ),
    "cell.has_mixed_numeric_text": re.compile(r"(?=.*\d)(?=.*[A-Za-z]).+"),
}


def tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[A-Za-z0-9]+", text.lower())
        if len(t) > 1 and t not in STOP
    }


def as_meta(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        return json.loads(raw)
    return raw or {}


def flatten_table(raw: dict[str, Any], dataset: str) -> tuple[list[str], list[list[str]]]:
    if dataset == "sqa":
        table = raw.get("table", {})
        return list(table.get("headers", table.get("header", [])) or []), list(table.get("rows", []) or [])
    if dataset == "tab_fact":
        table_text = raw.get("table_text", "")
        rows = []
        for line in str(table_text).splitlines():
            parts = [p for p in line.split("#") if p != ""]
            if parts:
                rows.append(parts)
        if not rows:
            return [], []
        return rows[0], rows[1:]
    if dataset == "hitab":
        table_content = raw.get("table_content", {})
        texts = table_content.get("texts", []) if isinstance(table_content, dict) else []
        if not texts:
            return [], []
        return list(texts[0] or []), list(texts[1:] or [])
    table = raw.get("table", {})
    return list(table.get("header", table.get("headers", [])) or []), list(table.get("rows", []) or [])


def question_text(raw: dict[str, Any], dataset: str) -> str:
    if dataset == "tab_fact":
        return str(raw.get("statement", "") or "")
    return str(raw.get("question", "") or "")


def bin_count(n: int, cuts: tuple[int, ...]) -> str:
    prev = 0
    for cut in cuts:
        if n <= cut:
            return f"{prev + 1}_{cut}" if prev else f"0_{cut}"
        prev = cut
    return f"gt_{cuts[-1]}"


def numeric_like(value: str) -> bool:
    cleaned = re.sub(r"[$£€¥,%]", "", value.strip())
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned))


def canonical_atoms(meta: dict[str, Any], dataset: str) -> dict[str, str]:
    raw = as_meta(meta.get("_raw", {}))
    text = question_text(raw, dataset)
    header, rows = flatten_table(raw, dataset)
    header_text = " ".join(map(str, header))
    flat_cells = [str(cell).strip() for row in rows[:80] for cell in row[:50]]
    flat_text = " ".join(flat_cells)

    out: dict[str, str] = {}

    # Intent and text literals.
    for name, pat in PATTERNS.items():
        if name.startswith("dialog."):
            continue
        out[name] = "yes" if pat.search(text) else "no"

    # Schema surface.
    norm_headers = [str(h).strip().lower() for h in header]
    for name, pat in HEADER_PATTERNS.items():
        out[name] = "yes" if any(pat.search(str(h)) for h in header) else "no"
    out["schema.has_duplicate_headers"] = "yes" if len(norm_headers) != len(set(norm_headers)) else "no"

    # Table numeric shape.
    n_rows = len(rows)
    n_cols = len(header)
    out["table.rows_bin"] = bin_count(n_rows, (5, 10, 25, 50))
    out["table.cols_bin"] = bin_count(n_cols, (3, 6, 10))
    out["table.shape"] = "empty"
    if n_rows and n_cols:
        ratio = n_cols / n_rows
        out["table.shape"] = "wide" if ratio > 0.5 else ("tall" if ratio < 0.1 else "balanced")
    numeric_cols = 0
    for ci in range(n_cols):
        vals = [str(rows[ri][ci]).strip() for ri in range(min(n_rows, 50)) if ci < len(rows[ri]) and str(rows[ri][ci]).strip()]
        if vals and sum(numeric_like(v) for v in vals) / len(vals) > 0.5:
            numeric_cols += 1
    out["table.numeric_cols_bin"] = "0" if numeric_cols == 0 else ("1" if numeric_cols == 1 else ("2_3" if numeric_cols <= 3 else "ge_4"))
    out["table.numeric_density_bin"] = "none"
    if n_cols:
        density = numeric_cols / n_cols
        out["table.numeric_density_bin"] = "none" if density == 0 else ("low" if density < 0.25 else ("mid" if density < 0.6 else "high"))

    # Cell surface. Empty cells need direct row scan, not flat text only.
    for name, pat in CELL_PATTERNS.items():
        if name == "cell.has_empty":
            out[name] = "yes" if any(str(cell).strip() == "" for row in rows[:80] for cell in row[:50]) else "no"
        else:
            out[name] = "yes" if pat.search(flat_text) else "no"

    # Lightweight grounding overlap.
    q_tokens = tokens(text)
    h_tokens = tokens(header_text)
    c_tokens = tokens(flat_text)
    out["grounding.header_overlap"] = "yes" if q_tokens & h_tokens else "no"
    out["grounding.cell_overlap"] = "yes" if q_tokens & c_tokens else "no"

    # Dialog state.
    if dataset == "sqa":
        pos = int(raw.get("position", meta.get("position", 0)) or 0)
        out["dialog.turn_bin"] = "first" if pos == 0 else ("early" if pos <= 2 else "late")
        out["dialog.has_reference_marker"] = "yes" if PATTERNS["dialog.has_reference_marker"].search(text) else "no"
    else:
        out["dialog.turn_bin"] = "na"
        out["dialog.has_reference_marker"] = "na"

    # Native metadata, marked but not shared transfer primitives.
    if dataset == "tablebench":
        out["native.tablebench_qtype"] = str(raw.get("qtype", meta.get("qtype", "unknown")) or "unknown")
        out["native.tablebench_qsubtype"] = str(raw.get("qsubtype", meta.get("qsubtype", "unknown")) or "unknown")
    elif dataset == "hitab":
        out["native.hitab_agg_type"] = str(raw.get("aggregation", meta.get("aggregation", "unknown")) or "unknown")
        out["native.hitab_source_family"] = str(raw.get("table_source", meta.get("table_source", "unknown")) or "unknown")

    return out


@dataclass
class DatasetReport:
    dataset: str
    n: int
    attr_values: dict[str, Counter]


def read_reports(db_path: Path, limit_per_dataset: int = 0) -> list[DatasetReport]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    reports: list[DatasetReport] = []
    for dataset in DATASETS:
        sql = "SELECT meta FROM query WHERE dataset=? ORDER BY query_id"
        params: list[Any] = [dataset]
        if limit_per_dataset:
            sql += " LIMIT ?"
            params.append(limit_per_dataset)
        values: dict[str, Counter] = defaultdict(Counter)
        n = 0
        for row in conn.execute(sql, params):
            meta = as_meta(row["meta"])
            atoms = canonical_atoms(meta, dataset)
            n += 1
            for k, v in atoms.items():
                values[k][v] += 1
        reports.append(DatasetReport(dataset=dataset, n=n, attr_values=dict(values)))
    return reports


def read_atom_sets(db_path: Path, limit_per_dataset: int = 0) -> dict[str, dict[str, set[int]]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    out: dict[str, dict[str, set[int]]] = {}
    for dataset in DATASETS:
        atom_sets: dict[str, set[int]] = defaultdict(set)
        sql = "SELECT meta FROM query WHERE dataset=? ORDER BY query_id"
        params: list[Any] = [dataset]
        if limit_per_dataset:
            sql += " LIMIT ?"
            params.append(limit_per_dataset)
        for idx, row in enumerate(conn.execute(sql, params)):
            atoms = canonical_atoms(as_meta(row["meta"]), dataset)
            for k, v in atoms.items():
                if v == "yes":
                    atom_sets[f"{k}=yes"].add(idx)
                elif not v.startswith("no") and v != "na" and not k.startswith("native."):
                    atom_sets[f"{k}={v}"].add(idx)
        out[dataset] = dict(atom_sets)
    return out


def near_duplicates(atom_sets: dict[str, set[int]], min_support: int, threshold: float) -> list[tuple[float, str, str, int, int, int]]:
    items = [(k, v) for k, v in atom_sets.items() if len(v) >= min_support]
    rows: list[tuple[float, str, str, int, int, int]] = []
    for i, (a, sa) in enumerate(items):
        for b, sb in items[i + 1:]:
            inter = len(sa & sb)
            union = len(sa | sb)
            if not union:
                continue
            jac = inter / union
            if jac >= threshold:
                rows.append((jac, a, b, len(sa), len(sb), inter))
    return sorted(rows, reverse=True)


def family_of(attr: str) -> str:
    return attr.split(".", 1)[0]


def esc(x: Any) -> str:
    return html.escape(str(x))


def prevalence_rows(reports: list[DatasetReport]) -> list[str]:
    rows: list[str] = []
    all_attrs = sorted({a for r in reports for a in r.attr_values if not a.startswith("native.")})
    for attr in all_attrs:
        cells = []
        for report in reports:
            counter = report.attr_values.get(attr)
            if not counter:
                cells.append('<td class="missing">not represented</td>')
                continue
            if set(counter) == {"na"}:
                cells.append('<td class="missing">not applicable</td>')
                continue
            if set(counter) <= {"yes", "no"}:
                yes = counter.get("yes", 0)
                pct = yes / report.n * 100 if report.n else 0
                cells.append(f"<td>{yes}/{report.n}<br><span>{pct:.1f}% yes</span></td>")
            else:
                top = counter.most_common(3)
                text = "<br>".join(f"{esc(k)}: {v}" for k, v in top)
                cells.append(f"<td>{text}</td>")
        rows.append(f"<tr><th>{esc(attr)}</th><td>{esc(family_of(attr))}</td>{''.join(cells)}</tr>")
    return rows


def is_applicable(counter: Counter) -> bool:
    return bool(counter) and set(counter) != {"na"}


def is_informative(counter: Counter, n: int) -> bool:
    if not is_applicable(counter) or n <= 0:
        return False
    if set(counter) <= {"yes", "no"}:
        yes_pct = counter.get("yes", 0) / n
        return 0.02 <= yes_pct <= 0.98
    top = counter.most_common(1)[0][1] / n
    return len(counter) > 1 and top < 0.98


def write_html(path: Path, reports: list[DatasetReport], atom_sets: dict[str, dict[str, set[int]]]) -> None:
    datasets = [r.dataset for r in reports]
    shared_attrs = sorted(
        attr for attr in {a for r in reports for a in r.attr_values}
        if not attr.startswith("native.")
    )
    by_dataset = {r.dataset: r for r in reports}
    representable_counts = {
        d: sum(
            1
            for attr in shared_attrs
            if is_applicable(by_dataset[d].attr_values.get(attr, Counter()))
        )
        for d in datasets
    }
    informative_counts = {
        d: sum(
            1
            for attr in shared_attrs
            if is_informative(by_dataset[d].attr_values.get(attr, Counter()), by_dataset[d].n)
        )
        for d in datasets
    }
    family_counts: dict[str, Counter] = defaultdict(Counter)
    for report in reports:
        for attr in report.attr_values:
            if not attr.startswith("native."):
                family_counts[report.dataset][family_of(attr)] += 1

    dup_sections = []
    for dataset in datasets:
        dups = near_duplicates(atom_sets[dataset], min_support=max(20, int(next(r.n for r in reports if r.dataset == dataset) * 0.02)), threshold=0.95)
        if not dups:
            body = '<p class="muted">No near-duplicate positive atoms above threshold.</p>'
        else:
            trs = []
            for jac, a, b, na, nb, inter in dups[:12]:
                trs.append(f"<tr><td>{jac:.3f}</td><td><code>{esc(a)}</code></td><td><code>{esc(b)}</code></td><td>{na}</td><td>{nb}</td><td>{inter}</td></tr>")
            body = "<table><thead><tr><th>Jaccard</th><th>Atom A</th><th>Atom B</th><th>n A</th><th>n B</th><th>intersection</th></tr></thead><tbody>" + "".join(trs) + "</tbody></table>"
        dup_sections.append(f"<section><h2>{esc(dataset)} overlap audit</h2>{body}</section>")

    cards = []
    for report in reports:
        fam = " ".join(f'<span class="pill">{esc(k)}: {v}</span>' for k, v in sorted(family_counts[report.dataset].items()))
        cards.append(
            f"<section class='card'><h2>{esc(report.dataset)}</h2>"
            f"<p><b>queries:</b> {report.n}</p>"
            f"<p><b>applicable shared attrs:</b> {representable_counts[report.dataset]} / {len(shared_attrs)}</p>"
            f"<p><b>informative candidates:</b> {informative_counts[report.dataset]} / {len(shared_attrs)}</p>"
            f"<p>{fam}</p></section>"
        )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Canonical context attributes v1 probe</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; background: #f8f9fb; }}
    h1 {{ margin-bottom: 4px; }}
    h2 {{ font-size: 18px; margin-top: 22px; }}
    .muted, td span {{ color: #666; font-size: 12px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin: 18px 0; }}
    .card {{ background: white; border: 1px solid #ddd; border-radius: 6px; padding: 12px; }}
    .pill {{ display: inline-block; padding: 2px 6px; margin: 2px; border: 1px solid #ddd; border-radius: 999px; background: #f2f4f7; font-size: 12px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #e2e4e8; padding: 7px 8px; vertical-align: top; text-align: left; font-size: 13px; }}
    thead th {{ background: #eef1f5; }}
    tbody th {{ background: #fafafa; font-family: ui-monospace, Menlo, Consolas, monospace; font-weight: 500; }}
    code {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }}
    .missing {{ color: #999; background: #fbfbfb; }}
  </style>
</head>
<body>
  <h1>Canonical Context Attributes v1 Probe</h1>
  <p class="muted">Input-only primitive context attributes over WTQ, SQA, TableBench, TabFact, and HiTab. Native metadata is excluded from shared counts. Applicable excludes all-NA atoms; informative candidates exclude saturated or near-constant atoms.</p>
  <div class="cards">{''.join(cards)}</div>
  <section>
    <h2>Representability and Prevalence</h2>
    <table>
      <thead><tr><th>canonical attribute</th><th>family</th>{''.join(f'<th>{esc(d)}</th>' for d in datasets)}</tr></thead>
      <tbody>{''.join(prevalence_rows(reports))}</tbody>
    </table>
  </section>
  {''.join(dup_sections)}
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_doc, encoding="utf-8")


def write_markdown(path: Path, reports: list[DatasetReport], atom_sets: dict[str, dict[str, set[int]]]) -> None:
    shared_attrs = sorted(attr for attr in {a for r in reports for a in r.attr_values} if not attr.startswith("native."))
    lines = [
        "# Canonical Context Attributes v1 Probe",
        "",
        "This probe tests a primitive, input-only context vocabulary across WTQ, SQA, TableBench, TabFact, and HiTab.",
        "",
        "## Benchmark Versions",
        "",
    ]
    for report in reports:
        represented = [a for a in shared_attrs if is_applicable(report.attr_values.get(a, Counter()))]
        informative = [a for a in shared_attrs if is_informative(report.attr_values.get(a, Counter()), report.n)]
        native = sorted(a for a in report.attr_values if a.startswith("native."))
        lines.append(f"### {report.dataset}")
        lines.append("")
        lines.append(f"- queries: {report.n}")
        lines.append(f"- applicable shared primitive attributes: {len(represented)} / {len(shared_attrs)}")
        lines.append(f"- informative candidate attributes: {len(informative)} / {len(shared_attrs)}")
        lines.append(f"- native metadata attributes: {', '.join(f'`{a}`' for a in native) if native else 'none'}")
        lines.append("")
    lines.extend([
        "## Decision",
        "",
        "- Use the same primitive extractor families for all five benchmarks where the input exposes question/statement text and table content.",
        "- Keep dialog attributes represented as `na` outside SQA instead of pretending they transfer.",
        "- Keep native metadata in a separate family. It is useful for within-benchmark analysis but not a shared transfer primitive.",
        "- Keep numeric values continuous for regression diagnostics; use bins only as derived atoms for rule mining / display.",
        "",
        "## Overlap Audit",
        "",
    ])
    for report in reports:
        min_support = max(20, int(report.n * 0.02))
        dups = near_duplicates(atom_sets[report.dataset], min_support=min_support, threshold=0.95)
        lines.append(f"### {report.dataset}")
        lines.append("")
        if not dups:
            lines.append("- No near-duplicate positive atoms above threshold.")
        else:
            for jac, a, b, na, nb, inter in dups[:8]:
                lines.append(f"- `{a}` overlaps `{b}`: Jaccard={jac:.3f}, n=({na},{nb}), intersection={inter}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("/data/users/jsu323/facet/wikitable_clean_surface_v1.db"))
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--md", type=Path, required=True)
    parser.add_argument("--limit-per-dataset", type=int, default=0)
    args = parser.parse_args()

    reports = read_reports(args.db, args.limit_per_dataset)
    atom_sets = read_atom_sets(args.db, args.limit_per_dataset)
    write_html(args.html, reports, atom_sets)
    write_markdown(args.md, reports, atom_sets)
    print(f"wrote {args.html}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
