#!/usr/bin/env python3
"""Build a registry-first context matrix and HTML report for v1.2."""

from __future__ import annotations

import argparse
import html
import inspect
import json
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from study_layer.context_attribute import lightweight_extractors as extractors  # noqa: E402

DEFAULT_DB = Path("/data/users/jsu323/facet/wikitable_reasoning_out_loud_addone_v1.db")
DEFAULT_REGISTRY = ROOT / "study_layer/context_attribute/context_attribute_registry_v1_2.json"
DEFAULT_MATRIX = ROOT / "study_layer/context_attribute/artifacts/context_matrix_v1_2.jsonl"
DEFAULT_REPORT = ROOT / "study_layer/context_attribute/artifacts/context_orthogonality_report_v1_2.html"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def query_rows(db_path: Path, datasets: list[str], limit_per_dataset: int) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    out: list[dict[str, Any]] = []
    for dataset in datasets:
        sql = "SELECT query_id, dataset, content, meta FROM query WHERE dataset=? ORDER BY query_id"
        params: list[Any] = [dataset]
        if limit_per_dataset:
            sql += " LIMIT ?"
            params.append(limit_per_dataset)
        for row in conn.execute(sql, params):
            out.append(dict(row))
    return out


def build_matrix(db_path: Path, registry: dict[str, Any], limit_per_dataset: int) -> list[dict[str, Any]]:
    atom_specs = {spec["atom"]: spec for spec in registry["atoms"]}
    matrix_rows: list[dict[str, Any]] = []
    for row in query_rows(db_path, registry["datasets"], limit_per_dataset):
        meta = json.loads(row.get("meta") or "{}")
        raw_atoms = extractors.canonical_atoms(meta, row["dataset"])
        indicators = sorted(extractors.indicator_atoms(raw_atoms, atom_specs))
        matrix_rows.append({
            "query_id": row["query_id"],
            "dataset": row["dataset"],
            "content": row.get("content") or "",
            "atoms": raw_atoms,
            "indicators": indicators,
        })
    return matrix_rows


def support_tables(rows: list[dict[str, Any]], registry: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    datasets = registry["datasets"]
    counts_by_dataset = Counter(row["dataset"] for row in rows)
    atom_specs = {spec["atom"]: spec for spec in registry["atoms"]}
    support_rows: list[dict[str, Any]] = []
    value_rows: list[dict[str, Any]] = []

    for atom, spec in atom_specs.items():
        by_dataset = {}
        values_by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            value = row["atoms"].get(atom, "")
            values_by_dataset[row["dataset"]][value] += 1
            present = value == "yes" if spec.get("value_type") == "binary" else value not in {"", "na", "unknown"}
            if present:
                by_dataset[row["dataset"]] = by_dataset.get(row["dataset"], 0) + 1
        total_present = sum(by_dataset.values())
        support_rows.append({
            "atom": atom,
            "family": spec.get("family", ""),
            "scope": spec.get("scope", ""),
            "decision": spec.get("decision", ""),
            "value_type": spec.get("value_type", ""),
            "support_total": total_present,
            "support_by_dataset": by_dataset,
            "rate_by_dataset": {
                ds: (by_dataset.get(ds, 0) / counts_by_dataset[ds] if counts_by_dataset[ds] else 0.0)
                for ds in datasets
            },
        })
        for dataset, counter in values_by_dataset.items():
            for value, count in counter.most_common(8):
                value_rows.append({"atom": atom, "dataset": dataset, "value": value, "count": count})

    support_rows.sort(key=lambda r: (r["family"], r["atom"]))
    return support_rows, value_rows


def indicator_index(rows: list[dict[str, Any]]) -> dict[str, set[int]]:
    idx: dict[str, set[int]] = defaultdict(set)
    for i, row in enumerate(rows):
        for indicator in row["indicators"]:
            idx[indicator].add(i)
    return dict(idx)


def phi(a: set[int], b: set[int], n: int) -> float:
    n11 = len(a & b)
    n10 = len(a - b)
    n01 = len(b - a)
    n00 = n - n11 - n10 - n01
    denom = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return ((n11 * n00 - n10 * n01) / denom) if denom else 0.0


def binary_nmi(a: set[int], b: set[int], n: int) -> float:
    cells = [len(a & b), len(a - b), len(b - a), n - len(a | b)]
    if n <= 0:
        return 0.0
    p11, p10, p01, p00 = [c / n for c in cells]
    px1, px0 = p11 + p10, p01 + p00
    py1, py0 = p11 + p01, p10 + p00
    mi = 0.0
    for pxy, px, py in [(p11, px1, py1), (p10, px1, py0), (p01, px0, py1), (p00, px0, py0)]:
        if pxy and px and py:
            mi += pxy * math.log2(pxy / (px * py))
    def entropy(p: float) -> float:
        return 0.0 if p in {0.0, 1.0} else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    hx = entropy(px1)
    hy = entropy(py1)
    return mi / math.sqrt(hx * hy) if hx and hy else 0.0


def redundancy_pairs(rows: list[dict[str, Any]], min_support: int) -> list[dict[str, Any]]:
    idx = {k: v for k, v in indicator_index(rows).items() if len(v) >= min_support}
    indicators = sorted(idx)
    n = len(rows)
    pairs: list[dict[str, Any]] = []
    for i, left in enumerate(indicators):
        a = idx[left]
        for right in indicators[i + 1:]:
            b = idx[right]
            inter = len(a & b)
            union = len(a | b)
            if not union:
                continue
            jaccard = inter / union
            p = phi(a, b, n)
            nmi = binary_nmi(a, b, n)
            if jaccard >= 0.45 or abs(p) >= 0.35 or nmi >= 0.25:
                pairs.append({
                    "left": left,
                    "right": right,
                    "support_left": len(a),
                    "support_right": len(b),
                    "intersection": inter,
                    "jaccard": jaccard,
                    "phi": p,
                    "nmi": nmi,
                })
    pairs.sort(key=lambda r: (max(abs(r["phi"]), r["jaccard"], r["nmi"]), r["intersection"]), reverse=True)
    return pairs


def esc(value: Any) -> str:
    return html.escape(str(value))


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def regex_flags(pattern: Any) -> str:
    flags: list[str] = []
    if pattern.flags & re.IGNORECASE:
        flags.append("re.I")
    return " | ".join(flags) or "0"


def pattern_snippet(mapping_name: str, atom: str, pattern: Any, source_value: str) -> str:
    return (
        f'{mapping_name}["{atom}"] = re.compile({pattern.pattern!r}, {regex_flags(pattern)})\n\n'
        f'# canonical_atoms(...)\n'
        f'out["{atom}"] = "yes" if {mapping_name}["{atom}"].search({source_value}) else "no"'
    )


def atom_implementation(atom_spec: dict[str, Any]) -> str:
    atom = atom_spec["atom"]
    if atom in extractors.TEXT_PATTERNS:
        return pattern_snippet("TEXT_PATTERNS", atom, extractors.TEXT_PATTERNS[atom], "text")
    if atom in extractors.HEADER_PATTERNS:
        return (
            pattern_snippet("HEADER_PATTERNS", atom, extractors.HEADER_PATTERNS[atom], "str(header)")
            .replace(
                f'out["{atom}"] = "yes" if HEADER_PATTERNS["{atom}"].search(str(header)) else "no"',
                f'out["{atom}"] = "yes" if any(HEADER_PATTERNS["{atom}"].search(str(header)) for header in headers) else "no"',
            )
        )
    if atom == "schema.header_repetition_marker":
        return 'normalized_headers = [str(header).strip().lower() for header in headers]\nout["schema.header_repetition_marker"] = "yes" if len(normalized_headers) != len(set(normalized_headers)) else "no"'
    if atom == "table.rows_bin":
        return 'n_rows = len(rows)\nout["table.rows_bin"] = bin_count(n_rows, (5, 10, 25, 50))'
    if atom == "table.cols_bin":
        return 'n_cols = len(headers)\nout["table.cols_bin"] = bin_count(n_cols, (3, 6, 10))'
    if atom == "table.shape":
        return 'if not n_rows or not n_cols:\n    out["table.shape"] = "empty"\nelse:\n    ratio = n_cols / n_rows\n    out["table.shape"] = "wide" if ratio > 0.5 else ("tall" if ratio < 0.1 else "balanced")'
    if atom in {"table.numeric_cols_bin", "table.numeric_density_bin"}:
        return '''numeric_cols = 0
for ci in range(n_cols):
    vals = [
        str(rows[ri][ci]).strip()
        for ri in range(min(n_rows, 50))
        if ci < len(rows[ri]) and str(rows[ri][ci]).strip()
    ]
    if vals and sum(numeric_like(v) for v in vals) / len(vals) > 0.5:
        numeric_cols += 1
out["table.numeric_cols_bin"] = "0" if numeric_cols == 0 else ("1" if numeric_cols == 1 else ("2_3" if numeric_cols <= 3 else "ge_4"))
if not n_cols or numeric_cols == 0:
    out["table.numeric_density_bin"] = "none"
else:
    density = numeric_cols / n_cols
    out["table.numeric_density_bin"] = "low" if density < 0.25 else ("mid" if density < 0.6 else "high")'''
    cell_pattern_atoms = {
        "cell.has_comma_number": ("COMMA_NUMBER_PATTERN", extractors.COMMA_NUMBER_PATTERN),
        "cell.has_percent": ("PERCENT_PATTERN", extractors.PERCENT_PATTERN),
        "cell.has_currency": ("CURRENCY_PATTERN", extractors.CURRENCY_PATTERN),
        "cell.has_date_like": ("DATE_LIKE_PATTERN", extractors.DATE_LIKE_PATTERN),
    }
    if atom in cell_pattern_atoms:
        name, pattern = cell_pattern_atoms[atom]
        return f'{name} = re.compile({pattern.pattern!r}, {regex_flags(pattern)})\nout["{atom}"] = "yes" if {name}.search(flat_text) else "no"'
    if atom == "cell.missing_value_marker":
        return 'MISSING_MARKERS = {"", "-", "--", "?", "n/a", "na", "none", "null"}\nout["cell.missing_value_marker"] = "yes" if any(cell.strip().lower() in MISSING_MARKERS for cell in visible_cells) else "no"'
    if atom in {"cell.has_numeric_range", "cell.has_score_surface"}:
        return inspect.getsource(extractors.split_range_and_score_atoms) + '\n# canonical_atoms(...)\nnumeric_range, score_surface = split_range_and_score_atoms(flat_text, out["schema.has_score_col"] == "yes")\nout["cell.has_numeric_range"] = numeric_range\nout["cell.has_score_surface"] = score_surface'
    if atom in {"grounding.header_overlap", "grounding.cell_overlap"}:
        return inspect.getsource(extractors.tokens) + '\n# canonical_atoms(...)\nquestion_tokens = tokens(text)\nout["grounding.header_overlap"] = "yes" if question_tokens & tokens(header_text) else "no"\nout["grounding.cell_overlap"] = "yes" if question_tokens & tokens(flat_text) else "no"'
    if atom in {"dialog.turn_bin", "dialog.has_reference_marker"}:
        return 'DIALOG_REFERENCE_PATTERN = re.compile(...)\nif dataset == "sqa":\n    position = int(raw.get("position", meta.get("position", 0)) or 0)\n    out["dialog.turn_bin"] = "first" if position == 0 else ("early" if position <= 2 else "late")\n    out["dialog.has_reference_marker"] = "yes" if DIALOG_REFERENCE_PATTERN.search(text) else "no"\nelse:\n    out["dialog.turn_bin"] = "na"\n    out["dialog.has_reference_marker"] = "na"'
    if atom in {"native.tablebench_qtype", "native.tablebench_qsubtype"}:
        return 'if dataset == "tablebench":\n    out["native.tablebench_qtype"] = str(raw.get("qtype", meta.get("qtype", "unknown")) or "unknown")\n    out["native.tablebench_qsubtype"] = str(raw.get("qsubtype", meta.get("qsubtype", "unknown")) or "unknown")'
    if atom in {"native.hitab_agg_type", "native.hitab_source_family"}:
        return 'if dataset == "hitab":\n    out["native.hitab_agg_type"] = str(raw.get("aggregation", meta.get("aggregation", "unknown")) or "unknown")\n    out["native.hitab_source_family"] = str(raw.get("table_source", meta.get("table_source", "unknown")) or "unknown")'
    return f'# Extractor path: {atom_spec.get("extractor", "unknown")}\n# See lightweight_extractors.canonical_atoms for implementation.'


def atom_cell_html(atom_spec: dict[str, Any]) -> str:
    atom = atom_spec["atom"]
    code = atom_implementation(atom_spec)
    return (
        '<details class="atom-code">'
        f'<summary><code>{esc(atom)}</code></summary>'
        f'<pre><code>{esc(code)}</code></pre>'
        '</details>'
    )


def write_report(path: Path, registry: dict[str, Any], rows: list[dict[str, Any]], support_rows: list[dict[str, Any]], redundancy: list[dict[str, Any]], matrix_path: Path, min_support: int) -> None:
    datasets = registry["datasets"]
    counts = Counter(row["dataset"] for row in rows)
    rule_pack = registry["bundled_rule_pack_experiment"]

    support_html = []
    for row in support_rows:
        rates = "".join(
            f"<td>{row['support_by_dataset'].get(ds, 0)}<div class='subtle'>{pct(row['rate_by_dataset'][ds])}</div></td>"
            for ds in datasets
        )
        support_html.append(
            "<tr>"
            f"<td><code>{esc(row['atom'])}</code><div class='subtle'>{esc(row['value_type'])}</div></td>"
            f"<td>{esc(row['family'])}</td>"
            f"<td>{esc(row['scope'])}</td>"
            f"<td><span class='decision'>{esc(row['decision'])}</span></td>"
            f"<td>{row['support_total']}</td>"
            f"{rates}"
            "</tr>"
        )

    registry_html = []
    for atom in registry["atoms"]:
        registry_html.append(
            "<tr>"
            f"<td>{atom_cell_html(atom)}</td>"
            f"<td>{esc(atom['family'])}</td>"
            f"<td>{esc(atom['value_type'])}</td>"
            f"<td>{esc(atom['scope'])}</td>"
            f"<td>{esc(atom['decision'])}</td>"
            f"<td>{esc(atom['extractor'])}</td>"
            f"<td>{esc(atom['description'])}</td>"
            "</tr>"
        )

    redundant_html = []
    for row in redundancy[:80]:
        redundant_html.append(
            "<tr>"
            f"<td><code>{esc(row['left'])}</code></td>"
            f"<td><code>{esc(row['right'])}</code></td>"
            f"<td>{row['support_left']}</td><td>{row['support_right']}</td><td>{row['intersection']}</td>"
            f"<td>{row['jaccard']:.3f}</td><td>{row['phi']:.3f}</td><td>{row['nmi']:.3f}</td>"
            "</tr>"
        )

    excluded_html = "".join(f"<li><code>{esc(item['name'])}</code>: {esc(item['reason'])}</li>" for item in registry.get("explicitly_excluded", []))
    rule_html = "".join(f"<li>{esc(rule)}</li>" for rule in rule_pack["rule_surface"])
    dataset_pills = "".join(f"<span class='pill'>{esc(ds)}: {counts[ds]}</span>" for ds in datasets)
    dataset_headers = "".join(f"<th>{esc(ds)}</th>" for ds in datasets)

    doc = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{esc(registry['registry_id'])}</title>
  <style>
    body {{ margin: 24px; font-family: Arial, sans-serif; color: #1f2933; background: #f6f7f9; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; }}
    h2 {{ margin-top: 28px; font-size: 20px; }}
    h3 {{ margin: 14px 0 8px; font-size: 15px; }}
    .subtle {{ color: #667085; font-size: 12px; }}
    .pills {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 14px 0; }}
    .pill {{ background: #e8eef7; border: 1px solid #cbd6e7; border-radius: 999px; padding: 5px 10px; font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }}
    .card {{ background: white; border: 1px solid #d8dee8; border-radius: 8px; padding: 14px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d8dee8; }}
    th, td {{ border-bottom: 1px solid #e5e9f0; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #edf2f7; font-size: 12px; text-transform: uppercase; }}
    code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
    pre {{ background: #111827; color: #e5e7eb; padding: 12px; overflow: auto; border-radius: 6px; max-height: 420px; }}
    details {{ background: white; border: 1px solid #d8dee8; border-radius: 8px; padding: 10px 12px; margin: 10px 0; }}
    summary {{ cursor: pointer; font-weight: 700; }}
    .atom-code {{ padding: 0; margin: 0; border: 0; background: transparent; }}
    .atom-code summary {{ list-style: none; display: inline-flex; align-items: center; gap: 8px; }}
    .atom-code summary::-webkit-details-marker {{ display: none; }}
    .atom-code summary::before {{ content: "+"; display: inline-grid; place-items: center; width: 16px; height: 16px; border: 1px solid #b8c2d4; border-radius: 4px; color: #344054; font-size: 12px; }}
    .atom-code[open] summary::before {{ content: "-"; }}
    .atom-code pre {{ margin: 8px 0 0; max-width: min(920px, 82vw); }}
    .decision {{ background: #edf7ee; border: 1px solid #c7e4cc; border-radius: 6px; padding: 2px 6px; font-size: 12px; }}
    .warn {{ background: #fff7e6; border-color: #f5cf85; }}
  </style>
</head>
<body>
  <h1>{esc(registry['registry_id'])}</h1>
  <div class=\"subtle\">Registry-first context matrix. Generated from <code>{esc(str(matrix_path))}</code>. Redundancy min support: {min_support}.</div>
  <div class=\"pills\"><span class=\"pill\">queries: {len(rows)}</span><span class=\"pill\">atoms: {len(registry['atoms'])}</span>{dataset_pills}</div>

  <div class=\"grid\">
    <section class=\"card\">
      <h3>Design Boundary</h3>
      <p>Primitive extractors only: text surface, table shape, value markers, header/cell overlap, scoped native metadata.</p>
      <p>No country/team/player/name-role ontology in the lightweight layer.</p>
    </section>
    <section class=\"card warn\">
      <h3>Bundled Rule-Pack Experiment</h3>
      <p><b>Feature:</b> <code>{esc(rule_pack['feature_canonical_id'])}</code></p>
      <p>{esc(rule_pack['policy'])}</p>
    </section>
  </div>

  <h2>Bundled Rule Surface</h2>
  <ol>{rule_html}</ol>

  <h2>Explicitly Excluded From Lightweight Context</h2>
  <ul>{excluded_html}</ul>

  <h2>Registry</h2>
  <p class="subtle">Click an atom to unfold the exact lightweight extractor code used for that atom.</p>
  <table><thead><tr><th>Atom + implementation</th><th>Family</th><th>Type</th><th>Scope</th><th>Decision</th><th>Extractor</th><th>Description</th></tr></thead><tbody>{''.join(registry_html)}</tbody></table>

  <h2>Support By Benchmark</h2>
  <table><thead><tr><th>Atom</th><th>Family</th><th>Scope</th><th>Decision</th><th>Total</th>{dataset_headers}</tr></thead><tbody>{''.join(support_html)}</tbody></table>

  <h2>Top Redundancy Pairs</h2>
  <p class=\"subtle\">Pairs shown when Jaccard >= 0.45, abs(phi) >= 0.35, or NMI >= 0.25.</p>
  <table><thead><tr><th>Left</th><th>Right</th><th>Support L</th><th>Support R</th><th>Both</th><th>Jaccard</th><th>Phi</th><th>NMI</th></tr></thead><tbody>{''.join(redundant_html)}</tbody></table>

</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--matrix-out", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--html-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit-per-dataset", type=int, default=0)
    parser.add_argument("--min-support", type=int, default=100)
    args = parser.parse_args()

    registry = load_json(args.registry)
    rows = build_matrix(args.db, registry, args.limit_per_dataset)
    write_jsonl(args.matrix_out, rows)
    support_rows, _value_rows = support_tables(rows, registry)
    redundancy = redundancy_pairs(rows, args.min_support)
    write_report(args.html_out, registry, rows, support_rows, redundancy, args.matrix_out, args.min_support)
    print(f"matrix: {args.matrix_out}")
    print(f"html: {args.html_out}")
    print(f"queries: {len(rows)}")
    print(f"redundancy_pairs: {len(redundancy)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
