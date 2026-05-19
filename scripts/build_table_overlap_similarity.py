#!/usr/bin/env python3
"""Build exact table-overlap diagnostics from cube query metadata.

This is the stricter precursor to embedding/cosine benchmark similarity:

1. exact full-table fingerprint overlap;
2. exact ordered header/schema overlap;
3. exact row-unit overlap;
4. exact column-unit overlap;
5. set-level Jaccard and overlap-coefficient summaries.

The script reads the unified cube `query` table only and writes local review
artifacts. It intentionally uses normalized string fingerprints rather than
semantic matching so it can answer: "do these benchmarks literally share table
surface?" before any embedding layer is considered.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sqlite3
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CUBE = Path("/data/users/jsu323/facet/wikitable_transfer_cube.db")
DEFAULT_OUT_DIR = ROOT / (
    "Obsidian/Transferability/Project/coding_agent_logs/codex/code/"
    "systematic_design/benchmark_design/domain_similarity_embedding"
)
TABLE_DATASETS = {"wtq", "sqa", "tab_fact", "tablebench", "hitab"}
SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class TableRecord:
    query_id: str
    dataset: str
    source_id: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    table_fp: str
    header_ordered_fp: str
    header_set_fp: str
    row_fps: frozenset[str]
    column_fps: frozenset[str]
    column_value_set_fps: frozenset[str]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube", type=Path, default=DEFAULT_CUBE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--datasets",
        default="wtq,sqa,tab_fact,tablebench,hitab",
        help="Comma-separated table datasets to include.",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    records = load_records(args.cube, datasets=datasets)
    if not records:
        raise SystemExit(f"No table records loaded from {args.cube}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(records, cube=args.cube)

    write_csv(args.out_dir / "table_overlap_v0_pairwise.csv", summary["pairwise"])
    write_csv(args.out_dir / "table_overlap_v0_exact_table_groups.csv", summary["exact_table_groups"])
    write_csv(args.out_dir / "table_overlap_v0_header_groups.csv", summary["header_groups"])
    (args.out_dir / "table_overlap_v0.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_html(args.out_dir / "table_overlap_v0.html", summary)
    print(f"loaded {len(records)} query-table records from {args.cube}")
    print(f"wrote {args.out_dir / 'table_overlap_v0.html'}")
    return 0


def load_records(cube: Path, *, datasets: list[str]) -> list[TableRecord]:
    conn = sqlite3.connect(str(cube))
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in datasets)
    rows = conn.execute(
        f"select query_id, dataset, meta from query where dataset in ({placeholders})",
        datasets,
    ).fetchall()
    conn.close()

    records: list[TableRecord] = []
    for row in rows:
        meta = parse_json(row["meta"], {})
        headers, table_rows, source_id = extract_table(str(row["dataset"]), meta)
        if not headers and not table_rows:
            continue
        headers_t = tuple(headers)
        rows_t = tuple(tuple(r) for r in table_rows)
        records.append(
            TableRecord(
                query_id=str(row["query_id"]),
                dataset=str(row["dataset"]),
                source_id=str(source_id),
                headers=headers_t,
                rows=rows_t,
                table_fp=fingerprint({"headers": headers_t, "rows": rows_t}),
                header_ordered_fp=fingerprint(headers_t),
                header_set_fp=fingerprint(sorted(set(headers_t))),
                row_fps=frozenset(fingerprint(r) for r in rows_t),
                column_fps=frozenset(full_column_fingerprints(headers_t, rows_t)),
                column_value_set_fps=frozenset(value_set_column_fingerprints(headers_t, rows_t)),
            )
        )
    return records


def extract_table(dataset: str, meta: dict[str, Any]) -> tuple[list[str], list[list[str]], str]:
    raw = meta.get("_raw", {}) if isinstance(meta.get("_raw"), dict) else {}
    if dataset == "wtq":
        table = raw.get("table", {}) if isinstance(raw.get("table"), dict) else {}
        return normalize_list(table.get("header", [])), normalize_rows(table.get("rows", [])), str(meta.get("table_name", ""))
    if dataset == "sqa":
        table = raw.get("table", {}) if isinstance(raw.get("table"), dict) else {}
        return normalize_list(table.get("headers", [])), normalize_rows(table.get("rows", [])), str(meta.get("table_file", ""))
    if dataset in {"tab_fact", "tabfact"}:
        headers, rows = parse_tabfact_text(str(raw.get("table_text", "")))
        return headers, rows, str(meta.get("table_id", ""))
    if dataset == "tablebench":
        table = raw.get("table", {}) if isinstance(raw.get("table"), dict) else {}
        return normalize_list(table.get("header", [])), normalize_rows(table.get("rows", [])), str(raw.get("id", ""))
    if dataset == "hitab":
        headers, rows = parse_hitab_records(raw.get("table_content", {}))
        return headers, rows, str(meta.get("table_id", ""))
    return [], [], ""


def parse_tabfact_text(table_text: str) -> tuple[list[str], list[list[str]]]:
    lines = [line.strip() for line in table_text.splitlines() if line.strip()]
    if not lines:
        return [], []
    headers = [normalize_text(part) for part in lines[0].split("#")]
    rows = [[normalize_text(part) for part in line.split("#")] for line in lines[1:]]
    return headers, rows


def parse_hitab_records(table_content: Any) -> tuple[list[str], list[list[str]]]:
    if not isinstance(table_content, dict):
        return [], []
    texts = table_content.get("texts", [])
    if not isinstance(texts, list) or not texts:
        return [], []
    n_cols = max((len(row) for row in texts if isinstance(row, list)), default=0)
    grid: list[list[str]] = []
    for row in texts:
        if not isinstance(row, list):
            continue
        padded = normalize_list(row) + [""] * n_cols
        grid.append(padded[:n_cols])
    top_header_n = min(int(table_content.get("top_header_rows_num", 1) or 1), len(grid))
    headers: list[str] = []
    for col_idx in range(n_cols):
        parts: list[str] = []
        for row_idx in range(top_header_n):
            value = grid[row_idx][col_idx]
            if value and value not in parts:
                parts.append(value)
        headers.append(" / ".join(parts) if parts else f"col_{col_idx + 1}")
    return headers, grid[top_header_n:]


def normalize_rows(rows: Any) -> list[list[str]]:
    if not isinstance(rows, list):
        return []
    return [normalize_list(row) for row in rows if isinstance(row, list)]


def normalize_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [normalize_text(value) for value in values]


def normalize_text(value: Any) -> str:
    return SPACE_RE.sub(" ", str(value).strip().lower())


def full_column_fingerprints(headers: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> list[str]:
    n_cols = max([len(headers), *(len(row) for row in rows)] or [0])
    out: list[str] = []
    for col_idx in range(n_cols):
        header = headers[col_idx] if col_idx < len(headers) else ""
        values = [row[col_idx] if col_idx < len(row) else "" for row in rows]
        out.append(fingerprint({"header": header, "values": values}))
    return out


def value_set_column_fingerprints(headers: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> list[str]:
    n_cols = max([len(headers), *(len(row) for row in rows)] or [0])
    out: list[str] = []
    for col_idx in range(n_cols):
        header = headers[col_idx] if col_idx < len(headers) else ""
        values = sorted({row[col_idx] for row in rows if col_idx < len(row) and row[col_idx]})
        out.append(fingerprint({"header": header, "value_set": values}))
    return out


def build_summary(records: list[TableRecord], *, cube: Path) -> dict[str, Any]:
    by_dataset: dict[str, list[TableRecord]] = defaultdict(list)
    for record in records:
        by_dataset[record.dataset].append(record)

    dataset_summary = []
    for dataset, rows in sorted(by_dataset.items()):
        dataset_summary.append({
            "dataset": dataset,
            "query_table_records": len(rows),
            "unique_full_tables": len({r.table_fp for r in rows}),
            "unique_ordered_headers": len({r.header_ordered_fp for r in rows}),
            "unique_header_sets": len({r.header_set_fp for r in rows}),
            "unique_rows": len(set().union(*(r.row_fps for r in rows))) if rows else 0,
            "unique_columns": len(set().union(*(r.column_fps for r in rows))) if rows else 0,
        })

    fp_maps = {
        "table": group_by(records, "table_fp"),
        "header": group_by(records, "header_ordered_fp"),
    }
    exact_table_groups = render_groups(fp_maps["table"], group_type="table")
    header_groups = render_groups(fp_maps["header"], group_type="header")

    pairwise = []
    datasets = sorted(by_dataset)
    sets_by_dataset = {
        dataset: {
            "full_tables": {r.table_fp for r in rows},
            "ordered_headers": {r.header_ordered_fp for r in rows},
            "header_sets": {r.header_set_fp for r in rows},
            "rows": set().union(*(r.row_fps for r in rows)) if rows else set(),
            "columns": set().union(*(r.column_fps for r in rows)) if rows else set(),
            "column_value_sets": set().union(*(r.column_value_set_fps for r in rows)) if rows else set(),
        }
        for dataset, rows in by_dataset.items()
    }
    for i, left in enumerate(datasets):
        for right in datasets[i + 1:]:
            row = {"dataset_a": left, "dataset_b": right}
            for key in ["full_tables", "ordered_headers", "header_sets", "rows", "columns", "column_value_sets"]:
                a = sets_by_dataset[left][key]
                b = sets_by_dataset[right][key]
                inter = len(a & b)
                row[f"{key}_intersection"] = inter
                row[f"{key}_jaccard"] = round(jaccard(a, b), 6)
                row[f"{key}_overlap_coef"] = round(overlap_coef(a, b), 6)
            pairwise.append(row)

    return {
        "cube": str(cube),
        "dataset_summary": dataset_summary,
        "pairwise": pairwise,
        "exact_table_groups": exact_table_groups,
        "header_groups": header_groups,
    }


def group_by(records: list[TableRecord], attr: str) -> dict[str, list[TableRecord]]:
    groups: dict[str, list[TableRecord]] = defaultdict(list)
    for record in records:
        groups[getattr(record, attr)].append(record)
    return groups


def render_groups(groups: dict[str, list[TableRecord]], *, group_type: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fp_value, members in groups.items():
        ds_counts = Counter(m.dataset for m in members)
        if len(ds_counts) <= 1:
            continue
        sample = [f"{m.dataset}:{m.source_id}" for m in members[:8]]
        rows.append({
            "group_type": group_type,
            "fingerprint": fp_value,
            "datasets": ",".join(sorted(ds_counts)),
            "dataset_counts": json.dumps(dict(sorted(ds_counts.items())), sort_keys=True),
            "n_query_records": len(members),
            "sample_sources": " | ".join(sample),
        })
    rows.sort(key=lambda r: (-int(r["n_query_records"]), r["datasets"], r["fingerprint"]))
    return rows


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap_coef(a: set[str], b: set[str]) -> float:
    denom = min(len(a), len(b))
    if denom <= 0:
        return 0.0
    return len(a & b) / denom


def fingerprint(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def parse_json(raw: Any, default: Any) -> Any:
    try:
        return json.loads(raw or "")
    except (TypeError, json.JSONDecodeError):
        return default


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, summary: dict[str, Any]) -> None:
    dataset_rows = "\n".join(
        "<tr>"
        f"<th>{esc(row['dataset'])}</th>"
        f"<td>{row['query_table_records']}</td>"
        f"<td>{row['unique_full_tables']}</td>"
        f"<td>{row['unique_ordered_headers']}</td>"
        f"<td>{row['unique_rows']}</td>"
        f"<td>{row['unique_columns']}</td>"
        "</tr>"
        for row in summary["dataset_summary"]
    )
    pair_rows = "\n".join(
        "<tr>"
        f"<th>{esc(row['dataset_a'])} - {esc(row['dataset_b'])}</th>"
        f"<td>{row['full_tables_intersection']}</td>"
        f"<td>{row['ordered_headers_intersection']}</td>"
        f"<td>{row['rows_intersection']}</td>"
        f"<td>{row['rows_overlap_coef']:.4f}</td>"
        f"<td>{row['columns_intersection']}</td>"
        f"<td>{row['columns_overlap_coef']:.4f}</td>"
        f"<td>{row['column_value_sets_intersection']}</td>"
        "</tr>"
        for row in summary["pairwise"]
    )
    top_tables = render_group_table(summary["exact_table_groups"][:20])
    top_headers = render_group_table(summary["header_groups"][:20])
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Table Overlap v0</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #202124; }}
h1, h2 {{ margin: 0 0 12px; }}
h2 {{ margin-top: 30px; }}
p {{ max-width: 980px; line-height: 1.45; }}
table {{ border-collapse: collapse; margin: 12px 0 24px; min-width: 900px; }}
th, td {{ border: 1px solid #d8dce3; padding: 7px 9px; text-align: right; vertical-align: top; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f1f3f6; }}
.section {{ overflow-x: auto; }}
.note {{ background: #eef7ee; border: 1px solid #b8d8b8; border-radius: 8px; padding: 12px; max-width: 980px; }}
</style>
</head>
<body>
<h1>Table Overlap v0</h1>
<p class="note">Exact normalized fingerprint overlap from cube query metadata. This is the first-layer check before vector or semantic similarity.</p>
<p><strong>Cube:</strong> {esc(summary['cube'])}</p>
<h2>Dataset Table Surface</h2>
<div class="section"><table><thead><tr><th>dataset</th><th>query-table records</th><th>unique full tables</th><th>unique ordered headers</th><th>unique rows</th><th>unique columns</th></tr></thead><tbody>{dataset_rows}</tbody></table></div>
<h2>Pairwise Exact Overlap</h2>
<div class="section"><table><thead><tr><th>pair</th><th>full tables</th><th>ordered headers</th><th>rows</th><th>row overlap coef</th><th>full columns</th><th>column overlap coef</th><th>column value sets</th></tr></thead><tbody>{pair_rows}</tbody></table></div>
<h2>Top Cross-Benchmark Exact Table Groups</h2>
{top_tables}
<h2>Top Cross-Benchmark Header Groups</h2>
{top_headers}
</body>
</html>
""",
        encoding="utf-8",
    )


def render_group_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No cross-benchmark groups.</p>"
    body = "\n".join(
        "<tr>"
        f"<td>{esc(row['datasets'])}</td>"
        f"<td>{esc(row['dataset_counts'])}</td>"
        f"<td>{row['n_query_records']}</td>"
        f"<td>{esc(row['sample_sources'])}</td>"
        "</tr>"
        for row in rows
    )
    return "<div class='section'><table><thead><tr><th>datasets</th><th>counts</th><th>n</th><th>sample sources</th></tr></thead><tbody>" + body + "</tbody></table></div>"


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
