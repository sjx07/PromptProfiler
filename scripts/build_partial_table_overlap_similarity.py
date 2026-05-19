#!/usr/bin/env python3
"""Build partial table-overlap diagnostics with visual examples.

This complements ``build_table_overlap_similarity.py``. The exact-overlap v0
answers whether benchmarks literally share whole tables, rows, or columns. This
v1 script asks a looser question:

    Do tables from two benchmarks share enough partial surface to plausibly
    transfer source/domain prompt features?

It uses blocked candidate search, not exhaustive all-pairs comparison:

1. deduplicate tables within each benchmark by exact full-table fingerprint;
2. build blocking keys from exact headers and non-trivial cell values;
3. score candidate cross-benchmark table pairs with separate components:
   header Jaccard, cell-value overlap, greedy column alignment, row-set match;
4. render top examples as side-by-side mini tables with highlighted overlap.

The scores are diagnostics, not a learned similarity model.
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
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CUBE = Path("/data/users/jsu323/facet/wikitable_transfer_cube.db")
DEFAULT_OUT_DIR = ROOT / (
    "Obsidian/Transferability/Project/coding_agent_logs/codex/code/"
    "systematic_design/benchmark_design/domain_similarity_embedding"
)
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z0-9]+")
ALPHA_RE = re.compile(r"[a-z]")
STOPWORDS = {"a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"}


@dataclass(frozen=True)
class Column:
    index: int
    header: str
    header_tokens: frozenset[str]
    values: frozenset[str]


@dataclass(frozen=True)
class TableRecord:
    idx: int
    query_id: str
    dataset: str
    source_id: str
    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    table_fp: str
    columns: tuple[Column, ...]
    header_set: frozenset[str]
    value_set: frozenset[str]
    row_sets: tuple[frozenset[str], ...]
    blocking_keys: frozenset[str]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube", type=Path, default=DEFAULT_CUBE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--datasets", default="wtq,sqa,tab_fact,tablebench,hitab")
    parser.add_argument("--max-key-frequency", type=int, default=180)
    parser.add_argument("--max-candidates-per-pair", type=int, default=1400)
    parser.add_argument("--top-examples-per-pair", type=int, default=5)
    parser.add_argument("--row-cap", type=int, default=80)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    tables = load_unique_tables(args.cube, datasets=datasets)
    if not tables:
        raise SystemExit(f"No tables loaded from {args.cube}")

    candidate_pairs = build_candidate_pairs(
        tables,
        max_key_frequency=args.max_key_frequency,
        max_candidates_per_pair=args.max_candidates_per_pair,
    )
    summaries, examples = score_candidates(
        tables,
        candidate_pairs,
        top_examples_per_pair=args.top_examples_per_pair,
        row_cap=args.row_cap,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "cube": str(args.cube),
        "n_unique_tables": len(tables),
        "parameters": {
            "max_key_frequency": args.max_key_frequency,
            "max_candidates_per_pair": args.max_candidates_per_pair,
            "top_examples_per_pair": args.top_examples_per_pair,
            "row_cap": args.row_cap,
        },
        "dataset_counts": dict(sorted(Counter(t.dataset for t in tables).items())),
        "pairwise": summaries,
        "examples": examples,
    }
    write_csv(args.out_dir / "partial_table_overlap_v1_pairwise.csv", summaries)
    write_jsonl(args.out_dir / "partial_table_overlap_v1_examples.jsonl", examples)
    (args.out_dir / "partial_table_overlap_v1.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_html(args.out_dir / "partial_table_overlap_v1.html", payload, tables)
    print(f"loaded {len(tables)} unique tables from {args.cube}")
    print(f"wrote {args.out_dir / 'partial_table_overlap_v1.html'}")
    return 0


def load_unique_tables(cube: Path, *, datasets: list[str]) -> list[TableRecord]:
    conn = sqlite3.connect(str(cube))
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in datasets)
    rows = conn.execute(
        f"select query_id, dataset, meta from query where dataset in ({placeholders})",
        datasets,
    ).fetchall()
    conn.close()

    seen: set[tuple[str, str]] = set()
    tables: list[TableRecord] = []
    for row in rows:
        meta = parse_json(row["meta"], {})
        headers, table_rows, source_id = extract_table(str(row["dataset"]), meta)
        if not headers and not table_rows:
            continue
        table_fp = fingerprint({"headers": headers, "rows": table_rows})
        dedupe_key = (str(row["dataset"]), table_fp)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        idx = len(tables)
        headers_t = tuple(headers)
        rows_t = tuple(tuple(r) for r in table_rows)
        columns = tuple(build_columns(headers_t, rows_t))
        value_set = frozenset(
            value
            for row_values in rows_t
            for value in row_values
            if useful_overlap_value(value)
        )
        row_sets = tuple(
            frozenset(value for value in row_values if useful_overlap_value(value))
            for row_values in rows_t
        )
        header_set = frozenset(h for h in headers_t if h)
        tables.append(
            TableRecord(
                idx=idx,
                query_id=str(row["query_id"]),
                dataset=str(row["dataset"]),
                source_id=str(source_id),
                headers=headers_t,
                rows=rows_t,
                table_fp=table_fp,
                columns=columns,
                header_set=header_set,
                value_set=value_set,
                row_sets=row_sets,
                blocking_keys=build_blocking_keys(header_set, value_set),
            )
        )
    return tables


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


def build_columns(headers: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> list[Column]:
    n_cols = max([len(headers), *(len(row) for row in rows)] or [0])
    columns: list[Column] = []
    for col_idx in range(n_cols):
        header = headers[col_idx] if col_idx < len(headers) else ""
        values = frozenset(
            row[col_idx]
            for row in rows
            if col_idx < len(row) and useful_overlap_value(row[col_idx])
        )
        columns.append(Column(col_idx, header, frozenset(tokenize(header)), values))
    return columns


def build_blocking_keys(headers: frozenset[str], values: frozenset[str]) -> frozenset[str]:
    keys: set[str] = set()
    for header in headers:
        if header:
            keys.add(f"h:{header}")
            for token in tokenize(header):
                keys.add(f"ht:{token}")
    for value in values:
        if useful_value_key(value):
            keys.add(f"v:{value}")
    return frozenset(keys)


def useful_value_key(value: str) -> bool:
    if len(value) < 3:
        return False
    if value in {"yes", "no", "n/a", "none", "unknown", "-"}:
        return False
    if value.isdigit() and not (1800 <= int(value) <= 2099):
        return False
    return True


def useful_overlap_value(value: str) -> bool:
    if not useful_value_key(value):
        return False
    # Primary partial-overlap scoring is entity/text-surface based. Numeric and
    # year-only values create false similarity across date, rank, score, count,
    # and medal tables, so keep them out of value, row, and column scores.
    return bool(ALPHA_RE.search(value))


def build_candidate_pairs(
    tables: list[TableRecord],
    *,
    max_key_frequency: int,
    max_candidates_per_pair: int,
) -> dict[tuple[str, str], list[tuple[int, int, int]]]:
    key_to_tables: dict[str, list[int]] = defaultdict(list)
    for table in tables:
        for key in table.blocking_keys:
            key_to_tables[key].append(table.idx)

    counts_by_pair: dict[tuple[str, str], Counter[tuple[int, int]]] = defaultdict(Counter)
    for ids in key_to_tables.values():
        if len(ids) < 2 or len(ids) > max_key_frequency:
            continue
        by_dataset: dict[str, list[int]] = defaultdict(list)
        for table_id in ids:
            by_dataset[tables[table_id].dataset].append(table_id)
        datasets = sorted(by_dataset)
        for i, left_ds in enumerate(datasets):
            for right_ds in datasets[i + 1:]:
                pair_key = (left_ds, right_ds)
                counter = counts_by_pair[pair_key]
                for left_id in by_dataset[left_ds]:
                    for right_id in by_dataset[right_ds]:
                        counter[(left_id, right_id)] += 1

    out: dict[tuple[str, str], list[tuple[int, int, int]]] = {}
    for pair_key, counter in counts_by_pair.items():
        top = counter.most_common(max_candidates_per_pair)
        out[pair_key] = [(left, right, count) for (left, right), count in top]
    return out


def score_candidates(
    tables: list[TableRecord],
    candidate_pairs: dict[tuple[str, str], list[tuple[int, int, int]]],
    *,
    top_examples_per_pair: int,
    row_cap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for pair_key, candidates in sorted(candidate_pairs.items()):
        scored: list[dict[str, Any]] = []
        for left_id, right_id, block_count in candidates:
            score = score_table_pair(tables[left_id], tables[right_id], block_count=block_count, row_cap=row_cap)
            scored.append(score)
        scored.sort(key=lambda row: row["partial_table_score"], reverse=True)
        top = scored[:top_examples_per_pair]
        examples.extend(top)
        top20 = scored[:20]
        summaries.append({
            "dataset_a": pair_key[0],
            "dataset_b": pair_key[1],
            "candidate_pairs": len(candidates),
            "best_partial_table_score": round(top[0]["partial_table_score"], 4) if top else 0.0,
            "mean_top20_partial_table_score": round(mean([r["partial_table_score"] for r in top20]), 4) if top20 else 0.0,
            "best_header_jaccard": round(top[0]["header_jaccard"], 4) if top else 0.0,
            "best_column_alignment_score": round(top[0]["column_alignment_score"], 4) if top else 0.0,
            "best_row_alignment_score": round(top[0]["row_alignment_score"], 4) if top else 0.0,
            "best_value_overlap_coef": round(top[0]["value_overlap_coef"], 4) if top else 0.0,
            "best_source_a": top[0]["source_a"] if top else "",
            "best_source_b": top[0]["source_b"] if top else "",
        })
    return summaries, examples


def score_table_pair(left: TableRecord, right: TableRecord, *, block_count: int, row_cap: int) -> dict[str, Any]:
    header_j = jaccard(set(left.header_set), set(right.header_set))
    value_j = jaccard(set(left.value_set), set(right.value_set))
    value_oc = overlap_coef(set(left.value_set), set(right.value_set))
    matched_columns = greedy_column_alignment(left, right)
    column_score = sum(match["score"] for match in matched_columns) / max(1, min(len(left.columns), len(right.columns)))
    row_score, row_matches = row_alignment(left, right, row_cap=row_cap)
    partial = 0.25 * header_j + 0.25 * value_oc + 0.35 * column_score + 0.15 * row_score
    shared_values = sorted(set(left.value_set) & set(right.value_set), key=lambda x: (len(x), x))[:80]
    return {
        "dataset_a": left.dataset,
        "dataset_b": right.dataset,
        "source_a": left.source_id,
        "source_b": right.source_id,
        "query_id_a": left.query_id,
        "query_id_b": right.query_id,
        "table_idx_a": left.idx,
        "table_idx_b": right.idx,
        "blocking_key_count": block_count,
        "partial_table_score": round(partial, 6),
        "header_jaccard": round(header_j, 6),
        "value_jaccard": round(value_j, 6),
        "value_overlap_coef": round(value_oc, 6),
        "column_alignment_score": round(column_score, 6),
        "matched_column_count": len(matched_columns),
        "row_alignment_score": round(row_score, 6),
        "row_match_count_050": sum(1 for m in row_matches if m["score"] >= 0.5),
        "shared_headers": sorted(set(left.header_set) & set(right.header_set)),
        "shared_values_sample": shared_values,
        "matched_columns": matched_columns[:8],
        "row_matches": row_matches[:8],
    }


def greedy_column_alignment(left: TableRecord, right: TableRecord) -> list[dict[str, Any]]:
    candidates: list[tuple[float, int, int, float, float]] = []
    for lcol in left.columns:
        for rcol in right.columns:
            header_score = 1.0 if lcol.header and lcol.header == rcol.header else jaccard(set(lcol.header_tokens), set(rcol.header_tokens))
            value_score = jaccard(set(lcol.values), set(rcol.values))
            score = 0.45 * header_score + 0.55 * value_score
            if score >= 0.12:
                candidates.append((score, lcol.index, rcol.index, header_score, value_score))
    candidates.sort(reverse=True)
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[dict[str, Any]] = []
    for score, left_idx, right_idx, header_score, value_score in candidates:
        if left_idx in used_left or right_idx in used_right:
            continue
        used_left.add(left_idx)
        used_right.add(right_idx)
        lcol = left.columns[left_idx]
        rcol = right.columns[right_idx]
        matches.append({
            "left_index": left_idx,
            "right_index": right_idx,
            "left_header": lcol.header,
            "right_header": rcol.header,
            "score": round(score, 4),
            "header_score": round(header_score, 4),
            "value_score": round(value_score, 4),
            "shared_values": sorted(set(lcol.values) & set(rcol.values))[:12],
        })
    return matches


def row_alignment(left: TableRecord, right: TableRecord, *, row_cap: int) -> tuple[float, list[dict[str, Any]]]:
    left_rows = [row for row in left.row_sets[:row_cap] if len(row) >= 2]
    right_rows = [row for row in right.row_sets[:row_cap] if len(row) >= 2]
    if not left_rows or not right_rows:
        return 0.0, []
    if len(left_rows) <= len(right_rows):
        small_label, large_label = "left", "right"
        small_rows, large_rows = left_rows, right_rows
    else:
        small_label, large_label = "right", "left"
        small_rows, large_rows = right_rows, left_rows
    best: list[dict[str, Any]] = []
    total = 0.0
    for i, row in enumerate(small_rows):
        best_j = -1
        best_score = 0.0
        for j, other in enumerate(large_rows):
            score = jaccard(set(row), set(other))
            if score > best_score:
                best_score = score
                best_j = j
        total += best_score
        if best_score > 0:
            best.append({
                f"{small_label}_row": i,
                f"{large_label}_row": best_j,
                "score": round(best_score, 4),
                "shared_values": sorted(set(row) & set(large_rows[best_j]))[:12] if best_j >= 0 else [],
            })
    best.sort(key=lambda x: x["score"], reverse=True)
    return total / len(small_rows), best


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


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS and len(token) > 1]


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def overlap_coef(left: set[str], right: set[str]) -> float:
    denom = min(len(left), len(right))
    return len(left & right) / denom if denom else 0.0


def fingerprint(value: Any) -> str:
    return hashlib.sha1(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


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
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_html(path: Path, payload: dict[str, Any], tables: list[TableRecord]) -> None:
    table_by_idx = {table.idx: table for table in tables}
    dataset_cards = "\n".join(
        f"<div class='card'><div class='metric'>{count}</div><div>{esc(dataset)}</div></div>"
        for dataset, count in payload["dataset_counts"].items()
    )
    pair_rows = "\n".join(
        "<tr>"
        f"<th>{esc(row['dataset_a'])} - {esc(row['dataset_b'])}</th>"
        f"<td>{row['candidate_pairs']}</td>"
        f"<td>{row['best_partial_table_score']:.4f}</td>"
        f"<td>{row['mean_top20_partial_table_score']:.4f}</td>"
        f"<td>{row['best_column_alignment_score']:.4f}</td>"
        f"<td>{row['best_row_alignment_score']:.4f}</td>"
        f"<td>{esc(row['best_source_a'])}<br>{esc(row['best_source_b'])}</td>"
        "</tr>"
        for row in payload["pairwise"]
    )
    examples = "\n".join(render_example(example, table_by_idx) for example in payload["examples"])
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Partial Table Overlap v1</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #202124; background: #fff; }}
h1, h2, h3 {{ margin: 0 0 12px; }}
h2 {{ margin-top: 28px; }}
p {{ line-height: 1.45; max-width: 1000px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; max-width: 1100px; }}
.card {{ border: 1px solid #d5d9e0; border-radius: 8px; padding: 12px; background: #f8fafc; }}
.metric {{ font-size: 22px; font-weight: 700; }}
table {{ border-collapse: collapse; }}
.summary {{ min-width: 1050px; }}
.summary th, .summary td {{ border: 1px solid #d8dce3; padding: 7px 9px; text-align: right; vertical-align: top; }}
.summary th:first-child, .summary td:first-child {{ text-align: left; }}
.summary th {{ background: #f1f3f6; }}
.section {{ overflow-x: auto; margin-bottom: 22px; }}
.note {{ background: #eef7ee; border: 1px solid #b8d8b8; border-radius: 8px; padding: 12px; max-width: 1000px; }}
.example {{ border: 1px solid #d7dce2; border-radius: 8px; margin: 14px 0; padding: 14px; }}
.example-title {{ display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: baseline; margin-bottom: 10px; }}
.badge {{ display: inline-block; border: 1px solid #c9d6ee; background: #f2f6ff; border-radius: 999px; padding: 2px 8px; font-size: 12px; }}
.twocol {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; align-items: start; }}
.mini {{ width: 100%; font-size: 12px; border-collapse: collapse; }}
.mini caption {{ text-align: left; font-weight: 700; margin-bottom: 6px; }}
.mini th, .mini td {{ border: 1px solid #dde2e8; padding: 4px 6px; max-width: 180px; overflow-wrap: anywhere; vertical-align: top; }}
.mini th {{ background: #f6f7f9; }}
.mini .hmatch {{ background: #dff2e5; }}
.mini .vmatch {{ background: #fff2bf; }}
.mini .bothmatch {{ background: #c9ead7; box-shadow: inset 0 0 0 2px #e5c64c; }}
.columns {{ font-size: 13px; line-height: 1.45; margin: 10px 0; }}
.small {{ font-size: 12px; color: #5f6670; }}
</style>
</head>
<body>
<h1>Partial Table Overlap v1</h1>
<p class="note">Candidate-search diagnostic for partial table similarity. Header, value, column-alignment, and row-alignment components are shown separately. Yellow cells are shared values; green headers are exact shared headers.</p>
<p><strong>Cube:</strong> {esc(payload['cube'])}<br><strong>Unique tables:</strong> {payload['n_unique_tables']}<br><strong>Parameters:</strong> {esc(json.dumps(payload['parameters'], sort_keys=True))}</p>
<h2>Unique Tables</h2>
<div class="grid">{dataset_cards}</div>
<h2>Pair Summary</h2>
<div class="section"><table class="summary"><thead><tr><th>pair</th><th>candidates</th><th>best score</th><th>mean top20</th><th>best column</th><th>best row</th><th>best sources</th></tr></thead><tbody>{pair_rows}</tbody></table></div>
<h2>Visual Examples</h2>
{examples}
</body>
</html>
""",
        encoding="utf-8",
    )


def render_example(example: dict[str, Any], table_by_idx: dict[int, TableRecord]) -> str:
    left = table_by_idx[int(example["table_idx_a"])]
    right = table_by_idx[int(example["table_idx_b"])]
    shared_headers = set(example.get("shared_headers", []))
    shared_values = set(example.get("shared_values_sample", []))
    cols = example.get("matched_columns", [])
    col_text = "".join(
        f"<li>{esc(c['left_header'])} ↔ {esc(c['right_header'])} "
        f"<span class='small'>score={c['score']} header={c['header_score']} values={c['value_score']}</span></li>"
        for c in cols
    ) or "<li>No matched columns over threshold.</li>"
    return f"""
<div class="example">
  <div class="example-title">
    <h3>{esc(example['dataset_a'])} - {esc(example['dataset_b'])}</h3>
    <span class="badge">score {example['partial_table_score']:.4f}</span>
    <span class="badge">headers {example['header_jaccard']:.4f}</span>
    <span class="badge">columns {example['column_alignment_score']:.4f}</span>
    <span class="badge">rows {example['row_alignment_score']:.4f}</span>
    <span class="badge">values {example['value_overlap_coef']:.4f}</span>
  </div>
  <div class="columns"><strong>Matched columns:</strong><ul>{col_text}</ul></div>
  <div class="twocol">
    {render_mini_table(left, shared_headers, shared_values)}
    {render_mini_table(right, shared_headers, shared_values)}
  </div>
</div>
"""


def render_mini_table(table: TableRecord, shared_headers: set[str], shared_values: set[str]) -> str:
    max_cols = min(8, max([len(table.headers), *(len(r) for r in table.rows)] or [0]))
    max_rows = min(12, len(table.rows))
    header_cells = []
    for ci in range(max_cols):
        header = table.headers[ci] if ci < len(table.headers) else ""
        cls = "hmatch" if header in shared_headers else ""
        header_cells.append(f"<th class='{cls}'>{esc(header)}</th>")
    body_rows = []
    for row in table.rows[:max_rows]:
        cells = []
        for ci in range(max_cols):
            value = row[ci] if ci < len(row) else ""
            hmatch = ci < len(table.headers) and table.headers[ci] in shared_headers
            vmatch = value in shared_values and bool(value)
            cls = "bothmatch" if hmatch and vmatch else "hmatch" if hmatch else "vmatch" if vmatch else ""
            cells.append(f"<td class='{cls}'>{esc(value)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    if len(table.rows) > max_rows:
        body_rows.append(f"<tr><td colspan='{max_cols}' class='small'>... {len(table.rows) - max_rows} more rows</td></tr>")
    return (
        f"<table class='mini'><caption>{esc(table.dataset)}: {esc(table.source_id)} "
        f"<span class='small'>({len(table.rows)}x{len(table.headers)})</span></caption>"
        "<thead><tr>" + "".join(header_cells) + "</tr></thead>"
        "<tbody>" + "".join(body_rows) + "</tbody></table>"
    )


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
