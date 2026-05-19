#!/usr/bin/env python3
"""Build typed benchmark-domain similarity artifacts from cube query metadata.

The script intentionally has no third-party dependencies. It treats typed text
views as local TF-IDF embeddings:

    query  = question / claim / statement text
    schema = table title, caption, source, headers, passage titles
    values = sampled table cells or evidence entities
    joint  = query + schema + values

This keeps the output reproducible in the experiment environment. A neural
embedding backend can replace ``build_vectors`` later without changing the
view extraction or output schema.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CUBE = Path("/data/users/jsu323/facet/wikitable_transfer_cube.db")
DEFAULT_OUT_DIR = ROOT / (
    "Obsidian/Transferability/Project/coding_agent_logs/codex/code/"
    "systematic_design/benchmark_design/domain_similarity_embedding"
)
VIEWS = ("query", "schema", "values", "joint")
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "how", "in", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "there", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "with",
}

DOMAIN_DESCRIPTIONS = {
    "wikitable.sports": (
        "sports score result record rank place position seed points medals race "
        "game match tournament season team player winner runner event"
    ),
    "wikitable.election": (
        "candidate party district riding incumbent votes percent share result "
        "majority election beat lost democrat republican"
    ),
    "wikitable.media": (
        "episode season series production code air date directed written viewers "
        "rating television film movie title"
    ),
    "wikitable.music": (
        "song single album artist producer featured guest sales chart peak position "
        "track music label"
    ),
    "wikitable.geo": (
        "country county city state province location venue nationality represent "
        "capital airport region"
    ),
    "wikitable.identifier": (
        "number no code route unicode symbol rank place position pick seed id "
        "identifier c string glyph"
    ),
    "wikipedia.title_alias": (
        "wikipedia title alias disambiguation entity proper noun page relation "
        "bridge article retrieved supporting fact"
    ),
}


@dataclass(frozen=True)
class Instance:
    dataset: str
    query_id: str
    split: str
    views: dict[str, str]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube", type=Path, default=DEFAULT_CUBE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--datasets",
        default="",
        help="Comma-separated dataset filter. Default: all datasets in cube.",
    )
    parser.add_argument(
        "--max-per-dataset",
        type=int,
        default=512,
        help="Stable sample size per dataset for pairwise kNN. Use 0 for all.",
    )
    parser.add_argument(
        "--cell-limit",
        type=int,
        default=120,
        help="Maximum sampled table/evidence values per instance.",
    )
    parser.add_argument(
        "--max-tokens-per-view",
        type=int,
        default=260,
        help="Maximum tokens retained from each typed view per instance.",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()] or None
    instances = load_instances(
        args.cube,
        datasets=datasets,
        max_per_dataset=args.max_per_dataset,
        cell_limit=args.cell_limit,
    )
    if not instances:
        raise SystemExit(f"No instances loaded from {args.cube}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {
        "cube": str(args.cube),
        "max_per_dataset": args.max_per_dataset,
        "cell_limit": args.cell_limit,
        "max_tokens_per_view": args.max_tokens_per_view,
        "datasets": {},
        "pairwise": [],
        "domain_scores": [],
        "distinctive_terms": {},
    }

    by_dataset: dict[str, list[Instance]] = defaultdict(list)
    for inst in instances:
        by_dataset[inst.dataset].append(inst)
    for dataset, rows in sorted(by_dataset.items()):
        summaries["datasets"][dataset] = len(rows)

    vectors_by_view: dict[str, list[dict[str, float]]] = {}
    idf_by_view: dict[str, dict[str, float]] = {}
    tokens_by_view: dict[str, list[list[str]]] = {}
    for view in VIEWS:
        docs = [
            tokenize(inst.views.get(view, ""), max_tokens=args.max_tokens_per_view)
            for inst in instances
        ]
        tokens_by_view[view] = docs
        vectors, idf = build_vectors(docs)
        vectors_by_view[view] = vectors
        idf_by_view[view] = idf

    pair_rows = compute_pairwise(instances, by_dataset, vectors_by_view)
    summaries["pairwise"] = pair_rows

    domain_rows = compute_domain_scores(
        instances,
        by_dataset,
        vectors_by_view["joint"],
        idf_by_view["joint"],
        tokens_by_view["joint"],
    )
    summaries["domain_scores"] = domain_rows

    term_summary = compute_distinctive_terms(
        instances,
        by_dataset,
        vectors_by_view,
        top_k=12,
    )
    summaries["distinctive_terms"] = term_summary

    write_csv(args.out_dir / "benchmark_domain_similarity_v0_pairwise.csv", pair_rows)
    write_csv(args.out_dir / "benchmark_domain_similarity_v0_domain_scores.csv", domain_rows)
    (args.out_dir / "benchmark_domain_similarity_v0.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_html(args.out_dir / "benchmark_domain_similarity_v0.html", summaries)

    print(f"loaded {len(instances)} instances from {args.cube}")
    print(f"wrote {args.out_dir / 'benchmark_domain_similarity_v0.html'}")
    return 0


def load_instances(
    cube: Path,
    *,
    datasets: list[str] | None,
    max_per_dataset: int,
    cell_limit: int,
) -> list[Instance]:
    conn = sqlite3.connect(str(cube))
    conn.row_factory = sqlite3.Row
    clauses: list[str] = []
    params: list[Any] = []
    if datasets:
        placeholders = ",".join("?" for _ in datasets)
        clauses.append(f"dataset in ({placeholders})")
        params.extend(datasets)
    where = f"where {' and '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        f"select query_id, dataset, content, meta from query {where}",
        params,
    ).fetchall()
    conn.close()

    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)

    selected: list[sqlite3.Row] = []
    for dataset, ds_rows in sorted(grouped.items()):
        ds_rows = sorted(ds_rows, key=lambda r: stable_hash(str(r["query_id"])))
        if max_per_dataset > 0:
            ds_rows = ds_rows[:max_per_dataset]
        selected.extend(ds_rows)

    instances: list[Instance] = []
    for row in selected:
        meta = parse_json(row["meta"], {})
        views = extract_views(
            dataset=str(row["dataset"]),
            content=str(row["content"] or ""),
            meta=meta,
            cell_limit=cell_limit,
        )
        instances.append(
            Instance(
                dataset=str(row["dataset"]),
                query_id=str(row["query_id"]),
                split=str(meta.get("split", "")),
                views=views,
            )
        )
    return instances


def extract_views(
    *,
    dataset: str,
    content: str,
    meta: dict[str, Any],
    cell_limit: int,
) -> dict[str, str]:
    raw = meta.get("_raw", {}) if isinstance(meta.get("_raw"), dict) else {}
    query_parts = [content]
    schema_parts: list[str] = []
    value_parts: list[str] = []

    if dataset == "sqa":
        for turn in raw.get("history", []) or []:
            if isinstance(turn, dict):
                query_parts.append(str(turn.get("question", "")))
                query_parts.extend(str(a) for a in turn.get("answer", [])[:5])
        schema_parts.extend([str(meta.get("table_file", "")), *headers_from_table(raw.get("table"))])
        value_parts.extend(values_from_table(raw.get("table"), cell_limit=cell_limit))
    elif dataset == "wtq":
        table = raw.get("table")
        schema_parts.extend([str(meta.get("table_name", "")), *headers_from_table(table)])
        value_parts.extend(values_from_table(table, cell_limit=cell_limit))
    elif dataset in {"tab_fact", "tabfact"}:
        schema_parts.extend([str(meta.get("table_caption", "")), str(meta.get("table_id", ""))])
        headers, values = parse_tabfact_text(str(raw.get("table_text", "")), cell_limit)
        schema_parts.extend(headers)
        value_parts.extend(values)
    elif dataset == "tablebench":
        table = raw.get("table")
        schema_parts.extend([
            str(raw.get("qtype", "")),
            str(raw.get("qsubtype", "")),
            str(raw.get("instruction_type", "")),
            *headers_from_table(table),
        ])
        value_parts.extend(values_from_table(table, cell_limit=cell_limit))
    elif dataset == "hitab":
        table_content = raw.get("table_content", {})
        title, headers, values = parse_hitab_table(table_content, cell_limit=cell_limit)
        schema_parts.extend([
            str(meta.get("table_source", "")),
            str(meta.get("aggregation", "")),
            title,
            *headers,
        ])
        value_parts.extend(values)
    elif dataset in {"hover_context", "hotpotqa_context"}:
        facts = raw.get("supporting_facts", []) or meta.get("supporting_facts", []) or []
        titles = []
        for fact in facts:
            if isinstance(fact, (list, tuple)) and fact:
                titles.append(str(fact[0]))
            elif isinstance(fact, dict):
                titles.append(str(fact.get("title", "")))
        schema_parts.extend(titles)
        schema_parts.append(str(meta.get("label", "")))
        value_parts.extend(titles[:cell_limit])
    else:
        schema_parts.extend(flatten_known_strings(meta, max_items=40))

    query = " ".join(query_parts)
    schema = " ".join(schema_parts)
    values = " ".join(value_parts)
    return {
        "query": query,
        "schema": schema,
        "values": values,
        "joint": " ".join([query, schema, values]),
    }


def headers_from_table(table: Any) -> list[str]:
    if not isinstance(table, dict):
        return []
    headers = table.get("header") or table.get("headers") or []
    if isinstance(headers, list):
        return [str(h) for h in headers]
    return []


def values_from_table(table: Any, *, cell_limit: int) -> list[str]:
    if not isinstance(table, dict):
        return []
    rows = table.get("rows") or table.get("data") or []
    values: list[str] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, list):
                continue
            for cell in row:
                text = str(cell).strip()
                if text:
                    values.append(text)
                    if len(values) >= cell_limit:
                        return values
    return values


def parse_tabfact_text(table_text: str, cell_limit: int) -> tuple[list[str], list[str]]:
    lines = [line.strip() for line in table_text.splitlines() if line.strip()]
    if not lines:
        return [], []
    headers = [part.strip() for part in lines[0].split("#") if part.strip()]
    values: list[str] = []
    for line in lines[1:]:
        for cell in line.split("#"):
            cell = cell.strip()
            if cell:
                values.append(cell)
                if len(values) >= cell_limit:
                    return headers, values
    return headers, values


def parse_hitab_table(table_content: Any, *, cell_limit: int) -> tuple[str, list[str], list[str]]:
    if not isinstance(table_content, dict):
        return "", [], []
    title = str(table_content.get("title", ""))
    texts = table_content.get("texts", [])
    if not isinstance(texts, list):
        return title, [], []
    header_n = int(table_content.get("top_header_rows_num", 1) or 1)
    headers: list[str] = []
    values: list[str] = []
    for i, row in enumerate(texts):
        if not isinstance(row, list):
            continue
        target = headers if i < header_n else values
        for cell in row:
            text = str(cell).strip()
            if not text:
                continue
            target.append(text)
            if len(values) >= cell_limit:
                return title, headers, values
    return title, headers, values


def flatten_known_strings(obj: Any, *, max_items: int) -> list[str]:
    out: list[str] = []

    def visit(value: Any) -> None:
        if len(out) >= max_items:
            return
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, dict):
            for k, v in value.items():
                if str(k).startswith("_"):
                    continue
                visit(v)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(obj)
    return out[:max_items]


def tokenize(text: str, *, max_tokens: int) -> list[str]:
    tokens = [
        token
        for token in TOKEN_RE.findall(text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]
    return tokens[:max_tokens]


def build_vectors(docs: list[list[str]]) -> tuple[list[dict[str, float]], dict[str, float]]:
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set(doc))
    n_docs = max(len(docs), 1)
    idf = {
        term: math.log((1 + n_docs) / (1 + freq)) + 1.0
        for term, freq in df.items()
    }
    vectors = [vectorize_doc(doc, idf) for doc in docs]
    return vectors, idf


def vectorize_doc(doc: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(doc)
    weighted = {term: count * idf.get(term, 0.0) for term, count in counts.items()}
    return normalize(weighted)


def normalize(vec: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm <= 0:
        return {}
    return {k: v / norm for k, v in vec.items() if v}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def centroid(vectors: list[dict[str, float]]) -> dict[str, float]:
    acc: defaultdict[str, float] = defaultdict(float)
    if not vectors:
        return {}
    scale = 1.0 / len(vectors)
    for vec in vectors:
        for term, val in vec.items():
            acc[term] += val * scale
    return normalize(dict(acc))


def compute_pairwise(
    instances: list[Instance],
    by_dataset: dict[str, list[Instance]],
    vectors_by_view: dict[str, list[dict[str, float]]],
) -> list[dict[str, Any]]:
    index_by_query = {inst.query_id: i for i, inst in enumerate(instances)}
    datasets = sorted(by_dataset)
    rows: list[dict[str, Any]] = []
    for view in VIEWS:
        vectors = vectors_by_view[view]
        ds_vectors = {
            ds: [vectors[index_by_query[inst.query_id]] for inst in rows_]
            for ds, rows_ in by_dataset.items()
        }
        centroids = {ds: centroid(vecs) for ds, vecs in ds_vectors.items()}
        for i, left in enumerate(datasets):
            for right in datasets[i + 1:]:
                lvecs = ds_vectors[left]
                rvecs = ds_vectors[right]
                l_to_r = directed_knn(lvecs, rvecs)
                r_to_l = directed_knn(rvecs, lvecs)
                rows.append({
                    "view": view,
                    "dataset_a": left,
                    "dataset_b": right,
                    "n_a": len(lvecs),
                    "n_b": len(rvecs),
                    "centroid_cosine": round(cosine(centroids[left], centroids[right]), 4),
                    "knn_a_to_b": round(l_to_r, 4),
                    "knn_b_to_a": round(r_to_l, 4),
                    "symmetric_knn": round((l_to_r + r_to_l) / 2.0, 4),
                })
    return rows


def directed_knn(left: list[dict[str, float]], right: list[dict[str, float]]) -> float:
    if not left or not right:
        return 0.0
    total = 0.0
    for vec in left:
        total += max((cosine(vec, candidate) for candidate in right), default=0.0)
    return total / len(left)


def compute_domain_scores(
    instances: list[Instance],
    by_dataset: dict[str, list[Instance]],
    joint_vectors: list[dict[str, float]],
    joint_idf: dict[str, float],
    joint_tokens: list[list[str]],
) -> list[dict[str, Any]]:
    index_by_query = {inst.query_id: i for i, inst in enumerate(instances)}
    domain_vectors = {
        domain: vectorize_doc(tokenize(text, max_tokens=200), joint_idf)
        for domain, text in DOMAIN_DESCRIPTIONS.items()
    }
    domain_terms = {
        domain: set(tokenize(text, max_tokens=200))
        for domain, text in DOMAIN_DESCRIPTIONS.items()
    }
    rows: list[dict[str, Any]] = []
    for dataset, ds_rows in sorted(by_dataset.items()):
        ds_vecs = [joint_vectors[index_by_query[inst.query_id]] for inst in ds_rows]
        ds_tokens = [joint_tokens[index_by_query[inst.query_id]] for inst in ds_rows]
        for domain, dvec in domain_vectors.items():
            scores = [cosine(vec, dvec) for vec in ds_vecs]
            terms = domain_terms[domain]
            hit_docs = 0
            hit_total = 0
            token_total = 0
            for doc in ds_tokens:
                hits = sum(1 for token in doc if token in terms)
                hit_total += hits
                token_total += len(doc)
                if hits:
                    hit_docs += 1
            rows.append({
                "dataset": dataset,
                "domain": domain,
                "n": len(scores),
                "mean_affinity": round(sum(scores) / len(scores), 4) if scores else 0.0,
                "p90_affinity": round(percentile(scores, 0.90), 4) if scores else 0.0,
                "hit_rate": round(hit_docs / len(ds_tokens), 4) if ds_tokens else 0.0,
                "hits_per_100_tokens": round(100.0 * hit_total / token_total, 4) if token_total else 0.0,
            })
    return rows


def compute_distinctive_terms(
    instances: list[Instance],
    by_dataset: dict[str, list[Instance]],
    vectors_by_view: dict[str, list[dict[str, float]]],
    *,
    top_k: int,
) -> dict[str, dict[str, list[str]]]:
    index_by_query = {inst.query_id: i for i, inst in enumerate(instances)}
    out: dict[str, dict[str, list[str]]] = {}
    for view, vectors in vectors_by_view.items():
        global_centroid = centroid(vectors)
        out[view] = {}
        for dataset, ds_rows in sorted(by_dataset.items()):
            c = centroid([vectors[index_by_query[inst.query_id]] for inst in ds_rows])
            ranked = sorted(
                c.items(),
                key=lambda kv: (kv[1] - 0.35 * global_centroid.get(kv[0], 0.0)),
                reverse=True,
            )
            out[view][dataset] = [term for term, _ in ranked[:top_k]]
    return out


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return values[idx]


def stable_hash(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def parse_json(raw: Any, default: Any) -> Any:
    if isinstance(raw, (dict, list)):
        return raw
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
    pairwise = summary["pairwise"]
    domain_scores = summary["domain_scores"]
    datasets = sorted(summary["datasets"])
    joint_pairs = [row for row in pairwise if row["view"] == "joint"]
    top_pairs = sorted(joint_pairs, key=lambda r: r["symmetric_knn"], reverse=True)[:12]

    matrix_html = render_matrix(joint_pairs, datasets, value_key="symmetric_knn")
    centroid_html = render_matrix(joint_pairs, datasets, value_key="centroid_cosine")
    domain_html = render_domain_table(domain_scores, value_key="mean_affinity")
    domain_hit_html = render_domain_table(domain_scores, value_key="hit_rate")
    terms_html = render_terms(summary["distinctive_terms"])
    top_pairs_html = "\n".join(
        "<tr>"
        f"<td>{esc(row['dataset_a'])} -> {esc(row['dataset_b'])}</td>"
        f"<td>{row['symmetric_knn']:.4f}</td>"
        f"<td>{row['centroid_cosine']:.4f}</td>"
        f"<td>{row['n_a']} / {row['n_b']}</td>"
        "</tr>"
        for row in top_pairs
    )
    dataset_cards = "\n".join(
        f"<div class='card'><div class='num'>{summary['datasets'][ds]}</div><div>{esc(ds)}</div></div>"
        for ds in datasets
    )
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Benchmark Domain Similarity v0</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 28px; color: #202124; }}
h1, h2 {{ margin: 0 0 12px; }}
h2 {{ margin-top: 30px; }}
p {{ max-width: 980px; line-height: 1.45; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; max-width: 1100px; }}
.card {{ border: 1px solid #d5d8de; border-radius: 8px; padding: 12px; background: #fafafa; }}
.num {{ font-size: 22px; font-weight: 700; }}
table {{ border-collapse: collapse; margin: 12px 0 24px; min-width: 760px; }}
th, td {{ border: 1px solid #d8dce3; padding: 7px 9px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f1f3f6; }}
.section {{ overflow-x: auto; }}
.note {{ background: #fff8df; border: 1px solid #ead88a; border-radius: 8px; padding: 12px; max-width: 980px; }}
.pill {{ display: inline-block; background: #eef3ff; border: 1px solid #cbd8ff; border-radius: 999px; padding: 2px 8px; margin: 2px; }}
</style>
</head>
<body>
<h1>Benchmark Domain Similarity v0</h1>
<p class="note">Local analysis artifact. Typed query/table views are embedded with dependency-free TF-IDF vectors. Treat this as benchmark-design diagnostics, not a final claim.</p>
<p><strong>Cube:</strong> {esc(summary['cube'])}<br>
<strong>Sample:</strong> max {summary['max_per_dataset']} per dataset; cell limit {summary['cell_limit']}; token cap {summary['max_tokens_per_view']} per view.</p>
<h2>Loaded Instances</h2>
<div class="grid">{dataset_cards}</div>
<h2>Joint Similarity Matrix: Symmetric kNN</h2>
<p>High values mean individual examples in one benchmark tend to find close neighbors in the other benchmark.</p>
<div class="section">{matrix_html}</div>
<h2>Joint Similarity Matrix: Centroid Cosine</h2>
<p>High values mean the aggregate benchmark distributions are lexically/source similar.</p>
<div class="section">{centroid_html}</div>
<h2>Top Benchmark Pairs</h2>
<table><thead><tr><th>pair</th><th>symmetric kNN</th><th>centroid</th><th>n</th></tr></thead><tbody>{top_pairs_html}</tbody></table>
<h2>Domain Subpack Affinity</h2>
<p>Soft context attributes computed by similarity between each instance and domain-subpack descriptions.</p>
<div class="section">{domain_html}</div>
<h2>Domain Subpack Hit Rate</h2>
<p>Fraction of instances whose joint query/table text contains at least one term from the domain-subpack lexicon. This is a rough activation sanity check, not the final trigger.</p>
<div class="section">{domain_hit_html}</div>
<h2>Distinctive Terms</h2>
<p>Top centroid terms per dataset and typed view. This is mainly for sanity-checking whether the view extraction is capturing the intended source/task surface.</p>
{terms_html}
</body>
</html>
""",
        encoding="utf-8",
    )


def render_matrix(rows: list[dict[str, Any]], datasets: list[str], *, value_key: str) -> str:
    lookup = {}
    for row in rows:
        lookup[(row["dataset_a"], row["dataset_b"])] = row[value_key]
        lookup[(row["dataset_b"], row["dataset_a"])] = row[value_key]
    header = "<tr><th>dataset</th>" + "".join(f"<th>{esc(ds)}</th>" for ds in datasets) + "</tr>"
    body_rows = []
    for left in datasets:
        cells = [f"<th>{esc(left)}</th>"]
        for right in datasets:
            value = 1.0 if left == right else lookup.get((left, right), 0.0)
            cells.append(f"<td>{value:.4f}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table><thead>" + header + "</thead><tbody>" + "\n".join(body_rows) + "</tbody></table>"


def render_domain_table(rows: list[dict[str, Any]], *, value_key: str) -> str:
    datasets = sorted({row["dataset"] for row in rows})
    domains = sorted({row["domain"] for row in rows})
    lookup = {(row["dataset"], row["domain"]): row for row in rows}
    header = "<tr><th>domain</th>" + "".join(f"<th>{esc(ds)}</th>" for ds in datasets) + "</tr>"
    body_rows = []
    for domain in domains:
        cells = [f"<th>{esc(domain)}</th>"]
        for dataset in datasets:
            row = lookup.get((dataset, domain), {})
            cells.append(f"<td>{float(row.get(value_key, 0.0)):.4f}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return "<table><thead>" + header + "</thead><tbody>" + "\n".join(body_rows) + "</tbody></table>"


def render_terms(summary: dict[str, dict[str, list[str]]]) -> str:
    parts: list[str] = []
    for view, by_dataset in summary.items():
        rows = []
        for dataset, terms in by_dataset.items():
            pills = " ".join(f"<span class='pill'>{esc(t)}</span>" for t in terms)
            rows.append(f"<tr><th>{esc(dataset)}</th><td>{pills}</td></tr>")
        parts.append(
            f"<h3>{esc(view)}</h3><table><thead><tr><th>dataset</th><th>terms</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
        )
    return "\n".join(parts)


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
