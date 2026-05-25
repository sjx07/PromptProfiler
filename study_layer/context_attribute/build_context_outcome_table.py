#!/usr/bin/env python3
"""Build paired feature-outcome tables joined with context attributes.

This is intentionally a sidecar analysis artifact. It does not mutate the cube's
legacy ``predicate`` table; the context registry has its own versioned schema and
can be persisted later if we decide the compatibility tradeoff is worth it.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from study_layer.context_attribute import lightweight_extractors as extractors  # noqa: E402

DEFAULT_DB = Path("/data/users/jsu323/facet/wikitable_reasoning_default_addone_v1.db")
DEFAULT_REGISTRY = ROOT / "study_layer/context_attribute/context_attribute_registry_v1_2.json"
DEFAULT_OUT_DIR = ROOT / "study_layer/context_attribute/artifacts"
DEFAULT_OBSIDIAN_HTML = ROOT / "Obsidian/Transferability/Project/coding_agent_logs/codex/code/systematic_design/context_attribute/outcome_table_7b_v1_2.html"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
ALLOWED_KINDS = {"reasoning_default_addone", "reasoning_default_decompose_only"}


@dataclass(frozen=True)
class ConfigInfo:
    config_id: int
    dataset: str
    label: str
    kind: str
    n_eval: int
    score_mean: float


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value: Any) -> str:
    return html.escape(str(value))


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def open_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def config_rows(conn: sqlite3.Connection, model: str) -> list[ConfigInfo]:
    rows = conn.execute(
        """
        SELECT
            c.config_id AS config_id,
            q.dataset AS dataset,
            COALESCE(json_extract(c.meta, '$.label'), '') AS label,
            COALESCE(json_extract(c.meta, '$.kind'), '') AS kind,
            COUNT(ev.eval_id) AS n_eval,
            AVG(ev.score) AS score_mean
        FROM config c
        JOIN execution e ON e.config_id = c.config_id
        JOIN query q ON q.query_id = e.query_id
        JOIN evaluation ev ON ev.execution_id = e.execution_id
        WHERE e.model = ?
        GROUP BY c.config_id, q.dataset
        ORDER BY q.dataset, c.config_id
        """,
        (model,),
    ).fetchall()
    out: list[ConfigInfo] = []
    for row in rows:
        kind = str(row["kind"] or "")
        label = str(row["label"] or "")
        if kind not in ALLOWED_KINDS:
            continue
        if not row["n_eval"]:
            continue
        out.append(
            ConfigInfo(
                config_id=int(row["config_id"]),
                dataset=str(row["dataset"]),
                label=label,
                kind=kind,
                n_eval=int(row["n_eval"]),
                score_mean=float(row["score_mean"] or 0.0),
            )
        )
    return out


def resolve_pairs(configs: list[ConfigInfo]) -> tuple[dict[str, ConfigInfo], dict[str, list[ConfigInfo]]]:
    bases: dict[str, ConfigInfo] = {}
    treatments: dict[str, list[ConfigInfo]] = defaultdict(list)
    for cfg in configs:
        if cfg.label == "base":
            # Prefer the fully evaluated/current block. If duplicates exist, use
            # the config with the most eval rows, then highest id as a tiebreaker.
            old = bases.get(cfg.dataset)
            if old is None or (cfg.n_eval, cfg.config_id) > (old.n_eval, old.config_id):
                bases[cfg.dataset] = cfg
        else:
            treatments[cfg.dataset].append(cfg)
    for dataset in treatments:
        treatments[dataset].sort(key=lambda c: (family_for_label(c.label), c.label, c.config_id))
    return bases, dict(treatments)


def family_for_label(label: str) -> str:
    if "." in label:
        return label.split(".", 1)[0]
    return label or "unknown"


def load_scores(
    conn: sqlite3.Connection,
    *,
    config_id: int,
    model: str,
    dataset: str,
) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT e.query_id, ev.score, e.prediction, ev.metrics
        FROM execution e
        JOIN query q ON q.query_id = e.query_id
        JOIN evaluation ev ON ev.execution_id = e.execution_id
        WHERE e.config_id = ? AND e.model = ? AND q.dataset = ?
        """,
        (config_id, model, dataset),
    ).fetchall()
    return {
        str(row["query_id"]): {
            "score": float(row["score"] or 0.0),
            "prediction": row["prediction"] or "",
            "metrics": row["metrics"] or "{}",
        }
        for row in rows
    }


def load_context(
    conn: sqlite3.Connection,
    *,
    registry: dict[str, Any],
    datasets: Iterable[str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    atom_specs = {spec["atom"]: spec for spec in registry["atoms"]}
    atom_names = list(atom_specs)
    placeholders = ",".join("?" for _ in datasets)
    rows = conn.execute(
        f"SELECT query_id, dataset, content, meta FROM query WHERE dataset IN ({placeholders})",
        list(datasets),
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        meta = json.loads(row["meta"] or "{}")
        atoms = extractors.canonical_atoms(meta, str(row["dataset"]))
        indicators = sorted(extractors.indicator_atoms(atoms, atom_specs))
        out[str(row["query_id"])] = {
            "dataset": str(row["dataset"]),
            "content": row["content"] or "",
            "atoms": atoms,
            "indicators": indicators,
        }
    return out, atom_names


def sign(delta: float) -> str:
    if delta > 0:
        return "win"
    if delta < 0:
        return "loss"
    return "tie"


class Agg:
    __slots__ = ("n", "base_sum", "score_sum", "delta_sum", "wins", "losses", "ties")

    def __init__(self) -> None:
        self.n = 0
        self.base_sum = 0.0
        self.score_sum = 0.0
        self.delta_sum = 0.0
        self.wins = 0
        self.losses = 0
        self.ties = 0

    def add(self, base_score: float, score: float, delta: float) -> None:
        self.n += 1
        self.base_sum += base_score
        self.score_sum += score
        self.delta_sum += delta
        if delta > 0:
            self.wins += 1
        elif delta < 0:
            self.losses += 1
        else:
            self.ties += 1

    def row(self) -> dict[str, Any]:
        n = self.n or 1
        return {
            "n": self.n,
            "base_mean": self.base_sum / n,
            "score_mean": self.score_sum / n,
            "delta_mean": self.delta_sum / n,
            "win_rate": self.wins / n,
            "loss_rate": self.losses / n,
            "tie_rate": self.ties / n,
        }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_rows(
    conn: sqlite3.Connection,
    *,
    model: str,
    registry: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    configs = config_rows(conn, model)
    bases, treatments = resolve_pairs(configs)
    datasets = sorted(set(bases) & set(treatments))
    context, atom_names = load_context(conn, registry=registry, datasets=datasets)

    outcome_rows: list[dict[str, Any]] = []
    feature_aggs: dict[tuple[str, str, int], Agg] = defaultdict(Agg)
    context_aggs: dict[tuple[str, str, str], Agg] = defaultdict(Agg)
    config_coverage: list[dict[str, Any]] = []

    for dataset in datasets:
        base_cfg = bases[dataset]
        base_scores = load_scores(conn, config_id=base_cfg.config_id, model=model, dataset=dataset)
        for cfg in treatments[dataset]:
            treat_scores = load_scores(conn, config_id=cfg.config_id, model=model, dataset=dataset)
            paired_ids = sorted(set(base_scores) & set(treat_scores))
            config_coverage.append({
                "dataset": dataset,
                "label": cfg.label,
                "family": family_for_label(cfg.label),
                "feature_label": cfg.label,
                "feature_family": family_for_label(cfg.label),
                "base_config_id": base_cfg.config_id,
                "config_id": cfg.config_id,
                "base_n": len(base_scores),
                "treatment_n": len(treat_scores),
                "paired_n": len(paired_ids),
                "base_mean": base_cfg.score_mean,
                "treatment_mean": cfg.score_mean,
                "delta_mean": cfg.score_mean - base_cfg.score_mean,
            })
            fkey = (dataset, cfg.label, cfg.config_id)
            for query_id in paired_ids:
                ctx = context.get(query_id)
                if ctx is None:
                    continue
                b = base_scores[query_id]["score"]
                s = treat_scores[query_id]["score"]
                d = s - b
                direction = sign(d)
                row = {
                    "dataset": dataset,
                    "model": model,
                    "query_id": query_id,
                    "feature_label": cfg.label,
                    "feature_family": family_for_label(cfg.label),
                    "config_id": cfg.config_id,
                    "base_config_id": base_cfg.config_id,
                    "base_score": b,
                    "score": s,
                    "delta": d,
                    "direction": direction,
                    "context_indicators": ctx["indicators"],
                    "atoms": ctx["atoms"],
                }
                outcome_rows.append(row)
                feature_aggs[fkey].add(b, s, d)
                for indicator in ctx["indicators"]:
                    context_aggs[(dataset, cfg.label, indicator)].add(b, s, d)

    feature_rows: list[dict[str, Any]] = []
    global_delta: dict[tuple[str, str], float] = {}
    for (dataset, label, config_id), agg in sorted(feature_aggs.items()):
        vals = agg.row()
        global_delta[(dataset, label)] = vals["delta_mean"]
        feature_rows.append({
            "dataset": dataset,
            "model": model,
            "feature_label": label,
            "feature_family": family_for_label(label),
            "config_id": config_id,
            **vals,
        })

    context_rows: list[dict[str, Any]] = []
    for (dataset, label, indicator), agg in sorted(context_aggs.items()):
        vals = agg.row()
        g = global_delta.get((dataset, label), 0.0)
        context_rows.append({
            "dataset": dataset,
            "model": model,
            "feature_label": label,
            "feature_family": family_for_label(label),
            "context_indicator": indicator,
            **vals,
            "global_delta_mean": g,
            "diff_from_global": vals["delta_mean"] - g,
        })

    metadata = {
        "model": model,
        "datasets": datasets,
        "base_configs": {ds: bases[ds].config_id for ds in datasets},
        "n_outcome_rows": len(outcome_rows),
        "n_feature_rows": len(feature_rows),
        "n_context_rows": len(context_rows),
        "atom_names": atom_names,
        "config_coverage": config_coverage,
    }
    return outcome_rows, feature_rows, context_rows, metadata


def compact_outcome_csv_rows(rows: list[dict[str, Any]], atom_names: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        atoms = row["atoms"]
        out.append({
            "dataset": row["dataset"],
            "model": row["model"],
            "query_id": row["query_id"],
            "feature_label": row["feature_label"],
            "feature_family": row["feature_family"],
            "config_id": row["config_id"],
            "base_config_id": row["base_config_id"],
            "base_score": row["base_score"],
            "score": row["score"],
            "delta": row["delta"],
            "direction": row["direction"],
            **{atom: atoms.get(atom, "") for atom in atom_names},
        })
    return out


def write_html(
    path: Path,
    *,
    metadata: dict[str, Any],
    feature_rows: list[dict[str, Any]],
    context_rows: list[dict[str, Any]],
    artifact_paths: dict[str, Path],
) -> None:
    def table(headers: list[str], rows: list[dict[str, Any]], limit: int | None = None) -> str:
        use_rows = rows if limit is None else rows[:limit]
        body = []
        for row in use_rows:
            cells = []
            for h in headers:
                val = row.get(h, "")
                if isinstance(val, float):
                    val = fmt(val, 4)
                cells.append(f"<td>{esc(val)}</td>")
            body.append("<tr>" + "".join(cells) + "</tr>")
        return "<table><thead><tr>" + "".join(f"<th>{esc(h)}</th>" for h in headers) + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"

    by_dataset = Counter(row["dataset"] for row in metadata["config_coverage"])
    coverage_rows = sorted(metadata["config_coverage"], key=lambda r: (r["dataset"], r["feature_family"], r["feature_label"]))
    feature_sorted = sorted(feature_rows, key=lambda r: (r["dataset"], -r["delta_mean"], r["feature_label"]))
    strong_context = [
        r for r in context_rows
        if r["n"] >= 100 and abs(r["diff_from_global"]) >= 0.03
    ]
    strong_context.sort(key=lambda r: (abs(r["diff_from_global"]), abs(r["delta_mean"]), r["n"]), reverse=True)

    links = "".join(
        f'<li><code>{esc(name)}</code>: <code>{esc(path)}</code></li>'
        for name, path in artifact_paths.items()
    )
    pills = "".join(
        f'<span class="pill">{esc(ds)}: {by_dataset[ds]} treatments</span>'
        for ds in metadata["datasets"]
    )
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>7B context outcome table v1.2</title>
  <style>
    body {{ margin: 24px; font-family: Arial, sans-serif; color: #1f2933; background: #f7f8fa; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    h2 {{ margin-top: 28px; font-size: 19px; }}
    .subtle {{ color: #667085; font-size: 12px; }}
    .pills {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 14px 0; }}
    .pill {{ background: #e8eef7; border: 1px solid #cbd6e7; border-radius: 999px; padding: 5px 10px; font-size: 13px; }}
    .note {{ background: #fff8df; border: 1px solid #ead88a; border-radius: 8px; padding: 12px; max-width: 980px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d8dee8; margin: 12px 0 24px; }}
    th, td {{ border-bottom: 1px solid #e5e9f0; padding: 7px 8px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #edf2f7; font-size: 11px; text-transform: uppercase; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>7B Context Outcome Table v1.2</h1>
  <div class="subtle">Model: <code>{esc(metadata['model'])}</code>. Outcome rows: {metadata['n_outcome_rows']}. Context rows: {metadata['n_context_rows']}.</div>
  <div class="pills">{pills}</div>
  <div class="note"><b>Persistence decision:</b> this build does not mutate the cube <code>predicate</code> table. The v1.2 context atoms remain a versioned sidecar artifact for now, because legacy predicates mix older task-specific names and the new registry has categorical atoms, support metadata, and extractor provenance. If we need old-view compatibility later, use a prefixed export such as <code>ctx_v1_2.table.rows_bin</code>.</div>

  <h2>Artifacts</h2>
  <ul>{links}</ul>

  <h2>Config Coverage</h2>
  {table(['dataset','feature_label','feature_family','base_config_id','config_id','base_n','treatment_n','paired_n','base_mean','treatment_mean','delta_mean'], coverage_rows)}

  <h2>Feature Summary</h2>
  {table(['dataset','feature_label','feature_family','n','base_mean','score_mean','delta_mean','win_rate','loss_rate','tie_rate'], feature_sorted)}

  <h2>Large Context-Conditional Deviations</h2>
  <div class="subtle">Shown when support n &gt;= 100 and abs(diff from feature global delta) &gt;= 3 points.</div>
  {table(['dataset','feature_label','feature_family','context_indicator','n','delta_mean','global_delta_mean','diff_from_global','win_rate','loss_rate'], strong_context, limit=250)}
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prefix", default="outcome_7b_v1_2")
    parser.add_argument("--obsidian-html", type=Path, default=DEFAULT_OBSIDIAN_HTML)
    args = parser.parse_args()

    registry = load_json(args.registry)
    conn = open_conn(args.db)
    outcome_rows, feature_rows, context_rows, metadata = build_rows(conn, model=args.model, registry=registry)
    conn.close()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "jsonl": out_dir / f"{args.prefix}.jsonl",
        "csv": out_dir / f"{args.prefix}.csv",
        "feature_summary_csv": out_dir / f"{args.prefix}.feature_summary.csv",
        "context_summary_csv": out_dir / f"{args.prefix}.context_summary.csv",
        "metadata_json": out_dir / f"{args.prefix}.metadata.json",
        "html": out_dir / f"{args.prefix}.html",
    }

    write_jsonl(paths["jsonl"], outcome_rows)
    csv_rows = compact_outcome_csv_rows(outcome_rows, metadata["atom_names"])
    base_fields = [
        "dataset", "model", "query_id", "feature_label", "feature_family",
        "config_id", "base_config_id", "base_score", "score", "delta", "direction",
    ]
    write_csv(paths["csv"], csv_rows, base_fields + metadata["atom_names"])
    write_csv(
        paths["feature_summary_csv"],
        feature_rows,
        ["dataset", "model", "feature_label", "feature_family", "config_id", "n", "base_mean", "score_mean", "delta_mean", "win_rate", "loss_rate", "tie_rate"],
    )
    write_csv(
        paths["context_summary_csv"],
        context_rows,
        ["dataset", "model", "feature_label", "feature_family", "context_indicator", "n", "base_mean", "score_mean", "delta_mean", "global_delta_mean", "diff_from_global", "win_rate", "loss_rate", "tie_rate"],
    )
    paths["metadata_json"].write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_html(paths["html"], metadata=metadata, feature_rows=feature_rows, context_rows=context_rows, artifact_paths=paths)
    if args.obsidian_html:
        write_html(args.obsidian_html, metadata=metadata, feature_rows=feature_rows, context_rows=context_rows, artifact_paths=paths)

    print(f"model: {args.model}")
    print(f"outcome_rows: {len(outcome_rows)}")
    print(f"feature_rows: {len(feature_rows)}")
    print(f"context_rows: {len(context_rows)}")
    for name, path in paths.items():
        print(f"{name}: {path}")
    if args.obsidian_html:
        print(f"obsidian_html: {args.obsidian_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
