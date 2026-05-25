#!/usr/bin/env python3
"""Rule learning for add-one FACET feature interventions.

This runner consumes an existing cube and treats each add-one config as a paired
feature intervention against the benchmark-local base config. It mines low-order
context rules for heterogeneous treatment effects within benchmarks and pooled
binary useful-rate transfer rules across benchmarks.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analyze.context_attributes import canonical_context_items
from analyze.facet_matrix_ops.itemsets import frequent_itemsets
from analyze.facet_matrix_ops.stats import bootstrap_mean_delta_ci, wilson_interval

DEFAULT_DB = Path("/data/users/jsu323/facet/wikitable_reasoning_default_addone_v1.db")
DEFAULT_OUT = Path("study_layer/artifacts/addone_context_rule_learning_7b_v1")
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DATASET_SCORERS = {
    "wtq": "denotation_acc",
    "sqa": "denotation_acc",
    "tablebench": "tb_official_acc",
    "tab_fact": "fv_acc",
    "hitab": "denotation_acc",
}
STRUCTURAL_PREFIXES = ("_section_", "facet_dp_scaffold")


@dataclass(frozen=True)
class ConfigInfo:
    config_id: int
    label: str
    canonical_ids: tuple[str, ...]
    added_atoms: tuple[str, ...]


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def load_json(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        if not value:
            return {}
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"_text": value}
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    return value or {}


def query_split(meta: Any) -> str | None:
    obj = load_json(meta)
    raw = load_json(obj.get("_raw", {}))
    return obj.get("split") or raw.get("split")


def canonical_ids_from_meta(meta: Mapping[str, Any]) -> tuple[str, ...]:
    values = []
    for key in ("canonical_ids", "canonical_id"):
        raw = meta.get(key)
        if isinstance(raw, str):
            values.append(raw)
        elif raw:
            values.extend(str(x) for x in raw)
    return tuple(sorted(set(values)))


def is_structural(atom: str) -> bool:
    return atom.startswith(STRUCTURAL_PREFIXES)


def feature_family(label: str, atom: str) -> str:
    if label.startswith("format.") or atom.startswith("prompt_format_"):
        return "format"
    if label.startswith("ser.") or atom.startswith("table_serialization_"):
        return "serialization"
    if label.startswith("ctx.") or atom.startswith("input_context_"):
        return "input_context"
    if label.startswith("reason.") or atom.startswith("reasoning_"):
        return "reasoning"
    return "other"


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def normal_ci(values: Sequence[float], center: float = 0.0) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    centered = [float(x) - center for x in values]
    mu = mean(centered)
    if len(centered) < 2:
        return mu, mu
    half = 1.96 * stdev(centered) / math.sqrt(len(centered))
    return mu - half, mu + half


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = {}
            for key in keys:
                val = row.get(key, "")
                if isinstance(val, (tuple, list, set)):
                    out[key] = "|".join(str(x) for x in val)
                else:
                    out[key] = val
            writer.writerow(out)


def load_configs(conn: sqlite3.Connection) -> dict[int, ConfigInfo]:
    rows = conn.execute("SELECT config_id, meta FROM config ORDER BY config_id").fetchall()
    raw: dict[int, tuple[str, tuple[str, ...]]] = {}
    for row in rows:
        meta = load_json(row["meta"])
        label = str(meta.get("label") or meta.get("config_label") or row["config_id"])
        raw[int(row["config_id"])] = (label, canonical_ids_from_meta(meta))
    # Base atoms are benchmark-local. We infer added atoms later relative to each dataset's base.
    return {
        cid: ConfigInfo(cid, label, atoms, tuple())
        for cid, (label, atoms) in raw.items()
    }


def find_base_config(conn: sqlite3.Connection, dataset: str, model: str, scorer: str) -> int:
    rows = conn.execute(
        """
        SELECT e.config_id, c.meta, COUNT(*) AS n_eval
        FROM execution e
        JOIN evaluation ev ON ev.execution_id = e.execution_id
        JOIN query q ON q.query_id = e.query_id
        JOIN config c ON c.config_id = e.config_id
        WHERE q.dataset = ? AND e.model = ? AND ev.scorer = ?
          AND (e.error IS NULL OR e.error = '') AND ev.score IS NOT NULL
        GROUP BY e.config_id
        ORDER BY n_eval DESC, e.config_id
        """,
        (dataset, model, scorer),
    ).fetchall()
    for row in rows:
        meta = load_json(row["meta"])
        if str(meta.get("label") or "") == "base":
            return int(row["config_id"])
    raise RuntimeError(f"no base config found for {dataset} {model} {scorer}")


def load_query_meta(conn: sqlite3.Connection, datasets: Sequence[str]) -> dict[tuple[str, str], dict[str, Any]]:
    qmarks = ",".join("?" for _ in datasets)
    rows = conn.execute(
        f"SELECT query_id, dataset, content, meta FROM query WHERE dataset IN ({qmarks})",
        tuple(datasets),
    ).fetchall()
    out = {}
    for row in rows:
        meta = load_json(row["meta"])
        content = load_json(row["content"])
        out[(str(row["dataset"]), str(row["query_id"]))] = {
            "query_id": str(row["query_id"]),
            "dataset": str(row["dataset"]),
            "split": query_split(meta),
            "meta": meta,
            "content": content,
        }
    return out


def build_context_maps(
    query_meta: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    include_negative: bool,
) -> tuple[dict[tuple[str, str], frozenset[str]], dict[tuple[str, str], frozenset[str]]]:
    within = {}
    shared = {}
    for key, row in query_meta.items():
        dataset = str(row["dataset"])
        meta = row["meta"]
        within[key] = frozenset(
            canonical_context_items(meta, dataset, view="all", include_negative=include_negative)
        )
        shared[key] = frozenset(
            canonical_context_items(meta, dataset, view="shared", include_negative=include_negative)
        )
    return within, shared


def load_addone_observations(
    conn: sqlite3.Connection,
    *,
    datasets: Sequence[str],
    model: str,
    configs: Mapping[int, ConfigInfo],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations: list[dict[str, Any]] = []
    feature_summaries: list[dict[str, Any]] = []
    for dataset in datasets:
        scorer = DATASET_SCORERS[dataset]
        base_config = find_base_config(conn, dataset, model, scorer)
        base_atoms = set(configs[base_config].canonical_ids)
        rows = conn.execute(
            """
            WITH base AS (
                SELECT e.query_id, ev.score AS base_score
                FROM execution e
                JOIN evaluation ev ON ev.execution_id = e.execution_id
                WHERE e.config_id = ? AND e.model = ? AND ev.scorer = ?
                  AND (e.error IS NULL OR e.error = '') AND ev.score IS NOT NULL
            )
            SELECT e.config_id, e.query_id, ev.score, base.base_score
            FROM execution e
            JOIN evaluation ev ON ev.execution_id = e.execution_id
            JOIN query q ON q.query_id = e.query_id
            JOIN base ON base.query_id = e.query_id
            WHERE q.dataset = ? AND e.model = ? AND ev.scorer = ?
              AND e.config_id != ?
              AND (e.error IS NULL OR e.error = '') AND ev.score IS NOT NULL
            ORDER BY e.config_id, e.query_id
            """,
            (base_config, model, scorer, dataset, model, scorer, base_config),
        ).fetchall()
        by_feature: dict[str, list[float]] = defaultdict(list)
        feature_meta: dict[str, dict[str, Any]] = {}
        for row in rows:
            cfg = configs[int(row["config_id"])]
            added = tuple(
                sorted(
                    atom for atom in set(cfg.canonical_ids) - base_atoms
                    if not is_structural(atom)
                )
            )
            if not added:
                continue
            feature_atom = added[0] if len(added) == 1 else "+".join(added)
            family = feature_family(cfg.label, feature_atom)
            delta = float(row["score"]) - float(row["base_score"])
            observations.append({
                "dataset": dataset,
                "model": model,
                "scorer": scorer,
                "base_config_id": base_config,
                "config_id": int(row["config_id"]),
                "feature_label": cfg.label,
                "feature_atom": feature_atom,
                "feature_family": family,
                "query_id": str(row["query_id"]),
                "score": float(row["score"]),
                "base_score": float(row["base_score"]),
                "delta": delta,
                "useful": 1 if delta > 1e-12 else 0,
                "harmed": 1 if delta < -1e-12 else 0,
            })
            by_feature[cfg.label].append(delta)
            feature_meta[cfg.label] = {
                "dataset": dataset,
                "model": model,
                "scorer": scorer,
                "base_config_id": base_config,
                "config_id": int(row["config_id"]),
                "feature_label": cfg.label,
                "feature_atom": feature_atom,
                "feature_family": family,
            }
        for label, vals in sorted(by_feature.items()):
            pos = sum(1 for v in vals if v > 1e-12)
            neg = sum(1 for v in vals if v < -1e-12)
            zero = len(vals) - pos - neg
            meta = dict(feature_meta[label])
            meta.update({
                "n_queries": len(vals),
                "mean_delta": mean(vals),
                "median_delta": sorted(vals)[len(vals) // 2] if vals else float("nan"),
                "useful_rate": pos / len(vals) if vals else float("nan"),
                "harmed_rate": neg / len(vals) if vals else float("nan"),
                "stable_rate": zero / len(vals) if vals else float("nan"),
            })
            feature_summaries.append(meta)
    return observations, feature_summaries


def itemset_label(itemset: Sequence[str]) -> str:
    return " AND ".join(itemset)


def support_maps(
    transactions: Mapping[Any, frozenset[str]],
    *,
    max_order: int,
    min_support: int,
) -> dict[int, dict[tuple[str, ...], set[Any]]]:
    """Return supported singleton/pair itemsets keyed by order.

    This runner only needs order 1/2. Direct counting is far faster than the
    generic Apriori candidate scan for dense context vectors.
    """
    from itertools import combinations

    out: dict[int, dict[tuple[str, ...], set[Any]]] = defaultdict(dict)
    singleton_qids: dict[tuple[str, ...], set[Any]] = defaultdict(set)
    pair_qids: dict[tuple[str, ...], set[Any]] = defaultdict(set)
    for key, atoms in transactions.items():
        items = tuple(sorted(set(atoms)))
        for atom in items:
            singleton_qids[(atom,)].add(key)
        if max_order >= 2:
            for left, right in combinations(items, 2):
                pair_qids[(left, right)].add(key)
    out[1] = {item: qids for item, qids in singleton_qids.items() if len(qids) >= min_support}
    if max_order >= 2:
        out[2] = {item: qids for item, qids in pair_qids.items() if len(qids) >= min_support}
    return out

def mine_within_rules(
    observations: Sequence[Mapping[str, Any]],
    context_within: Mapping[tuple[str, str], frozenset[str]],
    *,
    datasets: Sequence[str],
    min_support: int,
    max_order: int,
    n_bootstrap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    obs_by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        obs_by_key[(str(row["dataset"]), str(row["feature_label"]))].append(row)

    rules: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"candidate_itemsets": {}}
    for dataset in datasets:
        tx = {
            qid: atoms
            for (ds, qid), atoms in context_within.items()
            if ds == dataset
        }
        itemsets_by_order = support_maps(tx, max_order=max_order, min_support=min_support)
        diagnostics["candidate_itemsets"][dataset] = {
            str(order): len(itemsets_by_order.get(order, {})) for order in range(1, max_order + 1)
        }
        for (ds, feature), rows in obs_by_key.items():
            if ds != dataset:
                continue
            deltas = {str(row["query_id"]): float(row["delta"]) for row in rows}
            vals_all = list(deltas.values())
            if not vals_all:
                continue
            global_mean = mean(vals_all)
            global_useful = sum(1 for v in vals_all if v > 1e-12) / len(vals_all)
            global_harmed = sum(1 for v in vals_all if v < -1e-12) / len(vals_all)
            first = rows[0]
            for order, itemsets in itemsets_by_order.items():
                for itemset, qids in itemsets.items():
                    vals = [deltas[qid] for qid in qids if qid in deltas]
                    if len(vals) < min_support:
                        continue
                    useful = sum(1 for v in vals if v > 1e-12)
                    harmed = sum(1 for v in vals if v < -1e-12)
                    useful_rate = useful / len(vals)
                    harmed_rate = harmed / len(vals)
                    ci_lo, ci_hi, p_gt = bootstrap_mean_delta_ci(
                        vals,
                        center=global_mean,
                        n_bootstrap=n_bootstrap,
                        seed=42 + order,
                    )
                    if math.isnan(ci_lo):
                        ci_lo, ci_hi = normal_ci(vals, center=global_mean)
                        p_gt = float("nan")
                    w_lo, w_hi = wilson_interval(useful, len(vals))
                    lift = mean(vals) - global_mean
                    rate_lift = useful_rate - global_useful
                    rules.append({
                        "scope": "within",
                        "dataset": dataset,
                        "model": first["model"],
                        "scorer": first["scorer"],
                        "feature_label": feature,
                        "feature_atom": first["feature_atom"],
                        "feature_family": first["feature_family"],
                        "context_order": order,
                        "context_rule": itemset_label(itemset),
                        "context_items": itemset,
                        "n_queries": len(vals),
                        "mean_delta": mean(vals),
                        "global_mean_delta": global_mean,
                        "conditional_lift": lift,
                        "ci95_lift_low": ci_lo,
                        "ci95_lift_high": ci_hi,
                        "p_lift_gt_0": p_gt,
                        "useful_rate": useful_rate,
                        "global_useful_rate": global_useful,
                        "useful_rate_lift": rate_lift,
                        "useful_wilson_low": w_lo,
                        "useful_wilson_high": w_hi,
                        "harmed_rate": harmed_rate,
                        "global_harmed_rate": global_harmed,
                        "harmed_rate_lift": harmed_rate - global_harmed,
                        "rank_score": abs(lift) * math.sqrt(len(vals)) + abs(rate_lift) * math.sqrt(len(vals)),
                    })
    rules.sort(key=lambda r: (str(r["dataset"]), int(r["context_order"]), str(r["feature_label"]), -abs(float(r["conditional_lift"]))))
    return rules, diagnostics


def mine_across_rules(
    observations: Sequence[Mapping[str, Any]],
    context_shared: Mapping[tuple[str, str], frozenset[str]],
    *,
    datasets: Sequence[str],
    min_total_support: int,
    min_dataset_support: int,
    max_order: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tx = {key: atoms for key, atoms in context_shared.items() if key[0] in set(datasets)}
    all_itemsets = support_maps(tx, max_order=max_order, min_support=min_total_support)
    diagnostics = {
        "candidate_itemsets": {
            str(order): len(all_itemsets.get(order, {})) for order in range(1, max_order + 1)
        }
    }

    # Precompute itemset support by benchmark once. The first version scanned
    # every feature row for every itemset, which is much too slow for tab_fact.
    itemset_support_by_dataset: dict[tuple[str, ...], dict[str, set[str]]] = {}
    for itemsets in all_itemsets.values():
        for itemset, qkeys in itemsets.items():
            by_ds: dict[str, set[str]] = defaultdict(set)
            for ds, qid in qkeys:
                by_ds[str(ds)].add(str(qid))
            itemset_support_by_dataset[itemset] = by_ds

    by_feature: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in observations:
        by_feature[str(row["feature_label"])].append(row)

    rules: list[dict[str, Any]] = []
    for feature, rows in by_feature.items():
        first = rows[0]
        delta_by_dataset: dict[str, dict[str, float]] = defaultdict(dict)
        for row in rows:
            delta_by_dataset[str(row["dataset"])][str(row["query_id"])] = float(row["delta"])
        global_rates = {
            ds: sum(1 for v in qdelta.values() if v > 1e-12) / len(qdelta)
            for ds, qdelta in delta_by_dataset.items()
            if qdelta
        }
        global_harm_rates = {
            ds: sum(1 for v in qdelta.values() if v < -1e-12) / len(qdelta)
            for ds, qdelta in delta_by_dataset.items()
            if qdelta
        }
        for order, itemsets in all_itemsets.items():
            for itemset in itemsets:
                support_by_ds = itemset_support_by_dataset.get(itemset, {})
                per_dataset = []
                pooled_success = 0
                pooled_harm = 0
                pooled_total = 0
                for ds, qdelta in delta_by_dataset.items():
                    qids = support_by_ds.get(ds, set())
                    if len(qids) < min_dataset_support:
                        continue
                    vals = [qdelta[qid] for qid in qids if qid in qdelta]
                    if len(vals) < min_dataset_support:
                        continue
                    useful = sum(1 for v in vals if v > 1e-12)
                    harmed = sum(1 for v in vals if v < -1e-12)
                    total = len(vals)
                    local_rate = useful / total
                    local_harm = harmed / total
                    lift = local_rate - global_rates[ds]
                    harm_lift = local_harm - global_harm_rates[ds]
                    per_dataset.append({
                        "dataset": ds,
                        "n": total,
                        "useful_rate": local_rate,
                        "global_useful_rate": global_rates[ds],
                        "useful_rate_lift": lift,
                        "harmed_rate": local_harm,
                        "global_harmed_rate": global_harm_rates[ds],
                        "harmed_rate_lift": harm_lift,
                    })
                    pooled_success += useful
                    pooled_harm += harmed
                    pooled_total += total
                if len(per_dataset) < 2 or pooled_total < min_total_support:
                    continue
                weighted_lift = sum(d["useful_rate_lift"] * d["n"] for d in per_dataset) / pooled_total
                macro_lift = mean([d["useful_rate_lift"] for d in per_dataset])
                weighted_harm_lift = sum(d["harmed_rate_lift"] * d["n"] for d in per_dataset) / pooled_total
                macro_harm_lift = mean([d["harmed_rate_lift"] for d in per_dataset])
                pos_bench = sum(1 for d in per_dataset if d["useful_rate_lift"] > 0)
                neg_bench = sum(1 for d in per_dataset if d["useful_rate_lift"] < 0)
                w_lo, w_hi = wilson_interval(pooled_success, pooled_total)
                rules.append({
                    "scope": "across",
                    "feature_label": feature,
                    "feature_atom": first["feature_atom"],
                    "feature_family": first["feature_family"],
                    "context_order": order,
                    "context_rule": itemset_label(itemset),
                    "context_items": itemset,
                    "benchmark_coverage": len(per_dataset),
                    "benchmarks": ",".join(sorted(d["dataset"] for d in per_dataset)),
                    "n_queries": pooled_total,
                    "pooled_useful_rate": pooled_success / pooled_total,
                    "pooled_harmed_rate": pooled_harm / pooled_total,
                    "macro_useful_rate_lift": macro_lift,
                    "weighted_useful_rate_lift": weighted_lift,
                    "macro_harmed_rate_lift": macro_harm_lift,
                    "weighted_harmed_rate_lift": weighted_harm_lift,
                    "positive_benchmarks": pos_bench,
                    "negative_benchmarks": neg_bench,
                    "useful_wilson_low": w_lo,
                    "useful_wilson_high": w_hi,
                    "per_dataset": json.dumps(per_dataset, sort_keys=True),
                    "rank_score": abs(macro_lift) * math.sqrt(pooled_total) * (1 + len(per_dataset) / len(datasets)),
                })
    rules.sort(key=lambda r: (int(r["context_order"]), str(r["feature_label"]), -abs(float(r["macro_useful_rate_lift"]))))
    return rules, diagnostics

def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:+.1f}"
    except Exception:
        return ""


def fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return ""


def html_table(rows: Sequence[Mapping[str, Any]], cols: Sequence[str], *, limit: int = 60) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = []
    for row in rows[:limit]:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in cols) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def top_rules(rows: Sequence[Mapping[str, Any]], *, key: str, positive: bool, order: Optional[int] = None, limit: int = 40) -> list[Mapping[str, Any]]:
    filtered = [r for r in rows if order is None or int(r.get("context_order", 0)) == order]
    if positive:
        filtered = [r for r in filtered if float(r.get(key, 0.0)) > 0]
        return sorted(filtered, key=lambda r: (-float(r.get(key, 0.0)), -int(r.get("n_queries", 0))))[:limit]
    filtered = [r for r in filtered if float(r.get(key, 0.0)) < 0]
    return sorted(filtered, key=lambda r: (float(r.get(key, 0.0)), -int(r.get("n_queries", 0))))[:limit]


def build_html(out: Path, metadata: Mapping[str, Any], feature_summary: Sequence[Mapping[str, Any]], within_rules: Sequence[Mapping[str, Any]], across_rules: Sequence[Mapping[str, Any]]) -> None:
    feature_cols = ["dataset", "feature_label", "feature_family", "n_queries", "mean_delta", "useful_rate", "harmed_rate"]
    within_cols = ["dataset", "feature_label", "feature_family", "context_order", "context_rule", "n_queries", "conditional_lift", "ci95_lift_low", "ci95_lift_high", "useful_rate_lift"]
    across_cols = ["feature_label", "feature_family", "context_order", "context_rule", "benchmark_coverage", "benchmarks", "n_queries", "macro_useful_rate_lift", "weighted_useful_rate_lift", "positive_benchmarks", "negative_benchmarks"]
    feature_rows = []
    for r in sorted(feature_summary, key=lambda x: (str(x["dataset"]), -abs(float(x["mean_delta"])))):
        rr = dict(r)
        rr["mean_delta"] = pct(rr["mean_delta"])
        rr["useful_rate"] = pct(rr["useful_rate"])
        rr["harmed_rate"] = pct(rr["harmed_rate"])
        feature_rows.append(rr)
    def prep(rows, cols):
        out_rows = []
        for r in rows:
            rr = dict(r)
            for k in cols:
                if k.endswith("lift") or k.startswith("ci95"):
                    rr[k] = pct(rr.get(k))
            out_rows.append(rr)
        return out_rows

    html_text = f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>7B Add-One Context Rule Learning</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;line-height:1.35;color:#202124}}h1,h2{{margin-top:24px}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 28px}}th,td{{border:1px solid #ddd;padding:6px;vertical-align:top}}th{{background:#f4f6f8;text-align:left;position:sticky;top:0}}code{{background:#f4f6f8;padding:1px 3px}}.note{{color:#555;max-width:920px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}@media(max-width:1000px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>7B Add-One Context Rule Learning</h1>
<p class=\"note\">Existing 7B runs only. Each feature is treated as a paired add-one intervention against the benchmark-local base. Within-benchmark rules use raw paired delta; across-benchmark rules use useful-rate lift because scorer magnitudes differ.</p>
<pre>{html.escape(json.dumps(metadata, indent=2))}</pre>
<h2>Feature Effects</h2>
{html_table(feature_rows, feature_cols, limit=100)}
<div class=\"grid\"><section><h2>Within Positive Order 1</h2>{html_table(prep(top_rules(within_rules,key='conditional_lift',positive=True,order=1), within_cols), within_cols)}</section>
<section><h2>Within Negative Order 1</h2>{html_table(prep(top_rules(within_rules,key='conditional_lift',positive=False,order=1), within_cols), within_cols)}</section></div>
<div class=\"grid\"><section><h2>Within Positive Order 2</h2>{html_table(prep(top_rules(within_rules,key='conditional_lift',positive=True,order=2), within_cols), within_cols)}</section>
<section><h2>Within Negative Order 2</h2>{html_table(prep(top_rules(within_rules,key='conditional_lift',positive=False,order=2), within_cols), within_cols)}</section></div>
<div class=\"grid\"><section><h2>Across Positive Order 1</h2>{html_table(prep(top_rules(across_rules,key='macro_useful_rate_lift',positive=True,order=1), across_cols), across_cols)}</section>
<section><h2>Across Negative Order 1</h2>{html_table(prep(top_rules(across_rules,key='macro_useful_rate_lift',positive=False,order=1), across_cols), across_cols)}</section></div>
<div class=\"grid\"><section><h2>Across Positive Order 2</h2>{html_table(prep(top_rules(across_rules,key='macro_useful_rate_lift',positive=True,order=2), across_cols), across_cols)}</section>
<section><h2>Across Negative Order 2</h2>{html_table(prep(top_rules(across_rules,key='macro_useful_rate_lift',positive=False,order=2), across_cols), across_cols)}</section></div>
</body></html>"""
    (out / "rule_learning_dashboard.html").write_text(html_text)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Mine order-1/2 context rules for add-one feature interventions.")
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--datasets", default=",".join(DATASET_SCORERS))
    p.add_argument("--max-order", type=int, default=2)
    p.add_argument("--within-min-support", type=int, default=80)
    p.add_argument("--across-min-total-support", type=int, default=200)
    p.add_argument("--across-min-dataset-support", type=int, default=30)
    p.add_argument("--bootstrap", type=int, default=300)
    p.add_argument("--include-negative-context", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    datasets = split_csv(args.datasets)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    configs = load_configs(conn)
    query_meta = load_query_meta(conn, datasets)
    context_within, context_shared = build_context_maps(
        query_meta,
        include_negative=bool(args.include_negative_context),
    )
    observations, feature_summary = load_addone_observations(
        conn,
        datasets=datasets,
        model=args.model,
        configs=configs,
    )
    within_rules, within_diag = mine_within_rules(
        observations,
        context_within,
        datasets=datasets,
        min_support=args.within_min_support,
        max_order=args.max_order,
        n_bootstrap=args.bootstrap,
    )
    across_rules, across_diag = mine_across_rules(
        observations,
        context_shared,
        datasets=datasets,
        min_total_support=args.across_min_total_support,
        min_dataset_support=args.across_min_dataset_support,
        max_order=args.max_order,
    )
    write_csv(out / "feature_summary.csv", feature_summary)
    write_csv(out / "within_rules.csv", within_rules)
    write_csv(out / "within_rules_order1.csv", [r for r in within_rules if int(r["context_order"]) == 1])
    write_csv(out / "within_rules_order2.csv", [r for r in within_rules if int(r["context_order"]) == 2])
    write_csv(out / "across_rules.csv", across_rules)
    write_csv(out / "across_rules_order1.csv", [r for r in across_rules if int(r["context_order"]) == 1])
    write_csv(out / "across_rules_order2.csv", [r for r in across_rules if int(r["context_order"]) == 2])
    # Keep the row-level outcome table compact enough for downstream checks.
    write_csv(out / "observations.csv", observations)
    metadata = {
        "db": str(args.db),
        "out": str(out),
        "model": args.model,
        "datasets": datasets,
        "max_order": args.max_order,
        "within_min_support": args.within_min_support,
        "across_min_total_support": args.across_min_total_support,
        "across_min_dataset_support": args.across_min_dataset_support,
        "include_negative_context": bool(args.include_negative_context),
        "n_observations": len(observations),
        "n_feature_summaries": len(feature_summary),
        "n_within_rules": len(within_rules),
        "n_across_rules": len(across_rules),
        "within_diagnostics": within_diag,
        "across_diagnostics": across_diag,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    build_html(out, metadata, feature_summary, within_rules, across_rules)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
