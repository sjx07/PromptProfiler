"""Context-aware subgroup discovery over FACET observation cubes.

The pipeline is offline: it consumes cached executions/evaluations from a cube,
builds matched query-level deltas for prompt-feature transitions, attaches
context predicates, mines low-order subgroup rules, and writes analysis tables.

Default scope intentionally stays on the clean non-POT surface/protocol space.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from analyze.context_attributes import canonical_context_items
from analyze.transition_flip import (
    AtomEditSpec,
    DEFAULT_PREFIX_RULES,
    TransitionSpec,
    aggregate_query_effects,
    atom_family_value,
    build_atom_edit_paired_contrasts,
    build_paired_contrasts,
    connect,
    load_config_atoms,
    load_config_family_assignments,
    load_scored_executions,
)

DEFAULT_DATASETS = ("wtq", "sqa", "tablebench", "tab_fact", "hitab")
DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_EXCLUDE_CONFIG_FAMILIES = ("response_mode", "runtime_binding")


def split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def med(values: Sequence[float]) -> float:
    return float(median(values)) if values else float("nan")


def stdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def ci95(values: Sequence[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mu = mean(values)
    if len(values) < 2:
        return mu, mu
    half = 1.96 * stdev(values) / math.sqrt(len(values))
    return mu - half, mu + half


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def family_of(atom: str) -> str:
    mapped = atom_family_value(atom, rules=DEFAULT_PREFIX_RULES)
    return mapped[0] if mapped else "unknown"


def value_of(atom: str) -> str:
    mapped = atom_family_value(atom, rules=DEFAULT_PREFIX_RULES)
    return mapped[1] if mapped else atom


def config_has_family(atoms: Sequence[str], families: Sequence[str]) -> bool:
    blocked = set(families)
    return any(family_of(atom) in blocked for atom in atoms)


@dataclass(frozen=True)
class PipelineTransition:
    name: str
    family: str
    spec: TransitionSpec | AtomEditSpec


def default_transitions(preset: str) -> List[PipelineTransition]:
    transitions = [
        PipelineTransition("ser.columns_to_html", "table_serialization", TransitionSpec("table_serialization", "json_columns_data", "html")),
        PipelineTransition("ser.columns_to_records", "table_serialization", TransitionSpec("table_serialization", "json_columns_data", "json_records")),
        PipelineTransition("ser.html_to_records", "table_serialization", TransitionSpec("table_serialization", "html", "json_records")),
        PipelineTransition("fmt.json_to_plain", "prompt_format", TransitionSpec("prompt_format", "json", "plain")),
        PipelineTransition("fmt.plain_to_json", "prompt_format", TransitionSpec("prompt_format", "plain", "json")),
        PipelineTransition("ctx.add_column_statistics", "input_context", AtomEditSpec(add_atoms=("input_context_column_statistics",), family="input_context")),
        PipelineTransition("ctx.add_type_annotation", "input_context", AtomEditSpec(add_atoms=("input_context_type_annotation",), family="input_context")),
        PipelineTransition("ctx.add_relevant_columns", "input_context", AtomEditSpec(add_atoms=("input_context_column_selection_relevance_12",), family="input_context")),
        PipelineTransition("ctx.add_relevant_rows", "input_context", AtomEditSpec(add_atoms=("input_context_row_selection_relevance_50",), family="input_context")),
        PipelineTransition("reason.add_verify_before_output", "reasoning", AtomEditSpec(add_atoms=("reasoning_verify_before_output",), family="reasoning")),
        PipelineTransition("reason.add_candidate_enumeration", "reasoning", AtomEditSpec(add_atoms=("reasoning_candidate_enumeration",), family="reasoning")),
        PipelineTransition("reason.add_evidence_localization", "reasoning", AtomEditSpec(add_atoms=("reasoning_evidence_localization",), family="reasoning")),
        PipelineTransition("reason.add_extract_then_compute", "reasoning", AtomEditSpec(add_atoms=("reasoning_extract_then_compute",), family="reasoning")),
        PipelineTransition("reason.add_plan_then_answer", "reasoning", AtomEditSpec(add_atoms=("reasoning_plan_then_answer",), family="reasoning")),
        PipelineTransition("reason.add_intermediate_evidence_table", "reasoning", AtomEditSpec(add_atoms=("reasoning_intermediate_evidence_table",), family="reasoning")),
        PipelineTransition("reason.add_symbolic_operation", "reasoning", AtomEditSpec(add_atoms=("reasoning_symbolic_operation",), family="reasoning")),
        PipelineTransition("reason.add_critique_then_revise", "reasoning", AtomEditSpec(add_atoms=("reasoning_critique_then_revise",), family="reasoning")),
    ]
    if preset == "pilot":
        keep = {
            "ser.columns_to_html",
            "ser.columns_to_records",
            "fmt.json_to_plain",
            "ctx.add_column_statistics",
            "reason.add_verify_before_output",
            "reason.add_candidate_enumeration",
        }
        return [t for t in transitions if t.name in keep]
    if preset == "surface":
        return [t for t in transitions if t.family in {"table_serialization", "prompt_format", "input_context"}]
    return transitions


def bin_num(name: str, value: str) -> Optional[str]:
    try:
        n = int(str(value))
    except Exception:
        return None
    if name.endswith("question_length") or name.endswith("statement_length"):
        if n <= 8:
            return "<=8"
        if n <= 16:
            return "9..16"
        if n <= 24:
            return "17..24"
        return ">24"
    if name.endswith("n_rows"):
        if n <= 10:
            return "<=10"
        if n <= 25:
            return "11..25"
        if n <= 50:
            return "26..50"
        return ">50"
    if name.endswith("n_cols"):
        if n <= 4:
            return "<=4"
        if n <= 8:
            return "5..8"
        if n <= 12:
            return "9..12"
        return ">12"
    if name.endswith("n_numeric_cols"):
        if n == 0:
            return "0"
        if n <= 2:
            return "1..2"
        if n <= 5:
            return "3..5"
        return ">5"
    if name.endswith("position") or name.endswith("n_history"):
        if n == 0:
            return "0"
        if n == 1:
            return "1"
        return ">=2"
    return None


def context_atom(name: str, value: str) -> Optional[str]:
    categorical_suffixes = (
        "operation_type",
        "qtype",
        "qsubtype",
        "agg_type",
        "t_shape",
        "table_source",
        "has_aggregation",
        "has_arithmetic",
        "has_comparison",
        "has_filter",
        "has_negation",
        "has_numeric_cols",
        "has_reference",
        "has_superlative",
        "has_temporal",
        "is_count",
        "has_numeric_claim",
    )
    if any(name.endswith(suffix) for suffix in categorical_suffixes):
        return f"{name}={value}"
    binned = bin_num(name, value)
    if binned is not None:
        return f"{name}_bin={binned}"
    return None


def load_context_atoms(
    conn: sqlite3.Connection,
    query_ids: Iterable[str],
    *,
    context_view: str = "canonical_shared",
    include_negative: bool = True,
) -> Dict[str, Tuple[str, ...]]:
    ids = sorted(set(str(qid) for qid in query_ids))
    out: Dict[str, List[str]] = defaultdict(list)
    if not ids:
        return {}
    if context_view != "raw":
        view = {
            "canonical_shared": "shared",
            "canonical_scoped": "scoped",
            "canonical_all": "all",
        }.get(context_view)
        if view is None:
            raise ValueError(
                "context_view must be raw, canonical_shared, canonical_scoped, "
                f"or canonical_all; got {context_view!r}"
            )
        for i in range(0, len(ids), 800):
            chunk = ids[i : i + 800]
            rows = conn.execute(
                f"""
                SELECT query_id, dataset, meta
                FROM query
                WHERE query_id IN ({','.join('?' for _ in chunk)})
                """,
                tuple(chunk),
            ).fetchall()
            for row in rows:
                out[str(row["query_id"])].extend(
                    canonical_context_items(
                        row["meta"],
                        str(row["dataset"]),
                        view=view,
                        include_negative=include_negative,
                    )
                )
        return {qid: tuple(sorted(set(out.get(qid, [])))) for qid in ids}

    for i in range(0, len(ids), 800):
        chunk = ids[i : i + 800]
        rows = conn.execute(
            f"""
            SELECT query_id, name, value
            FROM predicate
            WHERE query_id IN ({','.join('?' for _ in chunk)})
            """,
            tuple(chunk),
        ).fetchall()
        for row in rows:
            atom = context_atom(str(row["name"]), str(row["value"]))
            if atom:
                out[str(row["query_id"])].append(atom)
    return {qid: tuple(sorted(set(out.get(qid, [])))) for qid in ids}


def transition_pairs(rows: Sequence[Any], assignments: Mapping[int, Mapping[str, str]], atoms_by_config: Mapping[int, Sequence[str]], transition: PipelineTransition) -> List[Dict[str, Any]]:
    if isinstance(transition.spec, TransitionSpec):
        pairs = build_paired_contrasts(rows, assignments, transition.spec)
    else:
        pairs = build_atom_edit_paired_contrasts(rows, atoms_by_config, transition.spec)
    for row in pairs:
        row["transition"] = transition.name
        row["transition_family"] = transition.family
    return pairs


def attach_query_context(query_effects: Sequence[Mapping[str, Any]], context: Mapping[str, Sequence[str]], transition: PipelineTransition) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in query_effects:
        rec = dict(row)
        rec["transition"] = transition.name
        rec["transition_family"] = transition.family
        rec["context_atoms"] = "|".join(context.get(str(row["query_id"]), ()))
        out.append(rec)
    return out


def rule_itemsets(atoms: Sequence[str], max_order: int) -> Iterable[Tuple[str, ...]]:
    import itertools

    atom_set = tuple(sorted(set(atoms)))
    for order in range(1, max_order + 1):
        if len(atom_set) < order:
            break
        yield from itertools.combinations(atom_set, order)


def mine_rules(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_order: int,
    min_query_support: int,
    top_k: int,
) -> List[Dict[str, Any]]:
    indexed = []
    for idx, row in enumerate(rows):
        atoms = [a for a in str(row.get("context_atoms") or "").split("|") if a]
        indexed.append((idx, row, tuple(sorted(set(atoms)))))
    if not indexed:
        return []
    deltas = [float(row.get("median_delta", row.get("mean_delta", 0.0))) for _idx, row, _atoms in indexed]
    global_mean = mean(deltas)
    global_positive_rate = sum(1 for d in deltas if d > 0) / len(deltas)
    global_negative_rate = sum(1 for d in deltas if d < 0) / len(deltas)

    support: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for idx, _row, atoms in indexed:
        for itemset in rule_itemsets(atoms, max_order):
            support[itemset].append(idx)

    rules = []
    for itemset, indices in support.items():
        if len(indices) < min_query_support:
            continue
        local_deltas = [deltas[i] for i in indices]
        pos = sum(1 for d in local_deltas if d > 0)
        neg = sum(1 for d in local_deltas if d < 0)
        stable = len(local_deltas) - pos - neg
        local_mean = mean(local_deltas)
        lo, hi = ci95(local_deltas)
        pos_rate = pos / len(local_deltas)
        neg_rate = neg / len(local_deltas)
        config_from = set()
        config_to = set()
        backgrounds = set()
        pair_support = 0
        for i in indices:
            row = indexed[i][1]
            backgrounds.add(str(row.get("background_id", "")))
            config_from.update(x for x in str(row.get("config_from_ids", "")).split("|") if x)
            config_to.update(x for x in str(row.get("config_to_ids", "")).split("|") if x)
            try:
                pair_support += int(row.get("n_pairs", 1))
            except Exception:
                pair_support += 1
        magnitude_lift = local_mean - global_mean
        signed_consistency = pos_rate - neg_rate
        rank_score = abs(magnitude_lift) * math.sqrt(len(local_deltas)) * (1.0 + abs(signed_consistency))
        rules.append(
            {
                "rule": " AND ".join(itemset),
                "order": len(itemset),
                "query_support": len(local_deltas),
                "pair_support": pair_support,
                "background_support": len(backgrounds),
                "config_from_support": len(config_from),
                "config_to_support": len(config_to),
                "mean_delta": local_mean,
                "median_delta": med(local_deltas),
                "global_mean_delta": global_mean,
                "delta_over_global": magnitude_lift,
                "ci95_low": lo,
                "ci95_high": hi,
                "positive_count": pos,
                "negative_count": neg,
                "stable_count": stable,
                "positive_rate": pos_rate,
                "negative_rate": neg_rate,
                "positive_lift": pos_rate / global_positive_rate if global_positive_rate else 0.0,
                "negative_lift": neg_rate / global_negative_rate if global_negative_rate else 0.0,
                "rank_score": rank_score,
            }
        )
    rules.sort(key=lambda r: (-float(r["rank_score"]), -int(r["query_support"]), str(r["rule"])))
    return rules[:top_k]


def summarize_query_effects(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    deltas = [float(row.get("median_delta", row.get("mean_delta", 0.0))) for row in rows]
    lo, hi = ci95(deltas)
    return {
        "n_queries": len(rows),
        "mean_delta": mean(deltas),
        "median_delta": med(deltas),
        "ci95_low": lo,
        "ci95_high": hi,
        "positive_queries": sum(1 for d in deltas if d > 0),
        "negative_queries": sum(1 for d in deltas if d < 0),
        "stable_queries": sum(1 for d in deltas if abs(d) <= 1e-12),
    }


def run_one(
    dataset: str,
    transition: PipelineTransition,
    rows: Sequence[Any],
    assignments: Mapping[int, Mapping[str, str]],
    atoms_by_config: Mapping[int, Sequence[str]],
    db_path: str,
    *,
    max_rule_order: int,
    min_query_support: int,
    min_pairs: int,
    top_rules: int,
    context_view: str,
    include_negative_context: bool,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    scoped_rows = [row for row in rows if row.dataset == dataset]
    pairs = transition_pairs(scoped_rows, assignments, atoms_by_config, transition)
    if len(pairs) < min_pairs:
        return None, [], []
    query_effects = aggregate_query_effects(pairs)
    with connect(db_path) as conn:
        context = load_context_atoms(
            conn,
            [str(row["query_id"]) for row in query_effects],
            context_view=context_view,
            include_negative=include_negative_context,
        )
    query_context = attach_query_context(query_effects, context, transition)
    if len(query_context) < min_query_support:
        return None, [], []
    rule_rows = mine_rules(
        query_context,
        max_order=max_rule_order,
        min_query_support=min_query_support,
        top_k=top_rules,
    )
    summary = summarize_query_effects(query_context)
    summary.update(
        {
            "dataset": dataset,
            "transition": transition.name,
            "transition_family": transition.family,
            "n_pairs": len(pairs),
            "n_rules": len(rule_rows),
        }
    )
    for rank, rule in enumerate(rule_rows, 1):
        rule.update(
            {
                "dataset": dataset,
                "transition": transition.name,
                "transition_family": transition.family,
                "rank": rank,
            }
        )
    return summary, rule_rows, query_context


def build_report(out_dir: Path, summaries: Sequence[Mapping[str, Any]], rules: Sequence[Mapping[str, Any]], runtime_s: float) -> None:
    lines = ["# FACET Subgroup Discovery Pipeline", ""]
    lines.append(f"Runtime: {runtime_s:.1f}s")
    lines.append("")
    lines.append("## Transition Effects")
    for row in sorted(summaries, key=lambda r: (str(r["dataset"]), str(r["transition"]))):
        lines.append(
            f"- `{row['dataset']}::{row['transition']}` n={row['n_queries']} pairs={row['n_pairs']} "
            f"mean={float(row['mean_delta']):+.4f} ci95=[{float(row['ci95_low']):+.4f},{float(row['ci95_high']):+.4f}] "
            f"pos={row['positive_queries']} neg={row['negative_queries']} rules={row['n_rules']}"
        )
    lines.extend(["", "## Top Positive Context Rules"])
    positive = [r for r in rules if float(r["delta_over_global"]) > 0 and float(r["positive_lift"]) >= 1.0]
    for row in sorted(positive, key=lambda r: (-float(r["rank_score"]), -abs(float(r["delta_over_global"]))))[:40]:
        lines.append(
            f"- `{row['dataset']}::{row['transition']}` rule={row['rule']} "
            f"q={row['query_support']} bg={row['background_support']} "
            f"mean={float(row['mean_delta']):+.4f} over_global={float(row['delta_over_global']):+.4f} "
            f"pos_lift={float(row['positive_lift']):.2f} neg_lift={float(row['negative_lift']):.2f}"
        )
    lines.extend(["", "## Top Negative Context Rules"])
    negative = [r for r in rules if float(r["delta_over_global"]) < 0 and float(r["negative_lift"]) >= 1.0]
    for row in sorted(negative, key=lambda r: (-float(r["rank_score"]), -abs(float(r["delta_over_global"]))))[:40]:
        lines.append(
            f"- `{row['dataset']}::{row['transition']}` rule={row['rule']} "
            f"q={row['query_support']} bg={row['background_support']} "
            f"mean={float(row['mean_delta']):+.4f} over_global={float(row['delta_over_global']):+.4f} "
            f"pos_lift={float(row['positive_lift']):.2f} neg_lift={float(row['negative_lift']):.2f}"
        )
    out_dir.joinpath("report.md").write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run context-aware subgroup discovery over matched prompt-feature transitions.")
    p.add_argument("--db", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    p.add_argument("--models", default=DEFAULT_MODEL)
    p.add_argument("--scorers", default="")
    p.add_argument("--preset", choices=("pilot", "surface", "core"), default="core")
    p.add_argument("--transitions", default="", help="Optional comma-separated transition names to run.")
    p.add_argument("--exclude-config-families", default=",".join(DEFAULT_EXCLUDE_CONFIG_FAMILIES))
    p.add_argument("--min-pairs", type=int, default=80)
    p.add_argument("--min-query-support", type=int, default=80)
    p.add_argument("--max-rule-order", type=int, default=2)
    p.add_argument("--top-rules", type=int, default=50)
    p.add_argument(
        "--context-view",
        choices=("raw", "canonical_shared", "canonical_scoped", "canonical_all"),
        default="canonical_shared",
        help="Context atom source for subgroup mining. raw preserves old predicate-table behavior.",
    )
    p.add_argument("--positive-context-only", action="store_true", help="Drop negative/no canonical atoms from mining.")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--save-query-effects", action="store_true")
    p.add_argument("--progress", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    t0 = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = split_csv(args.datasets)
    models = split_csv(args.models)
    scorers = split_csv(args.scorers)
    excluded = tuple(split_csv(args.exclude_config_families))
    transitions = default_transitions(args.preset)
    requested = set(split_csv(args.transitions))
    if requested:
        transitions = [t for t in transitions if t.name in requested]
    if not transitions:
        raise SystemExit("no transitions selected")

    print(f"[load] cube={args.db}", flush=True)
    with connect(args.db) as conn:
        atoms_by_config = load_config_atoms(conn)
        assignments = load_config_family_assignments(conn)
        rows = load_scored_executions(conn, datasets=datasets, models=models, scorers=scorers)
    rows = [row for row in rows if not config_has_family(atoms_by_config.get(row.config_id, ()), excluded)]
    print(f"[loaded] scored_rows={len(rows)} datasets={datasets} transitions={len(transitions)}", flush=True)

    jobs = [(dataset, transition) for dataset in datasets for transition in transitions]
    summaries: List[Dict[str, Any]] = []
    rule_rows: List[Dict[str, Any]] = []
    query_rows: List[Dict[str, Any]] = []
    max_workers = max(1, min(args.num_workers, len(jobs)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                run_one,
                dataset,
                transition,
                rows,
                assignments,
                atoms_by_config,
                args.db,
                max_rule_order=args.max_rule_order,
                min_query_support=args.min_query_support,
                min_pairs=args.min_pairs,
                top_rules=args.top_rules,
                context_view=args.context_view,
                include_negative_context=not args.positive_context_only,
            )
            for dataset, transition in jobs
        ]
        iterator = as_completed(futures)
        if args.progress:
            try:
                from tqdm import tqdm  # type: ignore

                iterator = tqdm(iterator, total=len(futures), desc="transitions")
            except Exception:
                pass
        for future in iterator:
            summary, rules, qrows = future.result()
            if summary is None:
                continue
            summaries.append(summary)
            rule_rows.extend(rules)
            if args.save_query_effects:
                query_rows.extend(qrows)

    summaries.sort(key=lambda r: (str(r["dataset"]), str(r["transition"])))
    rule_rows.sort(key=lambda r: (str(r["dataset"]), str(r["transition"]), int(r["rank"])))
    write_csv(out_dir / "transition_summary.csv", summaries)
    write_csv(out_dir / "subgroup_rules.csv", rule_rows)
    if args.save_query_effects:
        write_csv(out_dir / "query_effects_with_context.csv", query_rows)
    runtime_s = time.time() - t0
    build_report(out_dir, summaries, rule_rows, runtime_s)
    metadata = {
        "db": args.db,
        "datasets": datasets,
        "models": models,
        "scorers": scorers,
        "preset": args.preset,
        "transitions": [t.name for t in transitions],
        "exclude_config_families": list(excluded),
        "min_pairs": args.min_pairs,
        "min_query_support": args.min_query_support,
        "max_rule_order": args.max_rule_order,
        "top_rules": args.top_rules,
        "context_view": args.context_view,
        "positive_context_only": bool(args.positive_context_only),
        "num_workers": args.num_workers,
        "save_query_effects": bool(args.save_query_effects),
        "runtime_s": runtime_s,
        "n_transition_summaries": len(summaries),
        "n_rules": len(rule_rows),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
