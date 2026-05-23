"""Per-family, per-feature matched transition matrix for FACET cubes.

This script materializes the analysis table we need before subgroup mining:
query-level paired deltas for every estimable prompt-feature transition, plus
benchmark-level and overall summaries.

Two transition shapes are used:

* additive atoms: compare ``absent -> present`` while holding every other atom
  fixed.  This fits multi-hot families such as input_context, reasoning, and
  domain_heuristic.
* choice families: compare ``value_a -> value_b`` while holding every
  non-focal atom fixed.  This fits prompt_format, table_serialization,
  visible_reasoning, and output_contract.

By default POT/runtime-bound configs are excluded so the report stays on the
current clean surface/protocol space.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from analyze.transition_flip import (
    DEFAULT_BACKGROUND_IGNORE,
    DEFAULT_PREFIX_RULES,
    atom_family_value,
    connect,
    load_config_atoms,
    load_context_atoms,
)


DEFAULT_EXCLUSIVE_FAMILIES = (
    "prompt_format",
    "table_serialization",
    "visible_reasoning",
    "output_contract",
)

DEFAULT_ADDITIVE_FAMILIES = (
    "input_context",
    "reasoning",
    "domain_heuristic",
    "task_heuristic",
    "retrieval_workflow",
)

DEFAULT_EXCLUDE_CONFIG_FAMILIES = (
    "response_mode",
    "runtime_binding",
)


@dataclass(frozen=True)
class ScoreRow:
    config_id: int
    query_id: str
    dataset: str
    model: str
    scorer: str
    score: float


def split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def med(values: Sequence[float]) -> float:
    return float(median(values)) if values else float("nan")


def stdev(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (n - 1))


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


def atom_allowed(atom: str, families: Sequence[str]) -> bool:
    return not families or family_of(atom) in set(families)


def config_has_family(atoms: Sequence[str], families: Sequence[str]) -> bool:
    blocked = set(families)
    return any(family_of(atom) in blocked for atom in atoms)


def background_atoms(
    atoms: Sequence[str],
    *,
    focal_family: str,
    focal_atoms: Iterable[str] = (),
    ignore_families: Sequence[str] = DEFAULT_BACKGROUND_IGNORE,
) -> Tuple[str, ...]:
    ignored = set(ignore_families)
    removed = set(focal_atoms)
    kept = []
    for atom in atoms:
        fam = family_of(atom)
        if fam in ignored:
            continue
        if fam == focal_family:
            continue
        if atom in removed:
            continue
        kept.append(atom)
    return tuple(sorted(set(kept)))


def additive_background_atoms(
    atoms: Sequence[str],
    *,
    focal_atom: str,
    ignore_families: Sequence[str] = DEFAULT_BACKGROUND_IGNORE,
) -> Tuple[str, ...]:
    ignored = set(ignore_families)
    kept = []
    for atom in atoms:
        if atom == focal_atom:
            continue
        if family_of(atom) in ignored:
            continue
        kept.append(atom)
    return tuple(sorted(set(kept)))


def background_id(atoms: Sequence[str]) -> str:
    text = "|".join(sorted(atoms))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def load_score_rows(
    db_path: str,
    *,
    datasets: Sequence[str],
    models: Sequence[str],
    scorers: Sequence[str],
    eligible_configs: Optional[set[int]],
) -> List[ScoreRow]:
    params: List[Any] = []
    where = ["ev.score IS NOT NULL"]
    if datasets:
        where.append("q.dataset IN (%s)" % ",".join("?" for _ in datasets))
        params.extend(datasets)
    if models:
        where.append("e.model IN (%s)" % ",".join("?" for _ in models))
        params.extend(models)
    if scorers:
        where.append("ev.scorer IN (%s)" % ",".join("?" for _ in scorers))
        params.extend(scorers)
    sql = f"""
        SELECT e.config_id,
               e.query_id,
               q.dataset,
               e.model,
               ev.scorer,
               AVG(ev.score) AS score
        FROM evaluation ev
        JOIN execution e ON e.execution_id = ev.execution_id
        JOIN query q ON q.query_id = e.query_id
        WHERE {' AND '.join(where)}
        GROUP BY e.config_id, e.query_id, q.dataset, e.model, ev.scorer
    """
    rows: List[ScoreRow] = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for r in conn.execute(sql, tuple(params)):
            cid = int(r["config_id"])
            if eligible_configs is not None and cid not in eligible_configs:
                continue
            rows.append(
                ScoreRow(
                    config_id=cid,
                    query_id=str(r["query_id"]),
                    dataset=str(r["dataset"]),
                    model=str(r["model"]),
                    scorer=str(r["scorer"]),
                    score=float(r["score"]),
                )
            )
    return rows


def side_mean(rows: Sequence[ScoreRow]) -> float:
    return mean([row.score for row in rows])


def build_additive_contrasts(
    rows: Sequence[ScoreRow],
    atoms_by_config: Mapping[int, Sequence[str]],
    atom: str,
    *,
    ignore_families: Sequence[str],
) -> List[Dict[str, Any]]:
    family = family_of(atom)
    grouped: Dict[Tuple[str, str, str, str, Tuple[str, ...]], Dict[str, List[ScoreRow]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        atoms = tuple(atoms_by_config.get(row.config_id, ()))
        atom_set = set(atoms)
        bg = additive_background_atoms(atoms, focal_atom=atom, ignore_families=ignore_families)
        side = "to" if atom in atom_set else "from"
        key = (row.dataset, row.model, row.scorer, row.query_id, bg)
        grouped[key][side].append(row)

    out: List[Dict[str, Any]] = []
    for (dataset, model, scorer, query_id, bg), sides in grouped.items():
        if "from" not in sides or "to" not in sides:
            continue
        y_from = side_mean(sides["from"])
        y_to = side_mean(sides["to"])
        out.append(
            {
                "family": family,
                "transition_kind": "atom_add",
                "feature_atom": atom,
                "from_feature": "absent",
                "to_feature": atom,
                "from_value": "absent",
                "to_value": value_of(atom),
                "dataset": dataset,
                "model": model,
                "scorer": scorer,
                "query_id": query_id,
                "score_from": y_from,
                "score_to": y_to,
                "delta": y_to - y_from,
                "flip_type": classify_delta(y_from, y_to),
                "background_id": background_id(bg),
                "background_atoms": "|".join(bg),
                "config_from_ids": "|".join(str(x) for x in sorted({r.config_id for r in sides["from"]})),
                "config_to_ids": "|".join(str(x) for x in sorted({r.config_id for r in sides["to"]})),
                "n_from": len(sides["from"]),
                "n_to": len(sides["to"]),
            }
        )
    return out


def family_values(atoms: Sequence[str], family: str) -> Tuple[str, ...]:
    values = [atom for atom in atoms if family_of(atom) == family]
    return tuple(sorted(set(values))) if values else ("absent",)


def build_choice_contrasts(
    rows: Sequence[ScoreRow],
    atoms_by_config: Mapping[int, Sequence[str]],
    family: str,
    *,
    ignore_families: Sequence[str],
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, Tuple[str, ...]], Dict[str, List[ScoreRow]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        atoms = tuple(atoms_by_config.get(row.config_id, ()))
        bg = background_atoms(atoms, focal_family=family, ignore_families=ignore_families)
        values = family_values(atoms, family)
        value_label = "+".join(values)
        key = (row.dataset, row.model, row.scorer, row.query_id, bg)
        grouped[key][value_label].append(row)

    out: List[Dict[str, Any]] = []
    for (dataset, model, scorer, query_id, bg), by_value in grouped.items():
        if len(by_value) < 2:
            continue
        values = sorted(by_value)
        for from_value in values:
            for to_value in values:
                if from_value == to_value:
                    continue
                from_rows = by_value[from_value]
                to_rows = by_value[to_value]
                y_from = side_mean(from_rows)
                y_to = side_mean(to_rows)
                out.append(
                    {
                        "family": family,
                        "transition_kind": "family_choice",
                        "feature_atom": to_value,
                        "from_feature": from_value,
                        "to_feature": to_value,
                        "from_value": "|".join(value_of(x) for x in from_value.split("+") if x != "absent") or "absent",
                        "to_value": "|".join(value_of(x) for x in to_value.split("+") if x != "absent") or "absent",
                        "dataset": dataset,
                        "model": model,
                        "scorer": scorer,
                        "query_id": query_id,
                        "score_from": y_from,
                        "score_to": y_to,
                        "delta": y_to - y_from,
                        "flip_type": classify_delta(y_from, y_to),
                        "background_id": background_id(bg),
                        "background_atoms": "|".join(bg),
                        "config_from_ids": "|".join(str(x) for x in sorted({r.config_id for r in from_rows})),
                        "config_to_ids": "|".join(str(x) for x in sorted({r.config_id for r in to_rows})),
                        "n_from": len(from_rows),
                        "n_to": len(to_rows),
                    }
                )
    return out


def classify_delta(y_from: float, y_to: float, epsilon: float = 1e-12) -> str:
    if y_from in (0.0, 1.0) and y_to in (0.0, 1.0):
        if y_from == 0.0 and y_to == 1.0:
            return "up_flip"
        if y_from == 1.0 and y_to == 0.0:
            return "down_flip"
        return "stable_success" if y_to == 1.0 else "stable_fail"
    delta = y_to - y_from
    if delta > epsilon:
        return "positive_delta"
    if delta < -epsilon:
        return "negative_delta"
    return "stable_delta"


def summarize(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(k, "") for k in keys)].append(row)
    out: List[Dict[str, Any]] = []
    for key, items in grouped.items():
        deltas = [float(r["delta"]) for r in items]
        scores_from = [float(r["score_from"]) for r in items]
        scores_to = [float(r["score_to"]) for r in items]
        lo, hi = ci95(deltas)
        rec = {name: value for name, value in zip(keys, key)}
        rec.update(
            {
                "n_pairs": len(items),
                "n_unique_queries": len({r["query_id"] for r in items}),
                "n_backgrounds": len({r["background_id"] for r in items}),
                "mean_score_from": mean(scores_from),
                "mean_score_to": mean(scores_to),
                "mean_delta": mean(deltas),
                "median_delta": med(deltas),
                "ci95_low": lo,
                "ci95_high": hi,
                "up_count": sum(1 for r in items if str(r["flip_type"]) in {"up_flip", "positive_delta"} or float(r["delta"]) > 0),
                "down_count": sum(1 for r in items if str(r["flip_type"]) in {"down_flip", "negative_delta"} or float(r["delta"]) < 0),
                "stable_count": sum(1 for r in items if abs(float(r["delta"])) <= 1e-12),
            }
        )
        rec["up_rate"] = rec["up_count"] / rec["n_pairs"] if rec["n_pairs"] else 0.0
        rec["down_rate"] = rec["down_count"] / rec["n_pairs"] if rec["n_pairs"] else 0.0
        rec["net_rate"] = (rec["up_count"] - rec["down_count"]) / rec["n_pairs"] if rec["n_pairs"] else 0.0
        out.append(rec)
    out.sort(key=lambda r: (str(r.get("family", "")), -abs(float(r.get("mean_delta", 0.0))), str(r.get("feature_atom", ""))))
    return out


def attach_context(rows: Sequence[Mapping[str, Any]], context_atoms: Mapping[str, Sequence[str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        rec = dict(row)
        atoms = tuple(context_atoms.get(str(row["query_id"]), ()))
        rec["context_atoms"] = "|".join(atoms)
        out.append(rec)
    return out


def build_report(out_dir: Path, family_rows: Sequence[Mapping[str, Any]], feature_rows: Sequence[Mapping[str, Any]], transition_rows: Sequence[Mapping[str, Any]]) -> None:
    lines = ["# Per-Family Per-Feature Transition Matrix", ""]
    lines.append("## Family Summary")
    for row in sorted(family_rows, key=lambda r: -abs(float(r["mean_delta"]))):
        lines.append(
            f"- `{row['family']}`: n={row['n_pairs']} mean_delta={float(row['mean_delta']):+.4f} "
            f"ci95=[{float(row['ci95_low']):+.4f}, {float(row['ci95_high']):+.4f}] "
            f"up={float(row['up_rate']):.2f} down={float(row['down_rate']):.2f}"
        )
    lines.extend(["", "## Top Positive Feature Targets"])
    for row in sorted(feature_rows, key=lambda r: -float(r["mean_delta"]))[:30]:
        lines.append(
            f"- `{row['family']}::{row['feature_atom']}` kind={row['transition_kind']} "
            f"n={row['n_pairs']} mean_delta={float(row['mean_delta']):+.4f} "
            f"benchmarks={row.get('datasets', '')}"
        )
    lines.extend(["", "## Top Negative Feature Targets"])
    for row in sorted(feature_rows, key=lambda r: float(r["mean_delta"]))[:30]:
        lines.append(
            f"- `{row['family']}::{row['feature_atom']}` kind={row['transition_kind']} "
            f"n={row['n_pairs']} mean_delta={float(row['mean_delta']):+.4f} "
            f"benchmarks={row.get('datasets', '')}"
        )
    lines.extend(["", "## Top Positive Concrete Transitions"])
    for row in sorted(transition_rows, key=lambda r: -float(r["mean_delta"]))[:30]:
        lines.append(
            f"- `{row['family']}: {row['from_feature']} -> {row['to_feature']}` "
            f"n={row['n_pairs']} mean_delta={float(row['mean_delta']):+.4f}"
        )
    out_dir.joinpath("report.md").write_text("\n".join(lines) + "\n")


def run_family(
    family: str,
    mode: str,
    rows: Sequence[ScoreRow],
    atoms_by_config: Mapping[int, Sequence[str]],
    atoms: Sequence[str],
    ignore_families: Sequence[str],
) -> List[Dict[str, Any]]:
    if mode == "choice":
        return build_choice_contrasts(rows, atoms_by_config, family, ignore_families=ignore_families)
    out: List[Dict[str, Any]] = []
    for atom in atoms:
        if family_of(atom) != family:
            continue
        out.extend(build_additive_contrasts(rows, atoms_by_config, atom, ignore_families=ignore_families))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build per-family per-feature transition tables.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", default="")
    parser.add_argument("--models", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--scorers", default="")
    parser.add_argument("--families", default="")
    parser.add_argument("--exclusive-families", default=",".join(DEFAULT_EXCLUSIVE_FAMILIES))
    parser.add_argument("--additive-families", default=",".join(DEFAULT_ADDITIVE_FAMILIES))
    parser.add_argument("--ignore-families", default=",".join(DEFAULT_BACKGROUND_IGNORE))
    parser.add_argument("--exclude-config-families", default=",".join(DEFAULT_EXCLUDE_CONFIG_FAMILIES))
    parser.add_argument("--min-pairs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = split_csv(args.datasets)
    models = split_csv(args.models)
    scorers = split_csv(args.scorers)
    requested_families = split_csv(args.families)
    exclusive = split_csv(args.exclusive_families)
    additive = split_csv(args.additive_families)
    ignore_families = tuple(split_csv(args.ignore_families) or DEFAULT_BACKGROUND_IGNORE)
    exclude_config_families = tuple(split_csv(args.exclude_config_families))
    family_modes: Dict[str, str] = {family: "choice" for family in exclusive}
    family_modes.update({family: "additive" for family in additive})
    if requested_families:
        family_modes = {k: v for k, v in family_modes.items() if k in set(requested_families)}

    with connect(args.db) as conn:
        atoms_by_config_all = load_config_atoms(conn)
    eligible_configs = {
        cid
        for cid, atoms in atoms_by_config_all.items()
        if not config_has_family(atoms, exclude_config_families)
    }
    atoms_by_config = {cid: atoms for cid, atoms in atoms_by_config_all.items() if cid in eligible_configs}
    atoms = sorted({atom for atoms_ in atoms_by_config.values() for atom in atoms_ if atom_allowed(atom, family_modes.keys())})

    print(f"[load] scores from {args.db}", flush=True)
    rows = load_score_rows(
        args.db,
        datasets=datasets,
        models=models,
        scorers=scorers,
        eligible_configs=eligible_configs,
    )
    print(f"[loaded] {len(rows)} scored config/query rows; {len(eligible_configs)} eligible configs; {len(atoms)} focal atoms", flush=True)

    all_pairs: List[Dict[str, Any]] = []
    futures = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.num_workers, len(family_modes) or 1))) as pool:
        for family, mode in sorted(family_modes.items()):
            futures.append(pool.submit(run_family, family, mode, rows, atoms_by_config, atoms, ignore_families))
        iterator = as_completed(futures)
        if args.progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=len(futures), desc="families")
        for future in iterator:
            all_pairs.extend(future.result())

    all_pairs = [row for row in all_pairs if True]
    if not args.no_context:
        with connect(args.db) as conn:
            context = load_context_atoms(conn, [str(row["query_id"]) for row in all_pairs])
        all_pairs = attach_context(all_pairs, context)

    by_benchmark = summarize(
        all_pairs,
        ["family", "transition_kind", "feature_atom", "from_feature", "to_feature", "dataset", "model", "scorer"],
    )
    by_benchmark = [row for row in by_benchmark if int(row["n_pairs"]) >= args.min_pairs]
    feature_by_benchmark = summarize(
        all_pairs,
        ["family", "transition_kind", "feature_atom", "to_feature", "dataset", "model", "scorer"],
    )
    feature_by_benchmark = [row for row in feature_by_benchmark if int(row["n_pairs"]) >= args.min_pairs]
    feature_overall_base = summarize(all_pairs, ["family", "transition_kind", "feature_atom", "to_feature"])
    feature_overall: List[Dict[str, Any]] = []
    for row in feature_overall_base:
        if int(row["n_pairs"]) < args.min_pairs:
            continue
        matches = [
            r for r in feature_by_benchmark
            if r["family"] == row["family"]
            and r["transition_kind"] == row["transition_kind"]
            and r["feature_atom"] == row["feature_atom"]
            and r["to_feature"] == row["to_feature"]
        ]
        rec = dict(row)
        rec["n_benchmark_rows"] = len(matches)
        rec["positive_benchmark_rows"] = sum(1 for r in matches if float(r["mean_delta"]) > 0)
        rec["negative_benchmark_rows"] = sum(1 for r in matches if float(r["mean_delta"]) < 0)
        rec["datasets"] = "|".join(sorted({str(r["dataset"]) for r in matches}))
        feature_overall.append(rec)
    family_by_benchmark = summarize(all_pairs, ["family", "dataset", "model", "scorer"])
    family_by_benchmark = [row for row in family_by_benchmark if int(row["n_pairs"]) >= args.min_pairs]
    family_overall = summarize(all_pairs, ["family"])
    family_overall = [row for row in family_overall if int(row["n_pairs"]) >= args.min_pairs]

    all_pairs.sort(key=lambda r: (r["family"], r["dataset"], r["model"], r["scorer"], r["query_id"], r["from_feature"], r["to_feature"]))
    by_benchmark.sort(key=lambda r: (r["family"], r["dataset"], -abs(float(r["mean_delta"])), r["from_feature"], r["to_feature"]))
    feature_by_benchmark.sort(key=lambda r: (r["family"], r["dataset"], -abs(float(r["mean_delta"])), r["feature_atom"]))
    feature_overall.sort(key=lambda r: (r["family"], -abs(float(r["mean_delta"])), r["feature_atom"]))
    family_by_benchmark.sort(key=lambda r: (r["family"], r["dataset"]))
    family_overall.sort(key=lambda r: -abs(float(r["mean_delta"])))

    write_csv(out_dir / "query_level_transitions.csv", all_pairs)
    write_csv(out_dir / "transition_summary_by_benchmark.csv", by_benchmark)
    write_csv(out_dir / "feature_summary_by_benchmark.csv", feature_by_benchmark)
    write_csv(out_dir / "feature_summary_overall.csv", feature_overall)
    write_csv(out_dir / "family_summary_by_benchmark.csv", family_by_benchmark)
    write_csv(out_dir / "family_summary_overall.csv", family_overall)
    build_report(out_dir, family_overall, feature_overall, by_benchmark)

    summary = {
        "db": args.db,
        "output_dir": str(out_dir),
        "datasets": datasets,
        "models": models,
        "scorers": scorers,
        "families": sorted(family_modes),
        "excluded_config_families": list(exclude_config_families),
        "ignored_background_families": list(ignore_families),
        "n_score_rows": len(rows),
        "n_query_level_transitions": len(all_pairs),
        "n_transition_summary_rows": len(by_benchmark),
        "n_feature_summary_rows": len(feature_overall),
        "n_family_rows": len(family_overall),
        "min_pairs": args.min_pairs,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
