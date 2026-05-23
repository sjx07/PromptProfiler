"""Transition-local paired flip analysis over a FACET observation cube.

This module audits whether prompt-family transitions are estimable before
running heavier subgroup or causal-rule analysis.  A transition is represented
as a change in one focal prompt family while the non-focal prompt background is
held fixed.

Example:

    python -m analyze.transition_flip \
        --db /data/users/jsu323/facet/wikitable_clean_surface_v1.db \
        --family reasoning --from absent --to symbolic_operation \
        --datasets wtq,sqa \
        --models Qwen/Qwen2.5-14B-Instruct \
        --output-dir outputs/transition_flip/reasoning_symbolic

Use ``--inventory`` first to discover which transitions have enough exact
matched support.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


DEFAULT_FAMILIES = (
    "prompt_format",
    "table_serialization",
    "input_context",
    "reasoning",
    "visible_reasoning",
    "output_contract",
    "response_mode",
    "runtime_binding",
    "domain_heuristic",
    "task_heuristic",
    "retrieval_workflow",
    "base_scaffold",
    "base_shell",
)

DEFAULT_BACKGROUND_IGNORE = ("base_shell",)

PrefixRule = Tuple[str, str, str]

DEFAULT_PREFIX_RULES: Tuple[PrefixRule, ...] = (
    ("base_shell", "_section_", "_section_"),
    ("base_scaffold", "facet_dp_scaffold", ""),
    ("base_scaffold", "sqa_dialog_binding_base", ""),
    ("base_scaffold", "facet_retrieval_scaffold", ""),
    ("prompt_format", "prompt_format_", "prompt_format_"),
    ("table_serialization", "table_serialization_", "table_serialization_"),
    ("input_context", "input_context_", "input_context_"),
    ("visible_reasoning", "reasoning_scaffold_visible_", "reasoning_scaffold_visible_"),
    ("reasoning", "reasoning_", "reasoning_"),
    ("output_contract", "output_contract_", "output_contract_"),
    ("output_contract", "contract_", "contract_"),
    ("response_mode", "response_mode_", "response_mode_"),
    ("runtime_binding", "runtime_binding_", "runtime_binding_"),
    ("domain_heuristic", "dataset_domain_heuristics_", "dataset_domain_heuristics_"),
    ("domain_heuristic", "domain_heuristics_", "domain_heuristics_"),
    ("domain_heuristic", "domain_heuristics.", "domain_heuristics."),
    ("domain_heuristic", "domain_", "domain_"),
    ("task_heuristic", "task_heuristics_", "task_heuristics_"),
    ("retrieval_workflow", "query_decomposition_", ""),
    ("retrieval_workflow", "evidence_summarization_", ""),
    ("retrieval_workflow", "decision_heuristic_", ""),
    ("retrieval_workflow", "query_self_check", ""),
)


@dataclass(frozen=True)
class TransitionSpec:
    """A focal family transition under matched non-focal background."""

    family: str
    from_value: str
    to_value: str
    ignore_families: Tuple[str, ...] = DEFAULT_BACKGROUND_IGNORE
    use_thresholded_score: bool = False
    score_threshold: float = 1.0


@dataclass(frozen=True)
class AtomEditSpec:
    """A prompt feature-set edit under exact matched non-edit atom background.

    ``add_atoms`` are absent on the reference side and present on the
    treatment side. ``remove_atoms`` are present on the reference side and
    absent on the treatment side. All other non-ignored atoms are held fixed,
    including other atoms from the same prompt family.
    """

    add_atoms: Tuple[str, ...] = ()
    remove_atoms: Tuple[str, ...] = ()
    family: str = "atom_edit"
    ignore_families: Tuple[str, ...] = DEFAULT_BACKGROUND_IGNORE
    use_thresholded_score: bool = False
    score_threshold: float = 1.0


@dataclass(frozen=True)
class ScoredExecution:
    config_id: int
    query_id: str
    dataset: str
    task: str
    model: str
    scorer: str
    score: float
    error: str = ""
    metrics: str = "{}"


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def has_table(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _json_loads(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def _split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return sum(values) / len(values)


def _median(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(median(values))


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        out = float(value)
        if math.isnan(out):
            return None
        return out
    except Exception:
        return None


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_family_map(path: Optional[str | Path]) -> Tuple[Tuple[PrefixRule, ...], Tuple[str, ...]]:
    """Load an optional family mapping.

    Expected JSON shape:

        {
          "prefix_rules": [
            {"family": "reasoning", "prefix": "reasoning_", "strip": "reasoning_"}
          ],
          "families": ["reasoning", "input_context"],
          "ignore_families": ["base_shell"]
        }

    Missing fields fall back to defaults.
    """
    if not path:
        return DEFAULT_PREFIX_RULES, DEFAULT_FAMILIES
    data = _json_loads(Path(path).read_text(), {})
    rules = []
    for row in data.get("prefix_rules", []):
        rules.append((str(row["family"]), str(row["prefix"]), str(row.get("strip", row["prefix"]))))
    families = tuple(str(x) for x in data.get("families", []))
    return tuple(rules or DEFAULT_PREFIX_RULES), tuple(families or DEFAULT_FAMILIES)


def atom_family_value(canonical_id: str, rules: Sequence[PrefixRule] = DEFAULT_PREFIX_RULES) -> Optional[Tuple[str, str]]:
    """Infer ``(family, value)`` for a canonical prompt feature atom."""
    cid = str(canonical_id)
    for family, prefix, strip in rules:
        if cid == prefix or cid.startswith(prefix):
            value = cid[len(strip):] if strip and cid.startswith(strip) else cid
            return family, value or cid
    return None


def family_assignments(
    canonical_ids: Iterable[str],
    *,
    rules: Sequence[PrefixRule] = DEFAULT_PREFIX_RULES,
    families: Sequence[str] = DEFAULT_FAMILIES,
    absent_value: str = "absent",
) -> Dict[str, str]:
    """Convert canonical feature atoms into family assignment values.

    Multi-hot families are represented as sorted ``+``-joined values.
    """
    buckets: Dict[str, set[str]] = {family: set() for family in families}
    for cid in canonical_ids:
        mapped = atom_family_value(str(cid), rules=rules)
        if not mapped:
            continue
        family, value = mapped
        buckets.setdefault(family, set()).add(value)
    out: Dict[str, str] = {}
    for family in sorted(buckets):
        values = sorted(v for v in buckets[family] if v)
        out[family] = "+".join(values) if values else absent_value
    return out


def load_config_atoms(conn: sqlite3.Connection) -> Dict[int, List[str]]:
    """Load config_id -> canonical feature atoms from config_feature plus meta fallback."""
    out: Dict[int, set[str]] = defaultdict(set)
    for row in conn.execute("SELECT config_id, meta FROM config"):
        meta = _json_loads(row["meta"], {})
        for cid in meta.get("canonical_ids", []) or []:
            out[int(row["config_id"])].add(str(cid))

    if has_table(conn, "config_feature") and has_table(conn, "feature"):
        for row in conn.execute(
            """
            SELECT cf.config_id, f.canonical_id
            FROM config_feature cf
            JOIN feature f ON f.feature_id = cf.feature_id
            """
        ):
            out[int(row["config_id"])].add(str(row["canonical_id"]))
    return {cid: sorted(atoms) for cid, atoms in out.items()}


def load_config_family_assignments(
    conn: sqlite3.Connection,
    *,
    rules: Sequence[PrefixRule] = DEFAULT_PREFIX_RULES,
    families: Sequence[str] = DEFAULT_FAMILIES,
) -> Dict[int, Dict[str, str]]:
    atoms_by_config = load_config_atoms(conn)
    return {
        config_id: family_assignments(atoms, rules=rules, families=families)
        for config_id, atoms in atoms_by_config.items()
    }


def write_config_family_assignments(
    path: str | Path,
    assignments: Mapping[int, Mapping[str, str]],
) -> None:
    families = sorted({family for row in assignments.values() for family in row})
    rows = []
    for config_id in sorted(assignments):
        rec = {"config_id": config_id}
        rec.update({family: assignments[config_id].get(family, "absent") for family in families})
        rows.append(rec)
    write_csv(path, rows, fieldnames=["config_id"] + families)


def _where_in(column: str, values: Sequence[str], params: List[Any]) -> Optional[str]:
    if not values:
        return None
    params.extend(values)
    return f"{column} IN ({','.join('?' for _ in values)})"


def load_scored_executions(
    conn: sqlite3.Connection,
    *,
    datasets: Sequence[str] = (),
    models: Sequence[str] = (),
    scorers: Sequence[str] = (),
    split: Optional[str] = None,
    max_rows: int = 0,
) -> List[ScoredExecution]:
    """Load scored executions under optional dataset/model/scorer/split filters."""
    params: List[Any] = []
    where = ["ev.score IS NOT NULL"]
    for clause in (
        _where_in("q.dataset", list(datasets), params),
        _where_in("e.model", list(models), params),
        _where_in("ev.scorer", list(scorers), params),
    ):
        if clause:
            where.append(clause)
    if split:
        where.append("json_extract(q.meta, '$.split') = ?")
        params.append(split)
    limit_sql = ""
    if max_rows > 0:
        limit_sql = " LIMIT ?"
        params.append(int(max_rows))
    rows = conn.execute(
        f"""
        SELECT e.config_id,
               e.query_id,
               q.dataset,
               q.dataset AS task,
               e.model,
               ev.scorer,
               ev.score,
               COALESCE(e.error, '') AS error,
               COALESCE(ev.metrics, '{{}}') AS metrics
        FROM execution e
        JOIN query q ON q.query_id = e.query_id
        JOIN evaluation ev ON ev.execution_id = e.execution_id
        WHERE {' AND '.join(where)}
        {limit_sql}
        """,
        tuple(params),
    ).fetchall()
    out = []
    for row in rows:
        score = _safe_float(row["score"])
        if score is None:
            continue
        out.append(
            ScoredExecution(
                config_id=int(row["config_id"]),
                query_id=str(row["query_id"]),
                dataset=str(row["dataset"]),
                task=str(row["task"]),
                model=str(row["model"]),
                scorer=str(row["scorer"]),
                score=score,
                error=str(row["error"] or ""),
                metrics=str(row["metrics"] or "{}"),
            )
        )
    return out


def background_id(
    assignment: Mapping[str, str],
    *,
    focal_family: str,
    ignore_families: Sequence[str] = DEFAULT_BACKGROUND_IGNORE,
) -> str:
    ignored = set(ignore_families) | {focal_family}
    bg = {k: v for k, v in sorted(assignment.items()) if k not in ignored}
    return _stable_json(bg)


def _feature_family(canonical_id: str, rules: Sequence[PrefixRule]) -> str:
    mapped = atom_family_value(canonical_id, rules=rules)
    return mapped[0] if mapped else "unknown"


def infer_edit_family(
    add_atoms: Sequence[str],
    remove_atoms: Sequence[str],
    rules: Sequence[PrefixRule] = DEFAULT_PREFIX_RULES,
) -> str:
    families = {
        _feature_family(atom, rules)
        for atom in list(add_atoms) + list(remove_atoms)
        if atom
    }
    families.discard("unknown")
    if len(families) == 1:
        return next(iter(families))
    if len(families) > 1:
        return "+".join(sorted(families))
    return "atom_edit"


def atom_edit_background_id(
    atoms: Iterable[str],
    spec: AtomEditSpec,
    *,
    rules: Sequence[PrefixRule] = DEFAULT_PREFIX_RULES,
) -> str:
    """Background signature for atom-edit matching.

    The edited atoms are removed from the signature. Other atoms, including
    other atoms from the same family, remain in the background unless their
    family is explicitly ignored.
    """
    edit_atoms = set(spec.add_atoms) | set(spec.remove_atoms)
    ignored_families = set(spec.ignore_families)
    kept = []
    for atom in atoms:
        if atom in edit_atoms:
            continue
        family = _feature_family(atom, rules)
        if family in ignored_families:
            continue
        kept.append(atom)
    return _stable_json(sorted(set(kept)))


def atom_edit_side(atoms: Iterable[str], spec: AtomEditSpec) -> Optional[str]:
    """Return ``from``/``to`` if a config matches one side of an atom edit."""
    atom_set = set(atoms)
    add_atoms = set(spec.add_atoms)
    remove_atoms = set(spec.remove_atoms)
    if not add_atoms and not remove_atoms:
        return None

    is_from = remove_atoms <= atom_set and atom_set.isdisjoint(add_atoms)
    is_to = add_atoms <= atom_set and atom_set.isdisjoint(remove_atoms)
    if is_from and not is_to:
        return "from"
    if is_to and not is_from:
        return "to"
    return None


def atom_edit_labels(spec: AtomEditSpec) -> Tuple[str, str]:
    add = "+".join(spec.add_atoms) if spec.add_atoms else "none"
    remove = "+".join(spec.remove_atoms) if spec.remove_atoms else "none"
    from_label = f"without_add[{add}]__with_remove[{remove}]"
    to_label = f"with_add[{add}]__without_remove[{remove}]"
    return from_label, to_label


def classify_flip(
    y_from: float,
    y_to: float,
    *,
    use_thresholded_score: bool = False,
    score_threshold: float = 1.0,
    epsilon: float = 1e-12,
) -> str:
    if use_thresholded_score:
        a = 1 if y_from >= score_threshold else 0
        b = 1 if y_to >= score_threshold else 0
    elif y_from in (0.0, 1.0) and y_to in (0.0, 1.0):
        a = int(y_from)
        b = int(y_to)
    else:
        delta = y_to - y_from
        if delta > epsilon:
            return "positive_delta"
        if delta < -epsilon:
            return "negative_delta"
        return "stable_delta"

    if a == 0 and b == 1:
        return "up_flip"
    if a == 1 and b == 0:
        return "down_flip"
    if a == 0 and b == 0:
        return "stable_fail"
    return "stable_success"


def build_paired_contrasts(
    rows: Sequence[ScoredExecution],
    assignments: Mapping[int, Mapping[str, str]],
    spec: TransitionSpec,
) -> List[Dict[str, Any]]:
    """Build transition-local paired contrasts under exact matched backgrounds."""
    grouped: Dict[Tuple[str, str, str, str, str, str], Dict[str, List[ScoredExecution]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        assign = assignments.get(row.config_id)
        if not assign:
            continue
        value = assign.get(spec.family, "absent")
        if value not in {spec.from_value, spec.to_value}:
            continue
        bg = background_id(assign, focal_family=spec.family, ignore_families=spec.ignore_families)
        key = (row.query_id, row.dataset, row.task, row.model, row.scorer, bg)
        grouped[key][value].append(row)

    out: List[Dict[str, Any]] = []
    for key, by_value in grouped.items():
        if spec.from_value not in by_value or spec.to_value not in by_value:
            continue
        query_id, dataset, task, model, scorer, bg = key
        from_rows = by_value[spec.from_value]
        to_rows = by_value[spec.to_value]
        y_from = _mean([r.score for r in from_rows])
        y_to = _mean([r.score for r in to_rows])
        delta = y_to - y_from
        out.append({
            "query_id": query_id,
            "dataset": dataset,
            "task": task,
            "model": model,
            "scorer": scorer,
            "family": spec.family,
            "from_value": spec.from_value,
            "to_value": spec.to_value,
            "background_id": bg,
            "config_from_ids": "|".join(str(x) for x in sorted({r.config_id for r in from_rows})),
            "config_to_ids": "|".join(str(x) for x in sorted({r.config_id for r in to_rows})),
            "n_from": len(from_rows),
            "n_to": len(to_rows),
            "y_from": y_from,
            "y_to": y_to,
            "delta": delta,
            "flip_type": classify_flip(
                y_from,
                y_to,
                use_thresholded_score=spec.use_thresholded_score,
                score_threshold=spec.score_threshold,
            ),
        })
    out.sort(key=lambda r: (r["dataset"], r["model"], r["scorer"], r["query_id"], r["background_id"]))
    return out


def build_atom_edit_paired_contrasts(
    rows: Sequence[ScoredExecution],
    atoms_by_config: Mapping[int, Sequence[str]],
    spec: AtomEditSpec,
    *,
    rules: Sequence[PrefixRule] = DEFAULT_PREFIX_RULES,
) -> List[Dict[str, Any]]:
    """Build paired contrasts for an atom-level add/remove edit.

    This avoids treating multi-hot family states as unrelated categorical
    values. For example, adding ``input_context_column_statistics`` can match
    both ``absent -> stats`` and ``type_annotation -> type_annotation+stats``
    while holding the rest of the prompt atom set fixed.
    """
    from_label, to_label = atom_edit_labels(spec)
    grouped: Dict[Tuple[str, str, str, str, str, str], Dict[str, List[ScoredExecution]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        atoms = atoms_by_config.get(row.config_id)
        if atoms is None:
            continue
        side = atom_edit_side(atoms, spec)
        if side is None:
            continue
        bg = atom_edit_background_id(atoms, spec, rules=rules)
        key = (row.query_id, row.dataset, row.task, row.model, row.scorer, bg)
        grouped[key][side].append(row)

    out: List[Dict[str, Any]] = []
    for key, by_side in grouped.items():
        if "from" not in by_side or "to" not in by_side:
            continue
        query_id, dataset, task, model, scorer, bg = key
        from_rows = by_side["from"]
        to_rows = by_side["to"]
        y_from = _mean([r.score for r in from_rows])
        y_to = _mean([r.score for r in to_rows])
        delta = y_to - y_from
        out.append({
            "query_id": query_id,
            "dataset": dataset,
            "task": task,
            "model": model,
            "scorer": scorer,
            "transition_kind": "atom_edit",
            "family": spec.family,
            "from_value": from_label,
            "to_value": to_label,
            "add_atoms": "|".join(spec.add_atoms),
            "remove_atoms": "|".join(spec.remove_atoms),
            "background_id": bg,
            "config_from_ids": "|".join(str(x) for x in sorted({r.config_id for r in from_rows})),
            "config_to_ids": "|".join(str(x) for x in sorted({r.config_id for r in to_rows})),
            "n_from": len(from_rows),
            "n_to": len(to_rows),
            "y_from": y_from,
            "y_to": y_to,
            "delta": delta,
            "flip_type": classify_flip(
                y_from,
                y_to,
                use_thresholded_score=spec.use_thresholded_score,
                score_threshold=spec.score_threshold,
            ),
        })
    out.sort(key=lambda r: (r["dataset"], r["model"], r["scorer"], r["query_id"], r["background_id"]))
    return out


def summarize_background_effects(pairs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[str(row["background_id"])].append(row)
    out: List[Dict[str, Any]] = []
    for bg, rows in grouped.items():
        deltas = [float(r["delta"]) for r in rows]
        n = len(rows)
        up = sum(1 for r in rows if r["flip_type"] == "up_flip" or r["delta"] > 0)
        down = sum(1 for r in rows if r["flip_type"] == "down_flip" or r["delta"] < 0)
        stable_success = sum(1 for r in rows if r["flip_type"] == "stable_success")
        stable_fail = sum(1 for r in rows if r["flip_type"] == "stable_fail")
        out.append({
            "background_id": bg,
            "n_pairs": n,
            "n_unique_queries": len({r["query_id"] for r in rows}),
            "datasets": "|".join(sorted({str(r["dataset"]) for r in rows})),
            "models": "|".join(sorted({str(r["model"]) for r in rows})),
            "scorers": "|".join(sorted({str(r["scorer"]) for r in rows})),
            "mean_delta": _mean(deltas),
            "median_delta": _median(deltas),
            "up_rate": up / n if n else 0.0,
            "down_rate": down / n if n else 0.0,
            "net_flip": (up - down) / n if n else 0.0,
            "stable_success_rate": stable_success / n if n else 0.0,
            "stable_fail_rate": stable_fail / n if n else 0.0,
        })
    out.sort(key=lambda r: (-int(r["n_pairs"]), str(r["background_id"])))
    return out


def transition_summary(pairs: Sequence[Mapping[str, Any]], backgrounds: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    deltas = [float(r["delta"]) for r in pairs]
    net_flips = [float(r["net_flip"]) for r in backgrounds]
    n_pairs = len(pairs)
    return {
        "n_pairs": n_pairs,
        "n_unique_queries": len({r["query_id"] for r in pairs}),
        "n_backgrounds": len(backgrounds),
        "datasets": sorted({str(r["dataset"]) for r in pairs}),
        "models": sorted({str(r["model"]) for r in pairs}),
        "scorers": sorted({str(r["scorer"]) for r in pairs}),
        "mean_delta": _mean(deltas),
        "median_delta": _median(deltas),
        "up_count": sum(1 for r in pairs if r["flip_type"] == "up_flip" or r["delta"] > 0),
        "down_count": sum(1 for r in pairs if r["flip_type"] == "down_flip" or r["delta"] < 0),
        "stable_count": sum(1 for r in pairs if abs(float(r["delta"])) <= 1e-12),
        "mean_background_net_flip": _mean(net_flips),
        "median_background_net_flip": _median(net_flips),
        "sign_stability_background": (sum(1 for x in net_flips if x > 0) / len(net_flips)) if net_flips else 0.0,
        "harm_background": (sum(1 for x in net_flips if x < 0) / len(net_flips)) if net_flips else 0.0,
    }


def leave_one_background_out(
    pairs: Sequence[Mapping[str, Any]],
    backgrounds: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    out = []
    for bg_row in backgrounds:
        bg = str(bg_row["background_id"])
        kept = [r for r in pairs if str(r["background_id"]) != bg]
        bg_effects = summarize_background_effects(kept)
        summary = transition_summary(kept, bg_effects)
        out.append({
            "removed_background_id": bg,
            "removed_n_pairs": bg_row["n_pairs"],
            "remaining_pairs": summary["n_pairs"],
            "remaining_mean_delta": summary["mean_delta"],
            "remaining_median_delta": summary["median_delta"],
            "remaining_sign_stability_background": summary["sign_stability_background"],
        })
    return out


def aggregate_query_effects(pairs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[(str(row["query_id"]), str(row["dataset"]), str(row["task"]), str(row["model"]), str(row["scorer"]))].append(row)
    out = []
    for (query_id, dataset, task, model, scorer), rows in grouped.items():
        deltas = [float(r["delta"]) for r in rows]
        pos = sum(1 for d in deltas if d > 0)
        neg = sum(1 for d in deltas if d < 0)
        out.append({
            "query_id": query_id,
            "dataset": dataset,
            "task": task,
            "model": model,
            "scorer": scorer,
            "n_backgrounds": len({str(r["background_id"]) for r in rows}),
            "n_pairs": len(rows),
            "mean_delta": _mean(deltas),
            "median_delta": _median(deltas),
            "positive_backgrounds": pos,
            "negative_backgrounds": neg,
            "query_flip_score": (pos - neg) / len(rows) if rows else 0.0,
        })
    out.sort(key=lambda r: (r["dataset"], r["model"], r["scorer"], r["query_id"]))
    return out


def load_context_atoms(conn: sqlite3.Connection, query_ids: Iterable[str]) -> Dict[str, Tuple[str, ...]]:
    ids = sorted(set(str(q) for q in query_ids))
    if not ids or not has_table(conn, "predicate"):
        return {qid: tuple() for qid in ids}
    out: Dict[str, List[str]] = defaultdict(list)
    chunk_size = 800
    for i in range(0, len(ids), chunk_size):
        chunk = ids[i:i + chunk_size]
        rows = conn.execute(
            f"""
            SELECT query_id, name, value
            FROM predicate
            WHERE query_id IN ({','.join('?' for _ in chunk)})
            """,
            tuple(chunk),
        ).fetchall()
        for row in rows:
            out[str(row["query_id"])].append(f"{row['name']}={row['value']}")
    return {qid: tuple(sorted(set(out.get(qid, [])))) for qid in ids}


def join_query_effects_with_context(
    query_effects: Sequence[Mapping[str, Any]],
    context_atoms: Mapping[str, Sequence[str]],
) -> List[Dict[str, Any]]:
    out = []
    for row in query_effects:
        rec = dict(row)
        atoms = tuple(context_atoms.get(str(row["query_id"]), ()))
        rec["context_atoms"] = "|".join(atoms)
        out.append(rec)
    return out


def score_context_rules(
    query_effects_with_context: Sequence[Mapping[str, Any]],
    *,
    max_len: int = 3,
    min_support: int = 20,
    top_k: int = 100,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Mine low-order context CNF rules associated with positive/negative query effects."""
    rows = []
    for row in query_effects_with_context:
        atoms = [a for a in str(row.get("context_atoms") or "").split("|") if a]
        rows.append((row, tuple(sorted(set(atoms)))))
    n_total = len(rows)
    if n_total == 0:
        return [], []
    positives = {i for i, (r, _a) in enumerate(rows) if float(r.get("median_delta", 0.0)) > 0}
    negatives = {i for i, (r, _a) in enumerate(rows) if float(r.get("median_delta", 0.0)) < 0}
    n_pos = len(positives)
    n_neg = len(negatives)

    support: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for idx, (_row, atoms) in enumerate(rows):
        for k in range(1, max_len + 1):
            if len(atoms) < k:
                break
            for combo in itertools.combinations(atoms, k):
                support[combo].append(idx)

    scored = []
    for combo, indices in support.items():
        supp = len(indices)
        if supp < min_support:
            continue
        idx_set = set(indices)
        pos_count = len(idx_set & positives)
        neg_count = len(idx_set & negatives)
        deltas = [float(rows[i][0].get("median_delta", 0.0)) for i in indices]
        support_rate = supp / n_total
        p_phi_given_pos = pos_count / n_pos if n_pos else 0.0
        p_phi_given_neg = neg_count / n_neg if n_neg else 0.0
        pos_lift = p_phi_given_pos / support_rate if support_rate else 0.0
        neg_lift = p_phi_given_neg / support_rate if support_rate else 0.0
        pos_rate = pos_count / supp
        neg_rate = neg_count / supp
        net = pos_rate - neg_rate
        score = net * math.log1p(supp)
        rec = {
            "rule": " AND ".join(combo),
            "order": len(combo),
            "support": supp,
            "support_rate": support_rate,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "positive_rate": pos_rate,
            "negative_rate": neg_rate,
            "net_positive_rate": net,
            "positive_lift": pos_lift,
            "negative_lift": neg_lift,
            "mean_delta": _mean(deltas),
            "median_delta": _median(deltas),
            "rank_score": score,
        }
        scored.append(rec)

    positive = [r for r in scored if r["net_positive_rate"] > 0 and r["positive_lift"] > 1.0]
    negative = [r for r in scored if r["net_positive_rate"] < 0 and r["negative_lift"] > 1.0]
    positive.sort(key=lambda r: (-float(r["rank_score"]), -int(r["support"]), r["rule"]))
    negative.sort(key=lambda r: (float(r["rank_score"]), -int(r["support"]), r["rule"]))
    return positive[:top_k], negative[:top_k]


def _iter_progress(items: Iterable[Any], *, enabled: bool, total: Optional[int] = None, desc: str = "progress") -> Iterable[Any]:
    if not enabled:
        return items
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        return items
    return tqdm(items, total=total, desc=desc)


def _inventory_family(
    family: str,
    rows: Sequence[ScoredExecution],
    assignments: Mapping[int, Mapping[str, str]],
    *,
    ignore_families: Sequence[str] = DEFAULT_BACKGROUND_IGNORE,
    use_thresholded_score: bool = False,
    score_threshold: float = 1.0,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str, str, str], Dict[str, List[ScoredExecution]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        assign = assignments.get(row.config_id)
        if not assign:
            continue
        value = assign.get(family, "absent")
        bg = background_id(assign, focal_family=family, ignore_families=ignore_families)
        key = (row.query_id, row.dataset, row.task, row.model, row.scorer, bg)
        grouped[key][value].append(row)

    stats: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(lambda: {
        "n_pairs": 0,
        "queries": set(),
        "backgrounds": set(),
        "datasets": set(),
        "models": set(),
        "scorers": set(),
        "deltas": [],
        "up": 0,
        "down": 0,
    })
    for key, by_value in grouped.items():
        if len(by_value) < 2:
            continue
        query_id, dataset, _task, model, scorer, bg = key
        for from_value, to_value in itertools.permutations(sorted(by_value), 2):
            y_from = _mean([r.score for r in by_value[from_value]])
            y_to = _mean([r.score for r in by_value[to_value]])
            delta = y_to - y_from
            rec = stats[(from_value, to_value)]
            rec["n_pairs"] += 1
            rec["queries"].add(query_id)
            rec["backgrounds"].add(bg)
            rec["datasets"].add(dataset)
            rec["models"].add(model)
            rec["scorers"].add(scorer)
            rec["deltas"].append(delta)
            flip = classify_flip(
                y_from,
                y_to,
                use_thresholded_score=use_thresholded_score,
                score_threshold=score_threshold,
            )
            if flip == "up_flip" or delta > 0:
                rec["up"] += 1
            elif flip == "down_flip" or delta < 0:
                rec["down"] += 1

    out: List[Dict[str, Any]] = []
    for (from_value, to_value), rec in stats.items():
        n = int(rec["n_pairs"])
        deltas = rec["deltas"]
        out.append({
            "family": family,
            "from_value": from_value,
            "to_value": to_value,
            "n_pairs": n,
            "n_unique_queries": len(rec["queries"]),
            "n_backgrounds": len(rec["backgrounds"]),
            "datasets": "|".join(sorted(rec["datasets"])),
            "models": "|".join(sorted(rec["models"])),
            "scorers": "|".join(sorted(rec["scorers"])),
            "mean_delta": _mean(deltas),
            "median_delta": _median(deltas),
            "up_rate": rec["up"] / n if n else 0.0,
            "down_rate": rec["down"] / n if n else 0.0,
            "net_flip": (rec["up"] - rec["down"]) / n if n else 0.0,
            "suitable_for_rule_mining": "yes" if n >= 100 and len(rec["backgrounds"]) >= 2 else "no",
        })
    return out


def inventory_transitions(
    rows: Sequence[ScoredExecution],
    assignments: Mapping[int, Mapping[str, str]],
    *,
    families: Sequence[str],
    ignore_families: Sequence[str] = DEFAULT_BACKGROUND_IGNORE,
    use_thresholded_score: bool = False,
    score_threshold: float = 1.0,
    num_workers: int = 1,
    show_progress: bool = False,
) -> List[Dict[str, Any]]:
    """Audit exact matched support for every ordered value transition by family."""
    family_list = list(families)
    inventory: List[Dict[str, Any]] = []
    if num_workers <= 1 or len(family_list) <= 1:
        iterator = _iter_progress(family_list, enabled=show_progress, total=len(family_list), desc="families")
        for family in iterator:
            inventory.extend(_inventory_family(
                family,
                rows,
                assignments,
                ignore_families=ignore_families,
                use_thresholded_score=use_thresholded_score,
                score_threshold=score_threshold,
            ))
    else:
        max_workers = min(max(1, int(num_workers)), len(family_list))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _inventory_family,
                    family,
                    rows,
                    assignments,
                    ignore_families=ignore_families,
                    use_thresholded_score=use_thresholded_score,
                    score_threshold=score_threshold,
                ): family
                for family in family_list
            }
            iterator = _iter_progress(as_completed(futures), enabled=show_progress, total=len(futures), desc="families")
            for future in iterator:
                inventory.extend(future.result())
    inventory.sort(key=lambda r: (-int(r["n_pairs"]), r["family"], r["from_value"], r["to_value"]))
    return inventory


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], *, fieldnames: Optional[Sequence[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_report(
    path: str | Path,
    *,
    title: str,
    summary: Mapping[str, Any],
    top_backgrounds: Sequence[Mapping[str, Any]] = (),
    top_positive_rules: Sequence[Mapping[str, Any]] = (),
    top_negative_rules: Sequence[Mapping[str, Any]] = (),
    inventory: Sequence[Mapping[str, Any]] = (),
) -> None:
    lines = [f"# {title}", ""]
    lines.append("## Summary")
    lines.append("")
    for key, value in summary.items():
        lines.append(f"- **{key}:** `{value}`")
    if inventory:
        lines.extend(["", "## Top Transition Inventory", ""])
        for row in inventory[:20]:
            lines.append(
                f"- `{row['family']}:{row['from_value']} -> {row['to_value']}` "
                f"pairs={row['n_pairs']} backgrounds={row['n_backgrounds']} "
                f"net={float(row['net_flip']):.4f} mean_delta={float(row['mean_delta']):.4f}"
            )
    if top_backgrounds:
        lines.extend(["", "## Top Backgrounds", ""])
        for row in top_backgrounds[:10]:
            lines.append(
                f"- pairs={row['n_pairs']} net={float(row['net_flip']):.4f} "
                f"mean_delta={float(row['mean_delta']):.4f} bg=`{row['background_id']}`"
            )
    if top_positive_rules:
        lines.extend(["", "## Positive Context Rules", ""])
        for row in top_positive_rules[:10]:
            lines.append(
                f"- `{row['rule']}` support={row['support']} net={float(row['net_positive_rate']):.4f} "
                f"lift={float(row['positive_lift']):.3f} median_delta={float(row['median_delta']):.4f}"
            )
    if top_negative_rules:
        lines.extend(["", "## Negative Context Rules", ""])
        for row in top_negative_rules[:10]:
            lines.append(
                f"- `{row['rule']}` support={row['support']} net={float(row['net_positive_rate']):.4f} "
                f"lift_down={float(row['negative_lift']):.3f} median_delta={float(row['median_delta']):.4f}"
            )
    lines.extend([
        "",
        "## Interpretation Guardrails",
        "",
        "- This is paired contrast / transition-local effect analysis, not causal proof by itself.",
        "- Exact background matching can make support sparse; inspect inventory before mining rules.",
        "- Output-contract transitions should be read with parser/status risk in mind.",
    ])
    Path(path).write_text("\n".join(lines) + "\n")


def run_inventory(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rules, families_all = load_family_map(args.family_map)
    families = _split_csv(args.families) or [f for f in families_all if f not in {"base_shell", "base_scaffold"}]
    ignore_families = tuple(_split_csv(args.ignore_families) or DEFAULT_BACKGROUND_IGNORE)
    with connect(args.db) as conn:
        assignments = load_config_family_assignments(conn, rules=rules, families=families_all)
        write_config_family_assignments(output_dir / "config_family_assignments.csv", assignments)
        rows = load_scored_executions(
            conn,
            datasets=_split_csv(args.datasets),
            models=_split_csv(args.models),
            scorers=_split_csv(args.scorers),
            split=args.split,
            max_rows=args.max_rows,
        )
    inventory = inventory_transitions(
        rows,
        assignments,
        families=families,
        ignore_families=ignore_families,
        use_thresholded_score=args.use_thresholded_score,
        score_threshold=args.score_threshold,
        num_workers=args.num_workers,
        show_progress=args.progress,
    )
    write_csv(output_dir / "transition_inventory.csv", inventory)
    summary = {
        "mode": "inventory",
        "db": args.db,
        "n_scored_rows": len(rows),
        "n_configs_with_assignments": len(assignments),
        "families": ",".join(families),
        "n_transition_rows": len(inventory),
        "num_workers": args.num_workers,
        "progress": args.progress,
    }
    (output_dir / "transition_summary.json").write_text(json.dumps(summary, indent=2))
    write_report(output_dir / "report.md", title="Transition Inventory", summary=summary, inventory=inventory)


def run_transition(args: argparse.Namespace) -> None:
    add_atoms = tuple(_split_csv(args.add_atoms))
    remove_atoms = tuple(_split_csv(args.remove_atoms))
    use_atom_edit = bool(add_atoms or remove_atoms)
    if not use_atom_edit and (not args.family or args.from_value is None or args.to_value is None):
        raise SystemExit(
            "single-transition mode requires either --add-atoms/--remove-atoms "
            "or --family, --from, and --to"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rules, families_all = load_family_map(args.family_map)
    ignore_families = tuple(_split_csv(args.ignore_families) or DEFAULT_BACKGROUND_IGNORE)

    with connect(args.db) as conn:
        atoms_by_config = load_config_atoms(conn)
        assignments = {
            config_id: family_assignments(atoms, rules=rules, families=families_all)
            for config_id, atoms in atoms_by_config.items()
        }
        write_config_family_assignments(output_dir / "config_family_assignments.csv", assignments)
        rows = load_scored_executions(
            conn,
            datasets=_split_csv(args.datasets),
            models=_split_csv(args.models),
            scorers=_split_csv(args.scorers),
            split=args.split,
            max_rows=args.max_rows,
        )
        if use_atom_edit:
            edit_family = args.family or infer_edit_family(add_atoms, remove_atoms, rules=rules)
            atom_spec = AtomEditSpec(
                add_atoms=add_atoms,
                remove_atoms=remove_atoms,
                family=edit_family,
                ignore_families=ignore_families,
                use_thresholded_score=args.use_thresholded_score,
                score_threshold=args.score_threshold,
            )
            pairs = build_atom_edit_paired_contrasts(rows, atoms_by_config, atom_spec, rules=rules)
            from_label, to_label = atom_edit_labels(atom_spec)
            summary_extra = {
                "mode": "atom_edit_transition",
                "family": edit_family,
                "from_value": from_label,
                "to_value": to_label,
                "add_atoms": list(add_atoms),
                "remove_atoms": list(remove_atoms),
            }
            report_title = f"Atom-Edit Flip Audit: add={list(add_atoms)} remove={list(remove_atoms)}"
        else:
            spec = TransitionSpec(
                family=args.family,
                from_value=args.from_value,
                to_value=args.to_value,
                ignore_families=ignore_families,
                use_thresholded_score=args.use_thresholded_score,
                score_threshold=args.score_threshold,
            )
            pairs = build_paired_contrasts(rows, assignments, spec)
            summary_extra = {
                "mode": "family_value_transition",
                "family": args.family,
                "from_value": args.from_value,
                "to_value": args.to_value,
            }
            report_title = f"Transition Flip Audit: {args.family}:{args.from_value} -> {args.to_value}"

        backgrounds = summarize_background_effects(pairs)
        summary = transition_summary(pairs, backgrounds)
        summary.update({
            "db": args.db,
            "n_scored_rows": len(rows),
            **summary_extra,
        })
        query_effects = aggregate_query_effects(pairs)
        context_atoms = load_context_atoms(conn, [r["query_id"] for r in query_effects])
    query_with_context = join_query_effects_with_context(query_effects, context_atoms)
    pos_rules, neg_rules = score_context_rules(
        query_with_context,
        max_len=args.context_max_len,
        min_support=args.min_support,
        top_k=args.top_k_rules,
    )

    write_csv(output_dir / "paired_contrasts.csv", pairs)
    write_csv(output_dir / "background_effects.csv", backgrounds)
    write_csv(output_dir / "leave_background_out.csv", leave_one_background_out(pairs, backgrounds))
    write_csv(output_dir / "query_level_effects.csv", query_effects)
    write_csv(output_dir / "query_effects_with_context.csv", query_with_context)
    write_csv(output_dir / "context_rules_positive.csv", pos_rules)
    write_csv(output_dir / "context_rules_negative.csv", neg_rules)
    (output_dir / "transition_summary.json").write_text(json.dumps(summary, indent=2))
    write_report(
        output_dir / "report.md",
        title=report_title,
        summary=summary,
        top_backgrounds=backgrounds,
        top_positive_rules=pos_rules,
        top_negative_rules=neg_rules,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transition-local paired flip audit over a FACET cube")
    parser.add_argument("--db", required=True, help="SQLite observation cube path")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV/JSON/Markdown outputs")
    parser.add_argument("--inventory", action="store_true", help="Audit all ordered transitions for selected families")
    parser.add_argument("--family", help="Focal family for single-transition mode")
    parser.add_argument("--from", dest="from_value", help="From family value for single-transition mode")
    parser.add_argument("--to", dest="to_value", help="To family value for single-transition mode")
    parser.add_argument("--add-atoms", default="", help="Comma-separated canonical atoms that define the treatment side")
    parser.add_argument("--remove-atoms", default="", help="Comma-separated canonical atoms removed on the treatment side")
    parser.add_argument("--families", default="", help="Comma-separated families for inventory mode")
    parser.add_argument("--ignore-families", default=",".join(DEFAULT_BACKGROUND_IGNORE), help="Comma-separated families excluded from background matching")
    parser.add_argument("--family-map", default=None, help="Optional JSON family mapping")
    parser.add_argument("--datasets", default="", help="Comma-separated dataset filters")
    parser.add_argument("--models", default="", help="Comma-separated model filters")
    parser.add_argument("--scorers", default="", help="Comma-separated scorer filters")
    parser.add_argument("--split", default=None, help="Optional query split filter from query.meta.split")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional cap on scored execution rows loaded")
    parser.add_argument("--use-thresholded-score", action="store_true", help="Classify flips by score threshold")
    parser.add_argument("--score-threshold", type=float, default=1.0, help="Threshold used with --use-thresholded-score")
    parser.add_argument("--min-support", type=int, default=20, help="Minimum context rule support")
    parser.add_argument("--top-k-rules", type=int, default=100, help="Maximum context rules per direction")
    parser.add_argument("--context-max-len", type=int, default=3, help="Maximum context CNF length")
    parser.add_argument("--num-workers", type=int, default=1, help="Worker threads for inventory mode")
    parser.add_argument("--progress", action="store_true", help="Show tqdm progress bars when available")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.inventory:
        run_inventory(args)
    else:
        run_transition(args)


if __name__ == "__main__":  # pragma: no cover
    main()
