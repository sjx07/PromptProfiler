"""Predicate-aware experiment loop analysis.

Implements the analyze_fn interface for the experiment loop, providing:
    A. build_delta_table    — flat DataFrame of per-query deltas
    B. summarize_by_cell    — (primitive x predicate x value) stats
    C. classify_primitives  — always_on / always_off / gated
    D. pick_target_cells    — information-gain heuristic
    E. build_targeted_plan  — RunEntry list from target cells
    F. make_seed_plan       — cold-start initial plan

All planning logic is purely programmatic. No LLM calls.

Usage:
    # Define primitives (single or compound)
    specs = [
        PrimitiveSpec(name="enable_cot", func_ids=[cot_fid]),
        PrimitiveSpec(name="+error_context", func_ids=[field_fid, rule_fid]),
    ]

    # Cold start
    plan = make_seed_plan(store, base_ids, specs, query_ids,
                          n_primitives=3, n_queries=200)

    # Closed loop
    analyze_fn = make_predicate_aware_analyzer(
        base_cid=1, base_ids=base_ids,
        primitive_specs=specs, predicate_names=["is_count", "operation_type"],
    )
    run_experiment(store, plan, task_cls, model, llm_call, analyze_fn)
"""
from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import pandas as pd

from prompt_profiler.core.store import CubeStore
from prompt_profiler.experiment.loop import AnalysisResult
from prompt_profiler.experiment.planner import RunEntry

logger = logging.getLogger(__name__)


# ── dataclasses ───────────────────────────────────────────────────────


@dataclass
class PrimitiveSpec:
    """A named primitive — one or more func_ids that always appear together.

    Single-func primitives: func_ids has one element (classic add-one).
    Compound primitives: func_ids has multiple elements (e.g., add_input_field
    + explanation rule). The planner always adds the full set; the analyzer
    matches configs by the full set.
    """

    name: str
    func_ids: List[str]


@dataclass
class PrimitiveClassification:
    """Result of classify_primitives."""

    always_on: List[str]
    always_off: List[str]
    gated: List[str]
    undetermined: List[str]
    global_stats: pd.DataFrame  # per-primitive summary


@dataclass
class TargetCell:
    """A (primitive, predicate, value) cell selected for measurement."""

    primitive_id: str
    predicate_name: str
    predicate_value: str
    priority: float
    current_n: int


# ── A. build_delta_table ──────────────────────────────────────────────


def build_delta_table(
    store: CubeStore,
    model: str,
    scorer: str,
    base_cid: int,
    primitive_specs: Optional[List[PrimitiveSpec]] = None,
) -> pd.DataFrame:
    """Build flat DataFrame: one row per (query, primitive) with delta and predicates.

    Primitive identification:
        If primitive_specs is given, matches configs whose func_id diff
        against base equals a known spec's func_ids (supports compound primitives).
        If None, auto-discovers single-func diffs (classic add-one).

    Returns columns:
        query_id, primitive_id, config_id, baseline_score, primitive_score,
        delta, <predicate columns...>
    """
    conn = store._get_conn()

    # All scores in one query
    scores_df = pd.read_sql_query(
        """
        SELECT e.config_id, e.query_id, ev.score
        FROM execution e
        JOIN evaluation ev ON e.execution_id = ev.execution_id
        WHERE e.model = ? AND ev.scorer = ?
        """,
        conn,
        params=(model, scorer),
    )

    if scores_df.empty:
        return pd.DataFrame()

    # Baseline scores indexed by query_id
    base_scores = (
        scores_df[scores_df.config_id == base_cid]
        .drop_duplicates(subset="query_id", keep="first")
        .set_index("query_id")["score"]
    )

    if base_scores.empty:
        logger.warning("No baseline scores for config_id=%d", base_cid)
        return pd.DataFrame()

    # Identify primitive configs by diffing func_ids against base
    base_func_ids = set(store.get_config_func_ids(base_cid))
    configs = store.list_configs()

    # Build lookup: frozenset of added func_ids → primitive name
    if primitive_specs is not None:
        spec_lookup: Dict[frozenset, str] = {
            frozenset(spec.func_ids): spec.name for spec in primitive_specs
        }
    else:
        spec_lookup = None

    config_to_primitive: Dict[int, str] = {}
    for cfg in configs:
        cid = cfg["config_id"]
        if cid == base_cid:
            continue
        func_ids = set(json.loads(cfg["func_ids"]))
        added = func_ids - base_func_ids
        removed = base_func_ids - func_ids
        if removed:
            continue

        if spec_lookup is not None:
            # Match against known primitive specs (compound or single)
            key = frozenset(added)
            if key in spec_lookup:
                config_to_primitive[cid] = spec_lookup[key]
        else:
            # Auto-discover: single-func diffs only
            if len(added) == 1:
                config_to_primitive[cid] = added.pop()

    if not config_to_primitive:
        logger.warning("No primitive configs found relative to base %d", base_cid)
        return pd.DataFrame()

    # Build delta rows via vectorized merge instead of per-query loop
    prim_scores = scores_df[scores_df.config_id.isin(config_to_primitive)]
    prim_scores = prim_scores.assign(
        primitive_id=prim_scores.config_id.map(config_to_primitive)
    )

    merged = prim_scores.merge(
        base_scores.rename("baseline_score"),
        left_on="query_id",
        right_index=True,
        how="inner",
    )
    merged = merged.rename(columns={"score": "primitive_score"})
    merged["delta"] = merged["primitive_score"] - merged["baseline_score"]

    delta_df = merged[
        ["query_id", "primitive_id", "config_id", "baseline_score", "primitive_score", "delta"]
    ].copy()

    # Join predicates (wide format)
    pred_df = pd.read_sql_query(
        "SELECT query_id, name, value FROM predicate", conn
    )
    if not pred_df.empty:
        pred_wide = pred_df.pivot_table(
            index="query_id", columns="name", values="value", aggfunc="first"
        )
        delta_df = delta_df.merge(pred_wide, on="query_id", how="left")

    logger.info(
        "build_delta_table: %d rows, %d primitives, %d queries",
        len(delta_df),
        delta_df.primitive_id.nunique(),
        delta_df.query_id.nunique(),
    )
    return delta_df


# ── predicate type detection ──────────────────────────────────────────

_NUMERICAL_THRESHOLD = 8  # predicates with > this many distinct values are numerical


def detect_predicate_types(
    delta_df: pd.DataFrame,
    predicate_cols: List[str],
) -> Tuple[List[str], List[str]]:
    """Split predicates into categorical vs numerical.

    Numerical: > _NUMERICAL_THRESHOLD distinct values AND parseable as float.
    """
    categorical, numerical = [], []
    for col in predicate_cols:
        if col not in delta_df.columns:
            continue
        vals = delta_df[col].dropna().unique()
        if len(vals) <= _NUMERICAL_THRESHOLD:
            categorical.append(col)
            continue
        # Check if values are numeric
        try:
            pd.to_numeric(pd.Series(vals))
            numerical.append(col)
        except (ValueError, TypeError):
            categorical.append(col)
    return categorical, numerical


# ── B. summarize_by_cell ──────────────────────────────────────────────


def summarize_by_cell(
    delta_df: pd.DataFrame,
    predicate_cols: List[str],
) -> pd.DataFrame:
    """Aggregate to (primitive, predicate_name, predicate_value) level.

    Categorical predicates: per-value cell stats (n, mean, se).
    Numerical predicates: linear regression (slope, slope_se, p_value).

    Returns columns: primitive_id, predicate_name, predicate_value,
                     n, mean_delta, std_delta, se, coverage,
                     pred_type ("categorical" or "numerical"),
                     slope, slope_se, p_value (numerical only)
    """
    if delta_df.empty or not predicate_cols:
        return pd.DataFrame()

    cat_cols, num_cols = detect_predicate_types(delta_df, predicate_cols)

    rows = []

    # Categorical: per-value cells
    for prim_id, prim_df in delta_df.groupby("primitive_id"):
        n_total = len(prim_df)
        for pred_name in cat_cols:
            if pred_name not in prim_df.columns:
                continue
            for pred_val, cell_df in prim_df.groupby(pred_name):
                if pd.isna(pred_val):
                    continue
                n = len(cell_df)
                mean_d = cell_df["delta"].mean()
                std_d = cell_df["delta"].std() if n > 1 else 0.0
                se = std_d / math.sqrt(n) if n > 1 else float("inf")
                rows.append({
                    "primitive_id": prim_id,
                    "predicate_name": pred_name,
                    "predicate_value": str(pred_val),
                    "n": n,
                    "mean_delta": mean_d,
                    "std_delta": std_d,
                    "se": se,
                    "coverage": n / n_total if n_total > 0 else 0.0,
                    "pred_type": "categorical",
                    "slope": None,
                    "slope_se": None,
                    "p_value": None,
                })

    # Numerical: linear regression (delta ~ beta * x)
    for prim_id, prim_df in delta_df.groupby("primitive_id"):
        n_total = len(prim_df)
        for pred_name in num_cols:
            if pred_name not in prim_df.columns:
                continue
            x = pd.to_numeric(prim_df[pred_name], errors="coerce")
            y = prim_df["delta"]
            valid = x.notna() & y.notna()
            x_v, y_v = x[valid].values, y[valid].values
            n = len(x_v)
            if n < 10:
                continue

            slope, slope_se, p_val = _ols_slope(x_v, y_v)

            rows.append({
                "primitive_id": prim_id,
                "predicate_name": pred_name,
                "predicate_value": "_regression",
                "n": n,
                "mean_delta": y_v.mean(),
                "std_delta": y_v.std(),
                "se": y_v.std() / math.sqrt(n),
                "coverage": n / n_total if n_total > 0 else 0.0,
                "pred_type": "numerical",
                "slope": slope,
                "slope_se": slope_se,
                "p_value": p_val,
            })

    return pd.DataFrame(rows)


def _ols_slope(x, y):
    """Simple OLS: y = a + b*x. Returns (slope, slope_se, p_value).

    Uses numpy for speed. No scipy dependency — p_value from t-distribution
    approximated via normal for large n.
    """
    import numpy as np

    n = len(x)
    x_mean = x.mean()
    y_mean = y.mean()
    ss_xx = ((x - x_mean) ** 2).sum()
    ss_xy = ((x - x_mean) * (y - y_mean)).sum()

    if ss_xx == 0:
        return 0.0, float("inf"), 1.0

    slope = ss_xy / ss_xx
    y_hat = x_mean + slope * (x - x_mean)  # intercept absorbed
    residuals = y - y_hat
    mse = (residuals ** 2).sum() / max(n - 2, 1)
    slope_se = math.sqrt(mse / ss_xx) if mse > 0 else 0.0

    if slope_se > 0:
        t_stat = abs(slope / slope_se)
        # Normal approximation for p-value (good for n > 30)
        p_value = 2 * (1 - _normal_cdf(t_stat))
    else:
        p_value = 0.0 if slope != 0 else 1.0

    return float(slope), float(slope_se), float(p_value)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ── C. classify_primitives ────────────────────────────────────────────


def classify_primitives(
    delta_df: pd.DataFrame,
    cell_stats: pd.DataFrame,
    *,
    min_n: int = 30,
    pos_threshold: float = 0.02,
    neg_threshold: float = -0.02,
    gated_sign_min_n: int = 50,
    slope_p_threshold: float = 0.05,
) -> PrimitiveClassification:
    """Classify primitives using simple threshold rules.

    v1 heuristic — upgrade path: Bayesian hypothesis test, bootstrap CI.

    Thresholds:
        min_n=30             CLT minimum for SE reliability.
        pos_threshold=0.02   2pp meaningful at base acc ~0.47 (WTQ).
        neg_threshold=-0.02  Symmetric.
        gated_sign_min_n=50  Per-cell minimum to count toward sign disagreement.
        slope_p_threshold=0.05  Significance threshold for numerical predicates.

    Rules:
        1. global_n < min_n → undetermined
        2. Categorical sign disagreement OR significant numerical slope → gated
        3. global_mean > pos_threshold → always_on
        4. global_mean < neg_threshold → always_off
        5. else → undetermined (near-zero effect)
    """
    if delta_df.empty:
        return PrimitiveClassification([], [], [], [], pd.DataFrame())

    global_stats = (
        delta_df.groupby("primitive_id")["delta"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    global_stats.columns = ["primitive_id", "global_mean", "global_std", "global_n"]

    always_on, always_off, gated, undetermined = [], [], [], []

    for _, row in global_stats.iterrows():
        pid = row["primitive_id"]

        if row["global_n"] < min_n:
            undetermined.append(pid)
            continue

        is_gated = False

        if not cell_stats.empty:
            prim_cells = cell_stats[cell_stats.primitive_id == pid]

            # Categorical: sign disagreement across large-enough cells
            cat_cells = prim_cells[prim_cells.pred_type == "categorical"]
            sig_cells = cat_cells[cat_cells.n >= gated_sign_min_n]
            has_pos = (sig_cells.mean_delta > pos_threshold).any() if len(sig_cells) else False
            has_neg = (sig_cells.mean_delta < neg_threshold).any() if len(sig_cells) else False
            if has_pos and has_neg:
                is_gated = True

            # Numerical: significant slope on any predicate
            num_cells = prim_cells[prim_cells.pred_type == "numerical"]
            if not num_cells.empty:
                sig_slopes = num_cells[
                    num_cells.p_value.notna() & (num_cells.p_value < slope_p_threshold)
                ]
                if len(sig_slopes) > 0:
                    is_gated = True
        else:
            has_pos, has_neg = False, False

        if is_gated:
            gated.append(pid)
        elif row["global_mean"] > pos_threshold:
            always_on.append(pid)
        elif row["global_mean"] < neg_threshold:
            always_off.append(pid)
        else:
            undetermined.append(pid)

    return PrimitiveClassification(
        always_on=always_on,
        always_off=always_off,
        gated=gated,
        undetermined=undetermined,
        global_stats=global_stats,
    )


# ── D. pick_target_cells ─────────────────────────────────────────────


def pick_target_cells(
    cell_stats: pd.DataFrame,
    primitives: List[str],
    predicate_values: Dict[str, List[str]],
    *,
    top_k: int = 10,
    lambda_uncertainty: float = 1.0,
    unseen_bonus: float = 2.0,
    min_priority: float = 0.01,
) -> List[TargetCell]:
    """Select the most informative cells to measure next.

    v1 heuristic — upgrade path: UCB / Thompson sampling / LinUCB per-cell.

    Scoring:
        seen cells:   priority = |mean_delta| + lambda * se
        unseen cells: priority = unseen_bonus
    """
    # Index existing measurements for O(1) lookup
    seen: Dict[Tuple[str, str, str], Any] = {}
    if not cell_stats.empty:
        for _, row in cell_stats.iterrows():
            key = (row["primitive_id"], row["predicate_name"], row["predicate_value"])
            seen[key] = row

    targets = []
    for pid in primitives:
        for pred_name, values in predicate_values.items():
            for pred_val in values:
                key = (pid, pred_name, pred_val)
                if key in seen:
                    row = seen[key]
                    se = row["se"] if row["se"] != float("inf") else 1.0
                    priority = abs(row["mean_delta"]) + lambda_uncertainty * se
                    current_n = int(row["n"])
                else:
                    priority = unseen_bonus
                    current_n = 0

                if priority >= min_priority:
                    targets.append(TargetCell(
                        primitive_id=pid,
                        predicate_name=pred_name,
                        predicate_value=pred_val,
                        priority=priority,
                        current_n=current_n,
                    ))

    targets.sort(key=lambda t: -t.priority)
    return targets[:top_k]


# ── E. build_targeted_plan ────────────────────────────────────────────


def build_targeted_plan(
    store: CubeStore,
    model: str,
    base_cid: int,
    base_ids: List[str],
    targets: List[TargetCell],
    primitive_specs: Optional[List[PrimitiveSpec]] = None,
    *,
    query_budget_per_cell: int = 50,
    seed: int = 42,
) -> List[RunEntry]:
    """Convert target cells into RunEntry objects.

    For each cell:
        1. Find queries matching the predicate condition
        2. Exclude already-executed queries
        3. Sample up to query_budget from remaining
    Groups by primitive to avoid duplicate configs.
    Also creates baseline fill RunEntries for queries missing baseline scores.

    If primitive_specs is given, uses spec.func_ids to build configs
    (supports compound primitives). Otherwise treats primitive_id as
    a single func_id.
    """
    conn = store._get_conn()
    rng = random.Random(seed)

    # Lookup: primitive name → func_ids
    spec_map: Dict[str, List[str]] = {}
    if primitive_specs:
        for spec in primitive_specs:
            spec_map[spec.name] = spec.func_ids

    # Group targets by primitive
    prim_targets: Dict[str, List[TargetCell]] = {}
    for t in targets:
        prim_targets.setdefault(t.primitive_id, []).append(t)

    plan: List[RunEntry] = []
    base_executed = store.get_cached_query_ids(base_cid, model)

    for prim_name, cells in prim_targets.items():
        prim_func_ids = spec_map.get(prim_name, [prim_name])
        config_func_ids = base_ids + prim_func_ids
        cid = store.get_or_create_config(config_func_ids)
        executed = store.get_cached_query_ids(cid, model)

        # Collect query_ids across all target cells for this primitive
        target_qids: Set[str] = set()
        for cell in cells:
            rows = conn.execute(
                "SELECT query_id FROM predicate WHERE name = ? AND value = ?",
                (cell.predicate_name, cell.predicate_value),
            ).fetchall()
            cell_qids = {r[0] for r in rows} - executed
            cell_qids_list = sorted(cell_qids)
            if len(cell_qids_list) > query_budget_per_cell:
                cell_qids_list = rng.sample(cell_qids_list, query_budget_per_cell)
            target_qids.update(cell_qids_list)

        if not target_qids:
            continue

        # Baseline fill for queries that lack baseline scores
        base_needed = target_qids - base_executed
        if base_needed:
            plan.append(RunEntry(
                config_id=base_cid,
                func_ids=base_ids,
                query_ids=sorted(base_needed),
                meta={"role": "baseline_fill", "primitive": prim_name},
            ))

        plan.append(RunEntry(
            config_id=cid,
            func_ids=config_func_ids,
            query_ids=sorted(target_qids),
            meta={
                "role": "targeted",
                "primitive": prim_name,
                "n_target_cells": len(cells),
            },
        ))

    return plan


# ── F. make_seed_plan ─────────────────────────────────────────────────


def make_seed_plan(
    store: CubeStore,
    base_ids: List[str],
    primitive_specs: List[PrimitiveSpec],
    all_query_ids: List[str],
    *,
    n_primitives: int = 3,
    n_queries: int = 200,
    seed: int = 42,
    predicate_name: Optional[str] = None,
) -> List[RunEntry]:
    """Create a small cold-start seed plan.

    Samples n_primitives from primitive_specs and n_queries queries.
    If predicate_name is given, stratifies the query sample to
    cover major predicate values proportionally.

    Returns RunEntries for:
        1. Baseline config x sampled queries
        2. Each selected primitive config x same queries
    """
    rng = random.Random(seed)

    selected = (
        primitive_specs[:n_primitives]
        if len(primitive_specs) <= n_primitives
        else rng.sample(primitive_specs, n_primitives)
    )

    if predicate_name:
        sampled = _stratified_sample(store, all_query_ids, predicate_name, n_queries, rng)
    else:
        sampled = rng.sample(all_query_ids, min(n_queries, len(all_query_ids)))

    base_cid = store.get_or_create_config(base_ids)
    plan = [RunEntry(
        config_id=base_cid,
        func_ids=base_ids,
        query_ids=sampled,
        meta={"role": "seed_baseline"},
    )]

    for spec in selected:
        func_ids = base_ids + spec.func_ids
        cid = store.get_or_create_config(func_ids)
        plan.append(RunEntry(
            config_id=cid,
            func_ids=func_ids,
            query_ids=sampled,
            meta={"role": "seed_primitive", "primitive": spec.name},
        ))

    logger.info(
        "Seed plan: %d primitives x %d queries = %d configs",
        len(selected), len(sampled), len(plan),
    )
    return plan


# ── analyze_fn factory ────────────────────────────────────────────────


def make_predicate_aware_analyzer(
    base_cid: int,
    base_ids: List[str],
    primitive_specs: List[PrimitiveSpec],
    predicate_names: List[str],
    *,
    min_n: int = 30,
    pos_threshold: float = 0.02,
    neg_threshold: float = -0.02,
    gated_sign_min_n: int = 10,
    top_k: int = 10,
    lambda_uncertainty: float = 1.0,
    query_budget_per_cell: int = 50,
    max_new_queries: int = 500,
) -> Callable[[CubeStore, str, str, int], AnalysisResult]:
    """Factory returning an AnalyzeFn closure.

    The returned function:
        1. Builds delta table from DB (current state S_t)
        2. Summarizes by (primitive, predicate, value) cells
        3. Classifies primitives
        4. Picks target cells for next round
        5. Returns AnalysisResult with insights + next_plan

    Markov: next_plan depends only on S_t = {delta_table, cell_stats, coverage}.
    """
    prim_names = [s.name for s in primitive_specs]

    def analyze_fn(
        store: CubeStore, model: str, scorer: str, iteration: int,
    ) -> AnalysisResult:
        # ── A. Build current state from DB ────────────────────────
        delta_df = build_delta_table(store, model, scorer, base_cid,
                                     primitive_specs=primitive_specs)

        if delta_df.empty:
            logger.warning("Iteration %d: no delta data available", iteration)
            return AnalysisResult(insights={"error": "no_data"}, next_plan=None)

        pred_cols = [c for c in predicate_names if c in delta_df.columns]

        # ── B. Cell-level statistics ──────────────────────────────
        cell_stats = summarize_by_cell(delta_df, pred_cols)

        # ── C. Classify primitives ────────────────────────────────
        classification = classify_primitives(
            delta_df, cell_stats,
            min_n=min_n,
            pos_threshold=pos_threshold,
            neg_threshold=neg_threshold,
            gated_sign_min_n=gated_sign_min_n,
        )

        n_queries = delta_df.query_id.nunique()
        n_prims = delta_df.primitive_id.nunique()
        logger.info(
            "Iteration %d state: %d queries, %d primitives — "
            "%d always_on, %d always_off, %d gated, %d undetermined",
            iteration, n_queries, n_prims,
            len(classification.always_on), len(classification.always_off),
            len(classification.gated), len(classification.undetermined),
        )

        # ── D. Pick target cells ──────────────────────────────────
        predicate_values = _get_predicate_values(store, pred_cols)

        # Only target undetermined + gated (already-decided are lower priority)
        active_prims = classification.undetermined + classification.gated
        if not active_prims:
            active_prims = prim_names  # fallback: all

        targets = pick_target_cells(
            cell_stats, active_prims, predicate_values,
            top_k=top_k, lambda_uncertainty=lambda_uncertainty,
        )

        insights = _build_insights(delta_df, cell_stats, classification, pred_cols)

        if not targets:
            logger.info("Iteration %d: no target cells, converged", iteration)
            return AnalysisResult(insights=insights, next_plan=None)

        # ── E. Build targeted plan ────────────────────────────────
        next_plan = build_targeted_plan(
            store, model, base_cid, base_ids, targets,
            primitive_specs=primitive_specs,
            query_budget_per_cell=query_budget_per_cell,
            seed=42 + iteration,
        )

        # Enforce per-iteration query budget
        next_plan = _trim_plan(next_plan, max_new_queries)

        total = sum(len(e.query_ids) for e in next_plan)
        if total == 0:
            logger.info("Iteration %d: all target queries already executed", iteration)
            return AnalysisResult(insights=insights, next_plan=None)

        logger.info(
            "Iteration %d: next plan has %d entries, %d LLM calls",
            iteration, len(next_plan), total,
        )
        return AnalysisResult(insights=insights, next_plan=next_plan)

    return analyze_fn


# ── private helpers ───────────────────────────────────────────────────


def _get_predicate_values(
    store: CubeStore, predicate_names: List[str],
) -> Dict[str, List[str]]:
    """Get distinct values for each predicate from the DB."""
    conn = store._get_conn()
    result = {}
    for name in predicate_names:
        rows = conn.execute(
            "SELECT DISTINCT value FROM predicate WHERE name = ? ORDER BY value",
            (name,),
        ).fetchall()
        result[name] = [r[0] for r in rows]
    return result


def _build_insights(
    delta_df: pd.DataFrame,
    cell_stats: pd.DataFrame,
    classification: PrimitiveClassification,
    pred_cols: List[str],
) -> Dict[str, Any]:
    """Package analysis outputs into an insights dict."""
    prim_stats = (
        delta_df.groupby("primitive_id")["delta"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    prim_stats.columns = ["primitive_id", "mean_delta", "std_delta", "n"]

    if not cell_stats.empty:
        total_cells = len(cell_stats)
        measured_cells = int((cell_stats.n >= 10).sum())
        coverage_frac = measured_cells / total_cells if total_cells > 0 else 0.0
    else:
        coverage_frac = 0.0

    return {
        "primitive_stats": prim_stats.to_dict("records"),
        "cell_stats": cell_stats.to_dict("records") if not cell_stats.empty else [],
        "always_on": classification.always_on,
        "always_off": classification.always_off,
        "gated": classification.gated,
        "undetermined": classification.undetermined,
        "coverage": coverage_frac,
        "n_queries": delta_df.query_id.nunique(),
        "n_primitives": delta_df.primitive_id.nunique(),
    }


def _trim_plan(plan: List[RunEntry], max_queries: int) -> List[RunEntry]:
    """Trim plan entries to stay within query budget."""
    trimmed = []
    remaining = max_queries
    for entry in plan:
        if remaining <= 0:
            break
        if len(entry.query_ids) <= remaining:
            trimmed.append(entry)
            remaining -= len(entry.query_ids)
        else:
            trimmed.append(RunEntry(
                config_id=entry.config_id,
                func_ids=entry.func_ids,
                query_ids=entry.query_ids[:remaining],
                meta=entry.meta,
            ))
            remaining = 0
    return trimmed


def _stratified_sample(
    store: CubeStore,
    query_ids: List[str],
    predicate_name: str,
    n: int,
    rng: random.Random,
) -> List[str]:
    """Sample n queries stratified by a predicate (proportional allocation)."""
    conn = store._get_conn()
    qid_set = set(query_ids)

    rows = conn.execute(
        "SELECT query_id, value FROM predicate WHERE name = ?",
        (predicate_name,),
    ).fetchall()

    groups: Dict[str, List[str]] = {}
    for r in rows:
        if r["query_id"] in qid_set:
            groups.setdefault(r["value"], []).append(r["query_id"])

    if not groups:
        return rng.sample(query_ids, min(n, len(query_ids)))

    total = sum(len(v) for v in groups.values())
    sampled = []
    for val, qids in groups.items():
        k = max(1, round(n * len(qids) / total))
        k = min(k, len(qids))
        sampled.extend(rng.sample(qids, k))

    # Trim or pad to target n
    if len(sampled) > n:
        sampled = rng.sample(sampled, n)
    elif len(sampled) < n:
        remaining = [q for q in query_ids if q not in set(sampled)]
        extra = min(n - len(sampled), len(remaining))
        if extra > 0:
            sampled.extend(rng.sample(remaining, extra))

    return sampled


# ── G. fit_routing_model ──────────────────────────────────────────────


@dataclass
class RoutingModel:
    """Per-primitive decision tree routing model.

    Fitted on per-query deltas with predicate features as X.
    Each tree predicts delta(p, q) for its primitive p.
    Routing: pick primitive with max predicted delta (if > 0).
    """

    trees: Dict[str, Any]  # primitive_id -> DecisionTreeRegressor
    feature_names: List[str]  # columns used for fitting
    cat_columns: List[str]  # original categorical columns (before one-hot)
    num_columns: List[str]  # numerical columns (passed through)

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode predicate columns into model features.

        Same encoding used at fit time — categorical one-hot, numerical pass-through.
        Handles unseen categories gracefully (missing columns filled with 0).
        """
        parts = []
        for col in self.cat_columns:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, dtype=float)
                parts.append(dummies)
        for col in self.num_columns:
            if col in df.columns:
                parts.append(pd.to_numeric(df[col], errors="coerce").rename(col))

        if not parts:
            return pd.DataFrame(index=df.index)

        encoded = pd.concat(parts, axis=1)
        # Align to training features — fill missing with 0, drop extra
        for feat in self.feature_names:
            if feat not in encoded.columns:
                encoded[feat] = 0.0
        return encoded[self.feature_names]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict delta per primitive for each row.

        Returns DataFrame: index = df.index, columns = primitive_ids, values = predicted delta.
        """
        X = self.encode_features(df)
        preds = {}
        for pid, tree in self.trees.items():
            preds[pid] = tree.predict(X)
        return pd.DataFrame(preds, index=df.index)

    def route(self, df: pd.DataFrame) -> pd.Series:
        """Pick best primitive per row (or 'base' if all predicted deltas <= 0)."""
        pred_df = self.predict(df)
        best_pid = pred_df.idxmax(axis=1)
        best_val = pred_df.max(axis=1)
        best_pid[best_val <= 0] = "base"
        return best_pid

    def describe(self) -> Dict[str, str]:
        """Human-readable tree rules per primitive."""
        from sklearn.tree import export_text

        rules = {}
        for pid, tree in self.trees.items():
            rules[pid] = export_text(tree, feature_names=self.feature_names, max_depth=10)
        return rules


def fit_routing_model(
    delta_df: pd.DataFrame,
    predicate_cols: List[str],
    *,
    max_depth: int = 3,
    min_samples_leaf: int = 50,
) -> RoutingModel:
    """Fit a per-primitive decision tree on per-query deltas.

    Args:
        delta_df: Output of build_delta_table (one row per query × primitive).
        predicate_cols: Predicate columns to use as features.
        max_depth: Tree depth. 3 captures operation_type → is_count hierarchy.
        min_samples_leaf: Prevents overfitting on small subgroups.

    Returns:
        RoutingModel with fitted trees and feature encoding.
    """
    from sklearn.tree import DecisionTreeRegressor

    cat_cols, num_cols = detect_predicate_types(delta_df, predicate_cols)

    # Build feature matrix once (shared across primitives)
    parts = []
    for col in cat_cols:
        if col in delta_df.columns:
            parts.append(pd.get_dummies(delta_df[col], prefix=col, dtype=float))
    for col in num_cols:
        if col in delta_df.columns:
            parts.append(pd.to_numeric(delta_df[col], errors="coerce").rename(col))

    if not parts:
        raise ValueError("No valid predicate columns for feature encoding")

    X_all = pd.concat(parts, axis=1)
    feature_names = list(X_all.columns)

    # Fit one tree per primitive
    trees = {}
    for pid, prim_df in delta_df.groupby("primitive_id"):
        X = X_all.loc[prim_df.index]
        y = prim_df["delta"].values
        # Drop rows with NaN features
        valid = X.notna().all(axis=1)
        X_v, y_v = X[valid], y[valid.values]
        if len(X_v) < min_samples_leaf:
            continue
        tree = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
        )
        tree.fit(X_v, y_v)
        trees[pid] = tree

    logger.info("fit_routing_model: %d trees, %d features, max_depth=%d",
                len(trees), len(feature_names), max_depth)

    return RoutingModel(
        trees=trees,
        feature_names=feature_names,
        cat_columns=cat_cols,
        num_columns=num_cols,
    )


# ── H. evaluate_routing ──────────────────────────────────────────────


def evaluate_routing(
    delta_df: pd.DataFrame,
    model: RoutingModel,
    predicate_cols: List[str],
) -> Dict[str, Any]:
    """Evaluate routing model on measured data (offline).

    Metrics:
        routing_gain: mean score improvement from routed config over baseline.
        oracle_gain:  mean score improvement from perfect per-query routing.
        gain_ratio:   routing_gain / oracle_gain.
        base_acc:     mean baseline score.
    """
    # Pivot delta_df to wide: one row per query, one column per primitive
    pivot = delta_df.pivot_table(
        index="query_id", columns="primitive_id", values="delta", aggfunc="first"
    )
    primitives = [p for p in pivot.columns if p in model.trees]
    if not primitives:
        return {"error": "no primitives in model match delta_df"}

    pivot_prims = pivot[primitives].copy()

    # Get predicate features per query (deduplicated)
    query_features = delta_df.drop_duplicates(subset="query_id").set_index("query_id")

    # Model predictions
    pred_deltas = model.predict(query_features.loc[pivot.index])
    pred_deltas = pred_deltas[[p for p in primitives if p in pred_deltas.columns]]

    # Routing decision: best primitive or base
    routed_pid = pred_deltas.idxmax(axis=1)
    routed_val = pred_deltas.max(axis=1)
    routed_pid[routed_val <= 0] = "base"

    # Actual delta of routed choice
    routed_delta = pd.Series(0.0, index=pivot.index)
    for qid in pivot.index:
        pid = routed_pid[qid]
        if pid != "base" and pid in pivot_prims.columns:
            routed_delta[qid] = pivot_prims.loc[qid, pid]

    # Oracle: best actual delta per query (0 if all negative)
    oracle_delta = pivot_prims.max(axis=1).clip(lower=0)

    # Base accuracy
    base_scores = delta_df.drop_duplicates(subset="query_id").set_index("query_id")["baseline_score"]
    base_acc = base_scores.loc[pivot.index].mean()

    routing_gain = routed_delta.mean()
    oracle_gain = oracle_delta.mean()
    gain_ratio = routing_gain / oracle_gain if oracle_gain > 0 else 0.0

    # Per-primitive selection stats
    selection_counts = routed_pid.value_counts().to_dict()
    prim_gains = {}
    for pid in primitives:
        mask = routed_pid == pid
        if mask.sum() > 0:
            prim_gains[pid] = {
                "n_selected": int(mask.sum()),
                "mean_delta_when_selected": float(routed_delta[mask].mean()),
            }

    return {
        "base_acc": float(base_acc),
        "routing_gain": float(routing_gain),
        "routing_acc": float(base_acc + routing_gain),
        "oracle_gain": float(oracle_gain),
        "oracle_acc": float(base_acc + oracle_gain),
        "gain_ratio": float(gain_ratio),
        "n_queries": len(pivot),
        "selection_counts": selection_counts,
        "per_primitive": prim_gains,
    }


# ── I. cross_validate_routing ─────────────────────────────────────────


def cross_validate_routing(
    delta_df: pd.DataFrame,
    predicate_cols: List[str],
    *,
    n_folds: int = 5,
    max_depth: int = 3,
    min_samples_leaf: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """K-fold cross-validation of routing model within a single dataset.

    Splits by query_id (not by row) so train/test have disjoint queries.
    Reports mean ± std of routing_gain, plus baselines.
    """
    import numpy as np

    query_ids = sorted(delta_df.query_id.unique())
    rng = random.Random(seed)
    rng.shuffle(query_ids)

    fold_size = len(query_ids) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(query_ids)
        folds.append(set(query_ids[start:end]))

    fold_results = []
    for i in range(n_folds):
        test_qids = folds[i]
        train_qids = set()
        for j in range(n_folds):
            if j != i:
                train_qids |= folds[j]

        train_df = delta_df[delta_df.query_id.isin(train_qids)].copy()
        test_df = delta_df[delta_df.query_id.isin(test_qids)].copy()

        model = fit_routing_model(
            train_df, predicate_cols,
            max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        )
        result = evaluate_routing(test_df, model, predicate_cols)
        if "error" in result:
            logger.warning("Fold %d: %s", i, result["error"])
            continue
        result["fold"] = i
        fold_results.append(result)

    # Aggregate
    gains = [r["routing_gain"] for r in fold_results]
    oracle_gains = [r["oracle_gain"] for r in fold_results]
    ratios = [r["gain_ratio"] for r in fold_results]

    # Best-global baseline: always pick the single best primitive overall
    global_means = delta_df.groupby("primitive_id")["delta"].mean()
    best_global_pid = global_means.idxmax()
    best_global_gain = float(global_means.max()) if global_means.max() > 0 else 0.0

    return {
        "n_folds": n_folds,
        "routing_gain_mean": float(np.mean(gains)),
        "routing_gain_std": float(np.std(gains)),
        "oracle_gain_mean": float(np.mean(oracle_gains)),
        "gain_ratio_mean": float(np.mean(ratios)),
        "best_global_gain": best_global_gain,
        "best_global_primitive": best_global_pid,
        "fold_results": fold_results,
    }


# ── J. transfer_routing ──────────────────────────────────────────────


def transfer_routing(
    source_delta_df: pd.DataFrame,
    target_delta_df: pd.DataFrame,
    predicate_cols: List[str],
    *,
    max_depth: int = 3,
    min_samples_leaf: int = 50,
) -> Dict[str, Any]:
    """Train routing model on source task, evaluate on target (zero-shot transfer).

    Also fits a target-native model for comparison (upper bound).

    Returns:
        source_on_target: eval metrics of source model applied to target
        target_on_target: eval metrics of target's own model (k-fold)
        tree_comparison:  which splits are shared vs divergent
    """
    # Fit source model
    source_model = fit_routing_model(
        source_delta_df, predicate_cols,
        max_depth=max_depth, min_samples_leaf=min_samples_leaf,
    )

    # Evaluate source model on target data
    source_on_target = evaluate_routing(target_delta_df, source_model, predicate_cols)

    # Fit target-native model (for upper bound comparison)
    target_model = fit_routing_model(
        target_delta_df, predicate_cols,
        max_depth=max_depth, min_samples_leaf=min_samples_leaf,
    )
    target_on_target = evaluate_routing(target_delta_df, target_model, predicate_cols)

    # Compare tree structures: extract top splits
    source_rules = source_model.describe()
    target_rules = target_model.describe()

    # Find shared primitives
    shared_prims = set(source_model.trees.keys()) & set(target_model.trees.keys())
    tree_comparison = {}
    for pid in shared_prims:
        s_tree = source_model.trees[pid]
        t_tree = target_model.trees[pid]
        # Extract top split feature
        s_top = (source_model.feature_names[s_tree.tree_.feature[0]]
                 if s_tree.tree_.feature[0] >= 0 else "leaf")
        t_top = (target_model.feature_names[t_tree.tree_.feature[0]]
                 if t_tree.tree_.feature[0] >= 0 else "leaf")
        tree_comparison[pid] = {
            "source_top_split": s_top,
            "target_top_split": t_top,
            "same_top_split": s_top == t_top,
        }

    return {
        "source_on_target": source_on_target,
        "target_on_target": target_on_target,
        "tree_comparison": tree_comparison,
        "source_rules": source_rules,
        "target_rules": target_rules,
    }
