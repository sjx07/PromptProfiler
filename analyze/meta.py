"""Layer 1 — describe-what-exists.

Pure listings over a CubeStore: configs, models, scorers, phases, datasets,
features. No filters, no aggregations, no rankings. These are the "what do
I have in the cube?" primitives.

Every function takes a ``CubeStore`` instance as the first argument and
returns plain Python data structures (lists of dicts). Compose with
``analyze.query`` for filtered views and with ``analyze.compare`` for
cross-cutting summaries.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from prompt_profiler.core.store import CubeStore


# ── configs ───────────────────────────────────────────────────────────

def list_configs(store: CubeStore) -> List[Dict[str, Any]]:
    """Return every config row with meta parsed.

    Columns: config_id, func_ids (list), n_funcs, kind, canonical_ids,
    feature_ids, meta (raw dict).

    ``kind`` / ``canonical_ids`` / ``feature_ids`` are sourced from
    config.meta. For pre-round-1 (rule-level) configs those fields may be
    missing; they default to [].
    """
    rows = store._get_conn().execute(
        "SELECT config_id, func_ids, meta FROM config ORDER BY config_id"
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        func_ids = json.loads(r["func_ids"]) if r["func_ids"] else []
        meta = json.loads(r["meta"]) if r["meta"] else {}
        out.append({
            "config_id":     r["config_id"],
            "func_ids":      func_ids,
            "n_funcs":       len(func_ids),
            "kind":          meta.get("kind"),
            "canonical_ids": meta.get("canonical_ids", []),
            "feature_ids":   meta.get("feature_ids", []),
            "meta":          meta,
        })
    return out


def list_configs_with_features(store: CubeStore) -> List[Dict[str, Any]]:
    """Like ``list_configs`` but resolves feature_id hashes to canonical_ids
    via the feature table, and attaches a concise feature-type summary.

    Useful for one-glance audits of "what do configs in this cube stand
    for?" when config.meta only has hash feature_ids.
    """
    feat_rows = store._get_conn().execute(
        "SELECT feature_id, canonical_id, task FROM feature"
    ).fetchall()
    fid_to_cid = {r["feature_id"]: r["canonical_id"] for r in feat_rows}
    fid_to_task = {r["feature_id"]: r["task"] for r in feat_rows}

    out: List[Dict[str, Any]] = []
    for cfg in list_configs(store):
        resolved_cids = [fid_to_cid.get(fid, f"<unknown:{fid}>")
                         for fid in cfg["feature_ids"]]
        tasks = sorted({fid_to_task[fid] for fid in cfg["feature_ids"]
                        if fid in fid_to_task})
        cfg["resolved_canonical_ids"] = resolved_cids
        cfg["tasks"] = tasks
        out.append(cfg)
    return out


def list_configs_with_func_types(store: CubeStore) -> List[Dict[str, Any]]:
    """Count func_types per config (a lightweight structural summary).

    Returns rows like:
        {"config_id": 3, "n_funcs": 7,
         "by_type": {"insert_node": 6, "set_format": 1}}
    """
    conn = store._get_conn()
    func_rows = conn.execute("SELECT func_id, func_type FROM func").fetchall()
    fid_to_type = {r["func_id"]: r["func_type"] for r in func_rows}

    cfg_rows = conn.execute(
        "SELECT config_id, func_ids FROM config ORDER BY config_id"
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in cfg_rows:
        func_ids = json.loads(r["func_ids"]) if r["func_ids"] else []
        by_type = Counter(fid_to_type.get(fid, "<unknown>") for fid in func_ids)
        out.append({
            "config_id": r["config_id"],
            "n_funcs":   len(func_ids),
            "by_type":   dict(by_type),
        })
    return out


# ── models / scorers / phases ─────────────────────────────────────────

def list_models(store: CubeStore) -> List[Dict[str, Any]]:
    """Distinct models present in the execution table, each with an
    execution count and an earliest/latest created_at.
    """
    rows = store._get_conn().execute(
        """SELECT model,
                  COUNT(*) AS n_executions,
                  MIN(created_at) AS first_seen,
                  MAX(created_at) AS last_seen
           FROM execution
           GROUP BY model
           ORDER BY n_executions DESC"""
    ).fetchall()
    return [dict(r) for r in rows]


def list_scorers(store: CubeStore) -> List[Dict[str, Any]]:
    """Distinct scorers present in the evaluation table, with evaluation counts."""
    rows = store._get_conn().execute(
        """SELECT scorer,
                  COUNT(*) AS n_evaluations,
                  AVG(score) AS mean_score
           FROM evaluation
           GROUP BY scorer
           ORDER BY n_evaluations DESC"""
    ).fetchall()
    return [dict(r) for r in rows]


def list_phases(store: CubeStore) -> List[Dict[str, Any]]:
    """Unique phase tags across all executions, with tagged-execution counts.

    Phases are stored as JSON arrays in execution.phase_ids; this helper
    unpacks them.
    """
    rows = store._get_conn().execute(
        """SELECT phase_ids, model, created_at
           FROM execution
           WHERE phase_ids != '[]'"""
    ).fetchall()
    counter: Counter = Counter()
    first: Dict[str, str] = {}
    last: Dict[str, str] = {}
    models: Dict[str, set] = defaultdict(set)
    for r in rows:
        try:
            phases = json.loads(r["phase_ids"])
        except json.JSONDecodeError:
            continue
        for ph in phases:
            counter[ph] += 1
            ts = r["created_at"]
            if ph not in first or ts < first[ph]:
                first[ph] = ts
            if ph not in last or ts > last[ph]:
                last[ph] = ts
            models[ph].add(r["model"])

    return [
        {
            "phase":         ph,
            "n_executions": n,
            "first_seen":   first[ph],
            "last_seen":    last[ph],
            "models":       sorted(models[ph]),
        }
        for ph, n in counter.most_common()
    ]


# ── queries / datasets / predicates ───────────────────────────────────

def list_datasets(store: CubeStore) -> List[Dict[str, Any]]:
    """Distinct datasets with per-split query counts."""
    rows = store._get_conn().execute(
        """SELECT dataset,
                  COALESCE(json_extract(meta, '$.split'), '(no split)') AS split,
                  COUNT(*) AS n_queries
           FROM query
           GROUP BY dataset, split
           ORDER BY dataset, split"""
    ).fetchall()
    # Regroup by dataset for readability.
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        ds = r["dataset"]
        if ds not in out:
            out[ds] = {"dataset": ds, "splits": {}, "n_queries": 0}
        out[ds]["splits"][r["split"]] = r["n_queries"]
        out[ds]["n_queries"] += r["n_queries"]
    return list(out.values())


def list_predicates(store: CubeStore) -> List[Dict[str, Any]]:
    """Predicate names with distinct-value counts and coverage."""
    rows = store._get_conn().execute(
        """SELECT name,
                  COUNT(DISTINCT value) AS n_distinct_values,
                  COUNT(DISTINCT query_id) AS n_queries_tagged
           FROM predicate
           GROUP BY name
           ORDER BY n_queries_tagged DESC"""
    ).fetchall()
    return [dict(r) for r in rows]


# ── features (from the cube's feature table) ──────────────────────────

def list_features_in_cube(
    store: CubeStore,
    *,
    task: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Features synced into the cube's ``feature`` table.

    Args:
        task: Optional filter. Matches ``feature.task`` column.
    """
    conn = store._get_conn()
    if task is None:
        rows = conn.execute(
            "SELECT * FROM feature ORDER BY task, canonical_id"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM feature WHERE task = ? ORDER BY canonical_id",
            (task,),
        ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({
            "feature_id":     r["feature_id"],
            "canonical_id":   r["canonical_id"],
            "task":           r["task"],
            "requires":       json.loads(r["requires_json"] or "[]"),
            "conflicts_with": json.loads(r["conflicts_json"] or "[]"),
            "rationale":      r["rationale"],
            "source_path":    r["source_path"],
            "synced_at":      r["synced_at"],
        })
    return out


# ── cube-wide summary ─────────────────────────────────────────────────

def summary(store: CubeStore) -> Dict[str, Any]:
    """One-line-per-axis overview. Useful as a first sanity check on a cube.

    Returns:
        {
          "counts":    {func, query, config, execution, evaluation, feature},
          "models":    [{model, n_executions}, ...],
          "scorers":   [{scorer, n_evaluations, mean_score}, ...],
          "phases":    [{phase, n_executions}, ...],
          "datasets":  [{dataset, n_queries, splits}, ...],
          "tasks":     [<canonical tasks present in feature table>],
        }
    """
    counts = store.stats()
    tasks = sorted({
        r["task"] for r in store._get_conn().execute(
            "SELECT DISTINCT task FROM feature"
        ).fetchall()
    })
    return {
        "counts":   counts,
        "models":   list_models(store),
        "scorers":  list_scorers(store),
        "phases":   list_phases(store),
        "datasets": list_datasets(store),
        "tasks":    tasks,
    }
