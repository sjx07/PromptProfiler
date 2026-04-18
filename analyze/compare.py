"""Layer 3 — compose.

Operations that span multiple configs / features / predicates. Built on
top of ``analyze.query.ExecutionQuery`` and ``analyze.meta``.

Every function is pure over the cube — no writes.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from core.store import CubeStore
from analyze.query import ExecutionQuery


# ── config A vs config B on shared queries ────────────────────────────

def score_diff(
    store: CubeStore,
    *,
    config_a: int,
    config_b: int,
    model: str,
    scorer: str,
) -> Dict[str, Any]:
    """Compare two configs on queries evaluated under BOTH.

    Returns:
        {
          "config_a": int, "config_b": int,
          "n_a":      int,  # total evaluations for config_a
          "n_b":      int,
          "n_shared": int,  # queries evaluated under BOTH
          "avg_a":    float, "avg_b": float,
          "avg_delta":       float,  # avg_b - avg_a over shared queries
          "flipped_up":      [query_id, ...],  # 0→1 from a to b
          "flipped_down":    [query_id, ...],  # 1→0 from a to b
          "agree":           int,             # same score on shared queries
        }
    """
    a_rows = {r["query_id"]: r["score"]
              for r in ExecutionQuery(store).config(config_a).model(model).scorer(scorer).rows()
              if r.get("score") is not None}
    b_rows = {r["query_id"]: r["score"]
              for r in ExecutionQuery(store).config(config_b).model(model).scorer(scorer).rows()
              if r.get("score") is not None}
    shared = set(a_rows) & set(b_rows)

    flipped_up: List[str] = []
    flipped_down: List[str] = []
    agree = 0
    sum_a = 0.0
    sum_b = 0.0
    for qid in shared:
        a, b = a_rows[qid], b_rows[qid]
        sum_a += a
        sum_b += b
        if a == b:
            agree += 1
        elif b > a:
            flipped_up.append(qid)
        else:
            flipped_down.append(qid)

    n = len(shared) or 1
    return {
        "config_a":    config_a,
        "config_b":    config_b,
        "n_a":         len(a_rows),
        "n_b":         len(b_rows),
        "n_shared":    len(shared),
        "avg_a":       sum_a / n,
        "avg_b":       sum_b / n,
        "avg_delta":   (sum_b - sum_a) / n,
        "flipped_up":   sorted(flipped_up),
        "flipped_down": sorted(flipped_down),
        "agree":       agree,
    }


# ── feature add-one effect ranking ────────────────────────────────────

def feature_effect_ranking(
    store: CubeStore,
    *,
    model: str,
    scorer: str,
    task: Optional[str] = None,
):
    """Rank features by their average score in configs that activate them.

    Uses the ``feature_effect`` SQL view (join of feature ↔ config.meta.feature_ids
    ↔ evaluation). Returns a pandas DataFrame sorted by mean score descending.

    Columns: canonical_id, feature_id, task, n, mean_score.

    Args:
        task: Optional task filter (e.g. ``"table_qa"``).

    Note: this ranking is feature-marginal (averages across ALL configs
    that contain each feature), NOT a controlled add-one delta. For a
    proper add-one comparison, use ``score_diff(base, add_one_config)``.
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas required for feature_effect_ranking") from e

    sql = """
        SELECT canonical_id, feature_id, task,
               COUNT(*) AS n,
               AVG(score) AS mean_score
        FROM feature_effect
        WHERE model = ? AND scorer = ?
    """
    params: List[Any] = [model, scorer]
    if task is not None:
        sql += " AND task = ?"
        params.append(task)
    sql += " GROUP BY canonical_id, feature_id, task ORDER BY mean_score DESC"

    return pd.read_sql_query(sql, store._get_conn(), params=tuple(params))


# ── predicate-sliced scores ───────────────────────────────────────────

def predicate_slice(
    store: CubeStore,
    *,
    model: str,
    scorer: str,
    predicate_name: str,
    config_ids: Optional[List[int]] = None,
):
    """Mean score per (predicate_value × config_id).

    Useful to ask: "does enable_cot help more on aggregation queries
    than on lookup queries?"

    Args:
        predicate_name: Name of the predicate (e.g. ``"has_aggregation"``).
        config_ids: Optional filter to restrict to specific configs.

    Returns:
        pandas DataFrame with columns:
            predicate_value, config_id, n, mean_score
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas required for predicate_slice") from e

    sql = """
        SELECT p.value AS predicate_value,
               e.config_id,
               COUNT(*) AS n,
               AVG(ev.score) AS mean_score
        FROM execution e
        JOIN evaluation ev ON ev.execution_id = e.execution_id
        JOIN predicate  p  ON p.query_id = e.query_id
        WHERE e.model = ? AND ev.scorer = ? AND p.name = ?
    """
    params: List[Any] = [model, scorer, predicate_name]
    if config_ids:
        placeholders = ",".join("?" * len(config_ids))
        sql += f" AND e.config_id IN ({placeholders})"
        params.extend(config_ids)
    sql += " GROUP BY p.value, e.config_id ORDER BY p.value, e.config_id"

    return pd.read_sql_query(sql, store._get_conn(), params=tuple(params))


# ── add-one deltas vs a base config ───────────────────────────────────

def add_one_deltas(
    store: CubeStore,
    *,
    base_config_id: int,
    model: str,
    scorer: str,
    candidate_config_ids: Optional[List[int]] = None,
):
    """For every candidate config, compute the score_diff vs base on shared queries.

    Args:
        candidate_config_ids: Optional restriction. Defaults to "every
            config that isn't ``base_config_id``".

    Returns:
        pandas DataFrame sorted by avg_delta descending. Columns:
            config_id, canonical_id (if resolvable from config.meta), n_shared,
            avg_a, avg_b, avg_delta, flipped_up, flipped_down.
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas required for add_one_deltas") from e

    conn = store._get_conn()
    if candidate_config_ids is None:
        rows = conn.execute(
            "SELECT config_id FROM config WHERE config_id != ?",
            (base_config_id,),
        ).fetchall()
        candidate_config_ids = [r[0] for r in rows]

    # Build canonical-id lookup from config.meta (add_one_feature stores it).
    cfg_rows = conn.execute(
        "SELECT config_id, meta FROM config WHERE config_id IN ({})".format(
            ",".join("?" * len(candidate_config_ids))
        ),
        tuple(candidate_config_ids),
    ).fetchall()
    import json
    cid_to_canonical: Dict[int, Optional[str]] = {}
    for r in cfg_rows:
        try:
            meta = json.loads(r["meta"] or "{}")
        except json.JSONDecodeError:
            meta = {}
        cid_to_canonical[r["config_id"]] = meta.get("canonical_id")

    records: List[Dict[str, Any]] = []
    for cid in candidate_config_ids:
        d = score_diff(
            store,
            config_a=base_config_id, config_b=cid,
            model=model, scorer=scorer,
        )
        records.append({
            "config_id":    cid,
            "canonical_id": cid_to_canonical.get(cid),
            "n_shared":     d["n_shared"],
            "avg_a":        d["avg_a"],
            "avg_b":        d["avg_b"],
            "avg_delta":    d["avg_delta"],
            "flipped_up":   len(d["flipped_up"]),
            "flipped_down": len(d["flipped_down"]),
        })
    return pd.DataFrame(records).sort_values("avg_delta", ascending=False).reset_index(drop=True)


# ── per-query flip inspector ──────────────────────────────────────────

def flip_rows(
    store: CubeStore,
    *,
    base_config: int,
    target_config: int,
    model: str,
    scorer: str,
    direction: str = "both",
) -> List[Dict[str, Any]]:
    """Return rich per-query rows for cases where base and target disagreed.

    Use this to eyeball *why* a feature helped or hurt.

    Args:
        base_config:    Config without the feature (usually the base cid).
        target_config:  Config with the feature added.
        model / scorer: Pin the evaluation.
        direction:
          - ``"down"`` → base was right, target wrong  (feature HURT).
          - ``"up"``   → base was wrong, target right  (feature HELPED).
          - ``"both"`` (default) → all disagreements.

    Returns:
        List of dicts sorted by query_id, each with:
            query_id, question, direction ('up' | 'down'),
            base_score, target_score,
            base_prediction, target_prediction,
            base_raw, target_raw,
            gold (list, from query.meta._raw.answers if present).
    """
    if direction not in ("up", "down", "both"):
        raise ValueError(f"direction must be 'up' | 'down' | 'both'; got {direction!r}")

    def _index(config_id: int) -> Dict[str, Dict[str, Any]]:
        rows = (ExecutionQuery(store)
                .config(config_id).model(model).scorer(scorer)
                .columns(["query_id", "raw_response", "prediction", "score"])
                .rows())
        return {r["query_id"]: r for r in rows if r.get("score") is not None}

    a = _index(base_config)
    b = _index(target_config)
    shared = set(a) & set(b)
    if not shared:
        return []

    # Pull question text + gold answers from the query table for shared queries.
    placeholders = ",".join("?" * len(shared))
    q_rows = store._get_conn().execute(
        f"SELECT query_id, content, meta FROM query WHERE query_id IN ({placeholders})",
        tuple(shared),
    ).fetchall()
    q_map: Dict[str, Dict[str, Any]] = {}
    for qr in q_rows:
        try:
            meta = json.loads(qr["meta"] or "{}")
        except json.JSONDecodeError:
            meta = {}
        gold = (meta.get("_raw") or {}).get("answers") or meta.get("gold_answers") or []
        q_map[qr["query_id"]] = {"question": qr["content"], "gold": gold}

    out: List[Dict[str, Any]] = []
    for qid in sorted(shared):
        sa = a[qid]["score"]
        sb = b[qid]["score"]
        if sa == sb:
            continue
        dir_tag = "up" if sb > sa else "down"
        if direction != "both" and direction != dir_tag:
            continue
        q = q_map.get(qid, {})
        out.append({
            "query_id":          qid,
            "question":          q.get("question", ""),
            "direction":         dir_tag,
            "base_score":        sa,
            "target_score":      sb,
            "base_prediction":   a[qid].get("prediction", ""),
            "target_prediction": b[qid].get("prediction", ""),
            "base_raw":          a[qid].get("raw_response", ""),
            "target_raw":        b[qid].get("raw_response", ""),
            "gold":              q.get("gold", []),
        })
    return out


def harm_cases(store: CubeStore, **kwargs) -> List[Dict[str, Any]]:
    """Convenience alias for ``flip_rows(direction='down')`` —
    'base right, feature wrong' cases.  Kwargs: base_config, target_config,
    model, scorer.
    """
    return flip_rows(store, direction="down", **kwargs)


def help_cases(store: CubeStore, **kwargs) -> List[Dict[str, Any]]:
    """Convenience alias for ``flip_rows(direction='up')`` —
    'base wrong, feature right' cases.  Kwargs: base_config, target_config,
    model, scorer.
    """
    return flip_rows(store, direction="up", **kwargs)
