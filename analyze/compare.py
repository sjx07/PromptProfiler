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


# ══════════════════════════════════════════════════════════════════════
# Feature × Predicate analysis (round 5)
# ══════════════════════════════════════════════════════════════════════
#
# DESIGN CHOICES (per round 5 prompt):
#
#   * method="simple" — identify the "feature's config" by func_ids delta,
#     NOT by config.meta lookup. Tolerant to cubes that don't carry
#     add_one_feature meta. A feature f matches config c when
#     (set(c.func_ids) − base_ids) equals the set of func_ids produced by
#     materializing f.primitive_spec alone.
#
#   * method="marginal" — per-config pooling: for each config we compute
#     the mean score on each predicate slice first, then average across
#     the set of configs containing (or not containing) the feature.
#     This is robust to imbalanced per-config run sizes.
#
#   * metric="did" — reference value defaults to the alphabetically-first
#     predicate value when the predicate is binary. For multi-value
#     predicates, reference_values[predicate_name] must be supplied; if
#     omitted, DiD is undefined and the ``did`` column is NaN.
#
#   * Scope of predicates: list_predicates(store) used when
#     predicate_names=None; the user can restrict to a subset.
#
#   * Unobserved features: features declared in the `feature` table but
#     never matching any config in the cube appear in the output with NaN
#     metrics.
#
#   * Output shape: long-format DataFrame, one row per
#     (canonical_id, predicate_name, predicate_value). Caller pivots for
#     wide-format as needed.

def _canonicalize_edit(edit: dict) -> str:
    """Same canonicalization as feature_registry._canonical_edit."""
    from core.feature_registry import _canonical_edit
    return _canonical_edit(edit)


def _feature_add_funcs_from_spec(store: CubeStore,
                                 base_fids: set) -> Dict[str, set]:
    """For every feature in the cube, compute the set of func_ids it would
    contribute AFTER de-duping against base_fids.

    Uses content-addressed make_func_id to translate primitive_edits →
    func_ids, identical to what the runner does at materialize time.
    """
    import json as _json
    from core.func_registry import make_func_id

    out: Dict[str, set] = {}
    rows = store._get_conn().execute(
        "SELECT canonical_id, primitive_spec FROM feature"
    ).fetchall()
    for r in rows:
        try:
            edits = _json.loads(r["primitive_spec"])
        except _json.JSONDecodeError:
            continue
        fids = {make_func_id(e["func_type"], e.get("params", {})) for e in edits}
        add_funcs = fids - base_fids
        out[r["canonical_id"]] = add_funcs
    return out


def _simple_effect_configs(store: CubeStore,
                           base_config_id: int) -> Dict[str, int]:
    """canonical_id → config_id map for configs that match "base + exactly
    one feature's primitives."

    Identified by func_ids delta (choice 1b): find a config c whose
    set(func_ids) − base equals the feature's own add_funcs.
    """
    import json as _json

    conn = store._get_conn()
    base_row = conn.execute(
        "SELECT func_ids FROM config WHERE config_id = ?", (base_config_id,),
    ).fetchone()
    if base_row is None:
        raise ValueError(f"base_config_id {base_config_id} not found")
    base_fids = set(_json.loads(base_row["func_ids"]))

    feat_add = _feature_add_funcs_from_spec(store, base_fids)
    # Drop features whose add_funcs is empty (pure section features etc.)
    feat_add = {cid: f for cid, f in feat_add.items() if f}

    out: Dict[str, int] = {}
    rows = conn.execute(
        "SELECT config_id, func_ids FROM config WHERE config_id != ?",
        (base_config_id,),
    ).fetchall()
    for r in rows:
        cid = r["config_id"]
        c_fids = set(_json.loads(r["func_ids"]))
        delta = c_fids - base_fids
        for canonical, expected in feat_add.items():
            if delta == expected:
                out[canonical] = cid
                break
    return out


def _configs_containing_feature(store: CubeStore) -> Dict[str, set]:
    """canonical_id → set of config_ids whose func_ids is a superset of
    the feature's add_funcs. Used for the marginal method.
    """
    import json as _json

    conn = store._get_conn()
    # Build full func-set per config.
    cfg_fids: Dict[int, set] = {}
    for r in conn.execute("SELECT config_id, func_ids FROM config").fetchall():
        cfg_fids[r["config_id"]] = set(_json.loads(r["func_ids"]))

    # For each feature, the primitives it produces.
    from core.func_registry import make_func_id
    feat_fids: Dict[str, set] = {}
    for r in conn.execute(
        "SELECT canonical_id, primitive_spec FROM feature"
    ).fetchall():
        try:
            edits = _json.loads(r["primitive_spec"])
        except _json.JSONDecodeError:
            continue
        feat_fids[r["canonical_id"]] = {
            make_func_id(e["func_type"], e.get("params", {})) for e in edits
        }

    # containment map
    out: Dict[str, set] = {}
    for canonical, fids in feat_fids.items():
        if not fids:
            out[canonical] = set()
            continue
        out[canonical] = {cid for cid, cfids in cfg_fids.items() if fids.issubset(cfids)}
    return out


def _per_config_predicate_means(
    store: CubeStore, *, model: str, scorer: str, predicate_name: str,
) -> "pd.DataFrame":
    """(config_id, predicate_value) → mean score. Used by both methods."""
    import pandas as pd
    sql = """
        SELECT e.config_id      AS config_id,
               p.value           AS predicate_value,
               AVG(ev.score)     AS mean_score,
               COUNT(*)          AS n
        FROM execution e
        JOIN evaluation ev ON ev.execution_id = e.execution_id
        JOIN predicate  p  ON p.query_id = e.query_id
        WHERE e.model = ? AND ev.scorer = ? AND p.name = ?
          AND (e.error IS NULL OR e.error = '')
        GROUP BY e.config_id, p.value
    """
    return pd.read_sql_query(sql, store._get_conn(),
                             params=(model, scorer, predicate_name))


def _default_reference_value(predicate_values: List[str]) -> Optional[str]:
    """For binary predicates, return the alphabetically-first value.
    For multi-value, return None (caller must supply).
    """
    uniq = sorted(set(predicate_values))
    if len(uniq) == 2:
        return uniq[0]
    return None


def feature_predicate_table(
    store: CubeStore,
    *,
    model: str,
    scorer: str,
    method: str = "simple",                    # "simple" | "marginal"
    metric: str = "lift",                      # "lift" | "did"
    predicate_names: Optional[List[str]] = None,
    base_config_id: Optional[int] = None,      # required for method="simple"
    reference_values: Optional[Dict[str, str]] = None,
    task: Optional[str] = None,                # optional feature filter
):
    """Feature × Predicate analysis — long-format DataFrame.

    For every (feature, predicate, predicate_value) triple in scope,
    reports the mean score + chosen effect metric (lift or DiD).

    Args:
        method:
            "simple"   — effect = score(feature-only config, slice)
                         − score(base config, slice).
                         Identifies the feature's config via func_ids delta
                         against ``base_config_id`` (required).
            "marginal" — effect = mean over configs-containing-feature
                         − mean over configs-not-containing-feature,
                         per predicate slice, per-config pooling.
        metric:
            "lift" — column ``lift`` (score_with − score_without)
            "did"  — column ``did`` = lift_value − lift_reference
                     (for each non-reference predicate_value). Reference
                     is auto-detected for binary predicates; otherwise
                     must be supplied via ``reference_values``.
        predicate_names: restrict to these names. None → all predicates
                         present in the cube.
        base_config_id: required when method="simple".
        reference_values: {predicate_name: reference_value} overrides for
                          non-binary predicates under metric="did".
        task: optional filter on feature.task (only relevant when
              multiple tasks share the cube).

    Returns:
        pandas.DataFrame with columns:
            canonical_id, predicate_name, predicate_value,
            n_with, mean_with, n_without, mean_without, lift,
            (and ``did`` when metric="did")
        Features present in the `feature` table but never observed in any
        matching config appear as rows with NaN metrics.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas required for feature_predicate_table") from e

    if method not in ("simple", "marginal"):
        raise ValueError(f"method must be 'simple' | 'marginal'; got {method!r}")
    if metric not in ("lift", "did"):
        raise ValueError(f"metric must be 'lift' | 'did'; got {metric!r}")
    if method == "simple" and base_config_id is None:
        raise ValueError("method='simple' requires base_config_id")

    reference_values = reference_values or {}

    # ── feature universe ─────────────────────────────────────────────
    conn = store._get_conn()
    feat_sql = "SELECT canonical_id, feature_id, task FROM feature"
    params: List[Any] = []
    if task is not None:
        feat_sql += " WHERE task = ?"
        params.append(task)
    feat_sql += " ORDER BY canonical_id"
    feature_rows = conn.execute(feat_sql, tuple(params)).fetchall()
    all_features = [r["canonical_id"] for r in feature_rows]
    if not all_features:
        return pd.DataFrame(columns=[
            "canonical_id", "predicate_name", "predicate_value",
            "n_with", "mean_with", "n_without", "mean_without", "lift",
        ])

    # ── predicate universe ──────────────────────────────────────────
    if predicate_names is None:
        rows = conn.execute(
            "SELECT DISTINCT name FROM predicate ORDER BY name"
        ).fetchall()
        predicate_names = [r["name"] for r in rows]
    if not predicate_names:
        return pd.DataFrame(columns=[
            "canonical_id", "predicate_name", "predicate_value",
            "n_with", "mean_with", "n_without", "mean_without", "lift",
        ])

    # ── precompute per-config × predicate_value means ───────────────
    records: List[Dict[str, Any]] = []
    all_configs: List[int] = [
        r["config_id"] for r in conn.execute("SELECT config_id FROM config").fetchall()
    ]

    if method == "simple":
        simple_map = _simple_effect_configs(store, base_config_id)
    else:
        contain_map = _configs_containing_feature(store)

    for pname in predicate_names:
        pc = _per_config_predicate_means(
            store, model=model, scorer=scorer, predicate_name=pname,
        )
        # pc columns: config_id, predicate_value, mean_score, n
        predicate_values = sorted(pc["predicate_value"].unique().tolist()) \
            if not pc.empty else []
        # Resolve reference value for DiD.
        ref_value = reference_values.get(pname) or _default_reference_value(predicate_values)

        # Pivot (config_id, predicate_value) → score.
        pivot_score = pc.pivot(index="config_id", columns="predicate_value",
                               values="mean_score") if not pc.empty else \
                      pd.DataFrame(index=pd.Index([], name="config_id"))
        pivot_n = pc.pivot(index="config_id", columns="predicate_value",
                           values="n") if not pc.empty else \
                  pd.DataFrame(index=pd.Index([], name="config_id"))

        for canonical in all_features:
            for pv in predicate_values:
                if method == "simple":
                    target_cfg = simple_map.get(canonical)
                    s_with = (pivot_score.loc[target_cfg, pv]
                              if target_cfg is not None and target_cfg in pivot_score.index
                                 and pv in pivot_score.columns
                              else float("nan"))
                    n_with = (int(pivot_n.loc[target_cfg, pv])
                              if target_cfg is not None and target_cfg in pivot_n.index
                                 and pv in pivot_n.columns and pd.notna(pivot_n.loc[target_cfg, pv])
                              else 0)
                    s_without = (pivot_score.loc[base_config_id, pv]
                                 if base_config_id in pivot_score.index
                                    and pv in pivot_score.columns
                                 else float("nan"))
                    n_without = (int(pivot_n.loc[base_config_id, pv])
                                 if base_config_id in pivot_n.index
                                    and pv in pivot_n.columns
                                    and pd.notna(pivot_n.loc[base_config_id, pv])
                                 else 0)
                else:  # marginal
                    with_cfgs    = contain_map.get(canonical, set())
                    without_cfgs = set(all_configs) - with_cfgs
                    with_vals = pivot_score.reindex(list(with_cfgs)).get(pv, pd.Series(dtype=float)) \
                        if pv in pivot_score.columns else pd.Series(dtype=float)
                    without_vals = pivot_score.reindex(list(without_cfgs)).get(pv, pd.Series(dtype=float)) \
                        if pv in pivot_score.columns else pd.Series(dtype=float)
                    with_vals = with_vals.dropna()
                    without_vals = without_vals.dropna()
                    s_with = with_vals.mean() if len(with_vals) else float("nan")
                    s_without = without_vals.mean() if len(without_vals) else float("nan")
                    n_with = len(with_vals)
                    n_without = len(without_vals)

                lift = (s_with - s_without) if pd.notna(s_with) and pd.notna(s_without) \
                    else float("nan")
                records.append({
                    "canonical_id":     canonical,
                    "predicate_name":   pname,
                    "predicate_value":  pv,
                    "n_with":           n_with,
                    "mean_with":        s_with,
                    "n_without":        n_without,
                    "mean_without":     s_without,
                    "lift":             lift,
                    "_ref":             ref_value,
                })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # ── attach DiD if requested ─────────────────────────────────────
    if metric == "did":
        df["did"] = float("nan")
        # Compute per (canonical_id, predicate_name) — lift at ref vs at value.
        for (canon, pname), group in df.groupby(["canonical_id", "predicate_name"]):
            ref = group["_ref"].iloc[0]
            if ref is None:
                continue  # multi-value w/o explicit reference → leave NaN
            ref_lift_series = group.loc[group["predicate_value"] == ref, "lift"]
            if ref_lift_series.empty or pd.isna(ref_lift_series.iloc[0]):
                continue
            ref_lift = ref_lift_series.iloc[0]
            for idx in group.index:
                pv = df.at[idx, "predicate_value"]
                if pv == ref:
                    df.at[idx, "did"] = 0.0
                else:
                    lv = df.at[idx, "lift"]
                    if pd.notna(lv):
                        df.at[idx, "did"] = lv - ref_lift

    df = df.drop(columns=["_ref"])
    return df.sort_values(
        ["canonical_id", "predicate_name", "predicate_value"]
    ).reset_index(drop=True)
