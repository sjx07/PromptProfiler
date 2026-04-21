"""Layer 6 — thin public pipelines.

Each function here is a composition of lower layers:

  * ``analyze.data``        — raw SQL primitives.
  * ``analyze.resolve``     — feature ↔ config mapping.
  * ``analyze.effect``      — lift / DiD arithmetic.
  * ``analyze.confidence``  — bootstrap CI / P(lift>0).
  * ``analyze.rank``        — sort / filter / top_k.
  * ``analyze.report``      — markdown / text rendering.

Every function is pure over the cube — no writes.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from core.store import CubeStore
from analyze.query import ExecutionQuery
# Alias modules whose names collide with kwargs (``confidence``, ``report``).
from analyze import data, resolve, effect, rank
from analyze import confidence as _confidence
from analyze import report as _report


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

    Uses the ``feature_effect`` SQL view. Returns a pandas DataFrame
    sorted by mean score descending.

    Columns: canonical_id, feature_id, task, n, mean_score.

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

    Thin wrapper around ``data.scores_df`` + a groupby aggregation.
    """
    try:
        import pandas as pd  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas required for predicate_slice") from e

    sdf = data.scores_df(
        store, model=model, scorer=scorer,
        config_ids=config_ids, predicate_name=predicate_name,
    )
    if sdf.empty:
        import pandas as pd
        return pd.DataFrame(columns=["predicate_value", "config_id", "n", "mean_score"])
    out = sdf.groupby(["predicate_value", "config_id"], as_index=False).agg(
        n=("score", "count"),
        mean_score=("score", "mean"),
    )
    return out.sort_values(["predicate_value", "config_id"]).reset_index(drop=True)


# ── add-one deltas vs a base config ───────────────────────────────────

def add_one_deltas(
    store: CubeStore,
    *,
    base_config_id: int,
    model: str,
    scorer: str,
    candidate_config_ids: Optional[List[int]] = None,
    exclude_config_ids: Optional[List[int]] = None,
    kind_filter: Optional[Any] = None,   # str | list[str] | None
):
    """For every candidate config, compute the score_diff vs base on shared queries.

    The function name is a misnomer for historical reasons — by default it
    runs against EVERY config != base, including coalitions, leave-one-out
    runs, and alternative bases. Use ``kind_filter`` to restrict to a
    single generator kind (e.g. ``"add_one_feature"``) when you want
    true add-one semantics.

    Args:
        candidate_config_ids: explicit allow-list. Defaults to every
            config in the cube except ``base_config_id``.
        exclude_config_ids: explicit deny-list. Applied AFTER
            ``candidate_config_ids`` and ``kind_filter`` resolve.
            Useful for dropping known-bad runs.
        kind_filter: filter by ``config.meta.kind``. One of:
              * ``None``  — no filter (default; legacy behavior)
              * ``str``   — keep only configs with this kind
                (e.g. ``"add_one_feature"``, ``"coalition_feature"``,
                ``"leave_one_out_feature"``)
              * ``list[str]`` — keep configs whose kind is in the list
            Configs missing ``meta.kind`` are dropped when the filter
            is set.

    Returns a pandas DataFrame sorted by ``avg_delta`` descending.
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

    # Pre-parsed configs_df gives us meta as a dict + the canonical_id
    # lookup in one pass. Restricted to the candidate set so kind/meta
    # reads stay cheap on big cubes.
    cdf = data.configs_df(store, config_ids=candidate_config_ids)
    cid_to_canonical: Dict[int, Optional[str]] = {}
    cid_to_kind:      Dict[int, Optional[str]] = {}
    for _, r in cdf.iterrows():
        cid = int(r["config_id"])
        meta = r["meta"] if isinstance(r["meta"], dict) else {}
        cid_to_canonical[cid] = meta.get("canonical_id")
        cid_to_kind[cid]      = meta.get("kind")

    # ── kind_filter (config.meta.kind) ────────────────────────────────
    if kind_filter is not None:
        if isinstance(kind_filter, str):
            kinds = {kind_filter}
        else:
            kinds = set(kind_filter)
        candidate_config_ids = [
            cid for cid in candidate_config_ids
            if cid_to_kind.get(cid) in kinds
        ]

    # ── exclude_config_ids ────────────────────────────────────────────
    if exclude_config_ids:
        excl = set(int(c) for c in exclude_config_ids)
        candidate_config_ids = [cid for cid in candidate_config_ids if cid not in excl]

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
            "kind":         cid_to_kind.get(cid),
            "n_shared":     d["n_shared"],
            "avg_a":        d["avg_a"],
            "avg_b":        d["avg_b"],
            "avg_delta":    d["avg_delta"],
            "flipped_up":   len(d["flipped_up"]),
            "flipped_down": len(d["flipped_down"]),
        })
    cols = ["config_id", "canonical_id", "kind", "n_shared",
            "avg_a", "avg_b", "avg_delta", "flipped_up", "flipped_down"]
    if not records:
        return pd.DataFrame(columns=cols)
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
    """Convenience alias for ``flip_rows(direction='down')``."""
    return flip_rows(store, direction="down", **kwargs)


def help_cases(store: CubeStore, **kwargs) -> List[Dict[str, Any]]:
    """Convenience alias for ``flip_rows(direction='up')``."""
    return flip_rows(store, direction="up", **kwargs)


# ══════════════════════════════════════════════════════════════════════
# Feature × Predicate analysis — round 5 + round 6 behavior, now
# assembled from lower-layer primitives.
# ══════════════════════════════════════════════════════════════════════

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
    include_unmatched: bool = False,
    # R6 kwargs:
    confidence: bool = False,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
    sort_by: Optional[str] = None,
    top_k: Optional[int] = None,
    confidence_min: Optional[float] = None,
    min_effect: Optional[float] = None,        # magnitude gate on |lift| or |did|
    min_lift_in_pair: Optional[float] = None,  # pair-level: max(lift) over predicate_values ≥ threshold
    report: Optional[str] = None,              # "markdown" | "text" | "both"
    # R6.5 long-run kwargs:
    progress: bool = False,                    # tqdm bars on hot loops
    workers: int = 1,                          # >1 → ProcessPoolExecutor for bootstrap
    # R6.5 numeric-predicate guard:
    skip_numeric: bool = True,                 # drop numeric predicates (categorical-only analysis)
):
    """Feature × Predicate analysis — long-format DataFrame.

    Six-stage pipeline composed from ``analyze.{data,resolve,effect,
    confidence,rank,report}``.

    See ``docs/analysis.md`` (and the round-5 / round-6 response files)
    for the full design. Public behavior is preserved from R6.

    Long-run ergonomics (R6.5):
        progress: set True to draw tqdm bars on the lift loop and the
            bootstrap loop. No-op (silent) if tqdm isn't installed.
        workers: set > 1 to run the bootstrap across a ``ProcessPoolExecutor``.
            Helpful when the effect table has many rows (say ≥ 500)
            and n_bootstrap is also large. For small tables the
            per-task overhead dominates; stay at workers=1.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas required for feature_predicate_table") from e

    # R2 deprecation: prefer ``analyze.Pipeline`` for its stage-level cache
    # reuse (tweaking filter/rank/render does not re-run the bootstrap).
    # Kept as a shim this round; scheduled for removal in R4. Python
    # filters duplicate warnings to once per (category, location), so
    # this does not spam.
    import warnings as _warn
    _warn.warn(
        "feature_predicate_table is deprecated (analysis R2). Use "
        "analyze.Pipeline(store).source(...).scope(...).effect(...)"
        ".confidence(...).filter(...).rank(...).render(...).run() — "
        "the Pipeline caches each stage, so tweaking filter/rank/render "
        "does not re-run the bootstrap. Scheduled for removal in R4.",
        DeprecationWarning, stacklevel=2,
    )

    if method not in ("simple", "marginal"):
        raise ValueError(f"method must be 'simple' | 'marginal'; got {method!r}")
    if metric not in ("lift", "did"):
        raise ValueError(f"metric must be 'lift' | 'did'; got {metric!r}")
    if method == "simple" and base_config_id is None:
        raise ValueError("method='simple' requires base_config_id")

    # ── 1. fetch ──────────────────────────────────────────────────────
    cdf = data.configs_df(store)
    fdf = data.features_df(store, task=task)
    sdf = data.scores_df(store, model=model, scorer=scorer)

    # Feature universe + predicate universe.
    base_fids: frozenset = frozenset()
    if base_config_id is not None:
        base_fids = resolve.base_func_ids(cdf, base_config_id)

    # Drop base-like features (complete primitive set already in base).
    if not fdf.empty:
        fdf = fdf[fdf["func_ids"].map(
            lambda fids: not resolve.is_base_feature(fids, base_fids)
        )].reset_index(drop=True)

    canonicals = fdf["canonical_id"].tolist() if not fdf.empty else []
    empty_cols = [
        "canonical_id", "predicate_name", "predicate_value",
        "n_with", "mean_with", "n_without", "mean_without", "lift",
    ]
    if not canonicals:
        empty = pd.DataFrame(columns=empty_cols)
        return (empty, "") if report else empty

    if predicate_names is None:
        if sdf.empty:
            predicate_names = []
        else:
            predicate_names = sorted(sdf["predicate_name"].unique().tolist())

    # Numeric-predicate guard (R6.5 quick fix). Categorical grouping
    # over a continuous predicate produces n=1 rows per value — useless.
    # Until a regression-based path lands, skip them with a warning.
    if skip_numeric and predicate_names:
        kinds = data.predicate_kinds(store)
        numeric_in_scope = [p for p in predicate_names if kinds.get(p) == "numeric"]
        if numeric_in_scope:
            import warnings
            warnings.warn(
                "feature_predicate_table: dropping numeric predicate(s) "
                f"{numeric_in_scope} — current pipeline groups by exact "
                "predicate_value, which is meaningless for continuous "
                "data. Pass skip_numeric=False to force-include, or "
                "wait for the regression path. See analyze.data.predicate_kinds.",
                stacklevel=2,
            )
            predicate_names = [p for p in predicate_names if kinds.get(p) != "numeric"]

    if not predicate_names:
        empty = pd.DataFrame(columns=empty_cols)
        return (empty, "") if report else empty

    # ── 2. resolve ────────────────────────────────────────────────────
    if method == "simple":
        canonical_to_cid = resolve.simple_effect_configs(cdf, fdf, base_config_id)
        canonical_to_with_cids = None
    else:
        canonical_to_cid = None
        canonical_to_with_cids = resolve.configs_containing_feature(cdf, fdf)

    # ── 3. lift ───────────────────────────────────────────────────────
    all_cids = cdf["config_id"].astype(int).tolist()
    if method == "simple":
        df = effect.lift_simple(
            sdf,
            base_cid=base_config_id,
            canonicals=canonicals,
            canonical_to_cid=canonical_to_cid,
            predicate_names=predicate_names,
            reference_values=reference_values,
            progress=progress,
        )
    else:
        df = effect.lift_marginal(
            sdf,
            canonicals=canonicals,
            canonical_to_with_cids=canonical_to_with_cids,
            all_cids=all_cids,
            predicate_names=predicate_names,
            reference_values=reference_values,
            progress=progress,
        )

    if df.empty:
        df = df.drop(columns=["_ref"], errors="ignore")
        return (df, "") if report else df

    # ── 4a. drop unmatched (default) ──────────────────────────────────
    if not include_unmatched:
        df = df[(df["n_with"] > 0) & (df["n_without"] > 0)].reset_index(drop=True)
        if df.empty:
            df = df.drop(columns=["_ref"], errors="ignore")
            return (df, "") if report else df

    # ── 4b. DiD ────────────────────────────────────────────────────────
    if metric == "did":
        df = effect.did(df, drop_ref_col=True)
    else:
        df = df.drop(columns=["_ref"], errors="ignore")

    # ── 5. confidence ─────────────────────────────────────────────────
    if confidence:
        df = _confidence.attach_ci(
            df, sdf,
            method=method,
            base_config_id=base_config_id,
            canonical_to_cid=canonical_to_cid,
            canonical_to_with_cids=canonical_to_with_cids,
            all_cids=all_cids,
            n_boot=n_bootstrap,
            seed=random_seed,
            progress=progress,
            workers=workers,
        )

    # ── 6. rank / filter / top_k ──────────────────────────────────────
    df = rank.rank(
        df,
        sort_by=sort_by, top_k=top_k,
        confidence_min=confidence_min,
        min_effect=min_effect,
        min_lift_in_pair=min_lift_in_pair,
        metric=metric, confidence=confidence,
    )

    # ── 7. report (optional) ──────────────────────────────────────────
    if report is not None:
        if report not in ("markdown", "text", "both"):
            raise ValueError(
                f"report must be 'markdown' | 'text' | 'both'; got {report!r}"
            )
        run_meta = dict(
            model=model, scorer=scorer, method=method, metric=metric,
            confidence=confidence, n_bootstrap=n_bootstrap,
            sort_by=sort_by, top_k=top_k, confidence_min=confidence_min,
        )
        return df, _report.render(df, fmt=report, run_meta=run_meta)
    return df


# ══════════════════════════════════════════════════════════════════════
# Feature × Predicate interaction profile (analysis fork R1)
# ══════════════════════════════════════════════════════════════════════
#
# Wide-format view: one row per feature, one `did_<pname>` column per
# binary categorical predicate, plus depth/breadth stats and a simple
# classification. Built on the same lower layers as
# feature_predicate_table. Redundant-predicate warning comes from
# data.predicate_overlap.


def _summarize_per_feature(
    long_df,
    *,
    confidence: bool,
    magnitude_threshold: float,
    confidence_threshold: float,
):
    """Long-format (feature × predicate × value) → wide per-feature summary.

    Takes the output of ``feature_predicate_table(..., metric='did')`` and
    returns one row per feature with ``did_<pname>`` / ``p_gt_zero_<pname>``
    columns plus four summary columns (``n_sig_slices``, ``max_abs_did``,
    ``breadth_mean``, ``classification``).

    For each (canonical_id, predicate_name) pair we pick the representative
    row as the one with largest ``|did|`` (ignoring the always-zero
    reference row). Multi-value predicates without an explicit DiD reference
    carry ``did=NaN`` everywhere and flow through as NaN columns.

    Classification labels (per-feature, post-threshold):
        null       — no predicate with ``|did| ≥ magnitude_threshold``
        selective  — 1–2 predicates above threshold
        broad      — ≥ 3 predicates, all same sign
        mixed      — ≥ 2 predicates, mixed signs
    """
    import pandas as pd

    if long_df.empty:
        return pd.DataFrame(columns=[
            "canonical_id",
            "n_sig_slices", "max_abs_did", "breadth_mean", "classification",
        ])

    # Pick representative row per (canonical_id, predicate_name): the row
    # whose |did| is largest, excluding always-zero reference rows.
    def _pick(group):
        nz = group.loc[group["did"].abs() > 0]
        return (nz.loc[nz["did"].abs().idxmax()] if len(nz) else group.iloc[0])

    per_fp = (long_df
              .groupby(["canonical_id", "predicate_name"], as_index=False)
              .apply(_pick, include_groups=False)
              .reset_index())

    # Pivot to wide.
    did_wide = per_fp.pivot(index="canonical_id",
                            columns="predicate_name", values="did")
    did_wide.columns = [f"did_{c}" for c in did_wide.columns]
    blocks = [did_wide]
    if confidence and "p_gt_zero" in per_fp.columns:
        p_wide = per_fp.pivot(index="canonical_id",
                              columns="predicate_name", values="p_gt_zero")
        p_wide.columns = [f"p_gt_zero_{c}" for c in p_wide.columns]
        blocks.append(p_wide)
    wide = pd.concat(blocks, axis=1).reset_index()

    # Per-feature summary stats + classification.
    did_cols = [c for c in wide.columns if c.startswith("did_")]
    p_cols   = [c for c in wide.columns if c.startswith("p_gt_zero_")]

    def _stats(row):
        dids = [row[c] for c in did_cols if pd.notna(row[c])]
        big = [d for d in dids if abs(d) >= magnitude_threshold]
        if not big:
            cls = "null"
        elif len({1 if d > 0 else -1 for d in big}) > 1:
            cls = "mixed"
        elif len(big) <= 2:
            cls = "selective"
        else:
            cls = "broad"
        n_sig = sum(
            1 for c in p_cols
            if pd.notna(row[c]) and row[c] >= confidence_threshold
        ) if (confidence and p_cols) else 0
        return pd.Series({
            "n_sig_slices":   n_sig,
            "max_abs_did":    max((abs(d) for d in dids), default=float("nan")),
            "breadth_mean":   (sum(dids) / len(dids)) if dids else float("nan"),
            "classification": cls,
        })

    return pd.concat([wide, wide.apply(_stats, axis=1)], axis=1)


def feature_profile(
    store: CubeStore,
    *,
    model: str,
    scorer: str,
    base_config_id: int,
    predicate_names: Optional[List[str]] = None,
    reference_values: Optional[Dict[str, str]] = None,
    task: Optional[str] = None,
    # thresholds (tune per cube; defaults are conservative):
    magnitude_threshold: float = 0.01,   # |did| below this = "null" slot
    confidence_threshold: float = 0.80,  # P(lift>0) above this = "sig positive"
    # plumbed through to the bootstrap engine:
    confidence: bool = True,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
    progress: bool = False,
    workers: int = 1,
    # diagnostics:
    skip_numeric: bool = True,
    warn_redundant_predicates: bool = True,
    redundancy_threshold: float = 0.95,
):
    """Wide-format feature × predicate interaction profile.

    One row per feature. For each binary categorical predicate in scope,
    computes `did = lift(value != ref) − lift(ref)` and reports it as
    a column. Adds breadth/depth summary stats + classification.

    Returns a pandas.DataFrame with columns:
        canonical_id, feature_id,
        did_<pname_1>, did_<pname_2>, …
        p_gt_zero_<pname_1>, …             # if confidence=True
        n_sig_slices, max_abs_did, breadth_mean, classification

    Classification (post-hoc label based on thresholds):
        null       — no predicate with |did| ≥ magnitude_threshold
        selective  — 1–2 predicates with |did| above threshold; rest null
        broad      — ≥3 predicates same-signed (all positive or all negative)
        mixed      — ≥2 predicates above threshold with DIFFERENT signs

    Args:
        magnitude_threshold: |did| below this is considered "not meaningfully
            different". Default 0.01 matches the verdict logic in report.py.
        confidence_threshold: P(lift>0) at which a slice counts as
            statistically non-null (used for n_sig_slices).
        warn_redundant_predicates: if True, emit a UserWarning listing
            predicate-pair redundancies (e.g. is_count ↔ operation_type=count)
            detected by data.predicate_overlap. Does not modify the output.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("pandas required for feature_profile") from e

    # ── 0. redundancy diagnostic (opt-out) ─────────────────────────────
    if warn_redundant_predicates:
        overlap = data.predicate_overlap(store, threshold=redundancy_threshold)
        if not overlap.empty:
            import warnings
            pairs = "; ".join(
                f"{r['pred_a_name']}={r['pred_a_value']} ↔ "
                f"{r['pred_b_name']}={r['pred_b_value']} "
                f"(jaccard={r['jaccard']:.2f})"
                for _, r in overlap.head(10).iterrows()
            )
            warnings.warn(
                "feature_profile: redundant predicate pair(s) detected "
                f"(jaccard ≥ {redundancy_threshold}): {pairs}. "
                "Same queries double-counted across columns. "
                "Pass warn_redundant_predicates=False to silence, or "
                "call analyze.data.predicate_overlap(store) to audit.",
                stacklevel=2,
            )

    # ── 1. delegate to feature_predicate_table for the long-format cells ──
    # We ask for metric="did" + confidence=True so each (feature × predicate
    # × value) row carries did + p_gt_zero ready-to-use.
    long_df = feature_predicate_table(
        store,
        model=model, scorer=scorer,
        method="simple", metric="did",
        predicate_names=predicate_names,
        base_config_id=base_config_id,
        reference_values=reference_values,
        task=task,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        random_seed=random_seed,
        progress=progress, workers=workers,
        skip_numeric=skip_numeric,
    )
    if long_df.empty:
        return pd.DataFrame(columns=[
            "canonical_id", "feature_id",
            "n_sig_slices", "max_abs_did", "breadth_mean", "classification",
        ])

    # ── 2. long → wide summary via shared helper ──────────────────────
    wide = _summarize_per_feature(
        long_df,
        confidence=confidence,
        magnitude_threshold=magnitude_threshold,
        confidence_threshold=confidence_threshold,
    )

    # ── 3. attach feature_id from the registry ────────────────────────
    # Pull (canonical_id, feature_id) directly from the feature table
    # via data.features_df (already parses primitive_spec, materializes
    # func_ids — we only want canonical → feature_id here).
    fdf = data.features_df(store, task=task)
    if not fdf.empty:
        cid_to_fid = dict(zip(fdf["canonical_id"], fdf["feature_id"]))
        wide.insert(1, "feature_id",
                    wide["canonical_id"].map(cid_to_fid))
    else:
        wide.insert(1, "feature_id", None)

    # Sort by max_abs_did descending — the "deepest-effect" features first.
    wide = wide.sort_values("max_abs_did", ascending=False,
                            na_position="last").reset_index(drop=True)
    return wide
