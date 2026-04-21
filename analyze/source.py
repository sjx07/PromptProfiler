"""Layer 0.5 — lazy source handle for the analysis pipeline.

Analysis-fork R5 (option D — SQL-native source): the Pipeline's source
stage no longer eagerly materializes every DataFrame. It returns a
``SourceHandle`` instead — a lightweight query builder that pulls
rows from the SQL view layer only when a downstream stage actually
asks for them.

The handle is a facade over ``core.store.CubeStore`` + the
``v_query_scores`` / ``v_scored_executions`` / ``v_per_config_predicate_means``
SQL views (declared in ``core.schema``). Materialized DataFrames are
cached on the handle instance, so the same request from different
stages shares a single fetch.

Backwards compatibility: ``SourceHandle`` supports dict-style access
(``src["scores"]``, ``src["configs"]``, ``src["features"]``) so the
existing Pipeline stage code works with no edits. Explicit methods
are preferred going forward.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from core.store import CubeStore


# ── SQL builders (shared with analyze.data helpers) ───────────────────

def _sql_scored_executions(
    *,
    model: str,
    scorer: str,
    config_ids: Optional[List[int]] = None,
    predicate_name: Optional[str] = None,
) -> Tuple[str, tuple]:
    """Query rows from ``v_scored_executions`` for (model, scorer).

    Returns (sql, params) tuple. Caller applies `pd.read_sql_query`.
    Column set matches the legacy ``data.scores_df`` contract:
    ``config_id, query_id, predicate_name, predicate_value, score``
    — the (model, scorer) pair is fixed by the WHERE clause and not
    echoed back, so downstream groupbys don't accidentally pick them
    up as grouping keys.
    """
    sql = (
        "SELECT config_id, query_id, predicate_name, predicate_value, score "
        "FROM v_scored_executions WHERE model = ? AND scorer = ?"
    )
    params: List[Any] = [model, scorer]
    if predicate_name is not None:
        sql += " AND predicate_name = ?"
        params.append(predicate_name)
    if config_ids is not None:
        if not config_ids:
            return sql + " AND 1=0", tuple(params)   # empty-result short-circuit
        ph = ",".join("?" * len(config_ids))
        sql += f" AND config_id IN ({ph})"
        params.extend(config_ids)
    return sql, tuple(params)


def _sql_per_config_means(
    *,
    model: str,
    scorer: str,
    predicate_name: Optional[str] = None,
    config_ids: Optional[List[int]] = None,
) -> Tuple[str, tuple]:
    """Query per-(config × predicate-value) means via the view."""
    sql = (
        "SELECT config_id, predicate_name, predicate_value, mean_score, n "
        "FROM v_per_config_predicate_means "
        "WHERE model = ? AND scorer = ?"
    )
    params: List[Any] = [model, scorer]
    if predicate_name is not None:
        sql += " AND predicate_name = ?"
        params.append(predicate_name)
    if config_ids is not None:
        if not config_ids:
            return sql + " AND 1=0", tuple(params)
        ph = ",".join("?" * len(config_ids))
        sql += f" AND config_id IN ({ph})"
        params.extend(config_ids)
    return sql, tuple(params)


def _sql_query_scores(
    *,
    model: str,
    scorer: str,
    config_ids: Optional[List[int]] = None,
) -> Tuple[str, tuple]:
    """Query rows from ``v_query_scores`` (no predicate join — cheaper
    when the caller doesn't need predicate tags)."""
    sql = "SELECT config_id, query_id, score FROM v_query_scores WHERE model = ? AND scorer = ?"
    params: List[Any] = [model, scorer]
    if config_ids is not None:
        if not config_ids:
            return sql + " AND 1=0", tuple(params)
        ph = ",".join("?" * len(config_ids))
        sql += f" AND config_id IN ({ph})"
        params.extend(config_ids)
    return sql, tuple(params)


# ── the handle ────────────────────────────────────────────────────────

class SourceHandle:
    """Lazy, cached facade over a CubeStore for analysis pipelines.

    A handle holds ``(store, model, scorer, task)`` and defers SQL
    execution until a consumer calls an explicit method. Each call
    memoizes its result on the handle, so repeated requests (by
    different Pipeline stages) hit cache.

    Usage::

        src = SourceHandle(store, model="gemma-...", scorer="denotation_acc")
        scores = src.scores_df()         # full predicate fan-out
        cfgs   = src.configs_df()        # parsed func_ids + meta
        means  = src.per_config_predicate_means(predicate_name="has_agg")
        print(src.show_sql("scores"))    # inspect the SQL
    """

    __slots__ = ("_store", "_model", "_scorer", "_task", "_cache")

    def __init__(
        self,
        store: CubeStore,
        *,
        model: str,
        scorer: str,
        task: Optional[str] = None,
    ):
        self._store  = store
        self._model  = model
        self._scorer = scorer
        self._task   = task
        self._cache: Dict[str, Any] = {}

    # ── metadata ─────────────────────────────────────────────────────

    @property
    def store(self):  return self._store
    @property
    def model(self):  return self._model
    @property
    def scorer(self): return self._scorer
    @property
    def task(self):   return self._task

    # ── lazy materializers (all memoized) ────────────────────────────

    def scores_df(
        self,
        config_ids: Optional[List[int]] = None,
        predicate_name: Optional[str] = None,
    ):
        """Rows from ``v_scored_executions`` for this (model, scorer).

        Cached by (config_ids tuple, predicate_name) — repeated calls
        with the same filter return the same DataFrame object.
        """
        key = ("scores", tuple(config_ids) if config_ids is not None else None,
               predicate_name)
        if key not in self._cache:
            import pandas as pd
            sql, params = _sql_scored_executions(
                model=self._model, scorer=self._scorer,
                config_ids=config_ids, predicate_name=predicate_name,
            )
            self._cache[key] = pd.read_sql_query(
                sql, self._store._get_conn(), params=params,
            )
        return self._cache[key]

    def query_scores_df(
        self,
        config_ids: Optional[List[int]] = None,
    ):
        """Rows from ``v_query_scores`` — no predicate join.

        Use when the analysis doesn't need predicate tags (e.g.
        ``score_diff``, ``flip_rows``). Cheaper than ``scores_df``.
        """
        key = ("qscores", tuple(config_ids) if config_ids is not None else None)
        if key not in self._cache:
            import pandas as pd
            sql, params = _sql_query_scores(
                model=self._model, scorer=self._scorer,
                config_ids=config_ids,
            )
            self._cache[key] = pd.read_sql_query(
                sql, self._store._get_conn(), params=params,
            )
        return self._cache[key]

    def per_config_predicate_means(
        self,
        predicate_name: Optional[str] = None,
        config_ids: Optional[List[int]] = None,
    ):
        """Per-(config × predicate-value) means via the view."""
        key = ("means", predicate_name,
               tuple(config_ids) if config_ids is not None else None)
        if key not in self._cache:
            import pandas as pd
            sql, params = _sql_per_config_means(
                model=self._model, scorer=self._scorer,
                predicate_name=predicate_name, config_ids=config_ids,
            )
            self._cache[key] = pd.read_sql_query(
                sql, self._store._get_conn(), params=params,
            )
        return self._cache[key]

    def configs_df(self, config_ids: Optional[List[int]] = None):
        """Delegate to ``analyze.data.configs_df`` — unchanged path
        for now (small table, few rows, already cheap). Memoized on
        first call per handle."""
        if "configs" not in self._cache:
            from analyze import data as _d
            self._cache["configs"] = _d.configs_df(self._store, config_ids=config_ids)
        return self._cache["configs"]

    def features_df(self):
        """Delegate to ``analyze.data.features_df`` (task-filtered)."""
        if "features" not in self._cache:
            from analyze import data as _d
            self._cache["features"] = _d.features_df(self._store, task=self._task)
        return self._cache["features"]

    # ── inspection helpers (R5) ───────────────────────────────────────

    def show_sql(self, stage: str = "scores", **kwargs) -> str:
        """Return the SQL + params that would be run for a given stage.

        stage ∈ {"scores", "qscores", "means"}. kwargs are forwarded
        to the same builder the materializer uses — so you can preview
        the exact query for a specific filter.
        """
        builders = {
            "scores":  _sql_scored_executions,
            "qscores": _sql_query_scores,
            "means":   _sql_per_config_means,
        }
        if stage not in builders:
            raise ValueError(
                f"stage must be one of {sorted(builders)}; got {stage!r}"
            )
        sql, params = builders[stage](
            model=self._model, scorer=self._scorer, **kwargs,
        )
        return f"-- stage={stage}\n{sql}\n-- params={params}"

    # ── backwards-compat subscript access ────────────────────────────

    def __getitem__(self, key: str):
        """Dict-style access for legacy Pipeline stage code.

        Supported keys: "scores", "configs", "features".
        """
        if key == "scores":
            return self.scores_df()
        if key == "configs":
            return self.configs_df()
        if key == "features":
            return self.features_df()
        raise KeyError(
            f"SourceHandle has no bulk key {key!r}; use the explicit "
            "method (scores_df / configs_df / features_df / ...) instead."
        )

    def __repr__(self) -> str:
        return (f"SourceHandle(model={self._model!r}, scorer={self._scorer!r}, "
                f"task={self._task!r}, cached={len(self._cache)})")
