"""Layer 2 — filterable execution query.

``ExecutionQuery`` is a chainable builder that compiles filters into SQL.
Each ``.where_*`` method returns a NEW ExecutionQuery (immutable-ish);
terminal methods (``.count()``, ``.rows()``, ``.df()``, ``.agg()``)
materialize the result.

Quick example
─────────────

.. code-block:: python

    from analyze.query import ExecutionQuery

    q = (ExecutionQuery(store)
         .model("meta-llama/Llama-3.1-8B-Instruct")
         .scorer("denotation_acc")
         .has_feature("enable_cot")
         .predicate("has_aggregation", "true"))

    print(q.count())              # → int
    df = q.df()                   # → pandas DataFrame with score joined
    df_slice = q.columns(         # → projection
        ["config_id", "query_id", "score", "prediction"]).df()
    agg = q.agg(by=["config_id"], fn="avg", metric="score")

Column semantics
────────────────

Base SELECT pulls from ``execution``. If ``.scorer(...)`` is set, the query
LEFT-joins ``evaluation`` so ``score`` / ``metrics`` are available. Query-
level fields from ``query`` (content, meta) are NOT pulled by default; call
``.columns([..., "q.content"])`` if you need them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.store import CubeStore


# Columns available from the default SELECT without a scorer join.
_EXECUTION_COLS = (
    "execution_id", "config_id", "query_id", "model",
    "system_prompt", "user_content", "raw_response", "prediction",
    "latency_ms", "prompt_tokens", "completion_tokens", "error",
    "phase_ids", "created_at", "meta",
)
# Columns added when a scorer filter is present (evaluation is LEFT-joined).
_EVAL_COLS = ("score", "scorer", "metrics", "eval_id")


@dataclass(frozen=True)
class _Filters:
    phases:           Tuple[str, ...] = ()
    config_ids:       Tuple[int, ...] = ()
    models:           Tuple[str, ...] = ()
    scorer:           Optional[str] = None
    query_ids:        Tuple[str, ...] = ()
    has_funcs:        Tuple[str, ...] = ()        # ALL of these
    has_any_funcs:    Tuple[str, ...] = ()        # ANY of these
    has_features:     Tuple[str, ...] = ()        # ALL canonical_ids
    has_any_features: Tuple[str, ...] = ()        # ANY canonical_ids
    predicates:       Tuple[Tuple[str, Optional[str]], ...] = ()
    score_ops:        Tuple[Tuple[str, float], ...] = ()  # (op, value)
    error_state:      Optional[bool] = None       # None=any, True=has error, False=no error

    projection:       Optional[Tuple[str, ...]] = None
    order_by:         Optional[str] = None
    limit:            Optional[int] = None


class ExecutionQuery:
    """Chainable filter over the execution+evaluation tables.

    All ``.where_*`` / ``.has_*`` methods are pure and return a fresh
    ExecutionQuery. Nothing executes against the cube until a terminal
    method is called.
    """

    def __init__(
        self,
        store: CubeStore,
        filters: Optional[_Filters] = None,
    ) -> None:
        self.store = store
        self._f = filters if filters is not None else _Filters()

    # ── filter methods (fluent, immutable) ────────────────────────────

    def phase(self, phase: str) -> "ExecutionQuery":
        """Include executions tagged with this phase_id."""
        return self._extend("phases", (phase,))

    def phases(self, phases: Sequence[str]) -> "ExecutionQuery":
        return self._extend("phases", tuple(phases))

    def config(self, config_id: int) -> "ExecutionQuery":
        return self._extend("config_ids", (config_id,))

    def configs(self, config_ids: Sequence[int]) -> "ExecutionQuery":
        return self._extend("config_ids", tuple(config_ids))

    def model(self, model: str) -> "ExecutionQuery":
        return self._extend("models", (model,))

    def models(self, models: Sequence[str]) -> "ExecutionQuery":
        return self._extend("models", tuple(models))

    def scorer(self, scorer: str) -> "ExecutionQuery":
        """Select a scorer; causes a LEFT JOIN on evaluation so ``score``
        / ``metrics`` columns are available in ``.rows()`` / ``.df()``.
        """
        return self._replace(scorer=scorer)

    def query(self, query_id: str) -> "ExecutionQuery":
        return self._extend("query_ids", (query_id,))

    def queries(self, query_ids: Sequence[str]) -> "ExecutionQuery":
        return self._extend("query_ids", tuple(query_ids))

    # -- structural filters on the config -----------------------------

    def has_func(self, func_id: str) -> "ExecutionQuery":
        """Config must contain this func_id."""
        return self._extend("has_funcs", (func_id,))

    def has_all_funcs(self, func_ids: Sequence[str]) -> "ExecutionQuery":
        return self._extend("has_funcs", tuple(func_ids))

    def has_any_func(self, func_ids: Sequence[str]) -> "ExecutionQuery":
        return self._extend("has_any_funcs", tuple(func_ids))

    def has_feature(self, canonical_id: str) -> "ExecutionQuery":
        """Config must contain a feature with this canonical_id (resolved
        to content-hash via the feature table).
        """
        return self._extend("has_features", (canonical_id,))

    def has_all_features(self, canonical_ids: Sequence[str]) -> "ExecutionQuery":
        return self._extend("has_features", tuple(canonical_ids))

    def has_any_feature(self, canonical_ids: Sequence[str]) -> "ExecutionQuery":
        return self._extend("has_any_features", tuple(canonical_ids))

    # -- per-query semantic filters -----------------------------------

    def predicate(self, name: str, value: Optional[str] = None) -> "ExecutionQuery":
        """Query must have a predicate row with this name (and optionally value)."""
        return self._extend("predicates", ((name, value),))

    # -- score / error filters ----------------------------------------

    def where_score(self, op: str, value: float) -> "ExecutionQuery":
        """Filter by score. ``op`` ∈ {'=', '!=', '<', '<=', '>', '>='}.

        Requires a scorer to be selected via ``.scorer(...)`` first.
        """
        if op not in ("=", "!=", "<", "<=", ">", ">="):
            raise ValueError(f"unsupported score op: {op!r}")
        return self._extend("score_ops", ((op, float(value)),))

    def with_error(self) -> "ExecutionQuery":
        return self._replace(error_state=True)

    def without_error(self) -> "ExecutionQuery":
        return self._replace(error_state=False)

    # -- projection / order / limit -----------------------------------

    def columns(self, cols: Sequence[str]) -> "ExecutionQuery":
        """Override the default SELECT columns. Use fully-qualified names
        (e.g. ``"e.prediction"``, ``"ev.score"``) when ambiguity matters;
        the builder falls back to unqualified lookup otherwise.
        """
        return self._replace(projection=tuple(cols))

    def order_by(self, clause: str) -> "ExecutionQuery":
        """Raw ORDER BY clause (e.g. ``"ev.score DESC"`` or ``"e.created_at"``)."""
        return self._replace(order_by=clause)

    def limit(self, n: int) -> "ExecutionQuery":
        return self._replace(limit=int(n))

    # ── terminal methods ──────────────────────────────────────────────

    def count(self) -> int:
        sql, params = self._compile(select="COUNT(*) AS n")
        row = self.store._get_conn().execute(sql, params).fetchone()
        return int(row[0]) if row else 0

    def rows(self) -> List[Dict[str, Any]]:
        """Return the result as a list of dicts.

        Parses JSON columns (``meta``, ``metrics``, ``phase_ids``) into
        Python objects for convenience.
        """
        sql, params = self._compile()
        rows = self.store._get_conn().execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def df(self):
        """Return a pandas DataFrame.

        JSON columns remain strings in the DataFrame; materialize them with
        ``df['meta'].map(json.loads)`` if needed.
        """
        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover
            raise ImportError("pandas required for .df()") from e
        sql, params = self._compile()
        return pd.read_sql_query(sql, self.store._get_conn(), params=params)

    def agg(
        self,
        *,
        by: Sequence[str],
        fn: str = "avg",
        metric: str = "score",
    ) -> List[Dict[str, Any]]:
        """GROUP BY ``by`` and aggregate ``metric`` with ``fn``.

        ``fn`` ∈ {'avg', 'sum', 'min', 'max', 'count'}. With ``by=[]``, no
        GROUP BY is emitted; the result is a single-row global aggregate.

        .. code-block:: python

            q.agg(by=["config_id"], fn="avg", metric="score")
            q.agg(by=["model", "config_id"], fn="count", metric="*")
            q.agg(by=[], fn="avg", metric="score")  # single global mean
        """
        fn_upper = fn.upper()
        if fn_upper not in ("AVG", "SUM", "MIN", "MAX", "COUNT"):
            raise ValueError(f"unsupported agg fn: {fn!r}")
        metric_expr = "*" if metric == "*" else self._resolve_col(metric)
        agg_alias = f"{fn.lower()}_{metric.replace('*', 'all')}"
        agg_select = f"{fn_upper}({metric_expr}) AS {agg_alias}, COUNT(*) AS n"

        if by:
            group_exprs = [self._resolve_col(c) for c in by]
            select = (
                ", ".join(f"{g} AS {alias}" for g, alias in zip(group_exprs, by))
                + ", " + agg_select
            )
            group_by = ", ".join(group_exprs)
        else:
            select = agg_select
            group_by = None

        sql, params = self._compile(
            select=select,
            group_by=group_by,
            order_by=self._f.order_by,
        )
        rows = self.store._get_conn().execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ── SQL compilation ───────────────────────────────────────────────

    def _compile(
        self,
        *,
        select: Optional[str] = None,
        group_by: Optional[str] = None,
        order_by: Optional[str] = None,
    ) -> Tuple[str, Tuple[Any, ...]]:
        f = self._f
        need_eval_join = f.scorer is not None or any(
            metric in (self._f.projection or ()) for metric in ("score", "ev.score")
        ) or bool(f.score_ops)
        need_config_join = (
            bool(f.has_funcs) or bool(f.has_any_funcs)
            or bool(f.has_features) or bool(f.has_any_features)
        )

        # -- SELECT -----------------------------------------------------
        if select is not None:
            select_clause = select
        elif f.projection is not None:
            select_clause = ", ".join(self._resolve_col(c) for c in f.projection)
        else:
            cols = [f"e.{c}" for c in _EXECUTION_COLS]
            if need_eval_join:
                cols += ["ev.score", "ev.scorer", "ev.metrics", "ev.eval_id"]
            select_clause = ", ".join(cols)

        # -- FROM / JOIN -----------------------------------------------
        parts: List[str] = ["FROM execution e"]
        if need_eval_join:
            if f.scorer is not None:
                parts.append(
                    "LEFT JOIN evaluation ev "
                    "  ON ev.execution_id = e.execution_id AND ev.scorer = ?"
                )
            else:
                parts.append(
                    "LEFT JOIN evaluation ev ON ev.execution_id = e.execution_id"
                )
        if need_config_join:
            parts.append("JOIN config c ON c.config_id = e.config_id")

        # -- WHERE ------------------------------------------------------
        where: List[str] = []
        params: List[Any] = []
        if f.scorer is not None:
            params.append(f.scorer)  # for ev.scorer = ?

        if f.phases:
            # phase_ids is a JSON array; LIKE '%"<phase>"%' is sufficient for ASCII tags.
            clauses = []
            for ph in f.phases:
                clauses.append("e.phase_ids LIKE ?")
                params.append(f'%"{ph}"%')
            where.append("(" + " OR ".join(clauses) + ")")

        if f.config_ids:
            where.append("e.config_id IN (" + ",".join("?" * len(f.config_ids)) + ")")
            params.extend(f.config_ids)
        if f.models:
            where.append("e.model IN (" + ",".join("?" * len(f.models)) + ")")
            params.extend(f.models)
        if f.query_ids:
            where.append("e.query_id IN (" + ",".join("?" * len(f.query_ids)) + ")")
            params.extend(f.query_ids)

        if f.has_funcs:
            # Each func_id must be present in config.func_ids (sorted JSON array).
            for fid in f.has_funcs:
                where.append(
                    "EXISTS (SELECT 1 FROM json_each(c.func_ids) WHERE value = ?)"
                )
                params.append(fid)
        if f.has_any_funcs:
            placeholders = ",".join("?" * len(f.has_any_funcs))
            where.append(
                f"EXISTS (SELECT 1 FROM json_each(c.func_ids) WHERE value IN ({placeholders}))"
            )
            params.extend(f.has_any_funcs)

        if f.has_features:
            # Resolve canonical_id → feature_id hash(es) via the feature table.
            for cid in f.has_features:
                where.append(
                    "EXISTS (SELECT 1 "
                    "FROM json_each(json_extract(c.meta, '$.feature_ids')) j "
                    "JOIN feature f ON f.feature_id = j.value "
                    "WHERE f.canonical_id = ?)"
                )
                params.append(cid)
        if f.has_any_features:
            placeholders = ",".join("?" * len(f.has_any_features))
            where.append(
                f"EXISTS (SELECT 1 "
                f"FROM json_each(json_extract(c.meta, '$.feature_ids')) j "
                f"JOIN feature f ON f.feature_id = j.value "
                f"WHERE f.canonical_id IN ({placeholders}))"
            )
            params.extend(f.has_any_features)

        for name, value in f.predicates:
            if value is None:
                where.append(
                    "EXISTS (SELECT 1 FROM predicate p "
                    "WHERE p.query_id = e.query_id AND p.name = ?)"
                )
                params.append(name)
            else:
                where.append(
                    "EXISTS (SELECT 1 FROM predicate p "
                    "WHERE p.query_id = e.query_id AND p.name = ? AND p.value = ?)"
                )
                params.extend((name, value))

        for op, val in f.score_ops:
            where.append(f"ev.score {op} ?")
            params.append(val)

        if f.error_state is True:
            where.append("e.error IS NOT NULL AND e.error != ''")
        elif f.error_state is False:
            where.append("(e.error IS NULL OR e.error = '')")

        # -- assemble ---------------------------------------------------
        sql = f"SELECT {select_clause} " + " ".join(parts)
        if where:
            sql += " WHERE " + " AND ".join(where)
        if group_by:
            sql += f" GROUP BY {group_by}"
        ob = order_by or f.order_by
        if ob:
            sql += f" ORDER BY {ob}"
        if f.limit is not None:
            sql += f" LIMIT {int(f.limit)}"
        return sql, tuple(params)

    # ── helpers ───────────────────────────────────────────────────────

    def _extend(self, field_name: str, values: Tuple[Any, ...]) -> "ExecutionQuery":
        existing = getattr(self._f, field_name)
        return ExecutionQuery(self.store, replace(self._f, **{field_name: existing + values}))

    def _replace(self, **kwargs) -> "ExecutionQuery":
        return ExecutionQuery(self.store, replace(self._f, **kwargs))

    @staticmethod
    def _resolve_col(col: str) -> str:
        """Best-effort column-name resolution.

        - If already dotted (``e.prediction``), pass through.
        - If it's a known execution column → ``e.<col>``.
        - If it's a known evaluation column → ``ev.<col>``.
        - Otherwise pass through (user may be writing a SQL expression).
        """
        if "." in col or "(" in col or col == "*":
            return col
        if col in _EXECUTION_COLS:
            return f"e.{col}"
        if col in _EVAL_COLS:
            return f"ev.{col}"
        return col

    @staticmethod
    def _row_to_dict(row) -> Dict[str, Any]:
        d = dict(row)
        # Parse common JSON fields for convenience.
        for k in ("meta", "metrics"):
            if k in d and isinstance(d[k], str):
                try:
                    d[k] = json.loads(d[k])
                except (json.JSONDecodeError, TypeError):
                    pass
        if "phase_ids" in d and isinstance(d["phase_ids"], str):
            try:
                d["phase_ids"] = json.loads(d["phase_ids"])
            except (json.JSONDecodeError, TypeError):
                pass
        return d
