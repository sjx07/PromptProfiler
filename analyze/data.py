"""Layer 0 — raw SQL primitives.

Each function here turns a parameterized SQL query into a DataFrame (or
plain dict / list). No business logic, no feature/config resolution, no
arithmetic. The rest of `analyze/` builds on top of these.

Three primitives cover the SQL surface used by `compare.py`:

  * ``scores_df``    — the canonical 3-way join
                       (execution × evaluation × predicate).
  * ``configs_df``   — config table with parsed func_ids/meta.
  * ``features_df``  — feature table with parsed primitive_spec
                       + content-addressed func_ids materialized.

Two utility re-shapers operate on a ``scores_df`` already in memory:

  * ``per_config_predicate_means`` — group → mean per (config × pv).
  * ``per_query_scores``           — pivot → {pv: {qid: score}} for one config.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from core.store import CubeStore


# ── SQL primitives ────────────────────────────────────────────────────

def scores_df(
    store: CubeStore, *,
    model: str,
    scorer: str,
    config_ids: Optional[List[int]] = None,
    predicate_name: Optional[str] = None,
):
    """Long-format per-(execution × predicate-tag) score frame.

    The canonical 3-way join used everywhere in compare.py.

    Columns: ``config_id``, ``query_id``, ``predicate_name``,
             ``predicate_value``, ``score``.

    Filters:
      * errored executions dropped (``error IS NULL OR error = ''``);
      * NULL scores dropped;
      * ``predicate_name`` if given;
      * ``config_ids`` if given.

    Note: a query carrying multiple predicate names produces multiple
    rows per execution (one per name). Consumers normally filter to a
    single predicate_name first.
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas required for scores_df") from e

    sql = """
        SELECT e.config_id      AS config_id,
               e.query_id       AS query_id,
               p.name           AS predicate_name,
               p.value          AS predicate_value,
               ev.score         AS score
        FROM execution e
        JOIN evaluation ev ON ev.execution_id = e.execution_id
        JOIN predicate  p  ON p.query_id = e.query_id
        WHERE e.model = ? AND ev.scorer = ?
          AND (e.error IS NULL OR e.error = '')
          AND ev.score IS NOT NULL
    """
    params: List[Any] = [model, scorer]
    if predicate_name is not None:
        sql += " AND p.name = ?"
        params.append(predicate_name)
    if config_ids is not None:
        if not config_ids:
            return pd.DataFrame(columns=[
                "config_id", "query_id", "predicate_name",
                "predicate_value", "score",
            ])
        ph = ",".join("?" * len(config_ids))
        sql += f" AND e.config_id IN ({ph})"
        params.extend(config_ids)
    return pd.read_sql_query(sql, store._get_conn(), params=tuple(params))


def configs_df(
    store: CubeStore,
    config_ids: Optional[List[int]] = None,
):
    """Config table with parsed columns.

    Columns: ``config_id`` (int), ``func_ids`` (frozenset[str]),
             ``meta`` (dict).
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas required for configs_df") from e

    sql = "SELECT config_id, func_ids, meta FROM config"
    params: tuple = ()
    if config_ids is not None:
        if not config_ids:
            return pd.DataFrame(columns=["config_id", "func_ids", "meta"])
        ph = ",".join("?" * len(config_ids))
        sql += f" WHERE config_id IN ({ph})"
        params = tuple(config_ids)
    sql += " ORDER BY config_id"
    df = pd.read_sql_query(sql, store._get_conn(), params=params)
    if df.empty:
        return df

    def _parse_fids(s):
        if not s:
            return frozenset()
        try:
            return frozenset(json.loads(s))
        except json.JSONDecodeError:
            return frozenset()

    def _parse_meta(s):
        if not s:
            return {}
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return {}

    df["func_ids"] = df["func_ids"].map(_parse_fids)
    df["meta"] = df["meta"].map(_parse_meta)
    return df


def features_df(
    store: CubeStore,
    task: Optional[str] = None,
):
    """Feature table with parsed primitive_spec + materialized func_ids.

    Columns: ``canonical_id``, ``feature_id``, ``task``,
             ``primitive_spec`` (list of edits),
             ``func_ids`` (frozenset[str], content-addressed via
             ``core.func_registry.make_func_id``).
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas required for features_df") from e

    from core.func_registry import make_func_id

    sql = "SELECT canonical_id, feature_id, task, primitive_spec FROM feature"
    params: tuple = ()
    if task is not None:
        sql += " WHERE task = ?"
        params = (task,)
    sql += " ORDER BY canonical_id"
    df = pd.read_sql_query(sql, store._get_conn(), params=params)
    if df.empty:
        df["primitive_spec"] = []
        df["func_ids"] = []
        return df

    def _parse(s):
        if not s:
            return []
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return []

    df["primitive_spec"] = df["primitive_spec"].map(_parse)
    df["func_ids"] = df["primitive_spec"].map(
        lambda edits: frozenset(
            make_func_id(e["func_type"], e.get("params", {})) for e in edits
        )
    )
    return df


# ── reshapers (operate on an already-fetched scores_df) ───────────────

def per_config_predicate_means(scores_df_in, predicate_name: str):
    """Group a scores_df to (config_id × predicate_value) means.

    Returns columns: ``config_id``, ``predicate_value``,
                     ``mean_score``, ``n``.
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas required") from e

    sub = scores_df_in[scores_df_in["predicate_name"] == predicate_name]
    if sub.empty:
        return pd.DataFrame(columns=["config_id", "predicate_value",
                                     "mean_score", "n"])
    out = sub.groupby(["config_id", "predicate_value"], as_index=False).agg(
        mean_score=("score", "mean"),
        n=("score", "count"),
    )
    return out


def per_query_scores(
    scores_df_in, *, config_id: int, predicate_name: str,
) -> Dict[str, Dict[str, float]]:
    """Pivot a scores_df to ``{predicate_value: {query_id: score}}`` for one config.

    Used for paired-bootstrap inputs.
    """
    sub = scores_df_in[
        (scores_df_in["config_id"] == config_id) &
        (scores_df_in["predicate_name"] == predicate_name)
    ]
    out: Dict[str, Dict[str, float]] = {}
    for _, r in sub.iterrows():
        out.setdefault(r["predicate_value"], {})[r["query_id"]] = float(r["score"])
    return out


# ── predicate kind detection ──────────────────────────────────────────
#
# The current analysis pipeline groups by ``predicate_value`` as a
# categorical key. For genuinely numeric predicates (row counts,
# lengths, …) that produces one row per value with n=1, which is
# useless. Until we add a regression-based path, numeric predicates
# are detected here and skipped by default in ``feature_predicate_table``.

def _is_numeric_str(s) -> bool:
    if s is None:
        return False
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def predicate_kinds(store: CubeStore) -> Dict[str, str]:
    """Classify every predicate name in the cube as 'numeric' or 'categorical'.

    A predicate is numeric iff **every** observed value parses as a
    float. Predicates with zero observed values are labeled
    'categorical' (conservative default — they won't get auto-skipped).
    """
    rows = store._get_conn().execute(
        "SELECT name, value FROM predicate"
    ).fetchall()
    by_name: Dict[str, List[str]] = {}
    for r in rows:
        by_name.setdefault(r["name"], []).append(r["value"])
    out: Dict[str, str] = {}
    for name, vals in by_name.items():
        if not vals:
            out[name] = "categorical"
            continue
        out[name] = "numeric" if all(_is_numeric_str(v) for v in vals) else "categorical"
    return out
