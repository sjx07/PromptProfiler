"""Outcome matrix operators for query-level paired deltas."""
from __future__ import annotations

from typing import Optional, Sequence

from core.store import CubeStore

from .common import query_scope_clause, query_split
from .config import config_feature_sets, infer_baseline_config_id


def paired_delta_rows(
    store: CubeStore,
    *,
    model: str,
    scorer: str,
    base_config_id: Optional[int] = None,
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    config_ids: Optional[Sequence[int]] = None,
):
    """Return query-level paired deltas against a baseline config."""
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas required for paired_delta_rows") from e

    if base_config_id is None:
        base_config_id = infer_baseline_config_id(
            store, model=model, scorer=scorer, dataset=dataset, split=split,
        )

    where, params = query_scope_clause(dataset=dataset, split=split, alias="q")
    config_filter = ""
    if config_ids is not None:
        scoped_ids = [int(c) for c in config_ids if int(c) != int(base_config_id)]
        if not scoped_ids:
            return pd.DataFrame(columns=[
                "dataset", "split", "model", "scorer", "config_id", "query_id",
                "score", "base_score", "delta", "improved", "active_features", "label",
            ])
        ph = ",".join("?" * len(scoped_ids))
        config_filter = f" AND e.config_id IN ({ph})"
        params = params + scoped_ids

    df = pd.read_sql_query(
        f"""
        WITH base AS (
            SELECT e.query_id, ev.score AS base_score
            FROM execution e
            JOIN evaluation ev ON ev.execution_id = e.execution_id
            WHERE e.config_id = ? AND e.model = ? AND ev.scorer = ?
              AND (e.error IS NULL OR e.error = '')
              AND ev.score IS NOT NULL
        )
        SELECT q.dataset,
               q.meta AS query_meta,
               e.config_id,
               e.query_id,
               ev.score AS score,
               base.base_score AS base_score,
               ev.score - base.base_score AS delta
        FROM execution e
        JOIN evaluation ev ON ev.execution_id = e.execution_id
        JOIN base ON base.query_id = e.query_id
        JOIN query q ON q.query_id = e.query_id
        WHERE e.config_id != ?
          AND e.model = ? AND ev.scorer = ?
          AND (e.error IS NULL OR e.error = '')
          AND ev.score IS NOT NULL
          AND {' AND '.join(where)}
          {config_filter}
        ORDER BY e.config_id, e.query_id
        """,
        store._get_conn(),
        params=tuple([base_config_id, model, scorer, base_config_id, model, scorer] + params),
    )
    if df.empty:
        df["split"] = []
        df["model"] = []
        df["scorer"] = []
        df["improved"] = []
        df["active_features"] = []
        df["label"] = []
        return df

    df["split"] = df["query_meta"].map(query_split)
    df = df.drop(columns=["query_meta"])
    df["model"] = model
    df["scorer"] = scorer
    df["config_id"] = df["config_id"].astype(int)
    df["improved"] = df["delta"] > 0

    cdf = config_feature_sets(
        store,
        config_ids=sorted(df["config_id"].unique().tolist()),
        dataset=dataset,
    )
    features_by_config = dict(zip(cdf["config_id"], cdf["active_features"])) if not cdf.empty else {}
    labels_by_config = dict(zip(cdf["config_id"], cdf["label"])) if not cdf.empty else {}
    df["active_features"] = df["config_id"].map(
        lambda cid: features_by_config.get(int(cid), frozenset())
    )
    df["label"] = df["config_id"].map(lambda cid: labels_by_config.get(int(cid), str(cid)))
    return df[[
        "dataset", "split", "model", "scorer", "config_id", "query_id",
        "score", "base_score", "delta", "improved", "active_features", "label",
    ]]
