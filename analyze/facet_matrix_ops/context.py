"""Query-context vector operators backed by cube predicates."""
from __future__ import annotations

from typing import Optional, Sequence

from analyze.context_attributes import canonical_context_items
from core.store import CubeStore

from .common import query_scope_clause, query_split


def query_context_sets(
    store: CubeStore,
    *,
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    predicate_names: Optional[Sequence[str]] = None,
    max_values_per_predicate: Optional[int] = 32,
    context_view: str = "raw",
    include_negative: bool = True,
):
    """Return query-level context atom sets.

    ``context_view="raw"`` preserves the historical behavior and reads
    ``predicate`` rows directly. ``"canonical_shared"`` and
    ``"canonical_scoped"`` compute the canonical context layer from query
    metadata without modifying the cube.
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas required for query_context_sets") from e

    if context_view != "raw":
        view = {
            "canonical_shared": "shared",
            "canonical_scoped": "scoped",
            "canonical_all": "all",
        }.get(context_view)
        if view is None:
            raise ValueError(
                "context_view must be raw, canonical_shared, canonical_scoped, "
                f"or canonical_all; got {context_view!r}"
            )
        where, params = query_scope_clause(dataset=dataset, split=split, alias="q")
        rows = store._get_conn().execute(
            f"""
            SELECT q.query_id, q.dataset, q.meta AS query_meta
            FROM query q
            WHERE {' AND '.join(where)}
            ORDER BY q.query_id
            """,
            tuple(params),
        ).fetchall()
        records = []
        for row in rows:
            meta = row["query_meta"]
            records.append({
                "query_id": str(row["query_id"]),
                "dataset": row["dataset"],
                "split": query_split(meta),
                "context_atoms": frozenset(
                    canonical_context_items(
                        meta,
                        str(row["dataset"]),
                        view=view,
                        include_negative=include_negative,
                    )
                ),
            })
        return pd.DataFrame(records, columns=["query_id", "dataset", "split", "context_atoms"])

    where, params = query_scope_clause(dataset=dataset, split=split, alias="q")
    pred_filter = ""
    if predicate_names is not None:
        names = [str(name) for name in predicate_names]
        if not names:
            return pd.DataFrame(columns=["query_id", "dataset", "split", "context_atoms"])
        ph = ",".join("?" * len(names))
        pred_filter = f" AND p.name IN ({ph})"
        params.extend(names)

    rows = store._get_conn().execute(
        f"""
        SELECT p.query_id, q.dataset, q.meta AS query_meta, p.name, p.value
        FROM predicate p
        JOIN query q ON q.query_id = p.query_id
        WHERE {' AND '.join(where)}
          {pred_filter}
        ORDER BY p.query_id, p.name
        """,
        tuple(params),
    ).fetchall()

    values_by_name: dict[str, set[str]] = {}
    for row in rows:
        values_by_name.setdefault(str(row["name"]), set()).add(str(row["value"]))
    allowed = set(values_by_name)
    if max_values_per_predicate is not None:
        allowed = {
            name for name, values in values_by_name.items()
            if len(values) <= int(max_values_per_predicate)
        }

    by_query: dict[str, dict] = {}
    for row in rows:
        name = str(row["name"])
        if name not in allowed:
            continue
        qid = str(row["query_id"])
        record = by_query.setdefault(qid, {
            "query_id": qid,
            "dataset": row["dataset"],
            "split": query_split(row["query_meta"]),
            "context_atoms": set(),
        })
        record["context_atoms"].add(f"{name}={row['value']}")

    records = [
        {
            "query_id": rec["query_id"],
            "dataset": rec["dataset"],
            "split": rec["split"],
            "context_atoms": frozenset(rec["context_atoms"]),
        }
        for rec in by_query.values()
    ]
    return pd.DataFrame(records, columns=["query_id", "dataset", "split", "context_atoms"])
