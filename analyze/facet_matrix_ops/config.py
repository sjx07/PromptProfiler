"""Config-level feature vector and baseline operators."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from core.store import CubeStore
from analyze import data

from .common import (
    DEFAULT_DROP_IF_CONSTANT_PREFIXES,
    as_string_list,
    clean_config_map_labels,
    config_label,
    is_structural_feature,
    matches_dataset,
    query_scope_clause,
)


def config_feature_sets(
    store: CubeStore,
    *,
    config_ids: Optional[Sequence[int]] = None,
    dataset: Optional[str] = None,
    drop_structural: bool = True,
    drop_constant_prefixes: Sequence[str] = DEFAULT_DROP_IF_CONSTANT_PREFIXES,
):
    """Return config-level sparse feature vectors.

    ``active_features`` are resolved canonical ids with structural boilerplate
    removed. Features whose canonical id starts with one of
    ``drop_constant_prefixes`` are removed only when they are constant across the
    selected config scope; this treats fixed output contracts as measurement
    adapters while still preserving contract variation if it exists.
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas required for config_feature_sets") from e

    cdf = data.configs_df(
        store,
        config_ids=[int(c) for c in config_ids] if config_ids is not None else None,
    )
    if cdf.empty:
        return pd.DataFrame(columns=[
            "config_id", "label", "canonical_ids", "active_features", "meta",
        ])

    fdf = data.features_df(store)
    fid_to_cid = (
        dict(zip(fdf["feature_id"], fdf["canonical_id"]))
        if not fdf.empty and "feature_id" in fdf.columns
        else {}
    )
    clean_labels = clean_config_map_labels(
        store,
        config_ids=[int(c) for c in config_ids] if config_ids is not None else None,
        dataset=dataset,
    )

    records: List[Dict[str, Any]] = []
    for _, row in cdf.iterrows():
        meta = row["meta"] if isinstance(row["meta"], dict) else {}
        if not matches_dataset(meta, dataset):
            continue

        canonical_ids = set(as_string_list(meta.get("canonical_ids")))
        canonical_ids.update(as_string_list(meta.get("canonical_id")))
        for fid in row.get("feature_ids_set", frozenset()) or frozenset():
            cid = fid_to_cid.get(fid)
            if cid:
                canonical_ids.add(cid)

        if drop_structural:
            active = frozenset(
                cid for cid in canonical_ids if not is_structural_feature(cid)
            )
        else:
            active = frozenset(canonical_ids)

        records.append({
            "config_id": int(row["config_id"]),
            "label": clean_labels.get(
                int(row["config_id"]),
                config_label(meta, dataset=dataset),
            ),
            "canonical_ids": frozenset(canonical_ids),
            "active_features": active,
            "meta": meta,
        })

    out = pd.DataFrame(records)
    if out.empty or not drop_constant_prefixes:
        return out

    constant_candidates: Counter[str] = Counter()
    for fset in out["active_features"]:
        for fid in fset:
            if any(fid.startswith(prefix) for prefix in drop_constant_prefixes):
                constant_candidates[fid] += 1
    n_configs = len(out)
    constant = {fid for fid, count in constant_candidates.items() if count == n_configs}
    if constant:
        out["active_features"] = out["active_features"].map(
            lambda fset: frozenset(fid for fid in fset if fid not in constant)
        )
    return out


def infer_baseline_config_id(
    store: CubeStore,
    *,
    model: str,
    scorer: str,
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    label: str = "base",
) -> int:
    """Infer the evaluated baseline config for a benchmark scope."""
    where, params = query_scope_clause(dataset=dataset, split=split, alias="q")
    rows = store._get_conn().execute(
        f"""
        SELECT e.config_id, COUNT(*) AS n_eval
        FROM execution e
        JOIN evaluation ev ON ev.execution_id = e.execution_id
        JOIN query q ON q.query_id = e.query_id
        WHERE e.model = ? AND ev.scorer = ?
          AND (e.error IS NULL OR e.error = '')
          AND ev.score IS NOT NULL
          AND {' AND '.join(where)}
        GROUP BY e.config_id
        ORDER BY n_eval DESC, e.config_id
        """,
        tuple([model, scorer] + params),
    ).fetchall()
    if not rows:
        raise ValueError(
            "no evaluated configs found for "
            f"model={model!r}, scorer={scorer!r}, dataset={dataset!r}, split={split!r}"
        )

    config_ids = [int(r["config_id"]) for r in rows]
    n_eval = {int(r["config_id"]): int(r["n_eval"] or 0) for r in rows}
    cdf = config_feature_sets(store, config_ids=config_ids, dataset=dataset)
    if not cdf.empty:
        label_matches = cdf[cdf["label"] == label]
        if not label_matches.empty:
            return int(max(
                label_matches["config_id"].tolist(),
                key=lambda cid: (n_eval.get(int(cid), 0), -int(cid)),
            ))
        empty = cdf[cdf["active_features"].map(lambda fset: len(fset) == 0)]
        if not empty.empty:
            return int(max(
                empty["config_id"].tolist(),
                key=lambda cid: (n_eval.get(int(cid), 0), -int(cid)),
            ))
    return min(config_ids)
