"""Shared helpers for FACET matrix operators."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from core.store import CubeStore


STRUCTURAL_FEATURE_IDS = {
    "facet_dp_scaffold",
    "sqa_dialog_binding_base",
}

STRUCTURAL_PREFIXES = (
    "_section_",
)

DEFAULT_DROP_IF_CONSTANT_PREFIXES = (
    "output_contract_",
)


def json_loads(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def as_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(v) for v in value if str(v)]
    return []


def query_split(meta: Any) -> Optional[str]:
    parsed = json_loads(meta, {})
    if isinstance(parsed, dict) and parsed.get("split") is not None:
        return str(parsed["split"])
    return None


def is_structural_feature(canonical_id: str) -> bool:
    cid = str(canonical_id)
    return (
        cid in STRUCTURAL_FEATURE_IDS
        or any(cid.startswith(prefix) for prefix in STRUCTURAL_PREFIXES)
    )


def config_label(meta: Mapping[str, Any], *, dataset: Optional[str] = None) -> str:
    for key in ("clean_label", "surface_label", "canonical_id"):
        value = meta.get(key)
        if value:
            return str(value)

    aliases = meta.get("clean_aliases")
    if isinstance(aliases, list):
        for alias in aliases:
            if not isinstance(alias, dict):
                continue
            if dataset is not None and alias.get("dataset") not in (None, dataset):
                continue
            value = alias.get("clean_label") or alias.get("surface_label")
            if value:
                return str(value)

    canonical_ids = as_string_list(meta.get("canonical_ids"))
    return "+".join(canonical_ids) if canonical_ids else "base"


def clean_config_map_labels(
    store: CubeStore,
    *,
    config_ids: Optional[Sequence[int]] = None,
    dataset: Optional[str] = None,
) -> Dict[int, str]:
    conn = store._get_conn()
    tables = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "clean_config_map" not in tables:
        return {}
    where = ["1=1"]
    params: List[Any] = []
    if config_ids is not None:
        ids = [int(cid) for cid in config_ids]
        if not ids:
            return {}
        ph = ",".join("?" * len(ids))
        where.append(f"clean_config_id IN ({ph})")
        params.extend(ids)
    if dataset is not None:
        where.append("dataset = ?")
        params.append(dataset)
    rows = conn.execute(
        f"""
        SELECT clean_config_id, clean_label, surface_label
        FROM clean_config_map
        WHERE {' AND '.join(where)}
        ORDER BY map_id
        """,
        tuple(params),
    ).fetchall()
    out: Dict[int, str] = {}
    for row in rows:
        cid = int(row["clean_config_id"])
        label = row["clean_label"] or row["surface_label"]
        if label and cid not in out:
            out[cid] = str(label)
    return out


def matches_dataset(meta: Mapping[str, Any], dataset: Optional[str]) -> bool:
    if dataset is None:
        return True
    direct = meta.get("dataset")
    if direct:
        return direct == dataset
    datasets = set(as_string_list(meta.get("datasets")))
    if datasets:
        return dataset in datasets
    aliases = meta.get("clean_aliases")
    if isinstance(aliases, list):
        alias_datasets = {
            str(alias.get("dataset"))
            for alias in aliases
            if isinstance(alias, dict) and alias.get("dataset")
        }
        if alias_datasets:
            return dataset in alias_datasets
    return True


def query_scope_clause(
    *,
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    alias: str = "q",
) -> Tuple[List[str], List[Any]]:
    where = ["1=1"]
    params: List[Any] = []
    if dataset is not None:
        where.append(f"{alias}.dataset = ?")
        params.append(dataset)
    if split is not None:
        where.append(f"json_extract({alias}.meta, '$.split') = ?")
        params.append(split)
    return where, params
