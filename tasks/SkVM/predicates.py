"""SkVM predicate extractors.

SkVM predicates are generator params stashed in ``query.meta._raw.predicates``
at seed time. Extractors just read them out and stringify. No text
heuristics needed (unlike WTQ).

Derived predicates (``total_teams``, ``total_members``, ``size_bucket``)
are computed on the fly from the primary D/T/M values.

Module import-time side effect: every predicate in ``_PRED_NAMES_STRUCTURED``
is registered with ``experiment.query_cohorts.register_extractor``.
"""
from __future__ import annotations

import json
from typing import Callable, Dict

from experiment.query_cohorts import register_extractor


def _get_pred_dict(query: dict) -> Dict[str, object]:
    """Pull the predicates sub-dict from query.meta._raw.predicates, or {}."""
    meta = query.get("meta", {})
    if isinstance(meta, str):
        meta = json.loads(meta)
    return (meta or {}).get("_raw", {}).get("predicates", {}) or {}


# ── primary predicates (direct read) ─────────────────────────────────

def _make_primary(name: str) -> Callable[[dict], str]:
    def extractor(query: dict) -> str:
        preds = _get_pred_dict(query)
        val = preds.get(name)
        return "unknown" if val is None else str(val)
    return extractor


# ── derived predicates ───────────────────────────────────────────────

def _size_bucket(total_members: int) -> str:
    if total_members < 30:
        return "S"
    if total_members <= 60:
        return "M"
    return "L"


def _total_teams(query: dict) -> str:
    preds = _get_pred_dict(query)
    if "D" in preds and "T" in preds:
        try:
            return str(int(preds["D"]) * int(preds["T"]))
        except (TypeError, ValueError):
            return "unknown"
    return "unknown"


def _total_members(query: dict) -> str:
    preds = _get_pred_dict(query)
    if all(k in preds for k in ("D", "T", "M")):
        try:
            return str(int(preds["D"]) * int(preds["T"]) * int(preds["M"]))
        except (TypeError, ValueError):
            return "unknown"
    return "unknown"


def _size_bucket_extractor(query: dict) -> str:
    preds = _get_pred_dict(query)
    if all(k in preds for k in ("D", "T", "M")):
        try:
            n = int(preds["D"]) * int(preds["T"]) * int(preds["M"])
            return _size_bucket(n)
        except (TypeError, ValueError):
            return "unknown"
    return "unknown"


# ── registration ─────────────────────────────────────────────────────
# NOTE: register_extractor is a global registry. These extractors return
# "unknown" for any query whose meta doesn't carry the expected keys
# (e.g. WTQ rows), so they are safe to leave registered globally. Callers
# that want to seed a subset can pass extractors=[...] to seed_predicates.

_PRED_NAMES_STRUCTURED_PRIMARY = ("D", "T", "M")

for _name in _PRED_NAMES_STRUCTURED_PRIMARY:
    register_extractor(_name)(_make_primary(_name))

register_extractor("total_teams")(_total_teams)
register_extractor("total_members")(_total_members)
register_extractor("size_bucket")(_size_bucket_extractor)


# Full list — exposed for callers that want to seed exactly these.
PRED_NAMES_STRUCTURED_L3 = [
    "D", "T", "M",
    "total_teams", "total_members", "size_bucket",
]


# ── reason.spatial L3 ────────────────────────────────────────────────

_PRED_NAMES_SPATIAL_L3_PRIMARY = (
    "city_a", "city_b", "hemisphere_pair", "crosses_equator",
    "distance_bucket", "expected_km",
)

for _name in _PRED_NAMES_SPATIAL_L3_PRIMARY:
    register_extractor(_name)(_make_primary(_name))

PRED_NAMES_SPATIAL_L3 = list(_PRED_NAMES_SPATIAL_L3_PRIMARY)


# ── reason.logic L2 ──────────────────────────────────────────────────

_PRED_NAMES_LOGIC_L2_PRIMARY = (
    "K", "target_pos", "n_constraints", "has_adjacency",
    "has_negation", "answer_name",
)

for _name in _PRED_NAMES_LOGIC_L2_PRIMARY:
    register_extractor(_name)(_make_primary(_name))

PRED_NAMES_LOGIC_L2 = list(_PRED_NAMES_LOGIC_L2_PRIMARY)
