"""SkVM evaluators — Python ports of the TypeScript eval heredocs.

Each evaluator takes the LLM response text + eval_params (the generator
cell params) and returns a list of checkpoint dicts:
    {"name": str, "score": float|None, "reason": Optional[str]}

Round-6 patch applied to structured_l3:
  - member fields split into per-field checks (name, role, email, year)
  - joined_year must be int in 2018..2024
  - unique_dept_names, unique_team_names
  - response_byte_size logged as a metric (score=None, not aggregated)
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Tuple


# ── helpers ───────────────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    """Remove leading/trailing ```json ... ``` fences if present."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text


def _cp(name: str, score: float | None, reason: str | None = None) -> Dict[str, Any]:
    return {"name": name, "score": score, "reason": reason}


# ── gen.text.structured L3 ────────────────────────────────────────────

def eval_structured_l3(response: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evaluate a structured L3 response against its generator cell.

    Args:
        response: raw LLM response text.
        params: {"D": int, "T": int, "M": int} — the generator cell.

    Returns a list of checkpoints. One checkpoint with score=None
    (``response_byte_size``) is included as an unscored metric.
    """
    D = int(params["D"])
    T = int(params["T"])
    M = int(params["M"])

    checkpoints: List[Dict[str, Any]] = []
    text = _strip_fences(response)
    byte_len = len(text.encode())

    # json_valid — short-circuit if JSON is broken
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as e:
        checkpoints.append(_cp("json_valid", 0.0, f"invalid JSON: {e}"))
        checkpoints.append(_cp("response_byte_size", None, str(byte_len)))
        return checkpoints

    checkpoints.append(_cp("json_valid", 1.0))

    # byte_size of the re-serialized canonical JSON (>5KB requirement)
    raw_bytes = len(json.dumps(doc).encode())
    size_ok = raw_bytes > 5000
    checkpoints.append(_cp(
        "byte_size",
        1.0 if size_ok else 0.0,
        None if size_ok else f"output too small: {raw_bytes} bytes",
    ))

    # field_departments
    depts = doc.get("departments", []) if isinstance(doc, dict) else []
    if not isinstance(depts, list):
        depts = []
    dept_ok = len(depts) == D
    checkpoints.append(_cp(
        "field_departments",
        1.0 if dept_ok else 0.0,
        None if dept_ok else f"expected {D} departments, got {len(depts)}",
    ))

    # field_teams: every dept has exactly T teams
    teams_ok_all = dept_ok
    teams_reason = None
    for i, d in enumerate(depts):
        if not isinstance(d, dict):
            teams_ok_all = False
            teams_reason = f"dept {i} is not an object"
            break
        teams = d.get("teams", [])
        if not isinstance(teams, list) or len(teams) != T:
            teams_ok_all = False
            teams_reason = f"dept {i}: expected {T} teams, got {len(teams) if isinstance(teams, list) else 'non-list'}"
            break
    checkpoints.append(_cp(
        "field_teams",
        1.0 if teams_ok_all else 0.0,
        teams_reason,
    ))

    # Gather all members for per-field checks
    all_members: List[Dict[str, Any]] = []
    members_count_ok = teams_ok_all
    for d in depts:
        if not isinstance(d, dict):
            continue
        for t in d.get("teams", []) or []:
            if not isinstance(t, dict):
                continue
            members = t.get("members", []) or []
            if not isinstance(members, list) or len(members) != M:
                members_count_ok = False
            for m in members:
                if isinstance(m, dict):
                    all_members.append(m)

    checkpoints.append(_cp(
        "field_members_count",
        1.0 if members_count_ok else 0.0,
        None if members_count_ok else f"at least one team has != {M} members",
    ))

    # Per-field checks on members (patched split from bundled type_correct)
    def _all_members(key: str, pred: Callable[[Any], bool]) -> Tuple[bool, str | None]:
        if not all_members:
            return False, "no members found"
        bad_idx: List[int] = [i for i, m in enumerate(all_members) if not pred(m.get(key))]
        if bad_idx:
            example = all_members[bad_idx[0]].get(key)
            return False, (
                f"{len(bad_idx)}/{len(all_members)} members failed {key} check "
                f"(e.g. idx {bad_idx[0]} → {example!r})"
            )
        return True, None

    name_ok, reason = _all_members("name", lambda v: isinstance(v, str) and v.strip() != "")
    checkpoints.append(_cp("member_has_name", 1.0 if name_ok else 0.0, reason))

    role_ok, reason = _all_members("role", lambda v: isinstance(v, str) and v.strip() != "")
    checkpoints.append(_cp("member_has_role", 1.0 if role_ok else 0.0, reason))

    email_ok, reason = _all_members("email", lambda v: isinstance(v, str) and "@" in v)
    checkpoints.append(_cp("member_has_email", 1.0 if email_ok else 0.0, reason))

    year_int_ok, reason = _all_members(
        "joined_year", lambda v: isinstance(v, int) and not isinstance(v, bool),
    )
    checkpoints.append(_cp("member_year_is_int", 1.0 if year_int_ok else 0.0, reason))

    year_range_ok, reason = _all_members(
        "joined_year",
        lambda v: isinstance(v, int) and not isinstance(v, bool) and 2018 <= v <= 2024,
    )
    checkpoints.append(_cp("member_year_in_range", 1.0 if year_range_ok else 0.0, reason))

    # uniqueness — dept names
    dept_names = [d.get("name", "") for d in depts if isinstance(d, dict)]
    dept_unique = len(dept_names) == D and len(set(dept_names)) == D
    checkpoints.append(_cp(
        "unique_dept_names",
        1.0 if dept_unique else 0.0,
        None if dept_unique else f"dept names not unique: {dept_names}",
    ))

    # uniqueness — team names (globally across all depts, expected D*T unique)
    team_names_flat: List[str] = []
    for d in depts:
        if not isinstance(d, dict):
            continue
        for t in d.get("teams", []) or []:
            if isinstance(t, dict):
                team_names_flat.append(t.get("team_name", ""))
    expected_teams = D * T
    team_unique = (
        len(team_names_flat) == expected_teams
        and len(set(team_names_flat)) == expected_teams
    )
    checkpoints.append(_cp(
        "unique_team_names",
        1.0 if team_unique else 0.0,
        None if team_unique else (
            f"team names not unique ({len(set(team_names_flat))}/{len(team_names_flat)}, "
            f"expected {expected_teams})"
        ),
    ))

    # Unscored metric — raw response byte size (useful for length-affecting features)
    checkpoints.append(_cp("response_byte_size", None, str(byte_len)))

    return checkpoints


# ── aggregation ───────────────────────────────────────────────────────

def score_checkpoints(checkpoints: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    """Aggregate scored checkpoints into a single average; record metrics separately.

    Checkpoints with score=None are treated as unscored metrics and do NOT
    count toward the average. They are returned under metrics["metrics"] as
    {name: reason_string} for downstream analysis.

    Returns (score, metrics_dict).
    """
    scored = [c for c in checkpoints if c.get("score") is not None]
    unscored = [c for c in checkpoints if c.get("score") is None]

    if not scored:
        return 0.0, {
            "status": "no_scored_checkpoints",
            "checkpoints": checkpoints,
        }

    avg = sum(float(c["score"]) for c in scored) / len(scored)
    return avg, {
        "status": "ok",
        "checkpoints": scored,
        "metrics": {c["name"]: c.get("reason") for c in unscored},
    }


# ── reason.spatial L3 ─────────────────────────────────────────────────

def eval_spatial_l3(response: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Score a great-circle distance answer.

    Tolerance mirrors the TS eval: max(5 km, expected × 0.005).
    Returns 2 checkpoints: number_found, distance_correct.
    """
    expected = int(params["expected_km"])
    tol = max(5.0, expected * 0.005)
    text = response.strip().replace(",", "")

    import re
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if not nums:
        return [
            _cp("number_found", 0.0, "no number found in response"),
            _cp("distance_correct", 0.0, "n/a — no number"),
        ]

    try:
        actual = float(nums[-1])
    except ValueError:
        return [
            _cp("number_found", 0.0, f"un-parseable number: {nums[-1]!r}"),
            _cp("distance_correct", 0.0, "n/a — unparseable"),
        ]

    checkpoints: List[Dict[str, Any]] = [_cp("number_found", 1.0)]
    ok = abs(actual - expected) <= tol
    checkpoints.append(_cp(
        "distance_correct",
        1.0 if ok else 0.0,
        None if ok else f"expected ~{expected} km (tol {tol:.1f}), got {actual}",
    ))
    return checkpoints


# ── reason.logic L2 ───────────────────────────────────────────────────

def eval_logic_l2(response: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Score a seating-puzzle answer (name match, case-insensitive).

    Returns 1 checkpoint: answer_correct.
    """
    import re
    expected = str(params["answer"]).strip().lower()
    text = response.strip()
    words = re.findall(r"[A-Za-z]+", text)
    found = any(w.lower() == expected for w in words)
    return [
        _cp(
            "answer_correct",
            1.0 if found else 0.0,
            None if found else f"expected {params['answer']!r}, got {text!r}",
        ),
    ]
