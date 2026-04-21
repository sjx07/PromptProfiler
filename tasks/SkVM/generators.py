"""SkVM generators — Python port of the TypeScript microbenchmark generators.

One function per (primitive, level). Each takes generator parameters plus
an optional seed and returns an Instance with prompt + eval_params.

Wired:
  - gen.text.structured L3 — deeply nested org JSON (27-cell balanced grid)
  - reason.spatial L3      — great-circle distance (8 cities, 56 ordered pairs)
  - reason.logic L2        — seating arrangement (9-cell balanced grid)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Tuple


@dataclass
class Instance:
    """One generator output.

    Attributes:
        prompt:       LLM-facing prompt body.
        eval_params:  values needed by the evaluator.
        predicates:   generator-derived slicing predicates (copy goes to
                      query.meta._raw.predicates at seed time).
    """
    prompt: str
    eval_params: Dict[str, Any] = field(default_factory=dict)
    predicates: Dict[str, Any] = field(default_factory=dict)


# ── gen.text.structured L3 ────────────────────────────────────────────


_STRUCTURED_L3_PROMPT = (
    "Respond with a JSON object representing an organization with this structure:\n"
    '- "org_name": "GlobalTech"\n'
    '- "departments": an array of {D} department objects, each with:\n'
    '  - "name": a unique department name\n'
    "  - \"head\": a person's name\n"
    '  - "teams": an array of {T} team objects, each with:\n'
    '    - "team_name": a unique team name\n'
    "    - \"lead\": a person's name\n"
    '    - "members": an array of {M} member objects, each with:\n'
    "      - \"name\": a person's name\n"
    '      - "role": a job title\n'
    '      - "email": an email address\n'
    '      - "joined_year": a year between 2018-2024\n\n'
    "The total output must be valid JSON and larger than 5KB. "
    "Provide ONLY the JSON, nothing else."
)


def generate_structured_l3(D: int, T: int, M: int, seed: int | None = None) -> Instance:
    """Generate one `gen.text.structured` L3 instance.

    Args:
        D: number of departments (3-5 in the default grid).
        T: number of teams per department (2-4).
        M: number of members per team (3-5).
        seed: reserved for future use. The L3 prompt is fully determined by
              (D, T, M); seed is accepted so seed-bearing callers can pass
              it through uniformly.

    Returns:
        Instance(prompt=<rendered prompt>, eval_params={"D": D, "T": T, "M": M})

    Raises:
        ValueError if any of D, T, M is outside its supported range.
    """
    if not (3 <= D <= 5):
        raise ValueError(f"D must be in {{3,4,5}}, got {D}")
    if not (2 <= T <= 4):
        raise ValueError(f"T must be in {{2,3,4}}, got {T}")
    if not (3 <= M <= 5):
        raise ValueError(f"M must be in {{3,4,5}}, got {M}")

    prompt = _STRUCTURED_L3_PROMPT.format(D=D, T=T, M=M)
    preds = {
        "D": D, "T": T, "M": M,
        "total_teams": D * T,
        "total_members": D * T * M,
        "size_bucket": "S" if D*T*M < 30 else ("M" if D*T*M <= 60 else "L"),
    }
    return Instance(
        prompt=prompt,
        eval_params={"D": D, "T": T, "M": M},
        predicates=preds,
    )


def iter_structured_l3_grid(
    *, per_cell: int = 148, seed_base: int = 42,
) -> Iterator[Tuple[int, int, int, int, int]]:
    """Yield balanced (D, T, M, instance_idx, seed) tuples for the 27-cell grid.

    At per_cell=148 the total is 27 × 148 = 3996 instances.
    Seed for instance i of cell (D, T, M) = seed_base + cell_offset*per_cell + i,
    keeping seeds unique across cells even though the L3 prompt doesn't use
    them (reserved for future primitives where generation is seed-dependent).
    """
    cells = [
        (D, T, M)
        for D in (3, 4, 5)
        for T in (2, 3, 4)
        for M in (3, 4, 5)
    ]
    for cell_idx, (D, T, M) in enumerate(cells):
        for i in range(per_cell):
            seed = seed_base + cell_idx * per_cell + i
            yield D, T, M, i, seed


# ── reason.spatial L3 ─────────────────────────────────────────────────
# Great-circle distance between two of 8 fixed cities.
# The generator is fully determined by (cityA, cityB) — no seeded randomness.
# There are only 8·7 = 56 ordered pairs, so the native dataset is 56 rows.

_SPATIAL_CITIES: List[Dict[str, Any]] = [
    {"name": "New York",  "lat": 40.7128,  "lon": -74.006},
    {"name": "London",    "lat": 51.5074,  "lon": -0.1278},
    {"name": "Tokyo",     "lat": 35.6762,  "lon": 139.6503},
    {"name": "Sydney",    "lat": -33.8688, "lon": 151.2093},
    {"name": "Paris",     "lat": 48.8566,  "lon": 2.3522},
    {"name": "Cairo",     "lat": 30.0444,  "lon": 31.2357},
    {"name": "Mumbai",    "lat": 19.076,   "lon": 72.8777},
    {"name": "Sao Paulo", "lat": -23.5505, "lon": -46.6333},
]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> int:
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return int(round(R * c))


def generate_spatial_l3(city_a_idx: int, city_b_idx: int, seed: int | None = None) -> Instance:
    """Generate one reason.spatial L3 instance.

    Args:
        city_a_idx, city_b_idx: indices into _SPATIAL_CITIES, 0..7.
        seed: unused for this primitive (kept for API uniformity).

    Returns:
        Instance with prompt, eval_params (expected_km), and predicates.
    """
    if not (0 <= city_a_idx < 8 and 0 <= city_b_idx < 8):
        raise ValueError(f"city_*_idx must be in 0..7, got ({city_a_idx}, {city_b_idx})")
    if city_a_idx == city_b_idx:
        raise ValueError(f"city_a_idx == city_b_idx == {city_a_idx}")

    ca = _SPATIAL_CITIES[city_a_idx]
    cb = _SPATIAL_CITIES[city_b_idx]
    expected = _haversine_km(ca["lat"], ca["lon"], cb["lat"], cb["lon"])

    prompt = (
        f"City A ({ca['name']}) is at latitude {ca['lat']}, longitude {ca['lon']}. "
        f"City B ({cb['name']}) is at latitude {cb['lat']}, longitude {cb['lon']}. "
        "What is the great-circle distance between them in kilometers? "
        "Answer with just the number (integer), nothing else."
    )

    hemi_a = "N" if ca["lat"] >= 0 else "S"
    hemi_b = "N" if cb["lat"] >= 0 else "S"
    hemisphere_pair = "".join(sorted([hemi_a, hemi_b]))  # NN / NS / SS
    crosses_equator = "yes" if hemi_a != hemi_b else "no"

    if expected < 3000:
        dist_bucket = "near"
    elif expected < 10000:
        dist_bucket = "mid"
    else:
        dist_bucket = "far"

    preds = {
        "city_a": ca["name"],
        "city_b": cb["name"],
        "hemisphere_pair": hemisphere_pair,
        "crosses_equator": crosses_equator,
        "distance_bucket": dist_bucket,
        "expected_km": str(expected),
    }
    return Instance(
        prompt=prompt,
        eval_params={"expected_km": expected},
        predicates=preds,
    )


def iter_spatial_l3_grid() -> Iterator[Tuple[int, int, int]]:
    """Yield (city_a_idx, city_b_idx, pair_idx) for all 56 ordered pairs."""
    pair_idx = 0
    for a in range(8):
        for b in range(8):
            if a == b:
                continue
            yield a, b, pair_idx
            pair_idx += 1


# ── reason.logic L2 ───────────────────────────────────────────────────
# Seating arrangement puzzle. K ∈ {4,5}, target_pos ∈ {1..K}.
# Balanced grid = 9 cells. Seed controls name shuffle, arrangement, and
# per-constraint type choices — replicates the TS generator's structure.

_LOGIC_L2_NAMES = ("Alice", "Bob", "Carol", "Dave", "Eve")


def generate_logic_l2(K: int, target_pos: int, seed: int) -> Instance:
    """Generate one reason.logic L2 instance.

    Args:
        K: 4 or 5 — number of people.
        target_pos: 1..K — position whose occupant is asked.
        seed: randomness for name shuffle, arrangement, constraint types.

    Returns:
        Instance with prompt, eval_params (K, target_pos, answer), predicates.
    """
    if K not in (4, 5):
        raise ValueError(f"K must be 4 or 5, got {K}")
    if not (1 <= target_pos <= K):
        raise ValueError(f"target_pos must be in 1..{K}, got {target_pos}")

    rng = random.Random(seed)

    # Pick K names in random order — mirrors TS shuffle(names).slice(0, K)
    names = list(_LOGIC_L2_NAMES)
    rng.shuffle(names)
    people = names[:K]

    # A random valid arrangement of those K names
    arrangement = list(people)
    rng.shuffle(arrangement)
    answer = arrangement[target_pos - 1]

    # Generate constraints — mirrors the TS branch logic exactly
    constraints: List[str] = []
    fixed_indices = list(range(K))
    rng.shuffle(fixed_indices)
    fixed_indices = fixed_indices[:min(K, 4)]

    for idx in fixed_indices:
        person = arrangement[idx]
        pos = idx + 1
        constraint_type = rng.choice(("direct", "adjacent", "not"))

        if constraint_type == "direct" or len(constraints) < 2:
            constraints.append(f"{person} sits at position {pos}.")
        elif constraint_type == "adjacent" and idx < K - 1:
            constraints.append(
                f"{person} sits immediately before {arrangement[idx + 1]}."
            )
        else:
            wrong_pos = ((idx + rng.randint(1, K - 1)) % K) + 1
            constraints.append(f"{person} does not sit at position {wrong_pos}.")
            constraints.append(f"{person} sits at position {pos}.")

    c_lines = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(constraints))
    prompt = (
        f"{K} people sit at positions 1 to {K}: {', '.join(people)}.\n\n"
        f"Constraints:\n{c_lines}\n\n"
        f"Who sits at position {target_pos}? Answer with just the name, nothing else."
    )

    has_adjacency = any("immediately before" in c for c in constraints)
    has_negation = any("does not sit" in c for c in constraints)
    n_constraints = len(constraints)

    preds = {
        "K": K,
        "target_pos": target_pos,
        "n_constraints": n_constraints,
        "has_adjacency": "yes" if has_adjacency else "no",
        "has_negation": "yes" if has_negation else "no",
        "answer_name": answer,
    }
    return Instance(
        prompt=prompt,
        eval_params={"K": K, "target_pos": target_pos, "answer": answer},
        predicates=preds,
    )


def iter_logic_l2_grid(
    *, per_cell: int = 444, seed_base: int = 42,
) -> Iterator[Tuple[int, int, int, int]]:
    """Yield balanced (K, target_pos, instance_idx, seed) tuples.

    9 cells = (K=4, target∈{1..4}) + (K=5, target∈{1..5}).
    per_cell=444 yields 9 × 444 = 3996 instances.
    """
    cells: List[Tuple[int, int]] = (
        [(4, p) for p in range(1, 5)] + [(5, p) for p in range(1, 6)]
    )
    for cell_idx, (K, target_pos) in enumerate(cells):
        for i in range(per_cell):
            seed = seed_base + cell_idx * per_cell + i
            yield K, target_pos, i, seed
