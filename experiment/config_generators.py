"""Config generators — produce experiment configs from a pool of funcs.

Each generator is registered via @register and follows the same signature:
    generate(name, store, base_ids, rule_ids, **params) → list[ConfigEntry]

Generators:
    add_one      — base + one rule each
    leave_one_out — all rules minus one each
    coalition    — random subsets of rules (Shapley-style sampling)
"""
from __future__ import annotations

import hashlib
import logging
import random
from typing import Any, Callable, Dict, List, Tuple

from prompt_profiler.core.store import CubeStore

logger = logging.getLogger(__name__)

ConfigEntry = Tuple[int, List[str], Dict[str, Any]]  # (config_id, func_ids, meta)

GeneratorFn = Callable[..., List[ConfigEntry]]

# ── registry ──────────────────────────────────────────────────────────

REGISTRY: Dict[str, GeneratorFn] = {}


def register(name: str):
    """Decorator to register a config generator."""
    def wrapper(fn: GeneratorFn) -> GeneratorFn:
        REGISTRY[name] = fn
        return fn
    return wrapper


def generate(
    name: str,
    store: CubeStore,
    base_ids: List[str],
    rule_ids: List[str],
    **params,
) -> List[ConfigEntry]:
    """Dispatch to a registered config generator by name."""
    fn = REGISTRY.get(name)
    if fn is None:
        raise ValueError(f"Unknown generator: {name!r}. Available: {list(REGISTRY.keys())}")
    return fn(store, base_ids, rule_ids, **params)


# ── generators ────────────────────────────────────────────────────────

@register("base_only")
def base_only(
    store: CubeStore,
    base_ids: List[str],
    rule_ids: List[str],
    **_,
) -> List[ConfigEntry]:
    """Run only the base config, no add-ones."""
    return []


@register("add_one")
def add_one(
    store: CubeStore,
    base_ids: List[str],
    rule_ids: List[str],
    *,
    max_rules: int = 0,
    **_,
) -> List[ConfigEntry]:
    """Base + one rule each."""
    selected = rule_ids if max_rules <= 0 else rule_ids[:max_rules]
    configs: List[ConfigEntry] = []
    for rid in selected:
        func_ids = base_ids + [rid]
        cid = store.get_or_create_config(func_ids)
        configs.append((cid, func_ids, {"experiment": "add_one", "added_rule": rid}))
    logger.info("add_one: %d configs generated", len(configs))
    return configs


@register("leave_one_out")
def leave_one_out(
    store: CubeStore,
    base_ids: List[str],
    rule_ids: List[str],
    *,
    max_rules: int = 0,
    **_,
) -> List[ConfigEntry]:
    """All rules minus one each."""
    selected = rule_ids if max_rules <= 0 else rule_ids[:max_rules]
    configs: List[ConfigEntry] = []
    for rid in selected:
        func_ids = base_ids + [r for r in rule_ids if r != rid]
        cid = store.get_or_create_config(func_ids)
        configs.append((cid, func_ids, {"experiment": "leave_one_out", "removed_rule": rid}))
    logger.info("leave_one_out: %d configs generated", len(configs))
    return configs


@register("coalition")
def coalition(
    store: CubeStore,
    base_ids: List[str],
    rule_ids: List[str],
    *,
    n_samples: int = 500,
    seed: int = 42,
    min_rules: int = 1,
    max_rules: int = 0,
    **_,
) -> List[ConfigEntry]:
    """Random coalitions of rules (for Shapley value estimation).

    Samples random subsets of rule_ids, each added to base_ids.
    Strategy:
        - If n_samples >= total possible subsets: enumerate all
        - If n_samples >= 50% of total: generate all, shuffle, truncate
        - Otherwise: sample with rejection (while loop)
    """
    rng = random.Random(seed)
    n_rules = len(rule_ids)
    max_r = max_rules if max_rules > 0 else n_rules
    max_r = min(max_r, n_rules)

    # Estimate total possible subsets
    total_possible = sum(
        _comb(n_rules, k) for k in range(min_rules, max_r + 1)
    )

    # Strategy selection:
    #   - n >= total: enumerate all
    #   - n/total > 0.7 AND total <= 100K: enumerate, shuffle, truncate
    #     (rejection sampling > 2x overhead beyond 70% fill)
    #   - else: rejection sampling
    _ENUM_LIMIT = 100_000  # max subsets to hold in memory

    if n_samples >= total_possible:
        configs = _enumerate_all(store, base_ids, rule_ids, min_rules, max_r)
        rng.shuffle(configs)
        logger.info("coalition: full enumeration %d configs (pool=%d rules)",
                     len(configs), n_rules)
        return configs

    if total_possible <= _ENUM_LIMIT and n_samples > total_possible * 0.7:
        configs = _enumerate_all(store, base_ids, rule_ids, min_rules, max_r)
        rng.shuffle(configs)
        configs = configs[:n_samples]
        logger.info("coalition: shuffled enumeration, %d/%d configs (pool=%d rules)",
                     len(configs), total_possible, n_rules)
        return configs

    # Sparse sampling with rejection
    seen: set = set()
    configs: List[ConfigEntry] = []

    while len(configs) < n_samples:
        k = rng.randint(min_rules, max_r)
        subset = tuple(sorted(rng.sample(rule_ids, k)))

        if subset in seen:
            continue
        seen.add(subset)

        func_ids = base_ids + list(subset)
        cid = store.get_or_create_config(func_ids)
        configs.append((cid, func_ids, {
            "experiment": "coalition",
            "n_rules": len(subset),
            "rule_ids": list(subset),
        }))

    logger.info("coalition: sampled %d configs (pool=%d rules, total_possible=%d)",
                len(configs), n_rules, total_possible)
    return configs


def _comb(n: int, k: int) -> int:
    """Binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def _enumerate_all(
    store: CubeStore,
    base_ids: List[str],
    rule_ids: List[str],
    min_k: int,
    max_k: int,
) -> List[ConfigEntry]:
    """Generate all subsets of rule_ids with size in [min_k, max_k]."""
    from itertools import combinations
    configs: List[ConfigEntry] = []
    for k in range(min_k, max_k + 1):
        for subset in combinations(rule_ids, k):
            func_ids = base_ids + list(subset)
            cid = store.get_or_create_config(func_ids)
            configs.append((cid, func_ids, {
                "experiment": "coalition",
                "n_rules": k,
                "rule_ids": list(subset),
            }))
    return configs
