"""Config generators — produce experiment configs from a pool of funcs.

Two kinds of generators coexist:

  * rule-level generators (legacy): take ``rule_ids: List[str]`` (individual func_ids)
      - add_one, leave_one_out, coalition, base_only

  * feature-level generators (Phase 1 onward): take ``bundles: Dict[canonical_id, (feature_id_hash, [func_id, ...])]``
    where each bundle is all incremental func_ids contributed by that feature
    over the base config.
      - add_one_feature, leave_one_out_feature, coalition_feature, base_only

``generate()`` dispatches by name; rule-level vs feature-level is a property
of the registered function (it reads the kwargs it needs).
"""
from __future__ import annotations

import logging
import random
from typing import Any, Callable, Dict, List, Tuple

from core.store import CubeStore

logger = logging.getLogger(__name__)

ConfigEntry = Tuple[int, List[str], Dict[str, Any]]  # (config_id, func_ids, meta)
FeatureBundles = Dict[str, Tuple[str, List[str]]]   # canonical_id -> (feature_id_hash, add_funcs)
ConflictsMap   = Dict[str, frozenset]               # canonical_id -> frozenset[canonical_ids it conflicts with]

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
    *,
    base_ids: List[str],
    rule_ids: List[str] | None = None,
    bundles: FeatureBundles | None = None,
    conflicts: ConflictsMap | None = None,
    base_canonical_ids: List[str] | None = None,
    base_feature_ids: List[str] | None = None,
    **params,
) -> List[ConfigEntry]:
    """Dispatch to a registered config generator by name.

    Generators either consume ``rule_ids`` (rule-level) or ``bundles``
    (feature-level); pass whichever your generator expects.

    ``conflicts`` (feature-level only): canonical_id -> frozenset[canonical_id]
    mapping of declared conflicts_with relations.  Feature-aware generators
    (``leave_one_out_feature``, ``coalition_feature``) use it to avoid
    emitting configs that activate both sides of a conflict.

    ``base_canonical_ids`` / ``base_feature_ids`` (feature-level only):
    the canonical_ids / feature_id hashes that compose the base config.
    Feature-aware generators write ``meta.feature_ids`` / ``meta.canonical_ids``
    as the UNION of base + config-specific, so that downstream joins
    (e.g. ``feature_effect`` view, ``has_feature`` filter) reach every
    feature that actually influenced the prompt — not just the delta.
    """
    fn = REGISTRY.get(name)
    if fn is None:
        raise ValueError(f"Unknown generator: {name!r}. Available: {list(REGISTRY.keys())}")
    # Forward all kwargs; generators ignore what they don't need.
    return fn(
        store,
        base_ids=base_ids, rule_ids=rule_ids,
        bundles=bundles, conflicts=conflicts,
        base_canonical_ids=base_canonical_ids or [],
        base_feature_ids=base_feature_ids or [],
        **params,
    )


# ── shared conflict helper ────────────────────────────────────────────

def _resolve_conflicts(
    active: List[str],
    conflicts: ConflictsMap | None,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Resolve live conflicts deterministically.

    For each conflict pair where both sides are in ``active``, drop the
    lexicographically-larger canonical_id. Returns (kept, dropped_pairs)
    where dropped_pairs lists (dropped, kept_counterpart) for meta.

    The ordering rule is deterministic and stable across runs so that
    the same generator invocation produces the same resolved configs.
    """
    if not conflicts:
        return list(active), []

    kept = list(active)
    dropped_pairs: List[Tuple[str, str]] = []
    # Iterate in sorted order so the resolution is deterministic regardless
    # of input order.
    for cid in sorted(kept):
        if cid not in kept:
            continue
        for other in sorted(conflicts.get(cid, frozenset())):
            if other in kept and other != cid:
                loser = max(cid, other)
                winner = min(cid, other)
                dropped_pairs.append((loser, winner))
                kept.remove(loser)
    # Preserve original order for remaining entries.
    kept_set = set(kept)
    return [c for c in active if c in kept_set], dropped_pairs


def _has_live_conflict(active: List[str], conflicts: ConflictsMap | None) -> bool:
    """Return True if any declared conflict pair is fully active."""
    if not conflicts:
        return False
    active_set = set(active)
    for cid in active_set:
        if active_set & conflicts.get(cid, frozenset()):
            return True
    return False


# ── generators ────────────────────────────────────────────────────────

@register("base_only")
def base_only(
    store: CubeStore,
    *,
    base_ids: List[str],
    rule_ids: List[str] | None = None,
    bundles: FeatureBundles | None = None,
    **_,
) -> List[ConfigEntry]:
    """Run only the base config, no add-ones."""
    return []


@register("add_one")
def add_one(
    store: CubeStore,
    *,
    base_ids: List[str],
    rule_ids: List[str],
    max_rules: int = 0,
    **_,
) -> List[ConfigEntry]:
    """Base + one rule each (rule-level, legacy)."""
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
    *,
    base_ids: List[str],
    rule_ids: List[str],
    max_rules: int = 0,
    **_,
) -> List[ConfigEntry]:
    """All rules minus one each (rule-level, legacy)."""
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
    *,
    base_ids: List[str],
    rule_ids: List[str],
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


# ══════════════════════════════════════════════════════════════════════
# Feature-level generators — operate on bundles[canonical_id] = (fid_hash, [func_id, ...])
# ══════════════════════════════════════════════════════════════════════


@register("add_one_feature")
def add_one_feature(
    store: CubeStore,
    *,
    base_ids: List[str],
    bundles: FeatureBundles,
    base_canonical_ids: List[str] | None = None,
    base_feature_ids: List[str] | None = None,
    **_,
) -> List[ConfigEntry]:
    """Base + one feature bundle each.

    For each canonical feature in ``bundles``, produce one config whose
    func_ids are ``base_ids + incremental_funcs_of_that_feature``.
    ``meta.feature_ids`` / ``meta.canonical_ids`` hold the FULL active
    feature set for the config (base + added), so downstream filters and
    joins (``has_feature``, ``feature_effect`` view) see every feature
    that influenced the prompt.
    """
    if bundles is None or not bundles:
        raise ValueError("add_one_feature: 'bundles' is required and non-empty")
    base_canonical_ids = base_canonical_ids or []
    base_feature_ids = base_feature_ids or []

    configs: List[ConfigEntry] = []
    for canonical_id, (feature_hash, add_funcs) in bundles.items():
        if not add_funcs:
            logger.warning(
                "add_one_feature: feature %r has no incremental funcs over base; skipping",
                canonical_id,
            )
            continue
        func_ids = base_ids + add_funcs
        all_canonical_ids = list(base_canonical_ids) + [canonical_id]
        all_feature_ids   = list(base_feature_ids)   + [feature_hash]
        cid = store.get_or_create_config(
            func_ids,
            meta={
                "kind":          "add_one_feature",
                "canonical_id":  canonical_id,     # the single feature added
                "feature_id":    feature_hash,
                "canonical_ids": all_canonical_ids,  # FULL active set
                "feature_ids":   all_feature_ids,
                "added_funcs":   add_funcs,
            },
        )
        configs.append((cid, func_ids, {
            "experiment":     "add_one_feature",
            "canonical_id":   canonical_id,
            "feature_id":     feature_hash,
            "canonical_ids":  all_canonical_ids,
            "feature_ids":    all_feature_ids,
        }))
    logger.info("add_one_feature: %d configs generated", len(configs))
    return configs


@register("leave_one_out_feature")
def leave_one_out_feature(
    store: CubeStore,
    *,
    base_ids: List[str],
    bundles: FeatureBundles,
    conflicts: ConflictsMap | None = None,
    base_canonical_ids: List[str] | None = None,
    base_feature_ids: List[str] | None = None,
    **_,
) -> List[ConfigEntry]:
    """All feature bundles minus one each.

    When ``conflicts`` is provided, live conflict pairs in the remaining set
    are resolved by dropping the lexicographically-larger canonical_id.
    Dropped features are recorded in ``meta.conflict_resolutions``.
    """
    if bundles is None or not bundles:
        raise ValueError("leave_one_out_feature: 'bundles' is required and non-empty")
    base_canonical_ids = base_canonical_ids or []
    base_feature_ids = base_feature_ids or []

    all_feature_cids = list(bundles.keys())

    configs: List[ConfigEntry] = []
    for canonical_id, (feature_hash, _add_funcs) in bundles.items():
        active_cids = [c for c in all_feature_cids if c != canonical_id]
        kept_cids, dropped_pairs = _resolve_conflicts(active_cids, conflicts)

        # Compose func_ids from the kept feature set (dedup preserving order).
        seen: set = set(base_ids)
        func_ids: List[str] = list(base_ids)
        for k in kept_cids:
            for f in bundles[k][1]:
                if f not in seen:
                    seen.add(f)
                    func_ids.append(f)

        all_canonical_ids = list(base_canonical_ids) + list(kept_cids)
        all_feature_ids = list(base_feature_ids) + [bundles[k][0] for k in kept_cids]

        meta_common = {
            "kind":                  "leave_one_out_feature",
            "removed_canonical_id":  canonical_id,
            "removed_feature_id":    feature_hash,
            "active_canonical_ids":  kept_cids,
            "canonical_ids":         all_canonical_ids,  # FULL active set
            "feature_ids":           all_feature_ids,
        }
        if dropped_pairs:
            meta_common["conflict_resolutions"] = [
                {"dropped": loser, "kept": winner}
                for loser, winner in dropped_pairs
            ]

        cid = store.get_or_create_config(func_ids, meta=meta_common)
        configs.append((cid, func_ids, {
            "experiment":            "leave_one_out_feature",
            "removed_canonical_id":  canonical_id,
            "removed_feature_id":    feature_hash,
            "active_canonical_ids":  kept_cids,
            "canonical_ids":         all_canonical_ids,
            "feature_ids":           all_feature_ids,
            "conflict_resolutions":  meta_common.get("conflict_resolutions", []),
        }))
    logger.info("leave_one_out_feature: %d configs generated%s",
                len(configs),
                "" if conflicts is None else " (conflict-aware)")
    return configs


@register("coalition_feature")
def coalition_feature(
    store: CubeStore,
    *,
    base_ids: List[str],
    bundles: FeatureBundles,
    conflicts: ConflictsMap | None = None,
    base_canonical_ids: List[str] | None = None,
    base_feature_ids: List[str] | None = None,
    n_samples: int = 500,
    seed: int = 42,
    min_features: int = 1,
    max_features: int = 0,
    **_,
) -> List[ConfigEntry]:
    """Random coalitions of feature bundles (Shapley-style sampling).

    Each sampled coalition contributes a config whose func_ids are
    ``base_ids + ⋃ add_funcs for each sampled feature``.

    When ``conflicts`` is provided, subsets containing a live conflict pair
    are REJECTED (not silently resolved).  This keeps coalition semantics
    honest: a rejected subset means "this combination can't exist," not
    "this combination minus one side."
    """
    if bundles is None or not bundles:
        raise ValueError("coalition_feature: 'bundles' is required and non-empty")
    base_canonical_ids = base_canonical_ids or []
    base_feature_ids = base_feature_ids or []

    rng = random.Random(seed)
    names = list(bundles.keys())
    n_features = len(names)
    max_f = max_features if max_features > 0 else n_features
    max_f = min(max_f, n_features)

    def _is_valid(subset) -> bool:
        return not _has_live_conflict(list(subset), conflicts)

    total_possible = sum(_comb(n_features, k) for k in range(min_features, max_f + 1))
    _ENUM_LIMIT = 100_000

    def _configs_from_subsets(subsets):
        out: List[ConfigEntry] = []
        for subset in subsets:
            seen_fids: set = set(base_ids)
            func_ids = list(base_ids)
            for feat_name in subset:
                for f in bundles[feat_name][1]:
                    if f not in seen_fids:
                        seen_fids.add(f)
                        func_ids.append(f)

            all_canonical_ids = list(base_canonical_ids) + list(subset)
            all_feature_ids = list(base_feature_ids) + [bundles[n][0] for n in subset]

            cid = store.get_or_create_config(
                func_ids,
                meta={
                    "kind":            "coalition_feature",
                    "subset_canonical_ids": list(subset),  # the varying part
                    "subset_feature_ids":   [bundles[n][0] for n in subset],
                    "canonical_ids":   all_canonical_ids,  # FULL active set
                    "feature_ids":     all_feature_ids,
                },
            )
            out.append((cid, func_ids, {
                "experiment":      "coalition_feature",
                "n_features":      len(subset),
                "subset_canonical_ids": list(subset),
                "subset_feature_ids":   [bundles[n][0] for n in subset],
                "canonical_ids":   all_canonical_ids,
                "feature_ids":     all_feature_ids,
            }))
        return out

    # ── full enumeration branch ───────────────────────────────────────
    if n_samples >= total_possible:
        from itertools import combinations
        all_subsets = [
            s for k in range(min_features, max_f + 1) for s in combinations(names, k)
        ]
        pre_filter = len(all_subsets)
        all_subsets = [s for s in all_subsets if _is_valid(s)]
        rng.shuffle(all_subsets)
        configs = _configs_from_subsets(all_subsets)
        logger.info(
            "coalition_feature: full enumeration %d configs (pool=%d features, "
            "%d subsets rejected for conflicts)",
            len(configs), n_features, pre_filter - len(all_subsets),
        )
        return configs

    # ── shuffled truncation branch ────────────────────────────────────
    if total_possible <= _ENUM_LIMIT and n_samples > total_possible * 0.7:
        from itertools import combinations
        all_subsets = [
            s for k in range(min_features, max_f + 1) for s in combinations(names, k)
        ]
        all_subsets = [s for s in all_subsets if _is_valid(s)]
        rng.shuffle(all_subsets)
        all_subsets = all_subsets[:n_samples]
        configs = _configs_from_subsets(all_subsets)
        logger.info(
            "coalition_feature: shuffled enumeration, %d configs (pool=%d features)",
            len(configs), n_features,
        )
        return configs

    # ── rejection sampling branch ─────────────────────────────────────
    seen_subsets: set = set()
    subsets: List[tuple] = []
    n_rejected_conflict = 0
    max_rejections = max(n_samples * 50, 10_000)
    rejections = 0
    while len(subsets) < n_samples:
        if rejections > max_rejections:
            logger.warning(
                "coalition_feature: reached %d rejections without reaching n_samples=%d; "
                "returning %d configs (pool may be too constrained by conflicts)",
                rejections, n_samples, len(subsets),
            )
            break
        k = rng.randint(min_features, max_f)
        subset = tuple(sorted(rng.sample(names, k)))
        if subset in seen_subsets:
            rejections += 1
            continue
        if not _is_valid(subset):
            n_rejected_conflict += 1
            rejections += 1
            continue
        seen_subsets.add(subset)
        subsets.append(subset)
    configs = _configs_from_subsets(subsets)
    logger.info(
        "coalition_feature: sampled %d configs (pool=%d features, total_possible=%d, "
        "%d rejected for conflicts)",
        len(configs), n_features, total_possible, n_rejected_conflict,
    )
    return configs


# ══════════════════════════════════════════════════════════════════════
# Private helpers
# ══════════════════════════════════════════════════════════════════════


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
