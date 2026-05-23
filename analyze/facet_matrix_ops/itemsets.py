"""Feature and context itemset operators."""
from __future__ import annotations

import itertools
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Tuple


def feature_transactions(delta_df) -> Dict[int, frozenset[str]]:
    out: Dict[int, frozenset[str]] = {}
    if delta_df.empty:
        return out
    for config_id, group in delta_df.groupby("config_id"):
        out[int(config_id)] = frozenset(group["active_features"].iloc[0] or frozenset())
    return out


def context_transactions(context_df) -> Dict[str, frozenset[str]]:
    if context_df.empty:
        return {}
    return {
        str(row["query_id"]): frozenset(row["context_atoms"] or frozenset())
        for _, row in context_df.iterrows()
    }


def frequent_itemsets(
    transactions: Mapping[Any, Iterable[str]],
    *,
    max_order: int = 3,
    min_support: int = 20,
    max_candidates_per_level: int = 50000,
) -> List[Tuple[str, ...]]:
    """Discover frequent conjunctions with a small Apriori-style pass."""
    if max_order < 1:
        return []
    tx = [frozenset(items) for items in transactions.values()]
    if not tx:
        return []

    singleton_counts: Counter[str] = Counter()
    for items in tx:
        singleton_counts.update(items)
    current = {
        frozenset([item])
        for item, count in singleton_counts.items()
        if count >= min_support
    }
    out = list(current)

    order = 2
    while current and order <= max_order:
        current_lookup = set(current)
        current_list = sorted(current, key=lambda items: tuple(sorted(items)))
        candidates: set[frozenset[str]] = set()
        for i, left in enumerate(current_list):
            for right in current_list[i + 1:]:
                merged = left | right
                if len(merged) != order:
                    continue
                if all(
                    frozenset(subset) in current_lookup
                    for subset in itertools.combinations(merged, order - 1)
                ):
                    candidates.add(merged)
                if len(candidates) >= max_candidates_per_level:
                    break
            if len(candidates) >= max_candidates_per_level:
                break

        counts: Counter[frozenset[str]] = Counter()
        for items in tx:
            for candidate in candidates:
                if candidate.issubset(items):
                    counts[candidate] += 1
        current = {
            candidate for candidate, count in counts.items()
            if count >= min_support
        }
        out.extend(current)
        order += 1

    return [
        tuple(sorted(items))
        for items in sorted(out, key=lambda items: (len(items), tuple(sorted(items))))
    ]


def feature_itemsets_from_delta(
    delta_df,
    *,
    max_order: int = 3,
    min_config_support: int = 1,
) -> List[Tuple[str, ...]]:
    return frequent_itemsets(
        feature_transactions(delta_df),
        max_order=max_order,
        min_support=min_config_support,
    )


def context_itemsets_from_context(
    context_df,
    *,
    max_order: int = 3,
    min_query_support: int = 20,
    max_candidates_per_level: int = 50000,
) -> List[Tuple[str, ...]]:
    return frequent_itemsets(
        context_transactions(context_df),
        max_order=max_order,
        min_support=min_query_support,
        max_candidates_per_level=max_candidates_per_level,
    )
