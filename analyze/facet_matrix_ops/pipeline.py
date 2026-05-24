"""Chainable FACET matrix operator pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

from core.store import CubeStore

from .config import infer_baseline_config_id
from .context import query_context_sets
from .itemsets import context_itemsets_from_context, feature_itemsets_from_delta
from .outcomes import paired_delta_rows
from .scoring import (
    score_binary_contextual_rules,
    score_contextual_feature_itemsets,
    score_feature_itemsets,
)


@dataclass(frozen=True)
class FacetScope:
    """Benchmark/model/scorer scope for a FACET operator chain."""

    model: str
    scorer: str
    dataset: str
    split: Optional[str] = None
    base_config_id: Optional[int] = None


@dataclass
class FacetChainState:
    """Mutable state accumulated by :class:`FacetOperatorChain`."""

    base_config_id: Optional[int] = None
    delta_rows: Any = None
    context_rows: Any = None
    feature_itemsets: Optional[list[tuple[str, ...]]] = None
    context_itemsets: Optional[list[tuple[str, ...]]] = None
    global_effects: Any = None
    conditional_effects: Any = None
    binary_rules: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "base_config_id": self.base_config_id,
            "delta_rows": self.delta_rows,
            "context_rows": self.context_rows,
            "feature_itemsets": self.feature_itemsets,
            "context_itemsets": self.context_itemsets,
            "global_effects": self.global_effects,
            "conditional_effects": self.conditional_effects,
            "binary_rules": self.binary_rules,
            "meta": dict(self.meta),
        }


class FacetOperatorChain:
    """Chainable facade over FACET matrix primitives.

    The class is intentionally thin: each method runs one operator, stores its
    output in ``state``, and returns ``self`` so experiments can be composed as
    readable operator chains.
    """

    def __init__(self, store: CubeStore, scope: FacetScope):
        self.store = store
        self.scope = scope
        self.state = FacetChainState(base_config_id=scope.base_config_id)

    @classmethod
    def for_scope(
        cls,
        store: CubeStore,
        *,
        model: str,
        scorer: str,
        dataset: str,
        split: Optional[str] = None,
        base_config_id: Optional[int] = None,
    ) -> "FacetOperatorChain":
        return cls(
            store,
            FacetScope(
                model=model,
                scorer=scorer,
                dataset=dataset,
                split=split,
                base_config_id=base_config_id,
            ),
        )

    def with_baseline(self, *, label: str = "base") -> "FacetOperatorChain":
        if self.state.base_config_id is None:
            self.state.base_config_id = infer_baseline_config_id(
                self.store,
                model=self.scope.model,
                scorer=self.scope.scorer,
                dataset=self.scope.dataset,
                split=self.scope.split,
                label=label,
            )
        return self

    def with_paired_deltas(
        self,
        *,
        config_ids: Optional[Sequence[int]] = None,
    ) -> "FacetOperatorChain":
        self.with_baseline()
        self.state.delta_rows = paired_delta_rows(
            self.store,
            model=self.scope.model,
            scorer=self.scope.scorer,
            dataset=self.scope.dataset,
            split=self.scope.split,
            base_config_id=self.state.base_config_id,
            config_ids=config_ids,
        )
        self.state.meta["config_ids"] = list(config_ids) if config_ids is not None else None
        return self

    def with_context(
        self,
        *,
        predicate_names: Optional[Sequence[str]] = None,
        max_values_per_predicate: Optional[int] = 32,
        context_view: str = "raw",
        include_negative: bool = True,
    ) -> "FacetOperatorChain":
        self.state.context_rows = query_context_sets(
            self.store,
            dataset=self.scope.dataset,
            split=self.scope.split,
            predicate_names=predicate_names,
            max_values_per_predicate=max_values_per_predicate,
            context_view=context_view,
            include_negative=include_negative,
        )
        self.state.meta["predicate_names"] = (
            list(predicate_names) if predicate_names is not None else None
        )
        self.state.meta["max_values_per_predicate"] = max_values_per_predicate
        self.state.meta["context_view"] = context_view
        self.state.meta["include_negative_context"] = include_negative
        return self

    def with_itemsets(
        self,
        *,
        max_feature_order: int = 3,
        max_context_order: int = 3,
        min_config_support: int = 1,
        min_query_support: int = 20,
        max_candidates_per_level: int = 50000,
    ) -> "FacetOperatorChain":
        if self.state.delta_rows is None:
            self.with_paired_deltas()
        if self.state.context_rows is None:
            self.with_context()

        self.state.feature_itemsets = feature_itemsets_from_delta(
            self.state.delta_rows,
            max_order=max_feature_order,
            min_config_support=min_config_support,
        )
        self.state.context_itemsets = context_itemsets_from_context(
            self.state.context_rows,
            max_order=max_context_order,
            min_query_support=min_query_support,
            max_candidates_per_level=max_candidates_per_level,
        )
        self.state.meta.update({
            "max_feature_order": max_feature_order,
            "max_context_order": max_context_order,
            "min_config_support": min_config_support,
            "min_query_support": min_query_support,
            "max_candidates_per_level": max_candidates_per_level,
        })
        return self

    def with_global_effects(
        self,
        *,
        n_bootstrap: int = 500,
        seed: int = 42,
    ) -> "FacetOperatorChain":
        if self.state.feature_itemsets is None:
            self.with_itemsets()
        self.state.global_effects = score_feature_itemsets(
            self.state.delta_rows,
            self.state.feature_itemsets or [],
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        return self

    def with_conditional_effects(
        self,
        *,
        min_query_support: int = 20,
        min_observation_support: Optional[int] = None,
        n_bootstrap: int = 500,
        seed: int = 42,
    ) -> "FacetOperatorChain":
        if self.state.feature_itemsets is None or self.state.context_itemsets is None:
            self.with_itemsets(min_query_support=min_query_support)
        if min_observation_support is None:
            min_observation_support = min_query_support
        self.state.conditional_effects = score_contextual_feature_itemsets(
            self.state.delta_rows,
            self.state.context_rows,
            self.state.feature_itemsets or [],
            self.state.context_itemsets or [],
            min_query_support=min_query_support,
            min_observation_support=min_observation_support,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        return self

    def with_binary_rules(
        self,
        *,
        min_query_support: int = 20,
        min_observation_support: Optional[int] = None,
        positive_delta_threshold: float = 0.0,
    ) -> "FacetOperatorChain":
        if self.state.feature_itemsets is None or self.state.context_itemsets is None:
            self.with_itemsets(min_query_support=min_query_support)
        if min_observation_support is None:
            min_observation_support = min_query_support
        self.state.binary_rules = score_binary_contextual_rules(
            self.state.delta_rows,
            self.state.context_rows,
            self.state.feature_itemsets or [],
            self.state.context_itemsets or [],
            min_query_support=min_query_support,
            min_observation_support=min_observation_support,
            positive_delta_threshold=positive_delta_threshold,
        )
        return self

    def run_subgroups(
        self,
        *,
        max_feature_order: int = 3,
        max_context_order: int = 3,
        min_query_support: int = 50,
        min_config_support: int = 1,
        max_values_per_predicate: Optional[int] = 32,
        context_view: str = "raw",
        include_negative: bool = True,
        n_bootstrap: int = 500,
        seed: int = 42,
    ) -> "FacetOperatorChain":
        return (
            self.with_paired_deltas()
            .with_context(
                max_values_per_predicate=max_values_per_predicate,
                context_view=context_view,
                include_negative=include_negative,
            )
            .with_itemsets(
                max_feature_order=max_feature_order,
                max_context_order=max_context_order,
                min_query_support=min_query_support,
                min_config_support=min_config_support,
            )
            .with_global_effects(n_bootstrap=n_bootstrap, seed=seed)
            .with_conditional_effects(
                min_query_support=min_query_support,
                min_observation_support=min_query_support,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            .with_binary_rules(
                min_query_support=min_query_support,
                min_observation_support=min_query_support,
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.state.as_dict()


def discover_benchmark_subgroups(
    store: CubeStore,
    *,
    dataset: str,
    model: str,
    scorer: str,
    split: Optional[str] = None,
    base_config_id: Optional[int] = None,
    max_feature_order: int = 3,
    max_context_order: int = 3,
    min_query_support: int = 50,
    min_config_support: int = 1,
    max_values_per_predicate: Optional[int] = 32,
    n_bootstrap: int = 500,
    seed: int = 42,
):
    """Run the v0 within-benchmark FACET subgroup discovery stack."""
    chain = FacetOperatorChain.for_scope(
        store,
        model=model,
        scorer=scorer,
        dataset=dataset,
        split=split,
        base_config_id=base_config_id,
    ).run_subgroups(
        max_feature_order=max_feature_order,
        max_context_order=max_context_order,
        min_query_support=min_query_support,
        min_config_support=min_config_support,
        max_values_per_predicate=max_values_per_predicate,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    state = chain.state
    return {
        "delta_rows": state.delta_rows,
        "context_rows": state.context_rows,
        "feature_itemsets": state.feature_itemsets,
        "context_itemsets": state.context_itemsets,
        "global_effects": state.global_effects,
        "conditional_effects": state.conditional_effects,
        "binary_rules": state.binary_rules,
    }
