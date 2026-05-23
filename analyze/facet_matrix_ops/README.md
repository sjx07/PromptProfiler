# FACET Matrix Operators

This package is the modular version of the FACET feature/context matrix
analyzer. It replaces the previous long `analyze/facet_matrix.py`
implementation with a small operator chain.

The legacy import path still works:

```python
from analyze.facet_matrix import paired_delta_rows
```

New code should prefer the package:

```python
from analyze.facet_matrix_ops import FacetOperatorChain
```

## Mental Model

FACET analysis is a matrix over prompt features and query context attributes:

```text
row = (query_id, config_id, dataset, model, scorer)
y = score(config, query) - score(base_config, query)
F(config) = active prompt feature atoms
C(query) = context attribute atoms
rule = feature itemset useful under context itemset
```

The operators are read-only over `CubeStore`. They do not launch experiments,
write cube rows, or change configs.

## Operator Chain

The default chain is:

```text
config features
  -> selected baseline
  -> paired deltas
  -> query context atoms
  -> feature/context itemsets
  -> global feature effects
  -> contextual feature effects
  -> binary useful-rate rules
```

The chainable facade makes that explicit:

```python
from core.store import CubeStore
from analyze.facet_matrix_ops import FacetOperatorChain

store = CubeStore("/data/users/jsu323/facet/wikitable_clean_surface_v1.db")

chain = (
    FacetOperatorChain.for_scope(
        store,
        dataset="wtq",
        model="Qwen/Qwen2.5-14B-Instruct",
        scorer="denotation_acc",
        split="test",
    )
    .with_paired_deltas()
    .with_context(max_values_per_predicate=16)
    .with_itemsets(max_feature_order=2, max_context_order=3, min_query_support=50)
    .with_global_effects(n_bootstrap=0)
    .with_conditional_effects(min_query_support=50, n_bootstrap=500)
    .with_binary_rules(min_query_support=50)
)

rules = chain.state.conditional_effects
binary = chain.state.binary_rules
```

For the old one-shot behavior:

```python
from analyze.facet_matrix_ops import discover_benchmark_subgroups

result = discover_benchmark_subgroups(
    store,
    dataset="wtq",
    model="Qwen/Qwen2.5-14B-Instruct",
    scorer="denotation_acc",
    split="test",
    max_feature_order=2,
    max_context_order=3,
    min_query_support=50,
)
```

## Modules

| module | role |
| --- | --- |
| `common.py` | shared metadata, dataset, and query-scope helpers |
| `config.py` | config feature vectors and baseline selection |
| `outcomes.py` | paired query-level delta rows |
| `context.py` | query context atom vectors from cube predicates |
| `itemsets.py` | feature/context transactions and Apriori-style itemsets |
| `stats.py` | bootstrap and Wilson intervals |
| `scoring.py` | global effects, contextual effects, binary useful-rate rules, cross-benchmark comparison |
| `pipeline.py` | `FacetOperatorChain` and one-shot subgroup discovery |

## Design Notes

- Keep operators typed by unit of support. Feature itemsets are supported by
  configs; context itemsets are supported by queries.
- Keep paired deltas as the canonical outcome inside one benchmark. This avoids
  comparing unpaired query populations.
- Use binary useful-rate rules for cross-benchmark comparison when raw scorer
  magnitudes are not comparable.
- Treat output contracts that are constant inside a selected scope as
  measurement adapters rather than prompt features.
- Treat discovered rules as subgroup/conditional-effect screens. They are not
  causal claims unless a later validation design supports that interpretation.
