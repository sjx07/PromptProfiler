# Analysis & monitoring

Primitives for interrogating a `CubeStore` without opening a SQL REPL.
Everything lives in the `prompt_profiler.analyze` package.

```python
from prompt_profiler.analyze import (
    # meta
    summary, list_configs_with_features, list_models, list_scorers,
    list_phases, list_datasets, list_predicates, list_features_in_cube,
    # query
    ExecutionQuery,
    # compare
    score_diff, feature_effect_ranking, predicate_slice, add_one_deltas,
    # monitor
    ProgressMonitor,
)
```

## Taxonomy

```
Layer 1 · meta       describe-what-exists (no filters, no aggregation)
Layer 2 · query      chainable filter + projection + aggregate
Layer 3 · compare    cross-config deltas, rankings, predicate slices
Layer 4 · monitor    live view during / after a run
```

The layers build on each other: `compare` and `monitor` use `ExecutionQuery`
internally; `ExecutionQuery` composes raw SQL over the cube tables; `meta`
functions are independent one-liners. Every function takes a `CubeStore`
as the first argument and is pure (no writes).

## Layer 1 — describe-what-exists

One-shot inventories. Useful as the first thing you run on a new cube.

| function | returns |
| --- | --- |
| `summary(store)` | `{counts, models, scorers, phases, datasets, tasks}` |
| `list_configs(store)` | every config with `func_ids`, `meta`, `canonical_ids` |
| `list_configs_with_features(store)` | as above + `resolved_canonical_ids` via feature table |
| `list_configs_with_func_types(store)` | `{config_id, n_funcs, by_type}` per config |
| `list_models(store)` | distinct models + execution count + first/last seen |
| `list_scorers(store)` | distinct scorers + evaluation count + mean score |
| `list_phases(store)` | distinct phase tags + execution count + models |
| `list_datasets(store)` | datasets grouped by split with query counts |
| `list_predicates(store)` | predicate names + distinct-value / query-coverage counts |
| `list_features_in_cube(store, task=None)` | features synced into the cube |

```python
print(summary(store))
#   counts:   {'func': 32, 'query': 200, 'config': 11, 'execution': 2200, ...}
#   models:   [{'model': 'meta-llama/Llama-3.1-8B-Instruct', 'n_executions': 2200, ...}]
#   scorers:  [{'scorer': 'denotation_acc', 'n_evaluations': 2200, 'mean_score': 0.47}]
#   phases:   [{'phase': 'wtq_mvp_20260418_204500', 'n_executions': 2200, ...}]
#   ...
```

## Layer 2 — `ExecutionQuery`

Chainable filter builder. Every `.where_*` / `.has_*` method returns a
new query (immutable-ish). Terminal methods (`.count`, `.rows`, `.df`,
`.agg`) execute SQL against the cube.

### Scope filters

| method | effect |
| --- | --- |
| `.phase(phase)` / `.phases([p1, p2])` | OR over phase tags |
| `.config(id)` / `.configs([ids])` | WHERE `config_id IN (...)` |
| `.model(name)` / `.models([names])` | WHERE `model IN (...)` |
| `.scorer(name)` | LEFT JOIN `evaluation`; exposes `score`, `metrics`, `eval_id` |
| `.query(id)` / `.queries([ids])` | WHERE `query_id IN (...)` |

### Structural filters

| method | effect |
| --- | --- |
| `.has_func(func_id)` / `.has_all_funcs([ids])` | config must contain ALL these func_ids |
| `.has_any_func([ids])` | config must contain ANY of these func_ids |
| `.has_feature(canonical_id)` / `.has_all_features([cids])` | resolve canonical → hash, require ALL |
| `.has_any_feature([cids])` | resolve canonical → hash, require ANY |
| `.predicate(name, value=None)` | require a predicate row matching name (and value if given) |

### Score / error filters

| method | effect |
| --- | --- |
| `.where_score(op, value)` | `ev.score <op> <value>`; requires `.scorer(...)` first |
| `.with_error()` | `execution.error IS NOT NULL AND != ''` |
| `.without_error()` | inverse |

### Projection / order / limit

| method | effect |
| --- | --- |
| `.columns([cols])` | override SELECT columns; unqualified names auto-resolve to `e.*` or `ev.*` |
| `.order_by("clause")` | raw ORDER BY (e.g. `"ev.score DESC"`) |
| `.limit(n)` | LIMIT n |

### Terminals

| method | returns |
| --- | --- |
| `.count()` | `int` |
| `.rows()` | `List[dict]` with `meta` / `metrics` / `phase_ids` JSON parsed |
| `.df()` | `pandas.DataFrame` (JSON columns remain strings — materialize on demand) |
| `.agg(by=[cols], fn="avg", metric="score")` | grouped aggregate; `fn` ∈ `{avg, sum, min, max, count}`; `by=[]` → global |

### Example

```python
# Low-scoring CoT+aggregation cases under a specific model:
(ExecutionQuery(store)
 .model("meta-llama/Llama-3.1-8B-Instruct")
 .scorer("denotation_acc")
 .has_feature("enable_cot")
 .predicate("has_aggregation", "true")
 .where_score("<", 0.5)
 .columns(["config_id", "query_id", "prediction", "score"])
 .order_by("ev.score, e.query_id")
 .rows())
```

## Layer 3 — compose (`compare.py`)

Cross-cutting operations built from the above primitives.

### `score_diff(store, config_a, config_b, model, scorer)`

Compare two configs on their shared queries. Returns:

```python
{
  "config_a": 1, "config_b": 2,
  "n_a": 200, "n_b": 200, "n_shared": 200,
  "avg_a": 0.42, "avg_b": 0.48, "avg_delta": 0.06,
  "flipped_up":   ["q17", "q44", ...],   # 0 → 1 queries
  "flipped_down": ["q91"],               # 1 → 0 queries
  "agree": 181,
}
```

### `add_one_deltas(store, base_config_id, model, scorer)`

Sweep every non-base config; run `score_diff` against the base. Returns a
sorted DataFrame — the standard "feature impact" view for `add_one_feature`
experiments.

```
  config_id canonical_id  n_shared  avg_a  avg_b  avg_delta  flipped_up  flipped_down
0         3   enable_code       200   0.42   0.51      +0.09          22            4
1         5  enable_column_stats 200   0.42   0.46      +0.04          13            5
2         4    enable_cot       200   0.42   0.45      +0.03          11            4
...
```

### `feature_effect_ranking(store, model, scorer, task=None)`

Marginal mean score per feature across every config that activates it.
Uses the `feature_effect` SQL view. Returns a DataFrame sorted by
`mean_score` desc.

**Warning.** This is a marginal average, not a controlled effect. Useful
for a first look; for a real effect estimate, use `add_one_deltas`.

### `predicate_slice(store, model, scorer, predicate_name, config_ids=None)`

Mean score per `(predicate_value, config_id)`. Shows whether a feature
helps more on some slices than others — e.g. does `enable_cot` lift
aggregation queries more than lookups?

## Layer 4 — `ProgressMonitor`

Stateless facade for live or post-hoc inspection of a running experiment.

```python
mon = ProgressMonitor(store,
                      model="meta-llama/Llama-3.1-8B-Instruct",
                      scorer="denotation_acc",
                      phase="wtq_mvp_20260418_204500")

mon.overall()
# → {'n_executions': 450, 'n_configs': 3, 'n_queries': 150,
#    'n_errors': 2, 'n_evaluations': 448, 'mean_score': 0.47, ...}

for row in mon.by_config(total_queries_expected=200):
    print(row)
# → {'config_id': 1, 'canonical_id': 'base',       'n_done': 200, 'pct': 100.0, 'mean_score': 0.42}
# → {'config_id': 3, 'canonical_id': 'enable_cot', 'n_done': 150, 'pct':  75.0, 'mean_score': 0.45}
# → ...

mon.errors(limit=10)   # most recent errors
mon.recent(20)         # most recent executions (any status)
```

All three monitor methods scope to the `(model, scorer, phase)` combo
passed to the constructor. Leave any of them `None` for a broader view.

## Recipes

**"What's in this cube?"**

```python
from prompt_profiler.analyze import summary
pprint(summary(store))
```

**"Which features help on aggregation queries?"**

```python
from prompt_profiler.analyze import predicate_slice
df = predicate_slice(store,
                     model=MODEL, scorer=SCORER,
                     predicate_name="has_aggregation")
df_agg = df[df["predicate_value"] == "true"].sort_values("mean_score", ascending=False)
```

**"Show me the 20 worst predictions with enable_cot on."**

```python
(ExecutionQuery(store)
 .model(MODEL).scorer(SCORER)
 .has_feature("enable_cot")
 .without_error()
 .order_by("ev.score, e.query_id")
 .limit(20)
 .columns(["config_id", "query_id", "prediction", "raw_response", "score"])
 .rows())
```

**"Did enable_code help or hurt?"**

```python
from prompt_profiler.analyze import score_diff
score_diff(store,
           config_a=BASE_CID, config_b=CODE_CID,
           model=MODEL, scorer=SCORER)
# → {'avg_delta': +0.07, 'flipped_up': [...], 'flipped_down': [...], ...}
```

**"Live watch a running experiment."**

```python
import time
from prompt_profiler.analyze import ProgressMonitor
mon = ProgressMonitor(store, model=MODEL, scorer=SCORER, phase=CURRENT_PHASE)
while True:
    ov = mon.overall()
    print(f"{ov['n_executions']} executions, {ov['n_errors']} errors, "
          f"mean={ov['mean_score']:.3f}")
    time.sleep(30)
```

## Adding new primitives

- **Describe** (Layer 1) → new function in `analyze/meta.py`.
- **Filter axis** (Layer 2) → new method on `ExecutionQuery`; add a new
  field to `_Filters` and a WHERE clause to `_compile`.
- **Cross-cutting summary** (Layer 3) → new function in `analyze/compare.py`
  that builds on `ExecutionQuery`.
- **Monitor view** (Layer 4) → new method on `ProgressMonitor`.

Every addition should have at least one test in `tests/test_analyze.py`
using the seeded fixture.
