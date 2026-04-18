# Phase 1 Refactor: Paper-Aligned Primitives

## Overview

Phase 1 replaces ad-hoc `func_type` strings (`add_rule`, `define_section`,
`add_input_field`, `transform_input`) with three paper-aligned primitives
that map cleanly to the prompt-state tree model.

---

## Primitive Set

### `insert_node`

Inserts a node into the prompt state tree. Dispatches on `params.node_type`.

```json
{
  "func_type": "insert_node",
  "params": {
    "node_type": "rule",
    "parent_id": "<section_func_id>",
    "payload": { "content": "Use ORDER BY when sorting." }
  }
}
```

Allowed `node_type` values and their payload shapes:

| node_type     | payload fields                                             | sort priority |
|---------------|------------------------------------------------------------|---------------|
| `section`     | `title`, `ordinal`, `is_system`, `min_rules`, `max_rules`  | 0 (first)     |
| `input_field` | `name`, `description`                                      | 1             |
| `output_field`| `name`, `description`                                      | 1             |
| `rule`        | `content`                                                  | 2             |
| `example`     | `content`                                                  | 3 (last)      |

**`__root__` convention**: When `parent_id` is omitted or `null`, it defaults
to the string constant `"__root__"`. All top-level sections, input fields, and
output fields use `"__root__"` as their parent. Rules use their section's
`func_id` as `parent_id`.

**Subsection support (Phase 1 parse-only)**: A subsection is a section whose
`parent_id` is another section's `func_id` rather than `"__root__"`. Depth cap
is 2 for Phase 1 MVP. Render walk of subsections is deferred.

### `update_node`

Declared primitive; no consumer wired in Phase 1. Logs a warning if used.
First consumer planned for rule-rewrite features (Phase 2+).

```json
{
  "func_type": "update_node",
  "params": { "target_id": "<func_id>", "patch": { "content": "New text." } }
}
```

### `input_transform`

Pre-render input transform (renamed from `transform_input`). Single function
per primitive; multiple primitives compose deterministically by `func_id` sort order.

```json
{
  "func_type": "input_transform",
  "params": { "fn": "prune_cols", "kwargs": { "k": 5 } }
}
```

---

## Removed Handlers (Hard Break)

The following `func_type` strings are no longer registered. Any cube row
with these types will log a warning and be skipped at apply time:

| Removed         | Replacement                              |
|-----------------|------------------------------------------|
| `add_rule`      | `insert_node` with `node_type="rule"`    |
| `define_section`| `insert_node` with `node_type="section"` |
| `add_input_field` | `insert_node` with `node_type="input_field"` |
| `transform_input` | `input_transform`                      |

Old cubes (schema_version < 6) are **read-only**. Opening one without
`read_only=True` raises `RuntimeError`. See the migration section below.

---

## Flat-Rows-with-Parent-Hashes → Reconstructed Tree

The `func` table stores rows flat. The tree is reconstructed at apply time
using `parent_id` links:

```
func table (flat rows):
  func_id=a1b2c3  func_type=insert_node  params={"node_type":"section","parent_id":"__root__","payload":{"title":"reasoning",...}}
  func_id=d4e5f6  func_type=insert_node  params={"node_type":"rule","parent_id":"a1b2c3","payload":{"content":"Think step by step..."}}
  func_id=g7h8i9  func_type=insert_node  params={"node_type":"output_field","parent_id":"__root__","payload":{"name":"reasoning",...}}

Reconstructed tree (at apply_config time):
  __root__
  ├── section[a1b2c3] "reasoning" (ordinal=30)
  │   └── rule[d4e5f6] "Think step by step..."
  └── output_field[g7h8i9] "reasoning"
```

`apply_config` sorts rows by `(type_priority, node_type_priority, func_id)`
so sections are always processed before their child rules.

---

## Schema Version 6

`SCHEMA_VERSION = 6`. Changed views:

- `section_view` — projects from `insert_node` where `node_type='section'`
- `rule_view` — projects from `insert_node` where `node_type='rule'`
- `input_field_view` — new; projects from `insert_node` where `node_type='input_field'`
- `output_field_view` — new; projects from `insert_node` where `node_type='output_field'`

---

## Feature / Config File Shape

### Feature file (`features/<task>/<feature_id>.json`)

```json
{
  "feature_id": "enable_cot",
  "canonical_id": "enable_cot",
  "task": "table_qa",
  "requires": [],
  "conflicts_with": [],
  "primitive_edits": [
    {
      "func_type": "insert_node",
      "params": {
        "node_type": "rule",
        "parent": { "$ref": "_sections.reasoning" },
        "payload": { "content": "Think step by step before writing the final answer." }
      }
    }
  ]
}
```

`{"$ref": "_sections.X"}` is resolved at materialize time via
`features/<task>/_sections.json`.

### Sections catalog (`features/<task>/_sections.json`)

```json
{
  "version": 1,
  "task": "table_qa",
  "sections": {
    "reasoning": { "title": "reasoning", "ordinal": 30, "is_system": true, "min_rules": 0, "max_rules": 10 }
  }
}
```

### Config file (feature-level only, no `expanded_func_level`)

```json
{
  "config_id": "demo_table_qa_cot_only",
  "task": "table_qa",
  "feature_level": {
    "feature_ids": ["enable_cot"],
    "feature_version": "2026-04-17-v1"
  },
  "meta": {
    "created_by": "materializer:v0.1",
    "base_ir": "tool/PromptProfiler/features/table_qa/_sections.json"
  }
}
```

The DB still stores resolved `func_ids` (hashes) as the cache key — that
is a storage concern, not a config-file concern.

---

## Feature Dependency Semantics

Enforced at **materialize time** (compile time), zero runtime overhead.

1. **`requires`**: All listed `feature_ids` must be explicitly present in the
   config. If not, `validate_feature_set` raises `ValueError` naming the
   missing dependency. Transitive auto-include is intentionally NOT performed
   (per round 7, decision 1a) — the user must list all features explicitly.

2. **`conflicts_with`**: If any listed `feature_id` is also in the config,
   `ValueError` is raised naming the conflicting pair.

3. **Cross-task dependencies**: `requires` is task-scoped. A feature from
   `task_A` cannot `require` a feature from `task_B`. Such refs raise
   `ValueError`. Cross-task alignment uses `canonical_id` fields (later round).

4. **Cycles**: Detected implicitly — if `A requires B` and `B requires A`,
   both must be in the config; either one's `conflicts_with` can ban the other,
   but neither is auto-added. Explicit cycles in requires-DAG are not currently
   detected (no transitive closure is computed).

---

## Migration from v5 Cubes

Opening a v5 cube on the new code without `read_only=True` raises:

```
RuntimeError: cube schema_version=5 is older than required 6.
Open with read_only=True for migration scripting, or create a fresh cube.
See docs/refactor_phase1.md.
```

Migration path:
1. Open old cube with `CubeStore(path, read_only=True)`.
2. Read all `func` rows via `store.list_funcs()`.
3. Rewrite `define_section` rows as `insert_node(section)`, etc.
4. Write to a new v6 cube.

---

## Out of Scope (Phase 2+)

- `set_render` and retirement of `set_format` / `set_table_format` / `enable_*`
- Feature file population beyond `_sections.json`
- `update_node` consumer
- Subsection render walk (depth > 1)
- Full materializer feature→primitive expansion for user-defined features
