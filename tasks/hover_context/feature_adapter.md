# HoVer feature adapter

date: 2026-04-25
parent: `Obsidian/coding_agent_logs/claude_code/tasks/feature_decomposition/feature_taxonomy.md`
applies-to: `tasks/hover_context/`, cube `runs/hover_paper_repro_test_k15.db`
status: draft

Concrete features per (theme × module) for HoVer's 4-module pipeline. The shared taxonomy lives at the parent path; this file says **what each theme looks like in HoVer**.

## Module overview

HoVer is 3-hop fact verification (claim → 2–4 supporting Wikipedia titles, set-subset coverage metric).

| module | input | output | purpose |
|---|---|---|---|
| `summarize1` | claim, hop-1 retrieved passages | `reasoning`, `summary` | digest hop-1 evidence |
| `create_query_hop2` | claim, hop-1 summary | `reasoning`, `query` | hop-2 retrieval query |
| `summarize2` | claim, hop-1 summary, hop-2 retrieved passages | `reasoning`, `summary` | digest hop-2 evidence |
| `create_query_hop3` | claim, hop-2 summary | `reasoning`, `query` | hop-3 retrieval query |

The `reasoning` field is always-on per the parent taxonomy (Tier 0 floor).

## Theme × module matrix

`-` = not applicable for that module type. ⭐ = first-batch starting cells (see "First batch" below).

| theme | summarize1 | create_query_hop2 | summarize2 | create_query_hop3 |
|---|---|---|---|---|
| `decision-heuristic` | dh × 2 | dh × 2 ⭐ | dh × 2 | dh × 2 |
| `staged-reasoning` | sr × 1 (umbrella) | sr × 1 (umbrella) | sr × 1 | sr × 1 |
| `query-decomposition` | — | qd × 3 ⭐ | — | qd × 2 |
| `evidence-summarization` | es × 3 ⭐ | — | es × 2 | — |
| `verify-before-output` | vbo × 1 | vbo × 1 | vbo × 1 | vbo × 1 |
| `missing-domain-knowledge` | mdk × 2 (shared section) | mdk × 1 | mdk × 1 | mdk × 1 |
| `edge-case-handling` | ech × 1 | ech × 2 | ech × 1 | ech × 2 |

## Per-cell features

### `decision-heuristic` (`dh_`)

| canonical_id | module | rule body |
|---|---|---|
| `dh_summarize1_keep_claim_entities` | summarize1 | If a passage contains a named entity from the claim, retain that sentence verbatim. |
| `dh_summarize1_drop_no_overlap` | summarize1 | If a passage has no entity overlap with the claim, omit it from the summary. |
| `dh_create_query_hop2_drop_resolved` ⭐ | create_query_hop2 | If an entity is already grounded by hop-1 retrieved titles, do not include it in the hop-2 query. |
| `dh_create_query_hop2_target_unresolved` ⭐ | create_query_hop2 | Target the hop-2 query at the specific claim entity that was NOT covered by hop-1. |
| `dh_summarize2_check_bridge` | summarize2 | If hop-2 passages confirm the bridge entity, state it explicitly in the summary. |
| `dh_summarize2_drop_redundant` | summarize2 | Skip facts already established in the hop-1 summary. |
| `dh_create_query_hop3_target_remaining` | create_query_hop3 | Identify the still-uncovered claim component and write the hop-3 query against it. |
| `dh_create_query_hop3_use_aliases` | create_query_hop3 | If hop-2 surfaced an alias of the claim entity, prefer the alias in the hop-3 query. |

### `staged-reasoning` (`sr_`)

Umbrella procedure rules; one per module.

| canonical_id | module | rule body |
|---|---|---|
| `sr_summarize1_three_step` | summarize1 | Step 1: identify claim entities. Step 2: find each in passages. Step 3: write 1–2 sentences per matched entity. |
| `sr_create_query_hop2_three_step` | create_query_hop2 | Step 1: list resolved vs unresolved claim parts. Step 2: pick the most retrieval-tractable unresolved part. Step 3: write the query. |
| `sr_summarize2_consolidate` | summarize2 | Step 1: list new facts from hop-2. Step 2: drop facts already in hop-1 summary. Step 3: emit consolidated summary. |
| `sr_create_query_hop3_three_step` | create_query_hop3 | Step 1: identify what is STILL missing after hop-2. Step 2: choose a discriminative term. Step 3: write the query. |

### `query-decomposition` (`qd_`)

Sub-theme of staged-reasoning, retrieval-specific. Only applies to query-writing modules.

| canonical_id | module | rule body |
|---|---|---|
| `qd_create_query_hop2_split_conjunction` ⭐ | create_query_hop2 | Split compound claims at "and" or "while" — write the hop-2 query against the second clause only. |
| `qd_create_query_hop2_one_entity` ⭐ | create_query_hop2 | The hop-2 query should target exactly one bridging entity, not enumerate all claim entities. |
| `qd_create_query_hop2_avoid_negation` ⭐ | create_query_hop2 | Drop negation words from the query — Wikipedia abstracts are positive statements. |
| `qd_create_query_hop3_focus_residual` | create_query_hop3 | Hop-3 query should be the smallest residual that closes the claim, not a re-statement of the whole claim. |
| `qd_create_query_hop3_promote_specifics` | create_query_hop3 | Promote dates, numbers, and proper nouns over common nouns in the hop-3 query. |

### `evidence-summarization` (`es_`)

Only applies to summarize modules.

| canonical_id | module | rule body |
|---|---|---|
| `es_summarize1_quote_proper_nouns` ⭐ | summarize1 | Retain every proper noun verbatim; never paraphrase a name. |
| `es_summarize1_two_sentence_cap` ⭐ | summarize1 | Cap the summary at two sentences per relevant passage; cite the title before each sentence. |
| `es_summarize1_skip_irrelevant` ⭐ | summarize1 | Explicitly state "no relevant content" for passages that don't match any claim entity. |
| `es_summarize2_carry_hop1_facts` | summarize2 | Carry forward hop-1 facts only if hop-2 evidence corroborates or extends them. |
| `es_summarize2_emit_bridge_explicitly` | summarize2 | If hop-2 establishes a bridge between hop-1 entities, emit "<A> is connected to <B> via <C>" verbatim. |

### `verify-before-output` (`vbo_`)

| canonical_id | module | rule body |
|---|---|---|
| `vbo_summarize1_entity_check` | summarize1 | Before emitting summary, verify each retained sentence mentions ≥1 entity from the claim. |
| `vbo_create_query_hop2_query_check` | create_query_hop2 | Before emitting the query, verify it would not return the same titles as hop-1. |
| `vbo_summarize2_consistency` | summarize2 | Before emitting, verify the hop-2 summary does not contradict hop-1 facts; flag if it does. |
| `vbo_create_query_hop3_termination` | create_query_hop3 | Before emitting, verify hop-3 is actually needed; output an empty query if the claim is already fully grounded. |

### `missing-domain-knowledge` (`mdk_`)

Often goes in a shared "context" rule before any module-specific guidance. Module-level instances are tail-of-section rules.

| canonical_id | module | rule body |
|---|---|---|
| `mdk_section_corpus_year_floor` | (shared section) | The retrieval corpus is Wikipedia abstracts from 2017; events and works released after 2017 may not be indexed. |
| `mdk_section_alias_handling` | (shared section) | Wikipedia titles use canonical names; aliases, redirects, and disambiguators may not appear verbatim — match by substring or proper-noun head. |
| `mdk_create_query_hop2_year_token` | create_query_hop2 | When the claim mentions a year, include that year as a standalone token in the query. |
| `mdk_summarize2_relation_anchors` | summarize2 | Common Wikipedia relation anchors include "spouse", "directed", "founded", "located in" — flag these explicitly in the summary. |
| `mdk_create_query_hop3_decade_fallback` | create_query_hop3 | If a year fails to retrieve, broaden to the decade ("1980s") in hop-3. |

### `edge-case-handling` (`ech_`)

| canonical_id | module | rule body |
|---|---|---|
| `ech_summarize1_empty_retrieval` | summarize1 | If all hop-1 passages are irrelevant, emit summary "no supporting evidence found in hop-1" verbatim. |
| `ech_create_query_hop2_repeat_claim` | create_query_hop2 | If you cannot identify a bridging entity, fall back to re-querying the claim's main subject. |
| `ech_create_query_hop2_fictional` | create_query_hop2 | If the claim describes a fictional entity (work title, character), prefer the work's title as the query head. |
| `ech_summarize2_contradiction` | summarize2 | If hop-2 contradicts hop-1, prefer the source with more specific named entities and flag the contradiction. |
| `ech_create_query_hop3_zero_residual` | create_query_hop3 | If no claim component is unresolved after hop-2, emit a single neutral token ("relevant context") to satisfy the schema. |
| `ech_create_query_hop3_post_corpus` | create_query_hop3 | If the claim references events post-2017, hop-3 should query the most recent comparable event. |

## First batch (round 9a)

10 features across 3 cells. All semantic. All transferable to HotpotQA via the shared canonical_id stems.

| cell | features | rationale |
|---|---|---|
| `query-decomposition` × create_query_hop2 | qd_split_conjunction, qd_one_entity, qd_avoid_negation | Modules where GEPA's verbose prompts have the largest visible diffs. Pure retrieval-strategy probe. |
| `decision-heuristic` × create_query_hop2 | dh_drop_resolved, dh_target_unresolved | Conditional retrieval rules; tests "context-aware query writing." |
| `evidence-summarization` × summarize1 | es_quote_proper_nouns, es_two_sentence_cap, es_skip_irrelevant | Probes whether summary content (not summary structure) is the lever. |

Total: 8 features — uses 1 shared section per module + 1 rule per feature. Run as add-one sweep on the existing k=15 cube; Tier-A coherence check before publishing per-theme rankings.

## Modules deliberately not in first batch

- `summarize2` and `create_query_hop3` — defer until first-batch results show whether hop-1 features transfer to deeper hops. If they do, the same canonical_id pattern materializes for hop-2/3 in round 9b.
- `verify-before-output` — defer; verify-style features are tail-of-module and need a clean baseline first.
- `missing-domain-knowledge` shared section — defer; affects all 4 modules simultaneously and would confound module-level interpretation.

## Authoring instructions

1. Each feature file lives at `features/hover_context/<canonical_id>.json`.
2. `primitive_edits` is a list of one section reference + one or more `insert_node` rules. Use `{"$ref": "_sections.<module>"}` for the section anchor.
3. Set `requires: []` and `conflicts_with: []` unless a feature explicitly depends on or conflicts with another (none in first batch).
4. Cross-task: when authoring the matching HotpotQA features, reuse the canonical_id stem (drop the `dh_` / `qd_` / `es_` prefix and use the module-mapped stem). Example: `qd_create_query_hop2_split_conjunction` exists in both task adapters with the same descriptor.

## Sanity checks before launching the sweep

- `ProgressMonitor` against the cube to confirm config dedup is hitting (no error rows).
- `summary(store)` to confirm the 8 new feature_ids show up in the func table.
- Audit `gepa_qwen3_summarize1` parent_id hash hasn't shifted (parent-id stability is required for cube hashing per round-8 notes).
- One-shot dry run with `max_queries=10` to confirm parser handles the new prompt structure (round-7 nested-summary parser fix should already cover this).
