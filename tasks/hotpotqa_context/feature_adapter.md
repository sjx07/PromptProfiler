# HotpotQA feature adapter

date: 2026-04-25
parent: `Obsidian/coding_agent_logs/claude_code/tasks/feature_decomposition/feature_taxonomy.md`
applies-to: `tasks/hotpotqa_context/`, cube `runs/hotpotqa_paper_repro_v5*.db`
status: draft

Concrete features per (theme × module) for HotpotQA's 4-module pipeline. The shared taxonomy lives at the parent path; this file says **what each theme looks like in HotpotQA**.

## Module overview

HotpotQA is 2-hop bridge QA (question → bridge entity → final answer span; F1 + EM after `dspy.evaluate.metrics.normalize_text`).

| module | input | output | purpose |
|---|---|---|---|
| `summarize1` | question, hop-1 retrieved passages | `reasoning`, `summary` | digest hop-1 evidence |
| `create_query_hop2` | question, hop-1 summary | `reasoning`, `query` | hop-2 retrieval query |
| `summarize2` | question, hop-1 summary, hop-2 retrieved passages | `reasoning`, `summary` | digest hop-2 evidence |
| `final_answer` | question, hop-1 summary, hop-2 summary | `reasoning`, `answer` | final answer span |

The `reasoning` field is always-on per the parent taxonomy (Tier 0 floor). Note: round 7 found that *adding* a `reasoning` field to a setup that didn't have one killed GEPA's lift — so the floor is "reasoning is present in our v5 schema," not "reasoning is presence-toggled."

## Theme × module matrix

`-` = not applicable. ⭐ = first-batch starting cells (cross-task with HoVer).

| theme | summarize1 | create_query_hop2 | summarize2 | final_answer |
|---|---|---|---|---|
| `decision-heuristic` | dh × 2 | dh × 2 ⭐ | dh × 2 | dh × 3 |
| `staged-reasoning` | sr × 1 | sr × 1 | sr × 1 | sr × 1 |
| `query-decomposition` | — | qd × 3 ⭐ | — | — |
| `evidence-summarization` | es × 3 ⭐ | — | es × 2 | — |
| `verify-before-output` | vbo × 1 | vbo × 1 | vbo × 1 | vbo × 2 |
| `missing-domain-knowledge` | mdk × 1 | mdk × 1 | mdk × 1 | mdk × 1 (shared section) |
| `edge-case-handling` | ech × 1 | ech × 2 | ech × 1 | ech × 3 |

## Per-cell features

### `decision-heuristic` (`dh_`)

| canonical_id | module | rule body |
|---|---|---|
| `dh_summarize1_keep_question_entities` | summarize1 | If a passage contains an entity from the question, retain that sentence verbatim with the title prefix. |
| `dh_summarize1_drop_no_overlap` | summarize1 | If a passage has no entity overlap with the question, omit it. |
| `dh_create_query_hop2_target_bridge` ⭐ | create_query_hop2 | The hop-2 query should target the bridge entity (the one that joins hop-1 evidence to the answer), not the question's main subject. |
| `dh_create_query_hop2_use_hop1_answer` ⭐ | create_query_hop2 | If hop-1 surfaces a candidate bridge entity, use that entity as the hop-2 query head. |
| `dh_summarize2_extract_answer_candidates` | summarize2 | List candidate answer spans from hop-2 passages with their source titles. |
| `dh_summarize2_filter_question_type` | summarize2 | Filter answer candidates to those matching the question's expected answer type (date, person, location, count). |
| `dh_final_answer_match_question_type` | final_answer | If the question asks "who/which person", the answer is a person name; if "when", a date or year; if "where", a location. |
| `dh_final_answer_prefer_specific` | final_answer | Prefer the most specific answer (full name over surname; full date over year). |
| `dh_final_answer_yes_no_normalize` | final_answer | If the question is yes/no, answer with exactly "yes" or "no" (lowercase, no punctuation). |

### `staged-reasoning` (`sr_`)

| canonical_id | module | rule body |
|---|---|---|
| `sr_summarize1_three_step` | summarize1 | Step 1: identify question entities. Step 2: find each in passages. Step 3: write 1–2 sentences per matched entity. |
| `sr_create_query_hop2_three_step` | create_query_hop2 | Step 1: identify the bridge entity from hop-1. Step 2: identify what the question asks ABOUT that entity. Step 3: write the query. |
| `sr_summarize2_three_step` | summarize2 | Step 1: list answer candidates. Step 2: cross-reference each against hop-1. Step 3: emit consolidated summary. |
| `sr_final_answer_three_step` | final_answer | Step 1: identify question type. Step 2: extract candidate spans. Step 3: pick the most specific candidate matching the type. |

### `query-decomposition` (`qd_`)

Only applies to query-writing module (HotpotQA is 2-hop, no hop-3).

| canonical_id | module | rule body |
|---|---|---|
| `qd_create_query_hop2_split_conjunction` ⭐ | create_query_hop2 | Split compound questions at "and" or "while" — write the hop-2 query against the second clause. |
| `qd_create_query_hop2_one_entity` ⭐ | create_query_hop2 | The hop-2 query should target exactly one bridge entity, not enumerate all question entities. |
| `qd_create_query_hop2_avoid_negation` ⭐ | create_query_hop2 | Drop negation from the query — Wikipedia abstracts are positive statements. |

(Same canonical_id stems as HoVer's qd cell — explicit cross-task pairing for round 9a.)

### `evidence-summarization` (`es_`)

| canonical_id | module | rule body |
|---|---|---|
| `es_summarize1_quote_proper_nouns` ⭐ | summarize1 | Retain every proper noun verbatim; never paraphrase a name. |
| `es_summarize1_two_sentence_cap` ⭐ | summarize1 | Cap the summary at two sentences per relevant passage; cite the title before each sentence. |
| `es_summarize1_skip_irrelevant` ⭐ | summarize1 | Explicitly state "no relevant content" for passages that don't match any question entity. |
| `es_summarize2_carry_hop1_facts` | summarize2 | Carry forward hop-1 facts only if hop-2 evidence corroborates or extends them. |
| `es_summarize2_isolate_answer_span` | summarize2 | When a passage contains the candidate answer span, quote it verbatim with surrounding 1-sentence context. |

(Same `es_summarize1_*` canonical_id stems as HoVer.)

### `verify-before-output` (`vbo_`)

| canonical_id | module | rule body |
|---|---|---|
| `vbo_summarize1_entity_check` | summarize1 | Before emitting summary, verify each retained sentence mentions ≥1 entity from the question. |
| `vbo_create_query_hop2_query_check` | create_query_hop2 | Before emitting the query, verify it would not return the same titles as hop-1. |
| `vbo_summarize2_consistency` | summarize2 | Before emitting, verify the hop-2 summary does not contradict hop-1 facts. |
| `vbo_final_answer_type_check` | final_answer | Before emitting, verify the answer matches the expected type (e.g., a question asking "when" should not be answered with a person name). |
| `vbo_final_answer_grounded_check` | final_answer | Before emitting, verify the answer span appears in either hop-1 or hop-2 summary; if not, flag and pick the closest grounded candidate. |

### `missing-domain-knowledge` (`mdk_`)

| canonical_id | module | rule body |
|---|---|---|
| `mdk_section_corpus_year_floor` | (shared section) | The retrieval corpus is Wikipedia abstracts from 2017; events and works released after 2017 may not be indexed. |
| `mdk_section_alias_handling` | (shared section) | Wikipedia titles use canonical names; match by substring or proper-noun head when aliases differ. |
| `mdk_summarize1_disambiguators` | summarize1 | Common HotpotQA disambiguators include parenthetical type tags ("(film)", "(actor)") — use them when multiple titles match. |
| `mdk_create_query_hop2_year_token` | create_query_hop2 | When the question mentions a year, include that year as a standalone token in the query. |
| `mdk_summarize2_relation_anchors` | summarize2 | Common Wikipedia relation anchors include "spouse", "directed", "founded", "located in" — flag these explicitly. |

### `edge-case-handling` (`ech_`)

| canonical_id | module | rule body |
|---|---|---|
| `ech_summarize1_empty_retrieval` | summarize1 | If all hop-1 passages are irrelevant, emit "no supporting evidence found in hop-1" verbatim. |
| `ech_create_query_hop2_no_bridge` | create_query_hop2 | If you cannot identify a bridge entity, fall back to querying the question's main subject with a question-type token (e.g., "born", "directed", "founded"). |
| `ech_create_query_hop2_fictional` | create_query_hop2 | If the question is about a fictional work, prefer the work's title as the query head. |
| `ech_summarize2_contradiction` | summarize2 | If hop-2 contradicts hop-1, prefer the source with more specific named entities and flag the contradiction. |
| `ech_final_answer_no_evidence` | final_answer | If neither summary contains a grounded candidate, emit the most likely candidate from hop-1 with a "low-confidence" caveat in `reasoning`. |
| `ech_final_answer_multiple_candidates` | final_answer | If multiple candidates fit the question type, prefer the one with title overlap to both hops. |
| `ech_final_answer_strip_punct` | final_answer | Strip trailing periods and quote marks from the answer span before emitting. |

## First batch (round 9a) — cross-task with HoVer

8 features mirroring HoVer's first batch. **Same canonical_id stems** so we get within-feature cross-task comparison.

| cell | features | shared with HoVer |
|---|---|---|
| `query-decomposition` × create_query_hop2 | qd_split_conjunction, qd_one_entity, qd_avoid_negation | yes (3/3) |
| `decision-heuristic` × create_query_hop2 | dh_target_bridge, dh_use_hop1_answer | partial — HoVer cell is `dh_drop_resolved` + `dh_target_unresolved`. The semantics are similar but the rule bodies differ because HoVer is verification (set-subset) and HotpotQA is QA (final answer). Document the divergence. |
| `evidence-summarization` × summarize1 | es_quote_proper_nouns, es_two_sentence_cap, es_skip_irrelevant | yes (3/3) |

Total: 8 features. Add-one sweep on the v5 cube. Coherence check before publishing.

## Modules deliberately not in first batch

- `summarize2`, `final_answer` — defer until summarize1 features show within-task signal.
- `verify-before-output` and `missing-domain-knowledge` — defer for the same reason as HoVer (need clean baseline first).
- `final_answer × edge-case-handling` is tempting (HotpotQA's metric is sensitive to answer-span normalization) but interacts with the existing scorer; defer until the round-9a sweep clarifies the cleaner cells.

## Authoring instructions

1. Each feature file lives at `features/hotpotqa_context/<canonical_id>.json`.
2. `primitive_edits` mirrors HoVer: one `{"$ref": "_sections.<module>"}` + one or more `insert_node` rules.
3. **For features whose stem also exists in the HoVer adapter, the rule body should be byte-identical where the underlying semantics match.** This makes cross-task comparison clean. Where the rule body genuinely needs to differ (e.g., "claim entity" → "question entity"), document the divergence in the feature JSON's `notes` field.
4. Set `requires: []` and `conflicts_with: []` unless a feature explicitly depends on or conflicts with another.

## Sanity checks before launching

- v5 cube already has the round-7 parser fix; new features should not trigger nested-summary parsing issues. Confirm with a 10-query dry run.
- Cross-task A/B should run on **the same model + retrieval depth** — Qwen3-8B, k=15 — to keep the only varying factor the task itself.
- The HotpotQA scoring uses `dspy.evaluate.metrics.hotpot_f1_score` after normalize_text (round 5 fix); features that change answer formatting (e.g., `dh_final_answer_yes_no_normalize`) interact with the scorer and should not be in the first batch.

## Existing v5 baseline reference

- BASE n=300: 0.400 (matches paper 0.423 within 2pp)
- GEPA n=300: 0.457
- Δ +0.057
- Cube: `runs/hotpotqa_paper_repro_v5*.db` (audit + reuse)
