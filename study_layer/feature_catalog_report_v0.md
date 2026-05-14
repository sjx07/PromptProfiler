# Feature Catalog v0 Report

Generated from existing feature JSON specs only. This round does not add new features or modify the experiment runner.

## Inventory

- Rows: 921
- Tasks: 14
- Concepts: 57
- Cross-task concepts: 42
- Duplicate `task::canonical_id` keys namespace-qualified: 166
- Unknown concept rows: 0
- Low-confidence rows: 169

## Tasks

- `tablebench`       : 100
- `bird`             : 99
- `spider`           : 99
- `wtq`              : 97
- `sql_generation`   : 96
- `hotpotqa_context` : 88
- `hover_context`    : 88
- `sqa`              : 80
- `tabfact`          : 77
- `table_qa`         : 60
- `hitab`            : 27
- `sql_repair`       : 5
- `pupa`             : 4
- `tablebench_repro` : 1

## Semantic Families

- `rule_or_constraint`    : 482
- `reasoning_behavior`    : 232
- `response_mode`         : 104
- `input_context_builder` : 100
- `demonstration_policy`  : 3

## Implementation Surfaces

- `task_or_user_instruction` : 614
- `system_instruction`       : 117
- `auxiliary_output_field`   : 94
- `output_contract`          : 51
- `context_field`            : 34
- `input_transform`          : 11

## Cost Dependencies

- `prompt_only`          : 691
- `parser_change`        : 124
- `runtime_execution`    : 68
- `input_builder_change` : 38

## Concept Confidence

- `medium` : 665
- `low`    : 169
- `high`   : 87

## Top Concepts

- `rule.output_contract_control`                 : 103
- `response.program_of_thought.python`           : 56
- `rule.idiom_shortcut`                          : 54
- `reasoning.trace_guidance`                     : 52
- `rule.general_instruction`                     : 52
- `rule.decision_heuristic`                      : 49
- `reasoning.aggregation_guidance`               : 38
- `rule.edge_case_handling`                      : 38
- `reasoning.verification_guidance`              : 37
- `rule.construction_constraint`                 : 36
- `reasoning.verify_before_output`               : 36
- `rule.prompt_policy`                           : 33
- `input_context.domain_knowledge`               : 29
- `reasoning.staged_reasoning`                   : 28
- `input_context.table_or_schema_representation` : 28
- `response.visible_structured_trace`            : 25
- `input_context.evidence_summary`               : 23
- `reasoning.decomposition`                      : 21
- `structure.section.role`                       : 19
- `response.visible_cot`                         : 16
- `structure.section.rules`                      : 15
- `structure.section.task`                       : 15
- `structure.section.format_fix`                 : 13
- `structure.section.reasoning`                  : 12
- `structure.section.table_handling`             : 12
- `input_context.schema_use_guidance`            : 9
- `structure.section.strategy`                   : 8
- `response.direct_final_only`                   : 6
- `reasoning.extract_then_compute`               : 5
- `structure.section.examples`                   : 4
- `structure.section.procedure`                  : 4
- `demonstration.few_shot`                       : 3
- `input_context.type_annotation`                : 3
- `reasoning.candidate_enumeration`              : 3
- `structure.section.create_query_hop2`          : 2
- `structure.section.summarize1`                 : 2
- `structure.section.summarize2`                 : 2
- `structure.section.sql_rules`                  : 2
- `input_context.column_selection`               : 2
- `input_context.column_statistics`              : 2

## Cross-Task Transfer Candidates

- `structure.section.role`: 11 tasks (bird, hitab, spider, sqa, sql_generation, sql_repair, tabfact, table_qa, tablebench, tablebench_repro, wtq)
- `rule.output_contract_control`: 10 tasks (bird, hitab, hotpotqa_context, spider, sqa, sql_generation, tabfact, table_qa, tablebench, wtq)
- `structure.section.format_fix`: 7 tasks (hitab, sqa, sql_repair, tabfact, table_qa, tablebench, wtq)
- `structure.section.rules`: 7 tasks (bird, hitab, spider, sqa, tabfact, tablebench, wtq)
- `structure.section.strategy`: 7 tasks (sqa, sql_generation, sql_repair, tabfact, table_qa, tablebench, wtq)
- `structure.section.task`: 7 tasks (bird, hitab, spider, sqa, tabfact, tablebench, wtq)
- `input_context.table_or_schema_representation`: 6 tasks (sqa, sql_generation, tabfact, table_qa, tablebench, wtq)
- `reasoning.decomposition`: 6 tasks (hotpotqa_context, hover_context, sql_generation, table_qa, tablebench, wtq)
- `reasoning.verify_before_output`: 6 tasks (hotpotqa_context, hover_context, sql_generation, table_qa, tablebench, wtq)
- `response.visible_cot`: 6 tasks (hitab, sqa, tabfact, table_qa, tablebench, wtq)
- `structure.section.reasoning`: 6 tasks (hitab, sqa, tabfact, table_qa, tablebench, wtq)
- `structure.section.table_handling`: 6 tasks (hitab, sqa, tabfact, table_qa, tablebench, wtq)
- `reasoning.aggregation_guidance`: 5 tasks (bird, spider, sqa, tabfact, wtq)
- `reasoning.trace_guidance`: 5 tasks (bird, spider, sqa, tabfact, wtq)
- `reasoning.verification_guidance`: 5 tasks (bird, spider, sqa, tabfact, wtq)
- `response.program_of_thought.python`: 5 tasks (hitab, sqa, tabfact, tablebench, wtq)
- `response.visible_structured_trace`: 5 tasks (hitab, sqa, tabfact, tablebench, wtq)
- `rule.construction_constraint`: 5 tasks (bird, spider, sqa, tabfact, wtq)
- `rule.general_instruction`: 5 tasks (hotpotqa_context, hover_context, pupa, sql_generation, table_qa)
- `rule.idiom_shortcut`: 5 tasks (bird, spider, sqa, tabfact, wtq)
- `rule.prompt_policy`: 5 tasks (bird, spider, sqa, tabfact, wtq)
- `response.direct_final_only`: 4 tasks (sqa, tabfact, tablebench, wtq)
- `rule.decision_heuristic`: 4 tasks (hotpotqa_context, hover_context, sql_generation, tabfact)
- `input_context.domain_knowledge`: 3 tasks (hotpotqa_context, hover_context, sql_generation)
- `input_context.evidence_summary`: 3 tasks (hotpotqa_context, hover_context, sql_generation)
- `input_context.type_annotation`: 3 tasks (table_qa, tablebench, wtq)
- `reasoning.candidate_enumeration`: 3 tasks (table_qa, tablebench, wtq)
- `reasoning.extract_then_compute`: 3 tasks (table_qa, tablebench, wtq)
- `rule.edge_case_handling`: 3 tasks (hotpotqa_context, hover_context, sql_generation)
- `input_context.column_selection`: 2 tasks (tablebench, wtq)
- `input_context.column_statistics`: 2 tasks (tablebench, wtq)
- `input_context.row_selection`: 2 tasks (tablebench, wtq)
- `reasoning.enumerate_then_select`: 2 tasks (tablebench, wtq)
- `reasoning.evidence_localization`: 2 tasks (tablebench, wtq)
- `reasoning.filter_then_extract`: 2 tasks (tablebench, wtq)
- `reasoning.staged_reasoning`: 2 tasks (hotpotqa_context, hover_context)
- `structure.section.create_query_hop2`: 2 tasks (hotpotqa_context, hover_context)
- `structure.section.examples`: 2 tasks (bird, spider)
- `structure.section.procedure`: 2 tasks (bird, spider)
- `structure.section.sql_rules`: 2 tasks (sql_generation, sql_repair)
- `structure.section.summarize1`: 2 tasks (hotpotqa_context, hover_context)
- `structure.section.summarize2`: 2 tasks (hotpotqa_context, hover_context)

## Rows Needing Review

These rows are usable for inventory, but should not be treated as clean study treatments until reviewed.

- `bird::_section_examples@features` -> `structure.section.examples` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `bird::_section_examples@features_legacy` -> `structure.section.examples` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `bird::_section_procedure@features` -> `structure.section.procedure` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `bird::_section_procedure@features_legacy` -> `structure.section.procedure` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `bird::_section_role@features` -> `structure.section.role` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `bird::_section_role@features_legacy` -> `structure.section.role` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `bird::_section_rules@features` -> `structure.section.rules` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `bird::_section_rules@features_legacy` -> `structure.section.rules` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `bird::_section_task@features` -> `structure.section.task` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `bird::_section_task@features_legacy` -> `structure.section.task` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `hitab::_section_format_fix@features` -> `structure.section.format_fix` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `hitab::_section_format_fix@features_legacy` -> `structure.section.format_fix` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `hitab::_section_reasoning@features` -> `structure.section.reasoning` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `hitab::_section_reasoning@features_legacy` -> `structure.section.reasoning` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `hitab::_section_role@features` -> `structure.section.role` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `hitab::_section_role@features_legacy` -> `structure.section.role` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `hitab::_section_rules@features` -> `structure.section.rules` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `hitab::_section_rules@features_legacy` -> `structure.section.rules` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `hitab::_section_table_handling@features` -> `structure.section.table_handling` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `hitab::_section_table_handling@features_legacy` -> `structure.section.table_handling` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `hitab::_section_task@features` -> `structure.section.task` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `hitab::_section_task@features_legacy` -> `structure.section.task` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `hotpotqa_context::base_create_query_hop2` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `hotpotqa_context::base_summarize1` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `hotpotqa_context::base_summarize2` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `hotpotqa_context::gepa_qwen3_merge_create_query_hop2` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `hotpotqa_context::gepa_qwen3_merge_summarize1` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `hotpotqa_context::gepa_qwen3_merge_summarize2` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `hotpotqa_context::_section_create_query_hop2` -> `structure.section.create_query_hop2` (rule_or_constraint, system_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `hotpotqa_context::_section_final_answer` -> `structure.section.final_answer` (rule_or_constraint, system_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `hotpotqa_context::_section_summarize1` -> `structure.section.summarize1` (rule_or_constraint, system_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `hotpotqa_context::_section_summarize2` -> `structure.section.summarize2` (rule_or_constraint, system_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `hover_context::base_create_query_hop2` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `hover_context::base_create_query_hop3` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `hover_context::base_summarize1` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `hover_context::base_summarize2` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `hover_context::gepa_qwen3_create_query_hop2` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `hover_context::gepa_qwen3_create_query_hop3` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `hover_context::gepa_qwen3_summarize1` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `hover_context::gepa_qwen3_summarize2` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `hover_context::_section_create_query_hop2` -> `structure.section.create_query_hop2` (rule_or_constraint, system_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `hover_context::_section_create_query_hop3` -> `structure.section.create_query_hop3` (rule_or_constraint, system_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `hover_context::_section_summarize1` -> `structure.section.summarize1` (rule_or_constraint, system_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `hover_context::_section_summarize2` -> `structure.section.summarize2` (rule_or_constraint, system_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `pupa::base_craft_redacted_request` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `pupa::base_respond_to_query` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `pupa::_section_privacy_rewrite` -> `structure.section.privacy_rewrite` (rule_or_constraint, system_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `pupa::_section_response_synthesis` -> `structure.section.response_synthesis` (rule_or_constraint, system_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `spider::_section_examples@features` -> `structure.section.examples` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `spider::_section_examples@features_legacy` -> `structure.section.examples` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `spider::_section_procedure@features` -> `structure.section.procedure` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `spider::_section_procedure@features_legacy` -> `structure.section.procedure` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `spider::_section_role@features` -> `structure.section.role` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `spider::_section_role@features_legacy` -> `structure.section.role` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `spider::_section_rules@features` -> `structure.section.rules` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `spider::_section_rules@features_legacy` -> `structure.section.rules` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `spider::_section_task@features` -> `structure.section.task` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `spider::_section_task@features_legacy` -> `structure.section.task` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `sqa::_section_format_fix@features` -> `structure.section.format_fix` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `sqa::_section_format_fix@features_legacy` -> `structure.section.format_fix` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `sqa::_section_reasoning@features` -> `structure.section.reasoning` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `sqa::_section_reasoning@features_legacy` -> `structure.section.reasoning` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `sqa::_section_role@features` -> `structure.section.role` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `sqa::_section_role@features_legacy` -> `structure.section.role` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `sqa::_section_rules@features` -> `structure.section.rules` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `sqa::_section_rules@features_legacy` -> `structure.section.rules` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `sqa::_section_strategy` -> `structure.section.strategy` (rule_or_constraint, system_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `sqa::_section_table_handling@features` -> `structure.section.table_handling` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `sqa::_section_table_handling@features_legacy` -> `structure.section.table_handling` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `sqa::_section_task@features` -> `structure.section.task` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `sqa::_section_task@features_legacy` -> `structure.section.task` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `sql_generation::lv_sql_rules_case_insensitive_string_match` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `sql_generation::lv_sql_rules_quoted_phrase_is_exact` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `sql_generation::lv_sql_rules_substring_over_equality_for_names` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `sql_generation::lv_sql_rules_trim_and_collapse_whitespace` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `sql_generation::mq_sql_rules_no_defensive_not_null` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `sql_generation::mq_sql_rules_no_groupby_extras` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `sql_generation::mq_sql_rules_no_redundant_distinct` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `sql_generation::mq_sql_rules_no_unreferenced_joins` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `sql_generation::_section_output_format` -> `structure.section.output_format` (rule_or_constraint, system_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `sql_generation::_section_role` -> `structure.section.role` (rule_or_constraint, system_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `sql_generation::_section_schema_use` -> `structure.section.schema_use` (rule_or_constraint, system_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `sql_generation::_section_sql_rules` -> `structure.section.sql_rules` (rule_or_constraint, system_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `sql_generation::_section_strategy` -> `structure.section.strategy` (rule_or_constraint, system_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `sql_repair::_section_error_analysis` -> `structure.section.error_analysis` (rule_or_constraint, system_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `sql_repair::_section_format_fix` -> `structure.section.format_fix` (rule_or_constraint, system_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `sql_repair::_section_role` -> `structure.section.role` (rule_or_constraint, system_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `sql_repair::_section_sql_rules` -> `structure.section.sql_rules` (rule_or_constraint, system_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `sql_repair::_section_strategy` -> `structure.section.strategy` (rule_or_constraint, system_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `tabfact::_section_format_fix@features` -> `structure.section.format_fix` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `tabfact::_section_format_fix@features_legacy` -> `structure.section.format_fix` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `tabfact::_section_reasoning@features` -> `structure.section.reasoning` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `tabfact::_section_reasoning@features_legacy` -> `structure.section.reasoning` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `tabfact::_section_role@features` -> `structure.section.role` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `tabfact::_section_role@features_legacy` -> `structure.section.role` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `tabfact::_section_rules@features` -> `structure.section.rules` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `tabfact::_section_rules@features_legacy` -> `structure.section.rules` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `tabfact::_section_strategy` -> `structure.section.strategy` (rule_or_constraint, system_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `tabfact::_section_table_handling@features` -> `structure.section.table_handling` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `tabfact::_section_table_handling@features_legacy` -> `structure.section.table_handling` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `tabfact::_section_task@features` -> `structure.section.task` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `tabfact::_section_task@features_legacy` -> `structure.section.task` (rule_or_constraint, system_instruction, prompt_only): duplicate task::canonical_id; namespace-qualified with source root; low-confidence taxonomy inference; no semantic_labels metadata; structural section component
- `table_qa::alternate_phrasing` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `table_qa::cot_branch_strategy_by_type` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `table_qa::cot_classify_question_first` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `table_qa::cot_compute_aloud` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `table_qa::cot_plan_first` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `table_qa::cot_think_backward` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `table_qa::explicit_uncertainty_in_reasoning` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `table_qa::focus_header_term_mapping` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `table_qa::focus_negation_scope` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `table_qa::focus_primary_key_identification` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `table_qa::focus_row_uniqueness` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `table_qa::focus_superlative_full_scan` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `table_qa::imagine_as_sql_mental` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `table_qa::imagine_simpler_question` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `table_qa::no_aggregation_without_request` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `table_qa::no_external_knowledge` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `table_qa::no_invention` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- `table_qa::no_silent_unit_conversion` -> `rule.general_instruction` (rule_or_constraint, task_or_user_instruction, prompt_only): low-confidence taxonomy inference; no semantic_labels metadata
- ... 49 additional review rows omitted from report

## Interpretation

- `feature_id` remains the content/provenance key.
- `component_id` is the runnable row key. Duplicate `task::canonical_id` rows from active and legacy roots are suffixed with `@features` or `@features_legacy`.
- `concept_id` is the study-facing transfer label. It is intentionally coarser than concrete implementation.
- Response modes are represented as features, but they can still serve as baselines in later EffectSpec rows.
- Rows with low confidence or unknown concepts are the right next review target before controlled experiments.
