# GEPA-Style Constraint Rule Generation v1

This is a small failure-driven constraint discovery run over the current cube. Candidate constraints are hypotheses for later FACET validation, not accepted features.

Batches: `4`
Candidate constraints parsed: `16`

## Batches

- `wtq_surface_role_failures` (wtq): 6 failures, 2 success contrasts. Discover small source/table rules for atomic values, row-order navigation, rank-vs-value, and identifier-vs-name confusions.
- `sqa_followup_active_set_failures` (sqa): 6 failures, 2 success contrasts. Discover task-local follow-up rules for active-set filtering, one-answer-per-active-item, projection after history, and anchor exclusion.
- `tablebench_numeric_operation_failures` (tablebench): 6 failures, 2 success contrasts. Discover operation rules for bind-then-compute, unit-aware comparison, ranking direction, and aggregate scope.
- `hitab_hierarchy_source_failures` (hitab): 6 failures, 2 success contrasts. Discover hierarchy/source rules for header-path inheritance, source units, total/subcategory roles, and pair argmax/argmin cells.

## Candidate Constraints

- `wtq_surface_role_failures.surface_role_score_sum` [domain_heuristic, risk=low]: If the question asks about summing scores, then only sum the numeric values in the score column, not other columns. Trigger: Question contains words like 'score', 'points', 'total', and 'sum'
- `wtq_surface_role_failures.surface_role_rank_vs_value` [response_contract, risk=medium]: If the question asks about ranks, then do not treat them as numeric values for arithmetic operations. Trigger: Question contains words like 'rank', 'position', or 'place'
- `wtq_surface_role_failures.surface_role_identifier_vs_name` [task_heuristic, risk=low]: If the question asks about names or titles, then do not confuse them with identifiers or codes. Trigger: Question contains words like 'name', 'title', 'team', or 'school'
- `wtq_surface_role_failures.surface_role_row_order_navigation` [input_context, risk=low]: If the question asks about previous or next entries, then navigate based on row order, not value. Trigger: Question contains words like 'previous', 'next', 'before', or 'after'
- `sqa_followup_active_set_failures.followup_active_set_filter` [response_contract, risk=low]: If the follow-up question references a previous answer set, only use the items from that set for the new query. Trigger: Follow-up question mentions a previously answered set of items.
- `sqa_followup_active_set_failures.followup_project_correct_column` [task_heuristic, risk=low]: If the follow-up question asks about a specific attribute of the active set, project that column from the table. Trigger: Follow-up question asks for a specific attribute of the active set.
- `sqa_followup_active_set_failures.followup_answer_only_one_item` [response_contract, risk=low]: If the follow-up question asks for a single item from the active set, provide only one answer. Trigger: Follow-up question asks for a single item from the active set.
- `sqa_followup_active_set_failures.followup_handle_rank_questions` [domain_heuristic, risk=medium]: If the follow-up question involves ranking, ensure the answer reflects the correct rank position. Trigger: Follow-up question involves ranking or ordinal positions.
- `tablebench_numeric_operation_failures.bind_then_compute` [domain_heuristic, risk=low]: If the question asks for a ranked value, bind the relevant column before computing the final answer. Trigger: Question contains 'highest', 'lowest', 'ranked' or similar terms.
- `tablebench_numeric_operation_failures.unit_aware_comparison` [domain_heuristic, risk=low]: If comparing values with units, convert all values to the same unit before comparison. Trigger: Question mentions different units for the same quantity.
- `tablebench_numeric_operation_failures.time_filter_scope` [task_heuristic, risk=medium]: When filtering based on time, ensure the filter applies to the entire dataset, not just visible rows. Trigger: Question specifies a time range or period.
- `tablebench_numeric_operation_failures.aggregate_across_all` [task_heuristic, risk=medium]: For questions asking about an average or sum across all entries, include all relevant rows in the calculation. Trigger: Question asks for an average or sum of all entries.
- `hitab_hierarchy_source_failures.check_hierarchy_for_comparison` [domain_heuristic, risk=low]: If the question involves comparing values across categories, the model must identify the correct subcategories and compare them directly, not aggregate or misinterpret totals. Trigger: Question contains words like 'higher', 'lower', 'more', 'less' and mentions specific categories.
- `hitab_hierarchy_source_failures.respect_negative_values_in_tables` [response_contract, risk=low]: If the table contains negative values, the model must preserve these values and not invert their signs. Trigger: Table contains negative numbers.
- `hitab_hierarchy_source_failures.correctly_handle_aggregated_responses` [task_heuristic, risk=medium]: If the question requires aggregating multiple values (e.g., sum, difference), the model must perform the correct aggregation operation. Trigger: Question explicitly asks for an aggregated value (e.g., 'total', 'difference').
- `hitab_hierarchy_source_failures.identify_most_likely_category` [private_reasoning_rule, risk=low]: If the question asks for the most likely category based on percentages, the model must select the category with the highest percentage. Trigger: Question asks for the 'most likely' or 'highest probability' category.

## Raw Response Status

- `wtq_surface_role_failures` via port `8000`: ok, latency=40.204s, finish=stop
- `sqa_followup_active_set_failures` via port `8001`: ok, latency=41.269s, finish=stop
- `tablebench_numeric_operation_failures` via port `8002`: ok, latency=41.13s, finish=stop
- `hitab_hierarchy_source_failures` via port `8003`: ok, latency=40.134s, finish=stop
