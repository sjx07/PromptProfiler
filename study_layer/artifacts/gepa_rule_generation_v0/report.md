# GEPA-Style Rule Generation v0

This is a small failure-driven rule discovery run over the current cube. Candidate rules are hypotheses for later FACET validation, not accepted features.

Batches: `4`
Candidate rules parsed: `19`

## Batches

- `wtq_surface_role_failures` (wtq): 6 failures, 2 success contrasts. Discover small source/table rules for atomic values, row-order navigation, rank-vs-value, and identifier-vs-name confusions.
- `sqa_followup_active_set_failures` (sqa): 6 failures, 2 success contrasts. Discover task-local follow-up rules for active-set filtering, one-answer-per-active-item, projection after history, and anchor exclusion.
- `tablebench_numeric_operation_failures` (tablebench): 6 failures, 2 success contrasts. Discover operation rules for bind-then-compute, unit-aware comparison, ranking direction, and aggregate scope.
- `hitab_hierarchy_source_failures` (hitab): 6 failures, 2 success contrasts. Discover hierarchy/source rules for header-path inheritance, source units, total/subcategory roles, and pair argmax/argmin cells.

## Candidate Rules

- `wtq_surface_role_failures.check_for_named_entities` [input_context, risk=low]: Ensure that names are treated distinctly from numerical identifiers or scores. Trigger: When the question involves named entities like team names or player names.
- `wtq_surface_role_failures.verify_rank_vs_score` [private_reasoning_rule, risk=medium]: Distinguish between rank positions and actual scores or counts. Trigger: When the question asks about rankings or positions instead of totals or averages.
- `wtq_surface_role_failures.correct_row_order_navigation` [response_contract, risk=low]: Navigate tables correctly based on row order and position queries. Trigger: When the question involves specific row positions or sequences.
- `wtq_surface_role_failures.identify_and_summarize_scores` [domain_heuristic, risk=low]: Summarize scores correctly by identifying relevant columns and summing appropriate values. Trigger: When the question asks for a total or average score.
- `sqa_followup_active_set_failures.filter_by_prior_answer` [input_context, risk=low]: Filter the current query to only consider items mentioned in the previous answer set. Trigger: When the current question references a set of items previously answered.
- `sqa_followup_active_set_failures.project_on_prior_answer` [response_contract, risk=medium]: Project the current query results onto the prior answer set. Trigger: When the current question requires projecting results back onto a previously filtered set.
- `sqa_followup_active_set_failures.exclude_anchor_items` [private_reasoning_rule, risk=low]: Exclude items from the current query if they were anchors in the previous question. Trigger: When the current question follows a question that anchored on specific items.
- `tablebench_numeric_operation_failures.bind_then_compute` [private_reasoning_rule, risk=low]: Bind the relevant columns before performing any numerical operations. Trigger: When the question involves ranking, aggregation, or comparisons based on specific columns.
- `tablebench_numeric_operation_failures.unit_aware_comparison` [private_reasoning_rule, risk=low]: Ensure units match before comparing or aggregating numerical values. Trigger: When the question involves comparisons or aggregations with units.
- `tablebench_numeric_operation_failures.rank_direction` [private_reasoning_rule, risk=low]: Identify if the ranking should be ascending or descending based on the question. Trigger: When the question involves ranking.
- `tablebench_numeric_operation_failures.aggregate_scope` [private_reasoning_rule, risk=low]: Define the scope of aggregation clearly based on the question's timeframe or conditions. Trigger: When the question involves time-based calculations or conditional aggregations.
- `tablebench_numeric_operation_failures.count_condition` [private_reasoning_rule, risk=low]: Count items that meet specific conditions rather than listing them. Trigger: When the question involves counting items meeting certain criteria.
- `tablebench_numeric_operation_failures.total_across_series` [private_reasoning_rule, risk=low]: Sum up values across all series or entries as specified. Trigger: When the question involves summing values across multiple series or entries.
- `hitab_hierarchy_source_failures.check_aggregation_type` [private_reasoning_rule, risk=low]: Ensure the correct aggregation function is used based on the question type (e.g., sum, diff, div). Trigger: When the question involves numerical comparisons or differences.
- `hitab_hierarchy_source_failures.verify_comparison_direction` [private_reasoning_rule, risk=medium]: Identify if the comparison is asking for max/min or positive/negative values. Trigger: When the question asks for a comparative measure (e.g., higher/lower, more/less).
- `hitab_hierarchy_source_failures.correct_sign_errors` [private_reasoning_rule, risk=low]: Check for and correct any sign errors in the answer. Trigger: When the answer involves negative numbers or percentages.
- `hitab_hierarchy_source_failures.interpret_complete_language_transfers` [domain_heuristic, risk=high]: Understand 'complete language transfers' as a negative impact indicator. Trigger: When encountering terms like 'complete language transfers' in demographic contexts.
- `hitab_hierarchy_source_failures.handle_percentage_change` [private_reasoning_rule, risk=medium]: Calculate percentage changes correctly using the formula (new-old)/old. Trigger: When the question asks for percentage change over time or between groups.
- `hitab_hierarchy_source_failures.identify_most_likely_unmet_need` [private_reasoning_rule, risk=low]: Find the highest percentage in the 'unmet' column for each need type. Trigger: When the question asks about the most likely unmet need.

## Raw Response Status

- `wtq_surface_role_failures` via port `8000`: ok, latency=35.257s, finish=stop
- `sqa_followup_active_set_failures` via port `8001`: ok, latency=32.663s, finish=stop
- `tablebench_numeric_operation_failures` via port `8002`: ok, latency=48.704s, finish=stop
- `hitab_hierarchy_source_failures` via port `8003`: ok, latency=46.578s, finish=stop
