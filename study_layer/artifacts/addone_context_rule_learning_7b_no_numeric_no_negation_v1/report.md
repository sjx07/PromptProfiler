# 7B Add-One Context Rule Learning v1, Numeric and Negation Context Excluded

Existing 7B results only. Each feature is treated as a paired add-one intervention against the benchmark-local base. This rerun excludes numeric/table-size/value-surface context atoms and `intent.negation_marker` before mining order-1 and order-2 rules.

## Run Metadata
- `db`: `/data/users/jsu323/facet/wikitable_reasoning_default_addone_v1.db`
- `model`: `Qwen/Qwen2.5-7B-Instruct`
- `datasets`: `['wtq', 'sqa', 'tablebench', 'tab_fact', 'hitab']`
- `max_order`: `2`
- `within_min_support`: `80`
- `across_min_total_support`: `200`
- `across_min_dataset_support`: `30`
- `include_negative_context`: `False`
- `exclude_numeric_context`: `True`
- `excluded_context_names`: `['intent.negation_marker']`
- `excluded_numeric_context_names`: `['cell.has_comma_number', 'cell.has_percent', 'cell.has_range_or_score', 'intent.arithmetic_op_marker', 'intent.percent_or_ratio', 'schema.has_rank_col', 'schema.has_score_col', 'schema.has_unit_col', 'table.cols_bin', 'table.numeric_cols_bin', 'table.numeric_density_bin', 'table.rows_bin', 'table.shape', 'text.has_number', 'text.has_ordinal', 'text.has_year']`
- `n_observations`: `305298`
- `n_feature_summaries`: `70`
- `n_within_rules`: `4942`
- `n_across_rules`: `1008`

## Global Add-One Feature Effects

Global feature effects are unchanged by context filtering; shown for reference.

Top positive mean deltas:
- `hitab::ser.records` (serialization): mean +3.2 pts, useful 11.7%, harmed 8.5%
- `tab_fact::ser.records` (serialization): mean +3.0 pts, useful 10.3%, harmed 7.4%
- `wtq::reason.extract_compute` (reasoning): mean +2.3 pts, useful 10.7%, harmed 8.4%
- `wtq::ser.records` (serialization): mean +2.3 pts, useful 11.8%, harmed 9.5%
- `tab_fact::ser.html` (serialization): mean +2.2 pts, useful 9.7%, harmed 7.4%
- `sqa::reason.enumerate` (reasoning): mean +2.2 pts, useful 8.3%, harmed 6.1%
- `tablebench::reason.extract_compute` (reasoning): mean +2.1 pts, useful 14.2%, harmed 10.2%
- `tablebench::ser.records` (serialization): mean +1.9 pts, useful 14.0%, harmed 11.8%
- `wtq::ser.html` (serialization): mean +1.9 pts, useful 10.6%, harmed 8.7%
- `wtq::reason.verify` (reasoning): mean +1.8 pts, useful 8.9%, harmed 7.0%
- `sqa::reason.extract_compute` (reasoning): mean +1.8 pts, useful 7.2%, harmed 5.5%
- `sqa::reason.evidence_table` (reasoning): mean +1.6 pts, useful 7.9%, harmed 6.3%

Top negative mean deltas:
- `wtq::format.json` (format): mean -8.6 pts, useful 7.2%, harmed 15.8%
- `tablebench::reason.symbolic_op` (reasoning): mean -7.1 pts, useful 10.0%, harmed 18.9%
- `wtq::reason.plan_answer` (reasoning): mean -6.9 pts, useful 7.8%, harmed 14.7%
- `wtq::reason.symbolic_op` (reasoning): mean -3.7 pts, useful 8.4%, harmed 12.1%
- `hitab::reason.plan_answer` (reasoning): mean -3.6 pts, useful 7.2%, harmed 10.8%
- `hitab::format.json` (format): mean -2.9 pts, useful 8.2%, harmed 11.1%
- `tablebench::format.json` (format): mean -2.2 pts, useful 11.4%, harmed 14.7%
- `tab_fact::reason.plan_answer` (reasoning): mean -1.8 pts, useful 9.8%, harmed 11.7%
- `hitab::ctx.stats` (input_context): mean -1.8 pts, useful 8.6%, harmed 10.4%
- `hitab::reason.critique_revise` (reasoning): mean -1.5 pts, useful 5.3%, harmed 6.8%
- `sqa::reason.plan_answer` (reasoning): mean -1.2 pts, useful 7.1%, harmed 8.3%
- `wtq::reason.evidence_table` (reasoning): mean -1.2 pts, useful 9.5%, harmed 10.7%

## Within-Benchmark Order 1 Rules

Positive conditional lift:
- `tablebench::format.json` when `native.tablebench_qtype=FactChecking`: n=96, lift +7.4 pts, useful-rate lift -2.0 pts, CI [+0.1 pts, +14.8 pts]
- `hitab::reason.evidence_table` when `native.hitab_source_family=nsf`: n=97, lift +6.2 pts, useful-rate lift +0.7 pts, CI [-0.4 pts, +12.9 pts]
- `hitab::reason.extract_compute` when `native.hitab_source_family=nsf`: n=97, lift +6.2 pts, useful-rate lift +0.1 pts, CI [+0.2 pts, +12.2 pts]
- `tablebench::reason.symbolic_op` when `native.tablebench_qtype=DataAnalysis`: n=343, lift +6.1 pts, useful-rate lift +5.4 pts, CI [+4.1 pts, +8.1 pts]
- `sqa::reason.verify` when `dialog.turn_bin=late`: n=184, lift +5.8 pts, useful-rate lift +2.2 pts, CI [+1.1 pts, +10.5 pts]
- `tablebench::ctx.type` when `schema.has_date_col=yes`: n=298, lift +5.1 pts, useful-rate lift +3.5 pts, CI [+1.2 pts, +9.1 pts]
- `wtq::reason.critique_revise` when `intent.comparison=yes`: n=148, lift +5.1 pts, useful-rate lift +4.8 pts, CI [-2.1 pts, +12.3 pts]
- `tablebench::reason.symbolic_op` when `native.tablebench_qtype=FactChecking`: n=96, lift +5.0 pts, useful-rate lift -2.8 pts, CI [-3.2 pts, +13.2 pts]

Negative conditional lift:
- `wtq::ser.records` when `schema.header_repetition_marker=yes`: n=162, lift -17.1 pts, useful-rate lift -6.2 pts, CI [-24.6 pts, -9.6 pts]
- `wtq::format.json` when `intent.comparison=yes`: n=146, lift -11.3 pts, useful-rate lift -1.0 pts, CI [-19.9 pts, -2.6 pts]
- `hitab::ser.html` when `intent.reduction_marker=yes`: n=161, lift -8.9 pts, useful-rate lift -5.3 pts, CI [-15.0 pts, -2.8 pts]
- `sqa::reason.verify` when `intent.count=yes`: n=168, lift -7.3 pts, useful-rate lift -4.1 pts, CI [-12.3 pts, -2.3 pts]
- `wtq::reason.plan_answer` when `intent.comparison=yes`: n=148, lift -7.3 pts, useful-rate lift -1.1 pts, CI [-15.5 pts, +0.9 pts]
- `tablebench::reason.symbolic_op` when `native.tablebench_qtype=NumericalReasoning`: n=397, lift -6.5 pts, useful-rate lift -4.0 pts, CI [-11.3 pts, -1.7 pts]
- `tablebench::reason.symbolic_op` when `intent.superlative_rank=yes`: n=243, lift -5.8 pts, useful-rate lift -3.5 pts, CI [-11.9 pts, +0.2 pts]
- `wtq::reason.evidence_table` when `intent.comparison=yes`: n=148, lift -5.6 pts, useful-rate lift +0.6 pts, CI [-13.9 pts, +2.7 pts]

## Within-Benchmark Order 2 Rules

Positive conditional lift:
- `tablebench::reason.evidence_table` when `intent.reduction_marker=yes AND schema.has_date_col=yes`: n=103, lift +10.2 pts, useful-rate lift +2.9 pts, CI [+1.4 pts, +19.0 pts]
- `wtq::reason.critique_revise` when `grounding.header_overlap=yes AND intent.comparison=yes`: n=96, lift +10.1 pts, useful-rate lift +7.6 pts, CI [+1.2 pts, +19.1 pts]
- `tablebench::reason.plan_answer` when `intent.reduction_marker=yes AND schema.has_date_col=yes`: n=103, lift +10.1 pts, useful-rate lift +3.5 pts, CI [+1.3 pts, +18.9 pts]
- `tablebench::reason.plan_answer` when `native.tablebench_qtype=NumericalReasoning AND schema.has_date_col=yes`: n=126, lift +9.8 pts, useful-rate lift +5.1 pts, CI [+0.6 pts, +19.0 pts]
- `tablebench::ctx.type` when `intent.superlative_rank=yes AND schema.has_date_col=yes`: n=82, lift +9.6 pts, useful-rate lift +5.0 pts, CI [-0.3 pts, +19.6 pts]
- `tablebench::ctx.stats` when `native.tablebench_qtype=NumericalReasoning AND schema.has_date_col=yes`: n=126, lift +9.5 pts, useful-rate lift +3.4 pts, CI [+1.4 pts, +17.6 pts]
- `tablebench::reason.evidence_table` when `native.tablebench_qtype=NumericalReasoning AND schema.has_date_col=yes`: n=126, lift +9.2 pts, useful-rate lift +3.8 pts, CI [+0.4 pts, +18.1 pts]
- `hitab::format.json` when `grounding.header_overlap=yes AND native.hitab_source_family=nsf`: n=80, lift +9.2 pts, useful-rate lift +3.0 pts, CI [+0.4 pts, +17.9 pts]

Negative conditional lift:
- `wtq::ser.records` when `cell.missing_value_marker=yes AND schema.header_repetition_marker=yes`: n=93, lift -18.4 pts, useful-rate lift -7.5 pts, CI [-28.0 pts, -8.8 pts]
- `wtq::ser.records` when `schema.has_date_col=yes AND schema.header_repetition_marker=yes`: n=114, lift -17.2 pts, useful-rate lift -6.5 pts, CI [-26.1 pts, -8.3 pts]
- `wtq::ser.records` when `cell.has_date_like=yes AND schema.header_repetition_marker=yes`: n=147, lift -16.6 pts, useful-rate lift -5.7 pts, CI [-24.6 pts, -8.5 pts]
- `wtq::ser.records` when `grounding.cell_overlap=yes AND schema.header_repetition_marker=yes`: n=105, lift -16.6 pts, useful-rate lift -8.9 pts, CI [-24.7 pts, -8.4 pts]
- `wtq::ser.records` when `grounding.header_overlap=yes AND schema.header_repetition_marker=yes`: n=98, lift -16.6 pts, useful-rate lift -6.7 pts, CI [-26.0 pts, -7.1 pts]
- `wtq::ser.records` when `schema.has_entity_col=yes AND schema.header_repetition_marker=yes`: n=86, lift -16.2 pts, useful-rate lift -7.1 pts, CI [-26.1 pts, -6.4 pts]
- `hitab::ser.html` when `cell.has_date_like=yes AND intent.reduction_marker=yes`: n=100, lift -14.5 pts, useful-rate lift -6.7 pts, CI [-22.6 pts, -6.3 pts]
- `wtq::format.json` when `grounding.cell_overlap=yes AND intent.comparison=yes`: n=94, lift -13.7 pts, useful-rate lift -0.8 pts, CI [-24.9 pts, -2.6 pts]

## Across-Benchmark Order 1 Rules

Positive useful-rate lift:
- `reason.verify` when `intent.comparison=yes`: benches=hitab,sqa,tab_fact,tablebench,wtq, n=1187, macro useful lift +1.5 pts, weighted +0.8 pts, signs +3/-2
- `ctx.type` when `schema.header_repetition_marker=yes`: benches=hitab,sqa,wtq, n=1347, macro useful lift +1.0 pts, weighted +0.6 pts, signs +3/-0
- `format.json` when `intent.superlative_rank=yes`: benches=hitab,sqa,tab_fact,tablebench,wtq, n=4694, macro useful lift +0.7 pts, weighted +0.9 pts, signs +3/-2
- `reason.plan_answer` when `schema.header_repetition_marker=yes`: benches=hitab,sqa,wtq, n=1347, macro useful lift +0.7 pts, weighted +0.7 pts, signs +2/-1
- `ctx.type` when `intent.comparison=yes`: benches=hitab,sqa,tab_fact,tablebench,wtq, n=1187, macro useful lift +0.6 pts, weighted +0.4 pts, signs +2/-3
- `reason.symbolic_op` when `cell.missing_value_marker=yes`: benches=hitab,sqa,wtq, n=3963, macro useful lift +0.6 pts, weighted +0.4 pts, signs +2/-1
- `reason.symbolic_op` when `intent.comparison=yes`: benches=hitab,sqa,tab_fact,tablebench,wtq, n=1187, macro useful lift +0.5 pts, weighted +0.2 pts, signs +3/-2
- `ctx.type` when `schema.has_date_col=yes`: benches=hitab,sqa,tab_fact,tablebench,wtq, n=11385, macro useful lift +0.5 pts, weighted +0.1 pts, signs +3/-2

Negative useful-rate lift:
- `ser.records` when `schema.header_repetition_marker=yes`: benches=hitab,sqa,wtq, n=1346, macro useful lift -3.9 pts, weighted -0.3 pts, signs +1/-2
- `reason.enumerate` when `intent.reduction_marker=yes`: benches=hitab,sqa,tab_fact,tablebench,wtq, n=1666, macro useful lift -2.6 pts, weighted -1.0 pts, signs +2/-3
- `reason.evidence_localize` when `schema.header_repetition_marker=yes`: benches=hitab,sqa,wtq, n=1347, macro useful lift -2.3 pts, weighted -0.3 pts, signs +1/-2
- `reason.decompose` when `intent.reduction_marker=yes`: benches=hitab,sqa,tab_fact,tablebench,wtq, n=1666, macro useful lift -2.1 pts, weighted -1.5 pts, signs +1/-4
- `ctx.stats` when `intent.reduction_marker=yes`: benches=hitab,sqa,tab_fact,tablebench,wtq, n=1666, macro useful lift -2.1 pts, weighted -1.1 pts, signs +1/-4
- `reason.extract_compute` when `intent.reduction_marker=yes`: benches=hitab,sqa,tab_fact,tablebench,wtq, n=1666, macro useful lift -2.0 pts, weighted -1.4 pts, signs +1/-4
- `reason.enumerate` when `schema.header_repetition_marker=yes`: benches=hitab,sqa,wtq, n=1347, macro useful lift -2.0 pts, weighted -0.2 pts, signs +1/-2
- `reason.critique_revise` when `intent.reduction_marker=yes`: benches=hitab,sqa,tab_fact,tablebench,wtq, n=1666, macro useful lift -2.0 pts, weighted -1.0 pts, signs +0/-5

## Across-Benchmark Order 2 Rules

Positive useful-rate lift:
- `ctx.type` when `intent.count=yes AND schema.header_repetition_marker=yes`: benches=hitab,wtq, n=391, macro useful lift +5.0 pts, weighted +1.9 pts, signs +2/-0
- `reason.decompose` when `cell.has_date_like=yes AND intent.comparison=yes`: benches=sqa,tab_fact,wtq, n=791, macro useful lift +4.1 pts, weighted +0.6 pts, signs +2/-1
- `reason.evidence_table` when `intent.count=yes AND intent.superlative_rank=yes`: benches=hitab,tab_fact,tablebench,wtq, n=533, macro useful lift +3.9 pts, weighted +4.3 pts, signs +4/-0
- `reason.verify` when `cell.has_date_like=yes AND intent.comparison=yes`: benches=sqa,tab_fact,wtq, n=791, macro useful lift +3.7 pts, weighted +0.8 pts, signs +3/-0
- `reason.plan_answer` when `intent.count=yes AND intent.superlative_rank=yes`: benches=hitab,tab_fact,tablebench,wtq, n=533, macro useful lift +3.4 pts, weighted +3.3 pts, signs +4/-0
- `reason.critique_revise` when `intent.comparison=yes AND intent.temporal=yes`: benches=tab_fact,wtq, n=399, macro useful lift +3.4 pts, weighted +1.1 pts, signs +2/-0
- `reason.extract_compute` when `intent.comparison=yes AND schema.has_date_col=yes`: benches=tab_fact,wtq, n=533, macro useful lift +3.4 pts, weighted +2.9 pts, signs +2/-0
- `reason.critique_revise` when `intent.count=yes AND schema.header_repetition_marker=yes`: benches=hitab,wtq, n=391, macro useful lift +3.1 pts, weighted +0.2 pts, signs +1/-1

Negative useful-rate lift:
- `ser.records` when `schema.has_entity_col=yes AND schema.header_repetition_marker=yes`: benches=hitab,wtq, n=434, macro useful lift -6.0 pts, weighted -5.3 pts, signs +0/-2
- `ser.records` when `schema.has_date_col=yes AND schema.header_repetition_marker=yes`: benches=hitab,sqa,wtq, n=422, macro useful lift -5.7 pts, weighted -5.5 pts, signs +0/-3
- `reason.evidence_localize` when `intent.temporal=yes AND schema.header_repetition_marker=yes`: benches=hitab,wtq, n=675, macro useful lift -5.0 pts, weighted -1.9 pts, signs +0/-2
- `ser.records` when `grounding.header_overlap=yes AND schema.header_repetition_marker=yes`: benches=hitab,sqa,wtq, n=1015, macro useful lift -4.9 pts, weighted -0.8 pts, signs +1/-2
- `ser.records` when `intent.temporal=yes AND schema.header_repetition_marker=yes`: benches=hitab,wtq, n=675, macro useful lift -4.8 pts, weighted -0.6 pts, signs +1/-1
- `ser.records` when `cell.has_date_like=yes AND schema.header_repetition_marker=yes`: benches=hitab,sqa,wtq, n=911, macro useful lift -4.5 pts, weighted -2.6 pts, signs +0/-3
- `ser.records` when `grounding.cell_overlap=yes AND schema.header_repetition_marker=yes`: benches=hitab,wtq, n=1207, macro useful lift -4.0 pts, weighted -0.0 pts, signs +1/-1
- `ser.records` when `cell.missing_value_marker=yes AND schema.header_repetition_marker=yes`: benches=hitab,sqa,wtq, n=1250, macro useful lift -3.8 pts, weighted +0.1 pts, signs +1/-2

## Files
- `rule_learning_dashboard.html`
- `feature_summary.csv`
- `within_rules_order1.csv`
- `within_rules_order2.csv`
- `across_rules_order1.csv`
- `across_rules_order2.csv`
- `observations.csv`
- `metadata.json`
