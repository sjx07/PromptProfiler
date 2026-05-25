# 7B Add-One Context Rule Learning v1

Existing 7B results only. Each feature is treated as a paired add-one intervention against the benchmark-local base. Within-benchmark rules use raw paired delta; across-benchmark rules use useful-rate lift over the feature global rate within each benchmark.

## Run Metadata
- `db`: `/data/users/jsu323/facet/wikitable_reasoning_default_addone_v1.db`
- `model`: `Qwen/Qwen2.5-7B-Instruct`
- `datasets`: `['wtq', 'sqa', 'tablebench', 'tab_fact', 'hitab']`
- `max_order`: `2`
- `within_min_support`: `80`
- `across_min_total_support`: `200`
- `across_min_dataset_support`: `30`
- `include_negative_context`: `False`
- `n_observations`: `305298`
- `n_feature_summaries`: `70`
- `n_within_rules`: `33057`
- `n_across_rules`: `9177`

## Global Add-One Feature Effects

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
- `wtq::format.json` when `intent.arithmetic_op_marker=yes`: n=128, lift +10.9 pts, useful-rate lift +3.8 pts, CI [+3.3 pts, +18.6 pts]
- `tab_fact::ser.records` when `intent.percent_or_ratio=yes`: n=144, lift +10.9 pts, useful-rate lift +8.4 pts, CI [+3.3 pts, +18.6 pts]
- `wtq::reason.plan_answer` when `intent.arithmetic_op_marker=yes`: n=128, lift +10.0 pts, useful-rate lift +3.9 pts, CI [+2.2 pts, +17.8 pts]
- `sqa::reason.enumerate` when `intent.negation_marker=yes`: n=91, lift +9.9 pts, useful-rate lift +10.4 pts, CI [-0.2 pts, +20.0 pts]
- `tab_fact::reason.plan_answer` when `intent.percent_or_ratio=yes`: n=144, lift +9.5 pts, useful-rate lift +6.8 pts, CI [+1.2 pts, +17.7 pts]
- `hitab::format.json` when `cell.has_percent=yes`: n=154, lift +9.4 pts, useful-rate lift +4.8 pts, CI [+2.5 pts, +16.3 pts]
- `hitab::ctx.stats` when `cell.has_percent=yes`: n=154, lift +8.9 pts, useful-rate lift +5.1 pts, CI [+1.9 pts, +15.9 pts]
- `wtq::reason.verify` when `table.cols_bin=gt_10`: n=96, lift +8.6 pts, useful-rate lift +7.8 pts, CI [-0.8 pts, +18.0 pts]

Negative conditional lift:
- `wtq::ser.records` when `schema.header_repetition_marker=yes`: n=162, lift -17.1 pts, useful-rate lift -6.2 pts, CI [-24.6 pts, -9.6 pts]
- `hitab::ser.records` when `table.cols_bin=0_3`: n=107, lift -11.6 pts, useful-rate lift -6.1 pts, CI [-19.9 pts, -3.4 pts]
- `wtq::format.json` when `intent.comparison=yes`: n=146, lift -11.3 pts, useful-rate lift -1.0 pts, CI [-19.9 pts, -2.6 pts]
- `hitab::ser.html` when `table.rows_bin=gt_50`: n=103, lift -11.2 pts, useful-rate lift -0.9 pts, CI [-21.1 pts, -1.2 pts]
- `hitab::ser.html` when `intent.reduction_marker=yes`: n=161, lift -8.9 pts, useful-rate lift -5.3 pts, CI [-15.0 pts, -2.8 pts]
- `hitab::reason.evidence_localize` when `table.rows_bin=gt_50`: n=103, lift -8.4 pts, useful-rate lift -2.3 pts, CI [-16.1 pts, -0.7 pts]
- `hitab::ser.records` when `table.shape=tall`: n=81, lift -8.2 pts, useful-rate lift -6.7 pts, CI [-16.5 pts, +0.2 pts]
- `wtq::reason.decompose` when `cell.has_percent=yes`: n=129, lift -8.1 pts, useful-rate lift -2.9 pts, CI [-15.6 pts, -0.6 pts]

## Within-Benchmark Order 2 Rules

Positive conditional lift:
- `wtq::reason.plan_answer` when `intent.arithmetic_op_marker=yes AND table.cols_bin=4_6`: n=87, lift +14.9 pts, useful-rate lift +6.0 pts, CI [+5.8 pts, +24.1 pts]
- `wtq::format.json` when `grounding.cell_overlap=yes AND intent.arithmetic_op_marker=yes`: n=104, lift +14.4 pts, useful-rate lift +4.4 pts, CI [+6.4 pts, +22.3 pts]
- `tab_fact::reason.evidence_table` when `intent.negation_marker=yes AND table.numeric_density_bin=high`: n=122, lift +13.7 pts, useful-rate lift +12.2 pts, CI [+4.5 pts, +22.9 pts]
- `sqa::reason.symbolic_op` when `table.numeric_density_bin=high AND text.has_number=yes`: n=130, lift +13.1 pts, useful-rate lift +11.3 pts, CI [+4.8 pts, +21.4 pts]
- `tab_fact::reason.plan_answer` when `table.numeric_cols_bin=ge_4 AND text.has_ordinal=yes`: n=161, lift +13.0 pts, useful-rate lift +6.3 pts, CI [+6.1 pts, +19.9 pts]
- `wtq::reason.plan_answer` when `intent.arithmetic_op_marker=yes AND table.shape=balanced`: n=85, lift +12.8 pts, useful-rate lift +6.3 pts, CI [+2.7 pts, +22.8 pts]
- `tablebench::reason.extract_compute` when `native.tablebench_qtype=NumericalReasoning AND table.numeric_density_bin=mid`: n=83, lift +12.3 pts, useful-rate lift +0.2 pts, CI [+4.7 pts, +20.0 pts]
- `wtq::reason.plan_answer` when `intent.temporal=yes AND schema.has_unit_col=yes`: n=93, lift +12.3 pts, useful-rate lift +5.1 pts, CI [+3.1 pts, +21.4 pts]

Negative conditional lift:
- `hitab::format.json` when `cell.has_comma_number=yes AND schema.has_date_col=yes`: n=93, lift -24.0 pts, useful-rate lift -5.0 pts, CI [-34.4 pts, -13.5 pts]
- `hitab::format.json` when `cell.has_comma_number=yes AND native.hitab_source_family=totto`: n=95, lift -20.3 pts, useful-rate lift -4.0 pts, CI [-30.6 pts, -9.9 pts]
- `wtq::ser.records` when `cell.has_range_or_score=yes AND schema.header_repetition_marker=yes`: n=101, lift -20.1 pts, useful-rate lift -8.8 pts, CI [-29.0 pts, -11.2 pts]
- `wtq::ser.records` when `cell.missing_value_marker=yes AND schema.header_repetition_marker=yes`: n=93, lift -18.4 pts, useful-rate lift -7.5 pts, CI [-28.0 pts, -8.8 pts]
- `wtq::ser.records` when `schema.has_date_col=yes AND schema.header_repetition_marker=yes`: n=114, lift -17.2 pts, useful-rate lift -6.5 pts, CI [-26.1 pts, -8.3 pts]
- `wtq::ser.records` when `cell.has_date_like=yes AND schema.header_repetition_marker=yes`: n=147, lift -16.6 pts, useful-rate lift -5.7 pts, CI [-24.6 pts, -8.5 pts]
- `wtq::ser.records` when `grounding.cell_overlap=yes AND schema.header_repetition_marker=yes`: n=105, lift -16.6 pts, useful-rate lift -8.9 pts, CI [-24.7 pts, -8.4 pts]
- `wtq::ser.records` when `grounding.header_overlap=yes AND schema.header_repetition_marker=yes`: n=98, lift -16.6 pts, useful-rate lift -6.7 pts, CI [-26.0 pts, -7.1 pts]

## Across-Benchmark Order 1 Rules

Positive useful-rate lift:
- `reason.evidence_table` when `table.cols_bin=gt_10`: benches=hitab,tab_fact,tablebench,wtq, n=794, macro useful lift +3.8 pts, weighted +1.7 pts, signs +4/-0
- `reason.decompose` when `intent.negation_marker=yes`: benches=hitab,sqa,tab_fact,tablebench,wtq, n=1180, macro useful lift +3.3 pts, weighted +2.9 pts, signs +4/-1
- `ctx.stats` when `table.cols_bin=gt_10`: benches=hitab,tab_fact,tablebench,wtq, n=794, macro useful lift +2.6 pts, weighted +1.9 pts, signs +4/-0
- `reason.plan_answer` when `intent.negation_marker=yes`: benches=hitab,sqa,tab_fact,tablebench,wtq, n=1180, macro useful lift +2.4 pts, weighted +3.4 pts, signs +4/-1
- `ctx.stats` when `text.has_ordinal=yes`: benches=hitab,sqa,tab_fact,tablebench,wtq, n=2148, macro useful lift +2.2 pts, weighted +0.1 pts, signs +4/-1
- `reason.plan_answer` when `cell.has_comma_number=yes`: benches=hitab,sqa,tablebench,wtq, n=1700, macro useful lift +2.2 pts, weighted +0.5 pts, signs +3/-1
- `reason.enumerate` when `table.cols_bin=gt_10`: benches=hitab,tab_fact,tablebench,wtq, n=794, macro useful lift +2.2 pts, weighted +1.8 pts, signs +3/-1
- `reason.critique_revise` when `intent.negation_marker=yes`: benches=hitab,sqa,tab_fact,tablebench,wtq, n=1180, macro useful lift +2.1 pts, weighted +4.1 pts, signs +4/-1

Negative useful-rate lift:
- `ser.records` when `intent.arithmetic_op_marker=yes`: benches=tab_fact,tablebench,wtq, n=249, macro useful lift -5.1 pts, weighted -4.3 pts, signs +0/-3
- `reason.enumerate` when `table.rows_bin=0_5`: benches=hitab,tab_fact, n=1325, macro useful lift -4.3 pts, weighted -2.3 pts, signs +0/-2
- `ser.records` when `schema.header_repetition_marker=yes`: benches=hitab,sqa,wtq, n=1346, macro useful lift -3.9 pts, weighted -0.3 pts, signs +1/-2
- `reason.symbolic_op` when `intent.arithmetic_op_marker=yes`: benches=tab_fact,tablebench,wtq, n=250, macro useful lift -3.8 pts, weighted -2.7 pts, signs +0/-3
- `ctx.type` when `table.rows_bin=0_5`: benches=hitab,tab_fact, n=1325, macro useful lift -3.7 pts, weighted -2.4 pts, signs +0/-2
- `reason.symbolic_op` when `table.shape=tall`: benches=hitab,wtq, n=302, macro useful lift -3.3 pts, weighted -2.7 pts, signs +0/-2
- `ser.records` when `table.shape=tall`: benches=hitab,wtq, n=256, macro useful lift -3.3 pts, weighted -2.0 pts, signs +1/-1
- `ctx.stats` when `table.shape=tall`: benches=hitab,wtq, n=302, macro useful lift -3.3 pts, weighted -1.9 pts, signs +0/-2

## Across-Benchmark Order 2 Rules

Positive useful-rate lift:
- `reason.extract_compute` when `intent.comparison=yes AND table.rows_bin=6_10`: benches=tab_fact,wtq, n=427, macro useful lift +10.3 pts, weighted +3.0 pts, signs +2/-0
- `reason.extract_compute` when `intent.negation_marker=yes AND table.shape=wide`: benches=sqa,tab_fact,wtq, n=533, macro useful lift +9.6 pts, weighted +5.0 pts, signs +3/-0
- `ser.records` when `intent.negation_marker=yes AND table.cols_bin=7_10`: benches=hitab,tab_fact,wtq, n=303, macro useful lift +9.5 pts, weighted +7.8 pts, signs +3/-0
- `ser.records` when `intent.negation_marker=yes AND text.has_number=yes`: benches=hitab,tab_fact,wtq, n=599, macro useful lift +9.4 pts, weighted +6.7 pts, signs +3/-0
- `ctx.stats` when `intent.negation_marker=yes AND table.shape=wide`: benches=sqa,tab_fact,wtq, n=533, macro useful lift +9.3 pts, weighted +5.2 pts, signs +3/-0
- `ctx.stats` when `intent.negation_marker=yes AND schema.has_entity_col=yes`: benches=sqa,tab_fact,wtq, n=513, macro useful lift +9.3 pts, weighted +4.7 pts, signs +3/-0
- `reason.extract_compute` when `intent.comparison=yes AND table.shape=wide`: benches=tab_fact,wtq, n=540, macro useful lift +9.1 pts, weighted +3.4 pts, signs +2/-0
- `reason.critique_revise` when `intent.negation_marker=yes AND table.shape=wide`: benches=sqa,tab_fact,wtq, n=533, macro useful lift +8.9 pts, weighted +5.5 pts, signs +3/-0

Negative useful-rate lift:
- `ctx.stats` when `grounding.header_overlap=yes AND table.shape=tall`: benches=hitab,wtq, n=202, macro useful lift -6.4 pts, weighted -5.1 pts, signs +0/-2
- `reason.symbolic_op` when `cell.has_percent=yes AND table.numeric_density_bin=low`: benches=sqa,tab_fact, n=213, macro useful lift -6.2 pts, weighted -4.8 pts, signs +0/-2
- `reason.symbolic_op` when `cell.has_percent=yes AND table.numeric_cols_bin=1`: benches=sqa,tab_fact, n=213, macro useful lift -6.2 pts, weighted -4.8 pts, signs +0/-2
- `ser.html` when `intent.reduction_marker=yes AND table.rows_bin=26_50`: benches=hitab,tab_fact,tablebench,wtq, n=258, macro useful lift -6.1 pts, weighted -5.3 pts, signs +0/-4
- `reason.evidence_table` when `intent.comparison=yes AND table.numeric_density_bin=high`: benches=hitab,tab_fact,tablebench,wtq, n=366, macro useful lift -6.0 pts, weighted -2.7 pts, signs +0/-4
- `ser.records` when `schema.has_entity_col=yes AND schema.header_repetition_marker=yes`: benches=hitab,wtq, n=434, macro useful lift -6.0 pts, weighted -5.3 pts, signs +0/-2
- `ser.records` when `schema.has_date_col=yes AND schema.header_repetition_marker=yes`: benches=hitab,sqa,wtq, n=422, macro useful lift -5.7 pts, weighted -5.5 pts, signs +0/-3
- `ser.records` when `intent.superlative_rank=yes AND table.rows_bin=26_50`: benches=hitab,tab_fact,tablebench,wtq, n=632, macro useful lift -5.6 pts, weighted -3.5 pts, signs +0/-4

## Files
- `rule_learning_dashboard.html`
- `feature_summary.csv`
- `within_rules_order1.csv`
- `within_rules_order2.csv`
- `across_rules_order1.csv`
- `across_rules_order2.csv`
- `observations.csv`
- `metadata.json`
