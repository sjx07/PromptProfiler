# WTQ Fresh Base Features

Status: active runtime inventory for the WTQ/SQA fresh base-design round.

This directory contains the fixed base features plus the restored study
features for the WTQ/SQA fresh design round. Older active features were
archived under `legacy_features/active_pre_wtq_sqa_base_20260514/`.

The base fixes:
- direct-answer table QA scaffold;
- plain prompt rendering;
- `json_columns_data` table serialization;
- JSON-list answer interface contract.

Study features are organized as `features/wtq/<feature_family>/<feature>.json`
and should be selected explicitly in experiment configs, not silently folded
into the base:

- context builders: `input_context_*`;
- reasoning rules: `private_reasoning_rule/reasoning_*`;
- visible reasoning scaffolds: `visible_reasoning_scaffold/reasoning_scaffold_visible_*`;
- formatting/rendering alternatives: `prompt_format_*` and
  `table_serialization_*`;
- wiki-table blocked packs: `domain_wikitable_source_semantics_pack` and
  `contract_wikitable_answer_surface_pack`.

Active table serialization variants are limited to HTML and JSON forms. CSV and
Markdown table serialization specs are archived under
`legacy_features/round_7_removed_table_serialization/`.
