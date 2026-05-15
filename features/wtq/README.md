# WTQ Fresh Base Features

Status: active runtime inventory for the WTQ/SQA fresh base-design round.

This directory intentionally contains only the fixed base features needed to
run a measurement-stable WTQ base prompt. Older active features were archived
under `features_legacy/active_pre_wtq_sqa_base_20260514/`.

The base fixes:
- direct-answer table QA scaffold;
- plain prompt rendering;
- `json_columns_data` table serialization;
- JSON-list answer interface contract.

Reasoning, context-builder, domain-heuristic, and alternative formatting
features should be added in later rounds as study features, not silently folded
into this base.
