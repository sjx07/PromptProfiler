# Legacy Feature Archives

This directory stores archived active feature inventories that should not be
used by default runtime experiments.

- `active_pre_wtq_sqa_base_20260514/`: snapshot of the active `features/`
  tree before the WTQ/SQA fresh base-design reset.

The runtime `FeatureRegistry` still loads active features from `features/`.
The older `features_legacy/` directory remains a separate fallback path used by
the current registry when a task has no active `features/<task>` directory.
