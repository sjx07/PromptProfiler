# WTQ Systematic Feature Set v0

Status: active runtime feature set for the round 2.3 WTQ fixed-design matrix.

The full pre-cleanup WTQ feature snapshot is archived at:

```text
features_legacy/wtq_systematic_pre_v0/
```

Active set:

- structural sections: `_section_*`
- response scaffolds: `facet_dp_scaffold`, `facet_tcot_scaffold`, `facet_scot_scaffold`, `facet_pot_exec_scaffold`
- DP reasoning features: `reason_extract_then_compute`, `reason_filter_then_extract`, `reason_enumerate_then_select`, `reason_verify_before_output`
- input-context features: `ctx_annotate_types`, `ctx_filter_rows_relevance_50`, `ctx_prune_columns_12`, `ctx_prepend_column_stats`

`features/wtq` is intentionally small and executable. Cross-task concept mapping and matrix-level metadata should live under `study_layer/`, not in these JSON files.
