# tablebench Feature Set

Status: active runtime feature inventory for systematic prompt-feature experiments.

Formatting policy:

- prompt rendering is controlled by explicit `prompt_format_*` features.
- table serialization is controlled by explicit `table_serialization_*` features.
- response/scaffold/profile features should not include `set_format` or `set_table_format`; matrix specs select those axes separately.

Cross-task alignment policy:

- WTQ and TableBench share the same input-context canonical IDs where the transform semantics are the same.
- The current TableBench input-context features are DP-scaffold only. Do not combine them with TableBench PoT variants until the execution table used by the scorer is transformed consistently with the prompt table.
