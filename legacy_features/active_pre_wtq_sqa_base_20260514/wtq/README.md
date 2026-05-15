# wtq Feature Set

Status: active runtime feature inventory for systematic prompt-feature experiments.

Formatting policy:

- prompt rendering is controlled by explicit `prompt_format_*` features.
- table serialization is controlled by explicit `table_serialization_*` features.
- response/scaffold/profile features should not include `set_format` or `set_table_format`; matrix specs select those axes separately.
