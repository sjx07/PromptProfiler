# Aligned Benchmark Feature Taxonomy Coverage v1

Generated from executable JSON specs under `features/<benchmark>/`. The goal is to make empty benchmark x taxonomy slots explicit: implemented when a concrete feature exists, native when the behavior is supplied by the task renderer, and N/A when the benchmark does not expose that input/output surface.

Text2SQL note: following the context-construction distinction in `Obsidian/Transferability/Paper/Context Attribute Extraction.md`, schema item selection, schema type inventory, and schema summary are aligned to input-context features; row selection and table serialization are marked N/A because Spider/BIRD prompts render database schema rather than table rows.

## prompt_format

| Concept | wtq | tablebench | sqa | tabfact | hitab | spider | bird |
|---|---|---|---|---|---|---|---|
| `prompt_format.plain` | implemented `prompt_format_plain` | implemented `prompt_format_plain` | implemented `prompt_format_plain` | implemented `prompt_format_plain` | implemented `prompt_format_plain` | implemented `prompt_format_plain` | implemented `prompt_format_plain` |
| `prompt_format.markdown` | implemented `prompt_format_markdown` | implemented `prompt_format_markdown` | implemented `prompt_format_markdown` | implemented `prompt_format_markdown` | implemented `prompt_format_markdown` | implemented `prompt_format_markdown` | implemented `prompt_format_markdown` |
| `prompt_format.json` | implemented `prompt_format_json` | implemented `prompt_format_json` | implemented `prompt_format_json` | implemented `prompt_format_json` | implemented `prompt_format_json` | implemented `prompt_format_json` | implemented `prompt_format_json` |
| `prompt_format.yaml` | implemented `prompt_format_yaml` | implemented `prompt_format_yaml` | implemented `prompt_format_yaml` | implemented `prompt_format_yaml` | implemented `prompt_format_yaml` | implemented `prompt_format_yaml` | implemented `prompt_format_yaml` |
| `prompt_format.code_block` | implemented `prompt_format_code_block` | implemented `prompt_format_code_block` | implemented `prompt_format_code_block` | implemented `prompt_format_code_block` | implemented `prompt_format_code_block` | implemented `prompt_format_code_block` | implemented `prompt_format_code_block` |

## table_serialization

| Concept | wtq | tablebench | sqa | tabfact | hitab | spider | bird |
|---|---|---|---|---|---|---|---|
| `table_serialization.markdown` | implemented `table_serialization_markdown` | implemented `table_serialization_markdown` | implemented `table_serialization_markdown` | implemented `table_serialization_markdown` | implemented `table_serialization_markdown` | N/A | N/A |
| `table_serialization.csv` | implemented `table_serialization_csv` | implemented `table_serialization_csv` | implemented `table_serialization_csv` | implemented `table_serialization_csv` | implemented `table_serialization_csv` | N/A | N/A |
| `table_serialization.html` | implemented `table_serialization_html` | implemented `table_serialization_html` | implemented `table_serialization_html` | implemented `table_serialization_html` | implemented `table_serialization_html` | N/A | N/A |
| `table_serialization.json_records` | implemented `table_serialization_json_records` | implemented `table_serialization_json_records` | implemented `table_serialization_json_records` | implemented `table_serialization_json_records` | implemented `table_serialization_json_records` | N/A | N/A |
| `table_serialization.json_columns_data` | implemented `table_serialization_json_columns_data` | implemented `table_serialization_json_columns_data` | implemented `table_serialization_json_columns_data` | implemented `table_serialization_json_columns_data` | implemented `table_serialization_json_columns_data` | N/A | N/A |

## response_mode

| Concept | wtq | tablebench | sqa | tabfact | hitab | spider | bird |
|---|---|---|---|---|---|---|---|
| `response.direct_final_only` | implemented `facet_dp_scaffold` | implemented `tb_official_dp_full` | implemented `facet_dp_scaffold` | implemented `facet_dp_scaffold` | implemented `facet_dp_scaffold` | native `sql_query final field` | native `sql_query final field` |
| `response.visible_cot` | implemented `facet_tcot_scaffold` | implemented `tb_official_tcot_full` | implemented `facet_tcot_scaffold` | implemented `facet_tcot_scaffold` | implemented `hitab_tcot_profile` | N/A | N/A |
| `response.visible_structured_trace` | implemented `facet_scot_scaffold` | implemented `tb_official_scot_full` | implemented `facet_scot_scaffold` | implemented `facet_scot_scaffold` | implemented `hitab_scot_profile` | N/A | N/A |
| `response.program_of_thought.python` | implemented `facet_pot_exec_scaffold` | implemented `tb_official_pot_full` | implemented `facet_pot_exec_scaffold` | implemented `facet_pot_exec_scaffold` | implemented `hitab_pot_profile` | N/A | N/A |
| `response.sql_program` | N/A | N/A | N/A | N/A | N/A | native `sql_query final field` | native `sql_query final field` |

## reasoning

| Concept | wtq | tablebench | sqa | tabfact | hitab | spider | bird |
|---|---|---|---|---|---|---|---|
| `reasoning.extract_then_compute` | implemented `reasoning_extract_then_compute` | implemented `reasoning_extract_then_compute` | implemented `reasoning_extract_then_compute` | implemented `reasoning_extract_then_compute` | implemented `reasoning_extract_then_compute` | implemented `reasoning_extract_then_compute` | implemented `reasoning_extract_then_compute` |
| `reasoning.evidence_localization` | implemented `reasoning_evidence_localization` | implemented `reasoning_evidence_localization` | implemented `reasoning_evidence_localization` | implemented `reasoning_evidence_localization` | implemented `reasoning_evidence_localization` | implemented `reasoning_evidence_localization` | implemented `reasoning_evidence_localization` |
| `reasoning.candidate_enumeration` | implemented `reasoning_candidate_enumeration` | implemented `reasoning_candidate_enumeration` | implemented `reasoning_candidate_enumeration` | implemented `reasoning_candidate_enumeration` | implemented `reasoning_candidate_enumeration` | implemented `reasoning_candidate_enumeration` | implemented `reasoning_candidate_enumeration` |
| `reasoning.verify_before_output` | implemented `reasoning_verify_before_output` | implemented `reasoning_verify_before_output` | implemented `reasoning_verify_before_output` | implemented `reasoning_verify_before_output` | implemented `reasoning_verify_before_output` | implemented `reasoning_verify_before_output` | implemented `reasoning_verify_before_output` |

## input_context

| Concept | wtq | tablebench | sqa | tabfact | hitab | spider | bird |
|---|---|---|---|---|---|---|---|
| `input_context.type_annotation` | implemented `input_context_type_annotation` | implemented `input_context_type_annotation` | implemented `input_context_type_annotation` | implemented `input_context_type_annotation` | implemented `input_context_type_annotation` | implemented `input_context_type_annotation` | implemented `input_context_type_annotation` |
| `input_context.column_selection` | implemented `input_context_column_selection_relevance_12` | implemented `input_context_column_selection_relevance_12` | implemented `input_context_column_selection_relevance_12` | implemented `input_context_column_selection_relevance_12` | implemented `input_context_column_selection_relevance_12` | implemented `input_context_column_selection_relevance_12` | implemented `input_context_column_selection_relevance_12` |
| `input_context.row_selection` | implemented `input_context_row_selection_relevance_50` | implemented `input_context_row_selection_relevance_50` | implemented `input_context_row_selection_relevance_50` | implemented `input_context_row_selection_relevance_50` | implemented `input_context_row_selection_relevance_50` | N/A | N/A |
| `input_context.column_statistics` | implemented `input_context_column_statistics` | implemented `input_context_column_statistics` | implemented `input_context_column_statistics` | implemented `input_context_column_statistics` | implemented `input_context_column_statistics` | implemented `input_context_column_statistics` | implemented `input_context_column_statistics` |

## Remaining Missing

No missing applicable aligned slots in this matrix. Remaining empty cells are native defaults or N/A surfaces.
