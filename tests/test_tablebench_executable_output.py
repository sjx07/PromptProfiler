import os
from pathlib import Path

from core.func_registry import PromptBuildState, REGISTRY, _func_sort_key
from core.feature_registry import FeatureRegistry
from tasks.tablebench.official_parser import parse_code_output_prediction, parse_final_answer
from tasks.tablebench.official_scorer import score_one
from tasks.tablebench.table_bench import TableBench
from tasks.wtq.parsers import PARSER_REGISTRY


def test_task_profile_primitive_removed():
    assert "set_task_profile" not in REGISTRY
    assert "set_output_mode" not in REGISTRY


def test_code_parser_prefers_last_fenced_block_without_flattening():
    response = """Some reasoning.

```python
print("discard this")
```

### code

```python
for x in [1, 2]:
    print(x)
```
"""

    parsed = PARSER_REGISTRY["code"](response, _NoPromptStateTask())

    assert parsed.startswith("__CODE__")
    assert "for x in [1, 2]:\n    print(x)" in parsed
    assert "discard this" not in parsed


def test_code_parser_removes_json_string_top_level_indent():
    response = "{\"code\": \"import pandas as pd\\n df = pd.read_csv('table.csv')\\n for value in [1, 2]:\\n     print(value)\"}"

    parsed = PARSER_REGISTRY["code"](response, _JsonCodeTask())

    assert parsed.startswith("__CODE__")
    assert "\n df =" not in parsed
    assert "\nfor value in [1, 2]:" in parsed
    assert "\n    print(value)" in parsed


def test_tablebench_executes_code_prediction_before_official_scoring():
    exec_root = Path.cwd() / ".pytest_cache" / "tablebench_exec"
    exec_root.mkdir(parents=True, exist_ok=True)
    old_exec_root = os.environ.get("TABLEBENCH_EXEC_DIR")
    os.environ["TABLEBENCH_EXEC_DIR"] = str(exec_root)
    try:
        task = TableBench()
        prediction = """__CODE__
import csv

with open("table.csv", newline="") as f:
    rows = list(csv.DictReader(f))

total = sum(int(r["score"]) for r in rows)
print(total)
"""
        score, metrics = task.score(prediction, {
            "_raw": {
                "answer": "3",
                "qtype": "NumericalReasoning",
                "qsubtype": "Arithmetic",
                "table": {
                    "header": ["score"],
                    "rows": [["1"], ["2"]],
                },
            },
        })
    finally:
        if old_exec_root is None:
            os.environ.pop("TABLEBENCH_EXEC_DIR", None)
        else:
            os.environ["TABLEBENCH_EXEC_DIR"] = old_exec_root

    assert score == 1.0
    assert metrics["prediction"] == "3"
    assert metrics["ECR@1"] is True
    assert metrics["method"].startswith("python_exec_")
    assert metrics["output_mode"] == "python_exec"


def test_tablebench_code_output_field_executes_fenced_code():
    exec_root = Path.cwd() / ".pytest_cache" / "tablebench_exec"
    exec_root.mkdir(parents=True, exist_ok=True)
    old_exec_root = os.environ.get("TABLEBENCH_EXEC_DIR")
    os.environ["TABLEBENCH_EXEC_DIR"] = str(exec_root)
    try:
        task = TableBench()
        state = PromptBuildState()
        REGISTRY["insert_node"](state, {
            "_func_id": "test_code_output",
            "node_type": "output_field",
            "parent_id": "__root__",
            "payload": {
                "name": "code",
                "description": "Executable Python code that prints only the answer value.",
            },
        })
        task.bind(state)
        raw_response = """### Analytical Approach
Sum the score column.

```python
import csv

with open("table.csv", newline="") as f:
    rows = list(csv.DictReader(f))

total = sum(int(r["score"]) for r in rows)
print(total)
```
"""
        prediction = task.parse_response(raw_response)
        score, metrics = task.score(prediction, {
            "_raw": {
                "answer": "3",
                "qtype": "NumericalReasoning",
                "qsubtype": "Arithmetic",
                "table": {
                    "header": ["score"],
                    "rows": [["1"], ["2"]],
                },
            },
        })
    finally:
        if old_exec_root is None:
            os.environ.pop("TABLEBENCH_EXEC_DIR", None)
        else:
            os.environ["TABLEBENCH_EXEC_DIR"] = old_exec_root

    assert score == 1.0
    assert prediction.startswith("__CODE__")
    assert "print(total)" in prediction
    assert "Final Answer" not in prediction
    assert metrics["prediction"] == "3"
    assert metrics["ECR@1"] is True
    assert metrics["output_mode"] == "python_exec"
    assert "task_profile" not in metrics


def test_tablebench_code_output_parser_strips_legacy_answer_prefixes():
    assert parse_code_output_prediction("Final Answer: 3\n") == "3"
    assert parse_code_output_prediction("Final: 3\n") == "3"
    assert parse_code_output_prediction("answer: 3\n") == "3"
    assert parse_code_output_prediction("debug\nFinal: 3\n") == "3"


def test_tablebench_final_answer_parser_uses_last_match():
    response = """
The answer should follow this format:
Final Answer: AnswerName1, AnswerName2...

Reasoning...
Final Answer: Strong positive correlation, 0.87
"""

    assert parse_final_answer(response) == "Strong positive correlation, 0.87"


def test_tablebench_tcot_full_prompt_uses_facet_builder():
    task = TableBench()
    state, _ = _state_from_tablebench_features([
        "_section_role",
        "_section_task",
        "_section_table_handling",
        "_section_reasoning",
        "_section_format_fix",
        "_section_rules",
        "tb_official_tcot_full",
    ])
    task.bind(state)

    system_prompt, user_content = task.build_prompt({
        "content": "How many points?",
        "meta": {
            "_raw": {
                "question": "How many points?",
                "answer": "13",
                "qtype": "NumericalReasoning",
                "qsubtype": "Arithmetic",
                "table": {
                    "header": ["team", "points"],
                    "rows": [["a", "13"]],
                },
            },
        },
    })

    assert system_prompt
    assert "Act as a table analyst" in system_prompt
    assert "think step by step" in system_prompt
    assert "answer" in system_prompt
    assert "How many points?" in user_content
    assert '"columns": [' in user_content
    assert '"data": [' in user_content
    assert '"team"' in user_content
    assert '"a"' in user_content
    assert '"rows": [' not in user_content
    assert "[TABLE]" not in user_content
    assert "Let's get start!" not in user_content


def test_tablebench_dp_full_prompt_matches_official_semantics_without_literal_bypass():
    task = TableBench()
    state, _ = _state_from_tablebench_features([
        "_section_role",
        "_section_task",
        "_section_table_handling",
        "_section_format_fix",
        "_section_rules",
        "tb_official_dp_full",
    ])
    task.bind(state)

    system_prompt, user_content = task.build_prompt({
        "content": "What is the average number of tropical cyclones per season?",
        "meta": {
            "_raw": {
                "question": "What is the average number of tropical cyclones per season?",
                "answer": "10.6",
                "qtype": "NumericalReasoning",
                "qsubtype": "Arithmetic",
                "table": {
                    "header": ["season", "tropical cyclones"],
                    "rows": [["1990 - 91", "10"], ["1991 - 92", "10"], ["1992 - 93", "3"]],
                },
            },
        },
    })
    rendered = system_prompt + "\n" + user_content

    assert "You are a table analyst" in rendered
    assert "answer: <your_answer_here>" in rendered
    assert "You should follow the answer-format requirements below." in system_prompt
    assert system_prompt.index("You should follow the answer-format requirements below.") < system_prompt.index("Put the final answer")
    assert "AnswerName1, AnswerName2..." in rendered
    assert "Final Answer: AnswerName1" not in rendered
    assert "last output line" not in rendered
    assert "short as possible" in rendered
    assert '"columns": [' in user_content
    assert '"data": [' in user_content
    assert '"season"' in user_content
    assert '"rows": [' not in user_content
    assert "What is the average number of tropical cyclones per season?" in user_content
    assert "[TABLE]" not in user_content
    assert "Let's get start!" not in user_content


def test_tablebench_plain_renderer_skips_empty_sections():
    task = TableBench()
    state, _ = _state_from_tablebench_features([
        "_section_role",
        "_section_task",
        "_section_table_handling",
        "_section_reasoning",
        "_section_format_fix",
        "_section_rules",
        "_section_strategy",
        "tb_official_dp_full",
    ])
    task.bind(state)

    system_prompt, _ = task.build_prompt({
        "content": "How many points?",
        "meta": {
            "_raw": {
                "question": "How many points?",
                "answer": "13",
                "qtype": "NumericalReasoning",
                "qsubtype": "ArithmeticCalculation",
                "table": {
                    "header": ["team", "points"],
                    "rows": [["a", "13"]],
                },
            },
        },
    })

    assert "\ntask\n" not in system_prompt
    assert "\nrules\n" not in system_prompt
    assert "\nreasoning\n" not in system_prompt
    assert "\nstrategy\n" not in system_prompt
    assert "\nrole\n" in system_prompt
    assert "\ntable_handling\n" in system_prompt
    assert "\nformat_fix\n" in system_prompt


def test_tablebench_direct_answer_profiles_do_not_conflict_with_facet_output_contract():
    reg = FeatureRegistry.load(task="tablebench")
    base = [
        "_section_role",
        "_section_task",
        "_section_table_handling",
        "_section_reasoning",
        "_section_format_fix",
        "_section_rules",
    ]

    for feature_id in [
        "tb_official_dp_full",
        "tb_official_tcot_full",
        "tb_official_scot_full",
    ]:
        specs, _ = reg.materialize(base + [feature_id])
        rule_text = "\n".join(
            str(spec.get("params", {}).get("payload", {}).get("content", ""))
            for spec in specs
            if spec.get("func_type") == "insert_node"
            and spec.get("params", {}).get("node_type") == "rule"
        )

        assert "Final Answer: AnswerName1" not in rule_text
        assert "last output line" not in rule_text
        assert "adapt the `answer` value to the subtype" not in rule_text


def test_official_full_profiles_do_not_include_static_qsubtype_labels():
    reg = FeatureRegistry.load(task="tablebench")
    base = [
        "_section_role",
        "_section_task",
        "_section_table_handling",
        "_section_reasoning",
        "_section_format_fix",
        "_section_rules",
    ]

    for feature_id in [
        "tb_official_dp_full",
        "tb_official_tcot_full",
        "tb_official_scot_full",
        "tb_official_pot_full",
    ]:
        specs, _ = reg.materialize(base + [feature_id])
        rule_text = "\n".join(
            str(spec.get("params", {}).get("payload", {}).get("content", ""))
            for spec in specs
            if spec.get("func_type") == "insert_node"
            and spec.get("params", {}).get("node_type") == "rule"
        )

        assert "adapt the `answer` value to the subtype" not in rule_text
        assert "adapt the value to the DataAnalysis subtype" not in rule_text
        assert "ImpactAnalysis" not in rule_text
        assert "AnomalyDetection" not in rule_text
        assert "CorrelationAnalysis" not in rule_text
        assert "TrendForecasting" not in rule_text
        assert "CausalAnalysis" not in rule_text
        assert "DescriptiveAnalysis" not in rule_text


def test_tablebench_tcot_full_prompt_does_not_inline_qsubtype_contracts():
    task = TableBench()
    state, _ = _state_from_tablebench_features([
        "_section_role",
        "_section_task",
        "_section_table_handling",
        "_section_reasoning",
        "_section_format_fix",
        "_section_rules",
        "tb_official_tcot_full",
    ])
    task.bind(state)

    system_prompt, user_content = task.build_prompt({
        "content": "Question?",
        "meta": {
            "_raw": {
                "question": "Question?",
                "answer": "answer",
                "qtype": "DataAnalysis",
                "qsubtype": "AnomalyDetection",
                "table": {
                    "header": ["name", "score", "ratio", "pct", "rank"],
                    "rows": [["a", "13", "13.5", "43.68%", "1st"]],
                },
            },
        },
    })
    rendered = system_prompt + "\n" + user_content

    assert "adapt the `answer` value to the subtype" not in rendered
    assert "ImpactAnalysis" not in rendered
    assert "AnomalyDetection" not in rendered
    assert "CorrelationAnalysis" not in rendered
    assert "TrendForecasting" not in rendered
    assert "CausalAnalysis" not in rendered
    assert "DescriptiveAnalysis" not in rendered
    assert "StatisticalAnalysis" not in rendered
    assert "abnormal data with total number" in rendered
    assert "No anomalies are detected in the table" in rendered
    assert "The three anomalies are row 5" in rendered
    assert "Higher interest positively influences deposit balances change" not in rendered
    assert "shooting accuracy of 8 different bullet types" not in rendered


def test_tablebench_tcot_full_selects_record_specific_dataanalysis_contracts():
    task = TableBench()
    state, _ = _state_from_tablebench_features([
        "_section_role",
        "_section_task",
        "_section_table_handling",
        "_section_reasoning",
        "_section_format_fix",
        "_section_rules",
        "tb_official_tcot_full",
    ])
    task.bind(state)

    causal_system, causal_user = task.build_prompt({
        "content": "Does more funding cause higher output?",
        "meta": {
            "_raw": {
                "question": "Does more funding cause higher output?",
                "answer": "Yes, higher funding is associated with higher output.",
                "qtype": "DataAnalysis",
                "qsubtype": "CausalAnalysis",
                "table": {
                    "header": ["funding", "output"],
                    "rows": [["10", "20"], ["20", "45"]],
                },
            },
        },
    })
    causal_rendered = causal_system + "\n" + causal_user

    correlation_system, correlation_user = task.build_prompt({
        "content": "What is the correlation between funding and output?",
        "meta": {
            "_raw": {
                "question": "What is the correlation between funding and output?",
                "answer": "Strong positive correlation, 0.94",
                "qtype": "DataAnalysis",
                "qsubtype": "CorrelationAnalysis",
                "table": {
                    "header": ["funding", "output"],
                    "rows": [["10", "20"], ["20", "45"]],
                },
            },
        },
    })
    correlation_rendered = correlation_system + "\n" + correlation_user

    assert "Ensure answer should give the conclusion" in causal_rendered
    assert "Higher interest positively influences deposit balances change" in causal_rendered
    assert "CorrelationRelation, CorrelationCoefficient" not in causal_rendered
    assert "CausalAnalysis" not in causal_rendered
    assert "CorrelationAnalysis" not in causal_rendered
    assert causal_system.splitlines().count("format_fix") == 1

    assert "CorrelationRelation, CorrelationCoefficient" in correlation_rendered
    assert "between +0.3 to +0.7" in correlation_rendered
    assert "Strong positive correlation, 0.82" in correlation_rendered
    assert "Higher interest positively influences deposit balances change" not in correlation_rendered
    assert "CausalAnalysis" not in correlation_rendered
    assert "CorrelationAnalysis" not in correlation_rendered
    assert correlation_system.splitlines().count("format_fix") == 1

    assert task._prompt_state.semantic.output_fields["answer"].startswith("AnswerName1")


def test_tablebench_record_specific_format_rules_do_not_duplicate_format_fix_section():
    profiles = [
        "tb_official_dp_full",
        "tb_official_tcot_full",
        "tb_official_scot_full",
    ]
    for profile in profiles:
        task = TableBench()
        state, _ = _state_from_tablebench_features([
            "_section_role",
            "_section_task",
            "_section_table_handling",
            "_section_reasoning",
            "_section_format_fix",
            "_section_rules",
            profile,
        ])
        task.bind(state)

        system_prompt, _ = task.build_prompt({
            "content": "Which company was most impacted?",
            "meta": {
                "_raw": {
                    "question": "Which company was most impacted?",
                    "answer": "Negative impact",
                    "qtype": "DataAnalysis",
                    "qsubtype": "ImpactAnalysis",
                    "table": {
                        "header": ["company", "before", "after"],
                        "rows": [["a", "10", "4"], ["b", "5", "6"]],
                    },
                },
            },
        })

        assert system_prompt.splitlines().count("format_fix") == 1
        assert "AnswerName1, AnswerName2" in system_prompt
        assert "No clear impact" in system_prompt
        assert "Negtive impact" in system_prompt


def test_tablebench_pot_anomaly_prompt_uses_example_contracts_without_wrapper_noise():
    task = TableBench()
    state, _ = _state_from_tablebench_features([
        "_section_role",
        "_section_task",
        "_section_table_handling",
        "_section_reasoning",
        "_section_format_fix",
        "_section_rules",
        "tb_pot_fixed_scaffold",
        "tb_pot_python_persona_rule",
        "tb_pot_approach_then_code_rule",
        "tb_pot_code_concise_rule",
        "tb_pot_code_readable_rule",
        "tb_pot_code_comment_rule",
        "tb_pot_data_only_rule",
        "tb_pot_executable_code_rule",
    ])
    task.bind(state)

    system_prompt, _ = task.build_prompt({
        "content": "Find anomalies.",
        "meta": {
            "_raw": {
                "question": "Find anomalies.",
                "answer": "No anomalies are detected in the table.",
                "qtype": "DataAnalysis",
                "qsubtype": "AnomalyDetection",
                "table": {
                    "header": ["name", "score"],
                    "rows": [["a", "13"]],
                },
            },
        },
    })

    assert "Input Fields: Input Fields" not in system_prompt
    assert "Output Fields: Output Fields" not in system_prompt
    assert "Output Fields: code (" in system_prompt
    assert system_prompt.splitlines().count("format_fix") == 1
    assert system_prompt.count("Set `answer` to") == 1
    assert "Set `answer` to the requested result shaped as" not in system_prompt
    assert "Set `answer` to the requested result shaped as: Examples" not in system_prompt
    assert "The three anomalies are row 5" in system_prompt
    assert "abnormal data with total number" in system_prompt
    assert "No anomalies are detected in the table." in system_prompt


def test_tablebench_tcot_full_counting_contract_is_behavioral_not_taxonomy_label():
    task = TableBench()
    state, _ = _state_from_tablebench_features([
        "_section_role",
        "_section_task",
        "_section_table_handling",
        "_section_reasoning",
        "_section_format_fix",
        "_section_rules",
        "tb_official_tcot_full",
    ])
    task.bind(state)

    system_prompt, user_content = task.build_prompt({
        "content": "How many countries have at least one semifinalist?",
        "meta": {
            "_raw": {
                "question": "How many countries have at least one semifinalist?",
                "answer": "12",
                "qtype": "NumericalReasoning",
                "qsubtype": "Counting",
                "table": {
                    "header": ["country", "semifinalists"],
                    "rows": [["a", "1"], ["b", "0"], ["c", "2"]],
                },
            },
        },
    })
    rendered = system_prompt + "\n" + user_content

    assert "enumerate each table row or entity" in rendered
    assert "running count" in rendered
    assert "qsubtype" not in rendered
    assert "Counting" not in rendered


def test_tablebench_scorer_normalizes_country_case():
    score, method = score_one(
        "NumericalReasoning",
        "Domain-Specific",
        "united states",
        "United States",
    )

    assert score == 1.0
    assert method == "nr_em"


def test_tablebench_metrics_include_normalized_answer_values():
    task = TableBench()

    score, metrics = task.score("United States", {
        "_raw": {
            "answer": "united states",
            "qtype": "NumericalReasoning",
            "qsubtype": "Domain-Specific",
            "table": {
                "header": ["country"],
                "rows": [["united states"]],
            },
        },
    })

    assert score == 1.0
    assert metrics["prediction"] == "United States"
    assert metrics["normalized_prediction"] == "united states"
    assert metrics["normalized_gold"] == "united states"


def test_tablebench_feature_directory_only_has_sections_and_official_full_profiles():
    reg = FeatureRegistry.load(task="tablebench")

    expected_sections_and_profiles = {
        "_section_format_fix",
        "_section_reasoning",
        "_section_role",
        "_section_rules",
        "_section_strategy",
        "_section_table_handling",
        "_section_task",
        "tb_official_dp_full",
        "tb_official_tcot_full",
        "tb_official_scot_full",
        "tb_official_pot_full",
    }
    expected_pot_components = {
        "tb_pot_fixed_scaffold",
        "tb_pot_reasoning_field",
        "tb_pot_approach_then_code_rule",
        "tb_pot_code_concise_rule",
        "tb_pot_code_readable_rule",
        "tb_pot_code_comment_rule",
        "tb_pot_data_only_rule",
        "tb_pot_executable_code_rule",
        "tb_pot_python_persona_rule",
    }
    expected_tcot_components = {
        "tb_tcot_fixed_scaffold",
        "tb_tcot_step_by_step_rule",
        "tb_tcot_table_only_rule",
    }
    expected_scot_components = {
        "tb_scot_fixed_scaffold",
        "tb_scot_pattern_intro_rule",
        "tb_scot_thought_rule",
        "tb_scot_action_python_rule",
        "tb_scot_result_simulation_rule",
        "tb_scot_repeat_rule",
        "tb_scot_concluding_check_rule",
    }
    expected_pot_rescue_components = {
        "tb_pot_rescue_no_dataframe_print",
        "tb_pot_rescue_precision_units",
        "tb_pot_rescue_answer_type_gate",
        "tb_pot_rescue_numeric_normalization",
        "tb_pot_rescue_column_row_binding",
        "tb_pot_rescue_anomaly_conservative",
        "tb_pot_rescue_causal_evidence_shape",
        "tb_pot_rescue_minimum_delta_nonnegative",
    }

    features = set(reg.list_features())

    assert expected_sections_and_profiles <= features
    assert features <= (
        expected_sections_and_profiles
        | expected_pot_components
        | expected_tcot_components
        | expected_scot_components
        | expected_pot_rescue_components
    )


def test_official_pot_full_feature_uses_executable_tablebench_guidelines():
    reg = FeatureRegistry.load(task="tablebench")
    specs, _ = reg.materialize([
        "_section_role",
        "_section_task",
        "_section_table_handling",
        "_section_reasoning",
        "_section_format_fix",
        "_section_rules",
        "tb_official_pot_full",
    ])
    rule_text = "\n".join(
        str(spec.get("params", {}).get("payload", {}).get("content", ""))
        for spec in specs
        if spec.get("func_type") == "insert_node"
        and spec.get("params", {}).get("node_type") == "rule"
    )

    assert "df = pd.read_csv('table.csv')" in rule_text
    assert "Code blocks need to strictly start with ```python and end with ```" in rule_text
    assert "generate executable code" in rule_text
    assert "print function" in rule_text
    assert "JSON object with `columns` and `data`" in rule_text


def test_official_full_profile_features_materialize():
    reg = FeatureRegistry.load(task="tablebench")
    base = [
        "_section_role",
        "_section_task",
        "_section_table_handling",
        "_section_reasoning",
        "_section_format_fix",
        "_section_rules",
    ]
    dp_specs, _ = reg.materialize(base + ["tb_official_dp_full"])
    tcot_specs, _ = reg.materialize(base + ["tb_official_tcot_full"])
    scot_specs, _ = reg.materialize(base + ["tb_official_scot_full"])
    pot_specs, _ = reg.materialize(base + ["tb_official_pot_full"])

    all_specs = dp_specs + tcot_specs + scot_specs + pot_specs
    func_types = [spec["func_type"] for spec in all_specs]
    output_fields = [
        spec["params"]["payload"]["name"]
        for spec in all_specs
        if spec["func_type"] == "insert_node"
        and spec["params"]["node_type"] == "output_field"
    ]

    assert "set_task_profile" not in func_types
    assert "set_output_mode" not in func_types
    assert "reasoning" in output_fields
    assert "symbolic_trace" in output_fields
    assert "code" in output_fields


def _state_from_tablebench_features(canonical_ids):
    reg = FeatureRegistry.load(task="tablebench")
    specs, _ = reg.materialize(canonical_ids)
    state = PromptBuildState()
    for spec in sorted(
        specs,
        key=lambda s: _func_sort_key(s["func_id"], s["func_type"], s["params"]),
    ):
        params = dict(spec["params"])
        params["_func_id"] = spec["func_id"]
        REGISTRY[spec["func_type"]](state, params)
    return state, specs


class _NoPromptStateTask:
    _prompt_state = None


class _JsonCodePromptState:
    @staticmethod
    def parse_output(_response):
        return {
            "code": "import pandas as pd\n df = pd.read_csv('table.csv')\n for value in [1, 2]:\n     print(value)"
        }


class _JsonCodeTask:
    _prompt_state = _JsonCodePromptState()
