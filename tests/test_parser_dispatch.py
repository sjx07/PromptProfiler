"""test_parser_dispatch.py — unit tests for per-task PARSER_REGISTRY + @register_parser.

Verifies:
  1. @register_parser decorator populates PARSER_REGISTRY correctly
  2. Each registered parser is callable and returns the expected prefix/value
  3. DISPATCH_FIELDS matches the registry keys
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TOOL_DIR = str(Path(__file__).parent.parent.parent.parent)
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)


# ── WTQ parsers ───────────────────────────────────────────────────────

def test_wtq_registry_keys():
    from tasks.wtq.parsers import PARSER_REGISTRY, DISPATCH_FIELDS
    assert set(PARSER_REGISTRY.keys()) == DISPATCH_FIELDS
    assert DISPATCH_FIELDS == frozenset({"code", "sql", "answer"})


def test_wtq_parse_code_returns_prefix(tmp_mock_task):
    from tasks.wtq.parsers import PARSER_REGISTRY
    parser = PARSER_REGISTRY["code"]
    result = parser("df['wins'].sum()", tmp_mock_task)
    assert result.startswith("__CODE__")
    assert "df" in result


def test_wtq_parse_sql_returns_prefix(tmp_mock_task):
    from tasks.wtq.parsers import PARSER_REGISTRY
    parser = PARSER_REGISTRY["sql"]
    result = parser("SELECT MAX(score) FROM t", tmp_mock_task)
    assert result.startswith("__SQL__")
    assert "SELECT" in result


def test_wtq_parse_answer_returns_plain(tmp_mock_task):
    from tasks.wtq.parsers import PARSER_REGISTRY
    parser = PARSER_REGISTRY["answer"]
    result = parser("42", tmp_mock_task)
    assert not result.startswith("__CODE__")
    assert not result.startswith("__SQL__")
    assert "42" in result


def test_wtq_parse_code_markdown_block(tmp_mock_task):
    from tasks.wtq.parsers import PARSER_REGISTRY
    parser = PARSER_REGISTRY["code"]
    response = "```python\ndf['wins'].sum()\n```"
    result = parser(response, tmp_mock_task)
    assert result.startswith("__CODE__")
    assert "df" in result


def test_wtq_parse_answer_accepts_equals_label(tmp_mock_task):
    from tasks.wtq.parsers import PARSER_REGISTRY

    parser = PARSER_REGISTRY["answer"]
    result = parser("answer = 42", tmp_mock_task)

    assert result == "42"


def test_wtq_score_normalizes_unicode_dash_variants():
    from tasks.wtq.table_qa import TableQA

    score, metrics = TableQA().score("18–12", {"gold_answers": ["18-12"]})

    assert score == 1.0
    assert metrics["pred_normalized"] == ["18-12"]
    assert metrics["gold_normalized"] == ["18-12"]


def test_wtq_score_normalizes_space_grouped_thousands():
    from tasks.wtq.table_qa import TableQA

    score, metrics = TableQA().score('["4 000"]', {"gold_answers": ["4000"]})

    assert score == 1.0
    assert metrics["pred_normalized"] == ["4000"]
    assert metrics["gold_normalized"] == ["4000"]


# ── SQA parsers ───────────────────────────────────────────────────────

def test_sqa_registry_keys():
    from tasks.sqa.parsers import PARSER_REGISTRY, DISPATCH_FIELDS
    assert set(PARSER_REGISTRY.keys()) == DISPATCH_FIELDS
    assert DISPATCH_FIELDS == frozenset({"code", "answer"})


def test_sqa_parse_code_returns_prefix(tmp_mock_task):
    from tasks.sqa.parsers import PARSER_REGISTRY
    parser = PARSER_REGISTRY["code"]
    result = parser("df['points'].max()", tmp_mock_task)
    assert result.startswith("__CODE__")
    assert "df" in result


def test_sqa_parse_answer_returns_plain(tmp_mock_task):
    from tasks.sqa.parsers import PARSER_REGISTRY
    parser = PARSER_REGISTRY["answer"]
    result = parser("Answer: Lee", tmp_mock_task)
    assert result == "Lee"


def test_sqa_parse_answer_accepts_equals_label(tmp_mock_task):
    from tasks.sqa.parsers import PARSER_REGISTRY

    parser = PARSER_REGISTRY["answer"]
    result = parser("answer = Lee", tmp_mock_task)

    assert result == "Lee"


def test_sqa_parse_answer_preserves_json_list(tmp_mock_task):
    import json

    from tasks.sqa.parsers import PARSER_REGISTRY

    parser = PARSER_REGISTRY["answer"]
    result = parser('{"answer": ["20,000", "15,200"]}', tmp_mock_task)

    assert json.loads(result) == ["20,000", "15,200"]


def test_sqa_score_numeric_thousands_multi_answer_string():
    from tasks.sqa.sequential_qa import SequentialQA

    values = [
        "20,000", "20,000", "15,200", "290,000",
        "16,750", "30,000", "31,750", "114,430",
        "46,250", "51,000", "41,350", "47,500",
        "27,330", "38,350", "60,950", "34,250",
    ]

    score, metrics = SequentialQA().score(
        ", ".join(values),
        {"gold_answer": values},
    )

    assert score == 1.0
    assert metrics["parse_strategy"] == "numeric_thousands_list"
    assert "114430" in metrics["pred_normalized"]
    assert "20000" in metrics["pred_normalized"]


def test_sqa_score_normalizes_space_grouped_thousands():
    from tasks.sqa.sequential_qa import SequentialQA

    score, metrics = SequentialQA().score(
        '["4 000"]',
        {"gold_answer": ["4000"]},
    )

    assert score == 1.0
    assert metrics["parse_strategy"] == "json_list"
    assert metrics["pred_normalized"] == ["4000"]
    assert metrics["gold_normalized"] == ["4000"]

    score, metrics = SequentialQA().score("4 000", {"gold_answer": ["4000"]})

    assert score == 1.0
    assert metrics["parse_strategy"] == "numeric_thousands_list"
    assert metrics["pred_normalized"] == ["4000"]


def test_sqa_score_python_list_with_commas_inside_values():
    from tasks.sqa.sequential_qa import SequentialQA

    score, metrics = SequentialQA().score(
        "['Valley HS (Las Vegas, NV)', 'Saugus (CA) HS']",
        {"gold_answer": ["Valley HS (Las Vegas, NV)", "Saugus (CA) HS"]},
    )

    assert score == 1.0
    assert metrics["parse_strategy"] == "python_list"


def test_sqa_code_score_executes_dataframe():
    from tasks.sqa.sequential_qa import SequentialQA

    task = SequentialQA()
    score, metrics = task.score(
        "__CODE__answer = df.loc[df['Points'].idxmax(), 'Driver']",
        {
            "_raw": {
                "answer_text": ["Lee"],
                "table": {
                    "headers": ["Driver", "Points"],
                    "rows": [["Kim", "12"], ["Lee", "18"]],
                },
            }
        },
    )
    assert score == 1.0
    assert metrics["prediction"] == "Lee"


def test_sqa_code_score_normalizes_series_answer():
    from tasks.sqa.sequential_qa import SequentialQA

    task = SequentialQA()
    score, metrics = task.score(
        "__CODE__answer = df.loc[df['Team'] == 'Red Bull', 'Driver']",
        {
            "_raw": {
                "answer_text": ["Sergio Perez", "Max Verstappen"],
                "table": {
                    "headers": ["Driver", "Team", "Points"],
                    "rows": [
                        ["Sergio Perez", "Red Bull", "25"],
                        ["Carlos Sainz", "Ferrari", "18"],
                        ["Max Verstappen", "Red Bull", "15"],
                    ],
                },
            }
        },
    )
    assert score == 1.0
    assert metrics["prediction"] == "Sergio Perez, Max Verstappen"


def test_sqa_code_score_helper_visible_inside_generator():
    from tasks.sqa.sequential_qa import SequentialQA

    task = SequentialQA()
    score, metrics = task.score(
        "__CODE__def to_int(value):\n"
        "    return int(value)\n"
        "answer = sum(to_int(value) for value in df['Points'])",
        {
            "_raw": {
                "answer_text": ["55"],
                "table": {
                    "headers": ["Driver", "Points"],
                    "rows": [["Sergio Perez", "25"], ["Carlos Sainz", "18"], ["Max Verstappen", "12"]],
                },
            }
        },
    )
    assert score == 1.0
    assert metrics["prediction"] == "55"


def test_sqa_code_score_duplicate_headers_do_not_break_coercion():
    from tasks.sqa.sequential_qa import SequentialQA

    task = SequentialQA()
    score, metrics = task.score(
        "__CODE__answer = df.iloc[0, 1]",
        {
            "_raw": {
                "answer_text": ["second"],
                "table": {
                    "headers": ["Name", "Name", "Points"],
                    "rows": [["first", "second", "10"], ["third", "fourth", "12"]],
                },
            }
        },
    )
    assert score == 1.0
    assert metrics["prediction"] == "second"


# ── HiTab parsers ─────────────────────────────────────────────────────

def test_hitab_registry_keys():
    from tasks.hitab.parsers import PARSER_REGISTRY, DISPATCH_FIELDS
    assert set(PARSER_REGISTRY.keys()) == DISPATCH_FIELDS
    assert DISPATCH_FIELDS == frozenset({"code", "answer"})


def test_hitab_parse_code_returns_prefix(tmp_mock_task):
    from tasks.hitab.parsers import PARSER_REGISTRY
    parser = PARSER_REGISTRY["code"]
    result = parser("df['Points'].max()", tmp_mock_task)
    assert result.startswith("__CODE__")
    assert "df" in result


def test_hitab_parse_answer_returns_plain(tmp_mock_task):
    from tasks.hitab.parsers import PARSER_REGISTRY
    parser = PARSER_REGISTRY["answer"]
    result = parser("The answer is Lee.", tmp_mock_task)
    assert result == "Lee"


def test_hitab_code_score_executes_flattened_table():
    from tasks.hitab.table_qa import HiTabQA

    task = HiTabQA()
    score, metrics = task.score(
        "__CODE__answer = df.loc[df['Points'].idxmax(), 'Driver']",
        {
            "_raw": {
                "answer": "[\"Lee\"]",
                "table_content": {
                    "title": "Drivers",
                    "top_header_rows_num": 1,
                    "texts": [
                        ["Driver", "Points"],
                        ["Kim", "12"],
                        ["Lee", "18"],
                    ],
                    "merged_regions": [],
                },
            }
        },
    )
    assert score == 1.0
    assert metrics["prediction"] == "Lee"


# ── nl2sql parsers ────────────────────────────────────────────────────

def test_nl2sql_registry_keys():
    from tasks.nl2sql.parsers import PARSER_REGISTRY, DISPATCH_FIELDS
    assert set(PARSER_REGISTRY.keys()) == DISPATCH_FIELDS
    assert DISPATCH_FIELDS == frozenset({"sql_query"})


def test_nl2sql_parse_sql_query(tmp_mock_task):
    from tasks.nl2sql.parsers import PARSER_REGISTRY
    parser = PARSER_REGISTRY["sql_query"]
    result = parser("SELECT id FROM users WHERE age > 30", tmp_mock_task)
    assert "SELECT" in result


def test_nl2sql_parse_sql_query_from_json(tmp_mock_task):
    from tasks.nl2sql.parsers import PARSER_REGISTRY
    import json
    parser = PARSER_REGISTRY["sql_query"]
    payload = json.dumps({"sql_query": "SELECT COUNT(*) FROM t"})
    result = parser(payload, tmp_mock_task)
    assert "SELECT COUNT" in result


# ── tabfact parsers ───────────────────────────────────────────────────

def test_tabfact_registry_keys():
    from tasks.tabfact.parsers import PARSER_REGISTRY, DISPATCH_FIELDS
    assert set(PARSER_REGISTRY.keys()) == DISPATCH_FIELDS
    assert DISPATCH_FIELDS == frozenset({"code", "verdict"})


def test_tabfact_parse_verdict_true(tmp_mock_task):
    from tasks.tabfact.parsers import PARSER_REGISTRY
    parser = PARSER_REGISTRY["verdict"]
    result = parser("True", tmp_mock_task)
    assert result == "True"


def test_tabfact_parse_verdict_false(tmp_mock_task):
    from tasks.tabfact.parsers import PARSER_REGISTRY
    parser = PARSER_REGISTRY["verdict"]
    result = parser("The statement is False.", tmp_mock_task)
    assert result == "False"


def test_tabfact_parse_code_returns_prefix(tmp_mock_task):
    from tasks.tabfact.parsers import PARSER_REGISTRY
    parser = PARSER_REGISTRY["code"]
    result = parser("df['wins'].sum() > 3", tmp_mock_task)
    assert result.startswith("__CODE__")


def test_tabfact_code_score_accepts_answer_print_contract():
    from tasks.tabfact.fact_verification import FactVerification

    task = FactVerification()
    score, metrics = task.score(
        "__CODE__answer = df['points'].astype(int).max() > 20\nprint(answer)",
        {
            "_raw": {
                "label": 1,
                "table_text": "points\n10\n25",
            }
        },
    )
    assert score == 1.0
    assert metrics["status"] == "ok"
    assert metrics["prediction"] == "True"


def test_tabfact_code_score_exposes_data_alias():
    from tasks.tabfact.fact_verification import FactVerification

    task = FactVerification()
    score, metrics = task.score(
        "__CODE__answer = int(data['rows'][1]['points']) > 20\nprint(answer)",
        {
            "_raw": {
                "label": 1,
                "table_text": "points\n10\n25",
            }
        },
    )
    assert score == 1.0
    assert metrics["status"] == "ok"
    assert metrics["prediction"] == "True"


def test_tabfact_code_score_helper_visible_inside_generator():
    from tasks.tabfact.fact_verification import FactVerification

    task = FactVerification()
    score, metrics = task.score(
        "__CODE__def over_20(value):\n"
        "    return int(value) > 20\n"
        "answer = any(over_20(value) for value in df['points'])\n"
        "print(answer)",
        {
            "_raw": {
                "label": 1,
                "table_text": "points\n10\n25",
            }
        },
    )
    assert score == 1.0
    assert metrics["status"] == "ok"
    assert metrics["prediction"] == "True"


def test_tabfact_code_score_duplicate_headers_do_not_break_coercion():
    from tasks.tabfact.fact_verification import FactVerification

    task = FactVerification()
    score, metrics = task.score(
        "__CODE__answer = int(df.iloc[1, 0]) > 20\nprint(answer)",
        {
            "_raw": {
                "label": 1,
                "table_text": "points#points\n10#10\n25#25",
            }
        },
    )
    assert score == 1.0
    assert metrics["status"] == "ok"
    assert metrics["prediction"] == "True"


# ── fixture ───────────────────────────────────────────────────────────

class _MockTask:
    """Minimal task stub — _prompt_state=None so parsers use their fallback paths."""
    _prompt_state = None


@pytest.fixture
def tmp_mock_task():
    return _MockTask()
