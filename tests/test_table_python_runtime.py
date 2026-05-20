from tasks.parsing.code_output import CODE_PREFIX, clean_code, parse_code_output
from tasks.table_python_runtime import execute_python_table_code, runtime_from_rows


class _Task:
    _prompt_state = None


def test_runtime_executes_answer_assignment():
    runtime = runtime_from_rows(["name", "score"], [["a", "1"], ["b", "2"]])

    outcome = execute_python_table_code(
        "answer = df[df['score'] == 2]['name'].iloc[0]",
        runtime,
    )

    assert outcome.error is None
    assert outcome.value == "b"
    assert outcome.executed is True


def test_runtime_executes_printed_list():
    runtime = runtime_from_rows(["name", "score"], [["a", "1"], ["b", "2"]])

    outcome = execute_python_table_code("print(list(df['name']))", runtime)

    assert outcome.error is None
    assert outcome.value == "a, b"


def test_parse_code_output_uses_last_fenced_block():
    text = "scratch\n```python\nx = 1\n```\nfinal\n```python\nanswer = 2\n```"

    parsed = parse_code_output(text, _Task())

    assert parsed == f"{CODE_PREFIX}answer = 2"


def test_clean_code_preserves_block_indentation():
    code = clean_code(
        """
        if True:
            answer = 1
        """
    )

    assert code == "if True:\n    answer = 1"
