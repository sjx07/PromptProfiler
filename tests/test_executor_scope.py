"""test_executor_scope.py — runtime scope contract for code/sql executors.

Models frequently emit code that reaches for variable names/table names
DIFFERENT from what the executor exposes. These tests pin the charitable
aliases:

  * _execute_code exposes BOTH `df` (DataFrame) and `data` ({"table", "rows"}).
  * _execute_sql registers table `t` (canonical) AND a view `"table"` (alias),
    AND rewrites bare `FROM table` / `JOIN table` → `FROM t` / `JOIN t`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

_TOOL_DIR = str(Path(__file__).parent.parent.parent.parent)
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from prompt_profiler.tasks.wtq.table_qa import _execute_code, _execute_sql
from prompt_profiler.tasks.sqa.sequential_qa import _execute_code as _execute_sqa_code


TABLE = {
    "name": "F1 2022",
    "header": ["Position", "Driver", "Team", "Points"],
    "rows": [
        ["1", "Sergio Perez",    "Red Bull", "25"],
        ["2", "Carlos Sainz",    "Ferrari",  "18"],
        ["3", "Max Verstappen",  "Red Bull", "15"],
    ],
}


# ── _execute_code scope contract ──────────────────────────────────────

def test_execute_code_via_df():
    code = "winner = df.loc[df['Points'].astype(int).idxmax(), 'Driver']\nprint(winner)\nanswer = winner"
    result = _execute_code(code, TABLE)
    assert result == "Sergio Perez"


def test_execute_code_via_data_dict():
    """The new `data` alias — matches the JSON shape shown in the prompt."""
    code = (
        "rows = data['rows']\n"
        "best = max(rows, key=lambda r: int(r['Points']))\n"
        "answer = best['Driver']"
    )
    result = _execute_code(code, TABLE)
    assert result == "Sergio Perez"


def test_execute_code_data_has_table_name():
    """data['table'] carries the table's display name (as in the prompt)."""
    code = "answer = data['table']"
    result = _execute_code(code, TABLE)
    assert result == "F1 2022"


def test_execute_code_via_legacy_table_list():
    """Legacy `table` as list-of-records still works — kept for back-compat."""
    code = (
        "best = max(table, key=lambda r: int(r['Points']))\n"
        "answer = best['Driver']"
    )
    result = _execute_code(code, TABLE)
    assert result == "Sergio Perez"


def test_execute_code_model_redefining_data_still_has_access():
    """If a model redeclares `table = {...}` it shadows, but `df` and `data`
    from the globals remain, so charitable code still works by using them."""
    code = (
        "# model mistakenly redeclares table\n"
        "table = {'rows': []}\n"
        "# ... but if they use df, they still win\n"
        "answer = df.iloc[0]['Driver']"
    )
    result = _execute_code(code, TABLE)
    assert result == "Sergio Perez"


# ── stdout-capture contract (Finding F) ───────────────────────────────

def test_execute_code_captures_print_output_when_no_answer_var():
    """Model emits `print(...)` with no `answer = ...` assignment — the
    executor must capture stdout so the score pipeline gets a non-empty
    prediction. Before the fix this returned None → empty prediction → score=0.
    """
    code = (
        "count = df['Driver'].count()\n"
        "print(count)"
    )
    result = _execute_code(code, TABLE)
    assert result == "3"


def test_execute_code_uses_last_stdout_line():
    """Models often print debug then the answer — we take the LAST line."""
    code = (
        "print('debug:', len(df))\n"
        "winner = df.iloc[0]['Driver']\n"
        "print(winner)"
    )
    result = _execute_code(code, TABLE)
    assert result == "Sergio Perez"


def test_execute_code_normalizes_printed_list_when_no_answer_var():
    """Printed Python list reprs should be normalized like answer variables."""
    code = (
        "drivers = df.loc[df['Team'] == 'Red Bull', 'Driver'].tolist()\n"
        "print(drivers)"
    )
    result = _execute_code(code, TABLE)
    assert result == "Sergio Perez, Max Verstappen"


def test_execute_code_print_capture_is_thread_local():
    def run_one(i: int) -> str:
        return _execute_code(f"print('driver-{i}')", TABLE)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run_one, range(40)))

    assert results == [f"driver-{i}" for i in range(40)]


def test_execute_code_answer_variable_takes_priority():
    """If both `answer = ...` and print() are present, prefer the variable."""
    code = (
        "print('some scratch work')\n"
        "answer = 'canonical answer'"
    )
    result = _execute_code(code, TABLE)
    assert result == "canonical answer"


def test_execute_code_returns_none_on_exec_error():
    """A truly broken program still returns None (signals failure to score)."""
    code = "this is not valid python"
    assert _execute_code(code, TABLE) is None


def test_execute_code_single_expression_eval_path():
    """Pure expressions still round-trip via eval()."""
    code = "df.iloc[0]['Driver']"
    assert _execute_code(code, TABLE) == "Sergio Perez"


def test_execute_code_duplicate_headers_do_not_break_coercion():
    table = {
        "name": "duplicate headers",
        "header": ["Name", "Name", "Points"],
        "rows": [["first", "second", "10"], ["third", "fourth", "12"]],
    }
    code = "answer = df.iloc[0, 1]"
    assert _execute_code(code, table) == "second"


def test_execute_code_helper_visible_inside_generator_expression():
    code = (
        "def extract_score(value):\n"
        "    return int(value)\n"
        "answer = sum(extract_score(value) for value in df['Points'])"
    )
    assert _execute_code(code, TABLE) == 58


def test_execute_code_normalizes_series_answer_variable():
    code = "answer = df.loc[df['Team'] == 'Red Bull', 'Driver']"
    assert _execute_code(code, TABLE) == "Sergio Perez, Max Verstappen"


def test_execute_code_normalizes_single_column_dataframe():
    code = "answer = df.loc[df['Team'] == 'Red Bull', ['Driver']]"
    assert _execute_code(code, TABLE) == "Sergio Perez, Max Verstappen"


def test_execute_code_normalizes_numpy_array():
    code = "answer = df['Points'].to_numpy()"
    assert _execute_code(code, TABLE) == "25, 18, 15"


def test_sqa_execute_code_retries_raw_strings_after_typed_error():
    raw = {
        "table_file": "table_csv/example.csv",
        "history": [
            {
                "question": "what is the most amount of money spent?",
                "answer": ["152.5"],
            }
        ],
        "table": {
            "headers": ["Service", "2012/13 Total Cost (PSmillion)"],
            "rows": [
                ["BBC Radio 4", "122.1"],
                ["BBC Local Radio", "152.5"],
            ],
        },
    }
    code = (
        "answer = df.loc[df['2012/13 Total Cost (PSmillion)'] == '152.5', "
        "'Service'].iloc[0]"
    )
    assert _execute_sqa_code(code, raw) == "BBC Local Radio"


# ── _execute_sql scope contract ───────────────────────────────────────

def test_execute_sql_canonical_t():
    sql = "SELECT Driver FROM t ORDER BY CAST(Points AS INTEGER) DESC LIMIT 1;"
    result = _execute_sql(sql, TABLE)
    assert result == "Sergio Perez"


def test_execute_sql_accepts_bare_from_table():
    """Model writes `FROM table` despite rule saying `FROM t` — we rewrite."""
    sql = "SELECT Driver FROM table ORDER BY CAST(Points AS INTEGER) DESC LIMIT 1;"
    result = _execute_sql(sql, TABLE)
    assert result == "Sergio Perez"


def test_execute_sql_accepts_quoted_table():
    """Model writes `FROM "table"` — matches the registered view alias."""
    sql = 'SELECT Driver FROM "table" ORDER BY CAST(Points AS INTEGER) DESC LIMIT 1;'
    result = _execute_sql(sql, TABLE)
    assert result == "Sergio Perez"


def test_execute_sql_case_insensitive_from_table():
    """FROM table, FROM Table, from TABLE — all legal."""
    for clause in ("from table", "FROM Table", "From TABLE"):
        sql = f"SELECT Driver {clause} LIMIT 1"
        assert _execute_sql(sql, TABLE) == "Sergio Perez"


def test_execute_sql_bad_query_returns_none():
    assert _execute_sql("SELECT * FROM nonexistent_table", TABLE) is None
