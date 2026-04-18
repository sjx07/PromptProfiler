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

_TOOL_DIR = str(Path(__file__).parent.parent.parent.parent)
if _TOOL_DIR not in sys.path:
    sys.path.insert(0, _TOOL_DIR)

from prompt_profiler.tasks.wtq.table_qa import _execute_code, _execute_sql


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
