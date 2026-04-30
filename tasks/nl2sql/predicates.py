"""NL2SQL predicate extractors for the SQL-repair task.

Computes observable query features from:
  - Question text (intent predicates)
  - Wrong SQL AST (structural predicates, prefixed with ``wrong_``)
  - SQLite DB schema/statistics (DB introspection predicates)
  - Error metadata (error_type, source_model)

All predicate values are categorical strings ("yes"/"no" for booleans,
or a free-form string for categorical ones).

Register with query_cohorts so seed_predicates() can find them.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from collections import Counter
from functools import lru_cache
from typing import Dict, List, Optional

import sqlglot
import sqlglot.expressions  # noqa: F401 — ensures sqlglot.exp alias is populated


from experiment.query_cohorts import register_extractor

logger = logging.getLogger(__name__)


# ── task predicates (question intent) ────────────────────────────────────


def _extract_task_predicates(question: str, difficulty: str, evidence: str) -> Dict[str, str]:
    """Extract intent predicates from question text, difficulty label, and evidence.

    All values are "yes"/"no" strings (or a categorical string for ``difficulty``).
    Uses only information available at inference time — no gold SQL, no gold answer.

    Args:
        question: Natural language question.
        difficulty: Difficulty label (e.g. "simple", "moderate", "challenging").
        evidence: Domain-knowledge hint accompanying the question.

    Returns:
        Dict of predicate name → string value.
    """
    q_lower = (question or "").lower()
    difficulty_text = (difficulty or "").strip().lower()
    return {
        "has_evidence": "yes" if (evidence or "").strip() else "no",
        "difficulty": difficulty_text or "unknown",
        "long_question": "yes" if len((question or "").split()) >= 30 else "no",
        "q_asks_count": "yes" if re.search(
            r"\b(how many|count|number of|total number)\b", q_lower
        ) else "no",
        "q_asks_max_min": "yes" if re.search(
            r"\b(highest|lowest|most|least|maximum|minimum|greatest|smallest"
            r"|top|best|worst|largest|biggest|fewest)\b", q_lower
        ) else "no",
        "q_asks_ratio": "yes" if re.search(
            r"\b(percentage|percent|ratio|proportion|share|rate|fraction)\b", q_lower
        ) else "no",
        "q_asks_list": "yes" if re.search(
            r"\b(list|name all|show all|give me all|find all|what are the)\b", q_lower
        ) else "no",
        "q_asks_comparison": "yes" if re.search(
            r"\b(more than|less than|greater than|higher than|lower than"
            r"|at least|at most|between.*and|exceed|above|below)\b", q_lower
        ) else "no",
        "q_asks_distinct": "yes" if re.search(
            r"\b(different|unique|distinct|types?\s+of|kinds?\s+of|categories\s+of)\b", q_lower
        ) else "no",
        "q_asks_temporal": "yes" if re.search(
            r"\b(born|before|after|since|until|during|older|younger"
            r"|earliest|latest|recent)\b", q_lower
        ) else "no",
    }


# ── AST predicate names (used for defaults and registration) ──────────────

_AST_PRED_NAMES: List[str] = [
    "has_join",
    "has_subquery",
    "has_nested_subquery",
    "has_union",
    "has_limit",
    "has_distinct",
    "has_case",
    "has_where",
    "has_group_by",
    "has_having",
    "has_order_by",
    "has_aggregation",
    "has_like",
    "has_between",
    "has_in",
    "has_exists",
    "multi_join",
    "multi_table",
    "many_conditions",
    "wide_select",
    "has_math",
    "has_string_op",
    "has_date_func",
    "has_null_check",
    "has_or",
    "has_not",
]

_AST_DEFAULTS: Dict[str, str] = {k: "no" for k in _AST_PRED_NAMES}


def _extract_ast_predicates(sql: str) -> Dict[str, str]:
    """Parse *sql* with sqlglot and extract 26 structural AST features.

    Used on the *wrong* SQL — information observable at inference time.
    On parse failure, returns all "no".

    Args:
        sql: SQL string to analyse.

    Returns:
        Dict of short predicate name → "yes"/"no".
    """
    if not sql or not sql.strip():
        return dict(_AST_DEFAULTS)

    try:
        tree = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:
        return dict(_AST_DEFAULTS)

    joins = list(tree.find_all(sqlglot.exp.Join))
    subqueries = list(tree.find_all(sqlglot.exp.Subquery))
    agg_funcs = list(tree.find_all(sqlglot.exp.AggFunc))
    groups = list(tree.find_all(sqlglot.exp.Group))
    orders = list(tree.find_all(sqlglot.exp.Order))
    havings = list(tree.find_all(sqlglot.exp.Having))
    wheres = list(tree.find_all(sqlglot.exp.Where))
    limits = list(tree.find_all(sqlglot.exp.Limit))
    distincts = list(tree.find_all(sqlglot.exp.Distinct))
    unions = list(tree.find_all(sqlglot.exp.Union))
    cases = list(tree.find_all(sqlglot.exp.Case))
    likes = list(tree.find_all(sqlglot.exp.Like))
    betweens = list(tree.find_all(sqlglot.exp.Between))
    ins = list(tree.find_all(sqlglot.exp.In))
    exists_nodes = list(tree.find_all(sqlglot.exp.Exists))
    tables = list(tree.find_all(sqlglot.exp.Table))
    or_nodes = list(tree.find_all(sqlglot.exp.Or))
    not_nodes = list(tree.find_all(sqlglot.exp.Not))

    # Math: division, multiplication, subtraction
    math_ops = (
        list(tree.find_all(sqlglot.exp.Div))
        + list(tree.find_all(sqlglot.exp.Mul))
        + list(tree.find_all(sqlglot.exp.Sub))
    )

    sql_upper = sql.upper()

    # String functions — LIKE already in likes; also check named functions
    has_string_op = len(likes) >= 1 or bool(
        re.search(
            r"\b(REPLACE|SUBSTR|SUBSTRING|UPPER|LOWER|TRIM|LTRIM|RTRIM|INSTR|LENGTH|GLOB)\s*\(",
            sql_upper,
        )
    )

    # Date functions
    has_date_func = bool(
        re.search(r"\b(STRFTIME|DATE|DATETIME|JULIANDAY|TIME)\s*\(", sql_upper)
    )

    # NULL checks (IS NULL / IS NOT NULL)
    has_null_check = bool(re.search(r"\bIS\s+(NOT\s+)?NULL\b", sql_upper))

    def _count_conditions(expr: Optional[sqlglot.exp.Expression]) -> int:
        if expr is None:
            return 0
        logical = list(expr.find_all(sqlglot.exp.And))
        logical.extend(expr.find_all(sqlglot.exp.Or))
        return 1 + len(logical)

    def _is_nested(node: sqlglot.exp.Subquery) -> bool:
        parent = node.parent
        while parent is not None:
            if isinstance(parent, sqlglot.exp.Subquery):
                return True
            parent = parent.parent
        return False

    where_node = tree.find(sqlglot.exp.Where)
    having_node = tree.find(sqlglot.exp.Having)
    condition_count = _count_conditions(where_node.this if where_node else None)
    condition_count += _count_conditions(having_node.this if having_node else None)

    top_select = tree.find(sqlglot.exp.Select)
    select_width = len(top_select.expressions) if top_select else 0

    def _yn(flag: bool) -> str:
        return "yes" if flag else "no"

    return {
        "has_join": _yn(len(joins) >= 1),
        "has_subquery": _yn(len(subqueries) >= 1),
        "has_nested_subquery": _yn(any(_is_nested(n) for n in subqueries)),
        "has_union": _yn(len(unions) >= 1),
        "has_limit": _yn(len(limits) >= 1),
        "has_distinct": _yn(len(distincts) >= 1),
        "has_case": _yn(len(cases) >= 1),
        "has_where": _yn(len(wheres) >= 1),
        "has_group_by": _yn(len(groups) >= 1),
        "has_having": _yn(len(havings) >= 1),
        "has_order_by": _yn(len(orders) >= 1),
        "has_aggregation": _yn(len(agg_funcs) >= 1),
        "has_like": _yn(len(likes) >= 1),
        "has_between": _yn(len(betweens) >= 1),
        "has_in": _yn(len(ins) >= 1),
        "has_exists": _yn(len(exists_nodes) >= 1),
        "multi_join": _yn(len(joins) >= 2),
        "multi_table": _yn(len(tables) >= 3),
        "many_conditions": _yn(condition_count >= 3),
        "wide_select": _yn(select_width >= 5),
        "has_math": _yn(len(math_ops) >= 1),
        "has_string_op": _yn(has_string_op),
        "has_date_func": _yn(has_date_func),
        "has_null_check": _yn(has_null_check),
        "has_or": _yn(len(or_nodes) >= 1),
        "has_not": _yn(len(not_nodes) >= 1),
    }


# ── DB introspection predicates ───────────────────────────────────────────

_DB_PRED_NAMES: List[str] = [
    "complex_schema",
    "deep_schema",
    "fk_rich",
    "data_large",
    "giant_table",
    "wide_tables",
    "has_date_col",
    "has_real_col",
    "text_heavy",
    "has_special_cols",
    "many_tables",
    "ambiguous_cols",
    "col_has_space",
    "col_has_hyphen",
]

_DB_DEFAULTS: Dict[str, str] = {k: "no" for k in _DB_PRED_NAMES}

_DATE_KEYWORDS = frozenset({
    "date", "time", "year", "month", "day",
    "created", "updated", "timestamp",
    "born", "opened", "closed", "start", "end",
})


@lru_cache(maxsize=4096)
def _extract_db_predicates(db_path: str) -> Dict[str, str]:
    """Introspect a SQLite database and extract 14 schema/data predicates.

    Opens the actual DB at *db_path* — this is information observable at
    inference time (the schema is provided in the prompt). On any failure,
    returns all "no".

    Args:
        db_path: Filesystem path to the SQLite file.

    Returns:
        Dict of predicate name → "yes"/"no".
    """
    if not db_path:
        return dict(_DB_DEFAULTS)

    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as exc:
        logger.warning("Failed to open DB %s: %s", db_path, exc)
        return dict(_DB_DEFAULTS)

    try:
        table_rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        table_names = [row[0] for row in table_rows]
        if not table_names:
            return dict(_DB_DEFAULTS)

        total_cols = 0
        total_fk = 0
        total_rows = 0
        max_rows = 0
        has_date_col = False
        n_real_cols = 0
        n_text_cols = 0
        col_has_space = False
        col_has_hyphen = False
        col_has_special = False
        all_col_names: List[str] = []

        for tname in table_names:
            quoted = '"' + tname.replace('"', '""') + '"'
            cols = conn.execute(f"PRAGMA table_info({quoted})").fetchall()
            total_cols += len(cols)
            total_fk += len(conn.execute(f"PRAGMA foreign_key_list({quoted})").fetchall())

            for col in cols:
                col_name_raw: str = col[1] or ""
                col_name = col_name_raw.lower()
                col_type = (col[2] or "").upper()
                all_col_names.append(col_name)

                if " " in col_name_raw:
                    col_has_space = True
                    col_has_special = True
                if "-" in col_name_raw:
                    col_has_hyphen = True
                    col_has_special = True
                if "(" in col_name_raw or ")" in col_name_raw:
                    col_has_special = True

                if col_type in ("DATE", "DATETIME", "TIMESTAMP"):
                    has_date_col = True
                elif any(kw in col_name for kw in _DATE_KEYWORDS):
                    has_date_col = True

                if col_type in ("REAL", "FLOAT", "DOUBLE", "NUMERIC", "DECIMAL"):
                    n_real_cols += 1
                if col_type in ("TEXT", "VARCHAR", "CHAR", "NVARCHAR", "CLOB") or "CHAR" in col_type:
                    n_text_cols += 1

            try:
                row_count = conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
            except sqlite3.Error as exc:
                logger.warning("Row count failed for %s in %s: %s", tname, db_path, exc)
                row_count = 0

            if isinstance(row_count, int):
                total_rows += row_count
                if row_count > max_rows:
                    max_rows = row_count

        n_tables = len(table_names)
        avg_fk = total_fk / n_tables if n_tables else 0.0
        avg_cols = total_cols / n_tables if n_tables else 0.0
        has_ambiguous = any(v >= 2 for v in Counter(all_col_names).values())

        def _yn(flag: bool) -> str:
            return "yes" if flag else "no"

        return {
            "complex_schema": _yn(n_tables >= 10),
            "deep_schema": _yn(total_cols >= 55),
            "fk_rich": _yn(avg_fk >= 1.0),
            "data_large": _yn(total_rows >= 100_000),
            "giant_table": _yn(max_rows >= 50_000),
            "wide_tables": _yn(avg_cols >= 8.0),
            "has_date_col": _yn(has_date_col),
            "has_real_col": _yn(n_real_cols >= 2),
            "text_heavy": _yn(n_text_cols / max(total_cols, 1) >= 0.5),
            "has_special_cols": _yn(col_has_special),
            "many_tables": _yn(n_tables >= 8),
            "ambiguous_cols": _yn(has_ambiguous),
            "col_has_space": _yn(col_has_space),
            "col_has_hyphen": _yn(col_has_hyphen),
        }
    except sqlite3.Error as exc:
        logger.warning("Introspection failed for %s: %s", db_path, exc)
        return dict(_DB_DEFAULTS)
    finally:
        conn.close()


# ── main entry point ──────────────────────────────────────────────────────


def compute_predicates(meta: dict) -> Dict[str, str]:
    """Compute all predicates for a NL2SQL repair query.

    Combines:
      - Task predicates from question/difficulty/evidence text
      - Wrong-SQL AST predicates (prefixed ``wrong_``)
      - DB schema/data predicates
      - Error metadata (``error_type``, ``source_model``)

    All values are observable at inference time — no gold SQL used.

    Args:
        meta: Query metadata dict. Expected keys:
            ``_raw``       — raw dataset record (question, db_path, …)
            ``difficulty`` — difficulty label string
            ``evidence``   — hint/evidence string
            ``wrong_sql``  — the incorrect SQL to repair
            ``error_type`` — category of the SQL error
            ``source_model`` — model that produced the wrong SQL

    Returns:
        Flat dict of predicate name → string value.
    """
    raw = meta.get("_raw", {})
    question: str = raw.get("question", "")
    difficulty: str = meta.get("difficulty", "")
    evidence: str = meta.get("evidence", "")
    wrong_sql: str = meta.get("wrong_sql", raw.get("wrong_sql", ""))
    db_path: str = raw.get("db_path", "")
    error_type: str = meta.get("error_type", "unknown")
    source_model: str = meta.get("source_model", "unknown")

    preds: Dict[str, str] = _extract_task_predicates(question, difficulty, evidence)

    # Wrong-SQL AST (observable: the wrong SQL is the repair task input)
    wrong_ast = _extract_ast_predicates(wrong_sql)
    preds.update({f"wrong_{k}": v for k, v in wrong_ast.items()})

    # DB schema/statistics (defaults to "no" for all when db_path is absent)
    preds.update(_extract_db_predicates(db_path))

    # Error metadata
    preds["error_type"] = error_type
    preds["source_model"] = source_model

    return preds


# ── predicate name registry ───────────────────────────────────────────────

_TASK_PRED_NAMES: List[str] = [
    "has_evidence",
    "difficulty",
    "long_question",
    "q_asks_count",
    "q_asks_max_min",
    "q_asks_ratio",
    "q_asks_list",
    "q_asks_comparison",
    "q_asks_distinct",
    "q_asks_temporal",
]

_ALL_PRED_NAMES: List[str] = (
    _TASK_PRED_NAMES
    + [f"wrong_{k}" for k in _AST_PRED_NAMES]
    + _DB_PRED_NAMES
    + ["error_type", "source_model"]
)


# ── extractor registration ────────────────────────────────────────────────


_QUERY_PREDICATE_CACHE: Dict[str, Dict[str, str]] = {}


def _cache_key(query: dict, meta: dict) -> str:
    query_id = str(query.get("query_id", ""))
    try:
        meta_key = json.dumps(meta, sort_keys=True, default=str)
    except TypeError:
        meta_key = str(meta)
    return f"{query_id}\0{meta_key}"


def _make_extractor(pred_name: str):
    """Create an extractor closure for a single predicate name.

    Args:
        pred_name: Name of the predicate to extract.

    Returns:
        Extractor function (query: dict) → str.
    """
    def extractor(query: dict) -> str:
        meta = query.get("meta", {})
        if isinstance(meta, str):
            meta = json.loads(meta)
        key = _cache_key(query, meta)
        preds = _QUERY_PREDICATE_CACHE.get(key)
        if preds is None:
            preds = compute_predicates(meta)
            _QUERY_PREDICATE_CACHE[key] = preds
        return preds.get(pred_name, "unknown")
    return extractor


for _name in _ALL_PRED_NAMES:
    register_extractor(_name)(_make_extractor(_name))
