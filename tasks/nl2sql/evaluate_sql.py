"""SQL execution accuracy scorer — evaluate predicted SQL against gold.

Ported from transfer_pipeline (branch mvp/nl2sql-cube). Core logic:
execute both predicted and gold SQL on the actual sqlite DB, compare
result sets after normalization.

Also provides schema-linking F1 scorer (set_f1_with_unresolved).
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, Iterable, List, Set, Tuple

logger = logging.getLogger(__name__)


def markdown2sql(markdown_text: str) -> str:
    """Strip markdown code fences from SQL text."""
    if not markdown_text:
        return ""
    lines = markdown_text.strip().splitlines()
    sql_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```sql"):
            continue
        if stripped.startswith("```"):
            break
        sql_lines.append(line)
    return "\n".join(sql_lines).strip()


def _normalize_result(result: List[tuple]) -> Tuple[tuple, ...]:
    """Normalize SQL results for comparison (round floats, sort rows)."""
    def normalize_value(val: Any) -> Any:
        if isinstance(val, float):
            return round(val, 6)
        return val

    normalized = [tuple(normalize_value(v) for v in row) for row in result]
    try:
        normalized = sorted(normalized)
    except TypeError:
        pass
    return tuple(normalized)


def evaluate_execution(gold: Dict[str, Any], pred_sql: str) -> float:
    """Execute both SQLs on the DB and compare result sets.

    Args:
        gold: {"db_path": str, "gold_sql_query": str}
        pred_sql: Predicted SQL string.

    Returns:
        1.0 if results match, 0.0 otherwise.
    """
    db_path = gold.get("db_path")
    gold_sql = gold.get("gold_sql_query")
    if not db_path:
        raise ValueError("Gold record does not contain 'db_path'")
    if not gold_sql:
        raise ValueError("Gold record does not contain 'gold_sql_query'")

    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        cur = conn.cursor()
        cur.execute(markdown2sql(pred_sql))
        pred_result = cur.fetchall()

    with sqlite3.connect(uri, uri=True) as conn:
        cur = conn.cursor()
        cur.execute(gold_sql)
        gold_result = cur.fetchall()

    return 1.0 if _normalize_result(pred_result) == _normalize_result(gold_result) else 0.0


def _classify_sql_error(error_message: str) -> str:
    """Categorize SQL execution error by type."""
    lowered = error_message.lower()
    if "no such table" in lowered:
        return "table_not_found"
    if "no such column" in lowered:
        return "column_not_found"
    if "ambiguous column" in lowered:
        return "ambiguous_column"
    if "syntax error" in lowered:
        return "syntax_error"
    if "timeout" in lowered:
        return "timeout"
    return "execution_error"


def evaluate_execution_wrapper(
    gold: Dict[str, Any],
    pred_sql: str,
) -> Dict[str, Any]:
    """Normalize pred_sql and evaluate via execution accuracy.

    Returns dict with: score, status, error_type, error_message.
    """
    if isinstance(pred_sql, dict):
        pred_sql = pred_sql.get("pred_sql_query", "")
    if ";" in pred_sql:
        pred_sql = pred_sql.split(";")[0]
    pred_sql = markdown2sql(pred_sql)

    try:
        score = evaluate_execution(gold, pred_sql)
    except Exception as exc:
        logger.error("Execution evaluation failed: %s", exc)
        return {
            "score": 0.0,
            "status": "error",
            "error_type": _classify_sql_error(str(exc)),
            "error_message": str(exc),
        }

    if score == 1.0:
        return {
            "score": 1.0,
            "status": "success",
            "error_type": None,
            "error_message": None,
        }

    return {
        "score": 0.0,
        "status": "mismatch",
        "error_type": "result_mismatch",
        "error_message": "Query executed but results do not match gold standard",
    }


# ── Schema linking F1 scorer ─────────────────────────────────────────

TableColumnMap = Dict[str, Iterable[str]]
Column = Tuple[str, str]


def canonicalize(m: TableColumnMap) -> Set[Column]:
    """Normalize {table: [columns]} to a set of (table, column) pairs."""
    canon: Set[Column] = set()
    for table, cols in m.items():
        t = str(table).lower()
        if not isinstance(cols, Iterable):
            continue
        for col in cols:
            if not isinstance(col, str):
                continue
            canon.add((t, col.lower()))
    return canon


def set_f1_with_unresolved(
    pred: TableColumnMap,
    gold: TableColumnMap,
    unresolved_columns: Dict[str, List[str]],
) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 for schema linking with unresolved columns."""
    if not isinstance(pred, dict) or not isinstance(gold, dict):
        return 0.0, 0.0, 0.0

    predicted = canonicalize(pred)
    gold_resolved = canonicalize(gold)
    unresolved = {
        col.lower(): {t.lower() for t in tables}
        for col, tables in unresolved_columns.items()
    }

    correct_pred = 0
    for table, col in predicted:
        if (table, col) in gold_resolved:
            correct_pred += 1
        elif col in unresolved and table in unresolved[col]:
            correct_pred += 1

    precision = correct_pred / len(predicted) if predicted else 0.0

    recall_hits = len(predicted & gold_resolved)
    recall_total = len(gold_resolved) + len(unresolved)
    for col, candidates in unresolved.items():
        if any((table, col) in predicted for table in candidates):
            recall_hits += 1

    recall = recall_hits / recall_total if recall_total > 0 else 0.0
    if precision + recall == 0.0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)
