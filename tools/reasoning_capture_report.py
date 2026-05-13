#!/usr/bin/env python3
"""Summarize hidden-reasoning capture diagnostics from a FACET cube."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", type=Path, help="Path to FACET experiment cube")
    parser.add_argument(
        "--cap-tokens",
        type=int,
        default=None,
        help="Token cap used by the run. If omitted, uses MAX(completion_tokens).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a Markdown table.",
    )
    args = parser.parse_args()

    rows = summarize(args.db, cap_tokens=args.cap_tokens)
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_markdown(rows)


def summarize(db_path: Path, *, cap_tokens: int | None = None) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        if not _has_column(conn, "execution", "raw_reasoning"):
            raise SystemExit(
                "execution.raw_reasoning is missing. Open the cube with the updated "
                "CubeStore first, or run against a v10 cube."
            )
        if not _has_column(conn, "execution", "finish_reason"):
            raise SystemExit(
                "execution.finish_reason is missing. Open the cube with the updated "
                "CubeStore first, or run against a v10 cube."
            )

        if cap_tokens is None:
            row = conn.execute(
                "SELECT MAX(completion_tokens) AS max_completion FROM execution"
            ).fetchone()
            cap_tokens = int(row["max_completion"] or 0)

        rows = conn.execute(
            """
            WITH scored AS (
                SELECT
                    e.config_id,
                    COALESCE(json_extract(c.meta, '$.canonical_id'), 'BASE') AS feature,
                    e.raw_response,
                    e.raw_reasoning,
                    e.completion_tokens,
                    e.finish_reason,
                    ev.score
                FROM execution e
                JOIN config c USING(config_id)
                LEFT JOIN evaluation ev USING(execution_id)
            )
            SELECT
                config_id,
                feature,
                COUNT(*) AS n,
                AVG(score) AS score,
                SUM(raw_response = '') AS empty_content,
                SUM(COALESCE(raw_reasoning, '') != '') AS nonempty_reasoning,
                SUM(completion_tokens >= ?) AS capped,
                SUM(finish_reason = 'length') AS length_finished,
                SUM(finish_reason = 'stop' AND COALESCE(raw_response, '') != '') AS stop_answered,
                AVG(CASE
                    WHEN finish_reason = 'stop' AND COALESCE(raw_response, '') != ''
                    THEN score
                END) AS stop_answered_score,
                AVG(LENGTH(COALESCE(raw_response, ''))) AS avg_content_chars,
                AVG(LENGTH(COALESCE(raw_reasoning, ''))) AS avg_reasoning_chars,
                AVG(completion_tokens) AS avg_completion_tokens
            FROM scored
            GROUP BY config_id, feature
            ORDER BY config_id
            """,
            (cap_tokens,),
        ).fetchall()

        finish_rows = conn.execute(
            """
            SELECT config_id, COALESCE(finish_reason, '') AS finish_reason, COUNT(*) AS n
            FROM execution
            GROUP BY config_id, COALESCE(finish_reason, '')
            ORDER BY config_id, n DESC, finish_reason
            """
        ).fetchall()
    finally:
        conn.close()

    finish_by_config: dict[int, list[str]] = {}
    for row in finish_rows:
        config_id = int(row["config_id"])
        reason = row["finish_reason"] or "<missing>"
        finish_by_config.setdefault(config_id, []).append(f"{reason}:{row['n']}")

    out = []
    for row in rows:
        n = int(row["n"])
        out.append({
            "config_id": int(row["config_id"]),
            "feature": row["feature"],
            "n": n,
            "score": _round_or_none(row["score"]),
            "empty_content_rate": round((row["empty_content"] or 0) / n, 4) if n else None,
            "reasoning_present_rate": round((row["nonempty_reasoning"] or 0) / n, 4) if n else None,
            "cap_rate": round((row["capped"] or 0) / n, 4) if n else None,
            "length_n": int(row["length_finished"] or 0),
            "stop_answered_n": int(row["stop_answered"] or 0),
            "stop_answered_score": _round_or_none(row["stop_answered_score"]),
            "avg_content_chars": _round_or_none(row["avg_content_chars"], digits=1),
            "avg_reasoning_chars": _round_or_none(row["avg_reasoning_chars"], digits=1),
            "avg_completion_tokens": _round_or_none(row["avg_completion_tokens"], digits=1),
            "finish_reasons": ", ".join(finish_by_config.get(int(row["config_id"]), [])),
        })
    return out


def print_markdown(rows: list[dict[str, Any]]) -> None:
    headers = [
        "config",
        "feature",
        "n",
        "score",
        "empty_content_rate",
        "reasoning_present_rate",
        "cap_rate",
        "length_n",
        "stop_answered_n",
        "stop_answered_score",
        "avg_content_chars",
        "avg_reasoning_chars",
        "avg_completion_tokens",
        "finish_reasons",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        values = [
            row["config_id"],
            row["feature"],
            row["n"],
            _fmt(row["score"]),
            _fmt(row["empty_content_rate"]),
            _fmt(row["reasoning_present_rate"]),
            _fmt(row["cap_rate"]),
            row["length_n"],
            row["stop_answered_n"],
            _fmt(row["stop_answered_score"]),
            _fmt(row["avg_content_chars"]),
            _fmt(row["avg_reasoning_chars"]),
            _fmt(row["avg_completion_tokens"]),
            row["finish_reasons"],
        ]
        print("| " + " | ".join(str(v) for v in values) + " |")


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row["name"] == column for row in rows)


def _round_or_none(value: Any, *, digits: int = 4) -> float | None:
    return round(float(value), digits) if value is not None else None


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


if __name__ == "__main__":
    main()
