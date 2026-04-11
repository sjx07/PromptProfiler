"""Table serialization formats for WTQ.

Each function takes (header, rows, name) and returns a string representation.
The format affects how the model "sees" the table in the prompt.
"""
from __future__ import annotations

from typing import List


def table_to_markdown(header: List[str], rows: List[List[str]], name: str = "") -> str:
    """Markdown table format (default)."""
    if not header:
        return ""
    lines = []
    if name:
        lines.append(f"Table: {name}")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows:
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded[:len(header)]) + " |")
    return "\n".join(lines)


def table_to_csv(header: List[str], rows: List[List[str]], name: str = "") -> str:
    """CSV format — compact, no alignment overhead."""
    if not header:
        return ""
    lines = []
    if name:
        lines.append(f"# Table: {name}")
    lines.append(",".join(header))
    for row in rows:
        padded = row + [""] * (len(header) - len(row))
        # Escape commas inside values
        escaped = []
        for v in padded[:len(header)]:
            if "," in v or '"' in v:
                escaped.append('"' + v.replace('"', '""') + '"')
            else:
                escaped.append(v)
        lines.append(",".join(escaped))
    return "\n".join(lines)


def table_to_html(header: List[str], rows: List[List[str]], name: str = "") -> str:
    """HTML table format — explicit structure with tags."""
    if not header:
        return ""
    lines = []
    if name:
        lines.append(f"<caption>{name}</caption>")
    lines.append("<table>")
    lines.append("<thead><tr>" + "".join(f"<th>{h}</th>" for h in header) + "</tr></thead>")
    lines.append("<tbody>")
    for row in rows:
        padded = row + [""] * (len(header) - len(row))
        lines.append("<tr>" + "".join(f"<td>{v}</td>" for v in padded[:len(header)]) + "</tr>")
    lines.append("</tbody>")
    lines.append("</table>")
    return "\n".join(lines)


def table_to_json_records(header: List[str], rows: List[List[str]], name: str = "") -> str:
    """JSON array of row objects — each row is {col: val}."""
    import json
    if not header:
        return "[]"
    records = []
    for row in rows:
        padded = row + [""] * (len(header) - len(row))
        records.append(dict(zip(header, padded[:len(header)])))
    obj = {"table": name, "rows": records} if name else records
    return json.dumps(obj, indent=1)


# ── registry ─────────────────────────────────────────────────────────

TABLE_FORMATS = {
    "markdown": table_to_markdown,
    "csv": table_to_csv,
    "html": table_to_html,
    "json_records": table_to_json_records,
}


def get_table_formatter(fmt: str):
    """Get table format function by name. Defaults to markdown."""
    return TABLE_FORMATS.get(fmt, table_to_markdown)
