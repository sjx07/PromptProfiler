"""Table input transforms — modify table data before formatting.

These are structural changes to what the model sees, applied before
table serialization (markdown/csv/html/json). Each transform takes
(header, rows, question) and returns (header, rows).
"""
from __future__ import annotations

import re
from typing import List, Optional


def annotate_types(header: List[str], rows: List[List[str]]) -> List[str]:
    """Add type annotations to column headers based on value inspection.

    Scans column values and appends (int), (float), (date), or (str).
    Returns new header list; rows are unchanged.
    """
    if not header or not rows:
        return header

    annotated = []
    for ci, col_name in enumerate(header):
        values = [row[ci] for row in rows if ci < len(row) and row[ci].strip()]
        col_type = _detect_column_type(values)
        if col_type and col_type != "str":
            annotated.append(f"{col_name} ({col_type})")
        else:
            annotated.append(col_name)
    return annotated


def prune_columns(
    header: List[str],
    rows: List[List[str]],
    question: str,
    *,
    max_cols: int = 20,
    min_keep: int = 3,
) -> tuple[List[str], List[List[str]]]:
    """Remove columns unlikely to be relevant to the question.

    Scoring heuristic:
    1. Exact substring match of column name in question → high score
    2. Word overlap between column name and question → medium score
    3. All columns kept if table has <= min_keep columns

    Returns (pruned_header, pruned_rows).
    """
    if not header or len(header) <= min_keep:
        return header, rows

    q_lower = question.lower()
    q_words = set(re.findall(r'\w+', q_lower))

    scored = []
    for ci, col in enumerate(header):
        col_lower = col.lower()
        col_words = set(re.findall(r'\w+', col_lower))

        # Exact substring match
        if col_lower in q_lower:
            score = 3.0
        # Partial word overlap
        elif col_words & q_words:
            score = 1.0 + len(col_words & q_words)
        # No match — keep but low priority
        else:
            score = 0.5

        scored.append((ci, score))

    # Always keep top-scored columns; drop lowest-scored if too many
    scored.sort(key=lambda x: -x[1])
    n_keep = min(len(header), max(min_keep, max_cols))

    # Keep all if question mentions most columns or table is small
    if len(header) <= max_cols:
        return header, rows

    keep_indices = sorted([ci for ci, _ in scored[:n_keep]])
    new_header = [header[ci] for ci in keep_indices]
    new_rows = [[row[ci] if ci < len(row) else "" for ci in keep_indices] for row in rows]
    return new_header, new_rows


def compute_column_stats(header: List[str], rows: List[List[str]]) -> str:
    """Compute column statistics and return a compact summary string.

    Serves double duty:
    - For the LLM: helps validate answers against data distribution
    - For routing: cardinality/range are predicates for primitive selection

    Returns a multi-line string like:
        Columns: Name (str, 25 unique), Score (int, range 45–99, 25 unique), Year (int, 2 values: 2023, 2024)
    """
    if not header or not rows:
        return ""

    parts = []
    for ci, col_name in enumerate(header):
        values = [row[ci].strip() for row in rows if ci < len(row) and row[ci].strip()]
        if not values:
            parts.append(f"{col_name} (empty)")
            continue

        col_type = _detect_column_type(values)
        n_unique = len(set(values))
        n_total = len(values)

        if col_type in ("int", "float"):
            # Parse numeric values for range
            nums = []
            for v in values:
                cleaned = v.replace(",", "").replace("$", "").replace("£", "").replace("€", "").replace("%", "")
                try:
                    nums.append(float(cleaned))
                except ValueError:
                    pass
            if nums:
                lo, hi = min(nums), max(nums)
                if col_type == "int":
                    lo, hi = int(lo), int(hi)
                if n_unique <= 5:
                    unique_vals = sorted(set(str(int(n) if col_type == "int" else n) for n in nums))
                    parts.append(f"{col_name} ({col_type}, {n_unique} values: {', '.join(unique_vals)})")
                else:
                    parts.append(f"{col_name} ({col_type}, range {lo}–{hi}, {n_unique} unique)")
            else:
                parts.append(f"{col_name} (str, {n_unique} unique)")
        elif col_type == "date":
            parts.append(f"{col_name} (date, {n_unique} unique)")
        else:
            if n_unique <= 5:
                sample = sorted(set(values))[:5]
                parts.append(f"{col_name} (str, {n_unique} values: {', '.join(sample)})")
            elif n_unique == n_total:
                parts.append(f"{col_name} (str, all unique)")
            else:
                parts.append(f"{col_name} (str, {n_unique} unique)")

    return "Columns: " + "; ".join(parts)


def _detect_column_type(values: List[str], sample_size: int = 20) -> str:
    """Detect column type from sampled values.

    Returns: "int", "float", "date", or "str".
    """
    if not values:
        return "str"

    sample = values[:sample_size]
    n = len(sample)

    # Try integer
    n_int = 0
    for v in sample:
        cleaned = v.strip().replace(",", "").replace("$", "").replace("£", "").replace("€", "").replace("%", "")
        if cleaned.lstrip("-").isdigit():
            n_int += 1
    if n_int > n * 0.7:
        return "int"

    # Try float
    n_float = 0
    for v in sample:
        cleaned = v.strip().replace(",", "").replace("$", "").replace("£", "").replace("€", "").replace("%", "")
        try:
            float(cleaned)
            n_float += 1
        except ValueError:
            pass
    if n_float > n * 0.7:
        return "float"

    # Try date patterns
    date_patterns = [
        r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # 2024-01-15
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',  # 01/15/2024
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',  # Month names
    ]
    n_date = 0
    for v in sample:
        if any(re.search(p, v, re.IGNORECASE) for p in date_patterns):
            n_date += 1
    if n_date > n * 0.5:
        return "date"

    return "str"
