"""Preprocess registry — decoupled input transforms.

Two registries, same spec format:

    Table transforms:   (header, rows, text, **kwargs) → (header, rows)
    Record transforms:  (record, **kwargs) → record

Spec (func table): transform_input(fn="row_filter", max_rows=50)
Materialization (this registry): REGISTRY["row_filter"] → Python function

Adding a new transform = one function + one register/register_record call.
No changes to func_registry.py, task.py, or any config generator.

Calcite analogy:
    add_rule = optimizer hint (advisory text, model may ignore)
    transform_input = transformation rule (changes what the model sees, mandatory)
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

# ── Table transforms: (header, rows, text, **kwargs) -> (header, rows) ──

TransformFn = Callable[..., Tuple[List[str], List[List[str]]]]

REGISTRY: Dict[str, TransformFn] = {}


def register(name: str):
    """Decorator to register a table preprocess transform."""
    def wrapper(fn: TransformFn) -> TransformFn:
        REGISTRY[name] = fn
        return fn
    return wrapper


# ── Record transforms: (record, **kwargs) -> record ─────────────────────

RecordTransformFn = Callable[..., Dict[str, Any]]

RECORD_REGISTRY: Dict[str, RecordTransformFn] = {}


def register_record(name: str):
    """Decorator to register a record-level transform.

    Record transforms operate on the full build_record() dict — they can
    read any field and modify/replace any field. Used for transforms that
    operate on text fields (schema DDL, error messages) rather than
    structured table data.
    """
    def wrapper(fn: RecordTransformFn) -> RecordTransformFn:
        RECORD_REGISTRY[name] = fn
        return fn
    return wrapper


def apply_record_transforms(
    record: Dict[str, Any],
    transforms: List[dict],
) -> Dict[str, Any]:
    """Apply a sequence of record-level transforms.

    Args:
        record: The build_record() output dict.
        transforms: List of {"fn": name, "kwargs": {...}} dicts.

    Returns:
        Transformed record dict.
    """
    for spec in transforms:
        fn_name = spec.get("fn", "")
        kwargs = spec.get("kwargs", {})
        fn = RECORD_REGISTRY.get(fn_name)
        if fn is None:
            # Not a record transform — skip (might be a table transform)
            continue
        record = fn(record, **kwargs)
    return record


def apply_transforms(
    header: List[str],
    rows: List[List[str]],
    text: str,
    transforms: List[dict],
) -> Tuple[List[str], List[List[str]]]:
    """Apply a sequence of transforms to (header, rows).

    Args:
        header: Column names.
        rows: Row data.
        text: Question or statement (context for transforms).
        transforms: List of {"fn": name, "kwargs": {...}} dicts,
                    in the order stored in PromptBuildState.extras.

    Returns:
        (transformed_header, transformed_rows)
    """
    for spec in transforms:
        fn_name = spec.get("fn", "")
        kwargs = spec.get("kwargs", {})
        fn = REGISTRY.get(fn_name)
        if fn is None:
            logger.warning("Unknown preprocess transform: %s", fn_name)
            continue
        header, rows = fn(header, rows, text, **kwargs)
    return header, rows


# ── Built-in transforms ───────────────────────────────────────────────


@register("prune_columns")
def prune_columns(
    header: List[str],
    rows: List[List[str]],
    text: str,
    *,
    max_cols: int = 20,
    min_keep: int = 3,
) -> Tuple[List[str], List[List[str]]]:
    """Remove columns unlikely to be relevant to the question.

    Scoring: exact substring match > word overlap > no match.
    Keeps all columns if table has <= max_cols columns.
    """
    if not header or len(header) <= min_keep or len(header) <= max_cols:
        return header, rows

    q_lower = text.lower()
    q_words = set(re.findall(r'\w+', q_lower))

    scored = []
    for ci, col in enumerate(header):
        col_lower = col.lower()
        col_words = set(re.findall(r'\w+', col_lower))
        if col_lower in q_lower:
            score = 3.0
        elif col_words & q_words:
            score = 1.0 + len(col_words & q_words)
        else:
            score = 0.5
        scored.append((ci, score))

    scored.sort(key=lambda x: -x[1])
    n_keep = min(len(header), max(min_keep, max_cols))
    keep_indices = sorted([ci for ci, _ in scored[:n_keep]])
    new_header = [header[ci] for ci in keep_indices]
    new_rows = [[row[ci] if ci < len(row) else "" for ci in keep_indices] for row in rows]
    return new_header, new_rows


@register("annotate_types")
def annotate_types(
    header: List[str],
    rows: List[List[str]],
    text: str,
    **kwargs,
) -> Tuple[List[str], List[List[str]]]:
    """Add type annotations to column headers: (int), (float), (date).

    Inspects column values to detect type. Rows unchanged.
    """
    if not header or not rows:
        return header, rows

    annotated = []
    for ci, col_name in enumerate(header):
        values = [row[ci] for row in rows if ci < len(row) and row[ci].strip()]
        col_type = _detect_column_type(values)
        if col_type and col_type != "str":
            annotated.append(f"{col_name} ({col_type})")
        else:
            annotated.append(col_name)
    return annotated, rows


@register("prepend_stats")
def prepend_stats(
    header: List[str],
    rows: List[List[str]],
    text: str,
    **kwargs,
) -> Tuple[List[str], List[List[str]]]:
    """Compute column statistics. Returns unchanged (header, rows).

    The stats string is stored in kwargs["_stats_out"] for the caller
    to prepend to the formatted table. This keeps the transform signature
    clean while allowing the stats to be injected into the prompt.

    Note: The actual prepending happens in task.build_record() since
    it needs the formatted table string. This transform just computes.
    """
    # Stats computation delegated to existing function
    from prompt_profiler.tasks.wtq.table_transforms import compute_column_stats
    stats_str = compute_column_stats(header, rows)
    if stats_str:
        kwargs["_stats_out"] = stats_str
    return header, rows


@register("filter_rows")
def filter_rows(
    header: List[str],
    rows: List[List[str]],
    text: str,
    *,
    max_rows: int = 100,
    strategy: str = "relevance",
) -> Tuple[List[str], List[List[str]]]:
    """Filter rows to reduce table size.

    Calcite analog: filter pushdown — remove irrelevant rows before
    the model sees them. The model gets a smaller, more focused table.

    Strategies:
        relevance: Score rows by keyword overlap with text, keep top-K.
        head:      Keep first max_rows (preserves natural ordering).
        sample:    Uniform random sample (deterministic via hash).
    """
    if not rows or len(rows) <= max_rows:
        return header, rows

    if strategy == "head":
        return header, rows[:max_rows]

    if strategy == "sample":
        # Deterministic sample: hash row content
        scored = []
        for i, row in enumerate(rows):
            h = hash(tuple(row)) % 10000
            scored.append((h, i))
        scored.sort()
        keep = sorted([i for _, i in scored[:max_rows]])
        return header, [rows[i] for i in keep]

    # strategy == "relevance" (default)
    q_lower = text.lower()
    q_words = set(re.findall(r'\w+', q_lower))

    scored = []
    for i, row in enumerate(rows):
        row_text = " ".join(str(cell).lower() for cell in row)
        row_words = set(re.findall(r'\w+', row_text))

        # Exact value match (cell value appears in question)
        exact_matches = sum(
            1 for cell in row
            if len(str(cell).strip()) > 2 and str(cell).strip().lower() in q_lower
        )
        # Word overlap
        word_overlap = len(q_words & row_words)

        score = exact_matches * 3 + word_overlap
        scored.append((score, i))

    # Keep top-K by relevance, preserving original order
    scored.sort(key=lambda x: -x[0])
    keep = sorted([i for _, i in scored[:max_rows]])
    return header, [rows[i] for i in keep]


@register("sort_rows")
def sort_rows(
    header: List[str],
    rows: List[List[str]],
    text: str,
    *,
    by_relevance: bool = True,
) -> Tuple[List[str], List[List[str]]]:
    """Sort rows so most relevant appear first.

    Useful when the model has limited attention — relevant rows
    at the top are less likely to be missed.
    """
    if not rows or not by_relevance:
        return header, rows

    q_lower = text.lower()
    q_words = set(re.findall(r'\w+', q_lower))

    scored = []
    for i, row in enumerate(rows):
        row_text = " ".join(str(cell).lower() for cell in row)
        row_words = set(re.findall(r'\w+', row_text))
        exact = sum(1 for c in row if len(str(c).strip()) > 2 and str(c).strip().lower() in q_lower)
        overlap = len(q_words & row_words)
        scored.append((exact * 3 + overlap, i))

    scored.sort(key=lambda x: -x[0])
    sorted_indices = [i for _, i in scored]
    return header, [rows[i] for i in sorted_indices]


# ── helpers ───────────────────────────────────────────────────────────

def _detect_column_type(values: List[str], sample_size: int = 20) -> str:
    """Detect column type from sampled values."""
    if not values:
        return "str"
    sample = values[:sample_size]
    n = len(sample)

    # Integer
    n_int = sum(1 for v in sample if v.strip().replace(",", "").replace("$", "").replace("%", "").lstrip("-").isdigit())
    if n_int > n * 0.7:
        return "int"

    # Float
    n_float = 0
    for v in sample:
        cleaned = v.strip().replace(",", "").replace("$", "").replace("%", "")
        try:
            float(cleaned)
            n_float += 1
        except ValueError:
            pass
    if n_float > n * 0.7:
        return "float"

    # Date
    date_pats = [r'\d{4}[-/]\d{1,2}[-/]\d{1,2}', r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
                 r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)']
    n_date = sum(1 for v in sample if any(re.search(p, v, re.IGNORECASE) for p in date_pats))
    if n_date > n * 0.5:
        return "date"

    return "str"


# ══════════════════════════════════════════════════════════════════════
# Record transforms — operate on the full build_record() dict
# ══════════════════════════════════════════════════════════════════════


@register_record("focus_schema")
def focus_schema(record: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Filter schema DDL to only tables referenced in wrong_sql + FK neighbors.

    Calcite analog: filter pushdown — reduce noise in the input so the
    model sees only the relevant subset of the schema.
    """
    schema_text = record.get("schema", "")
    wrong_sql = record.get("wrong_sql", "")
    if not schema_text or not wrong_sql:
        return record

    referenced_tables = _extract_tables_from_sql(wrong_sql)
    if not referenced_tables:
        return record

    table_blocks = _parse_ddl_blocks(schema_text)
    if not table_blocks:
        return record

    fk_neighbors = _find_fk_neighbors(schema_text, referenced_tables)
    keep_tables = referenced_tables | fk_neighbors

    focused_blocks = []
    for table_name, block in table_blocks:
        if table_name.lower() in {t.lower() for t in keep_tables}:
            focused_blocks.append(block)

    if focused_blocks:
        record["schema"] = "\n".join(focused_blocks)

    return record


@register_record("localize_error")
def localize_error(record: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Parse error_message + schema into a structured hint.

    Calcite analog: predicate simplification — turn a vague error into
    a precise, actionable pointer.
    """
    error_msg = record.get("error_message", "")
    schema_text = record.get("schema", "")
    if not error_msg:
        return record

    hint = error_msg

    # column_not_found → suggest similar columns
    m = re.match(r"no such column:?\s*(.+)", error_msg, re.IGNORECASE)
    if m:
        missing_col = m.group(1).strip()
        table_qual = None
        col_name = missing_col
        if "." in missing_col:
            table_qual, col_name = missing_col.rsplit(".", 1)
            table_qual = table_qual.strip()
            col_name = col_name.strip()

        # Resolve alias (T1 → actual table name) from wrong SQL
        real_table = table_qual
        if table_qual:
            wrong_sql = record.get("wrong_sql", "")
            alias_map = _resolve_aliases(wrong_sql)
            real_table = alias_map.get(table_qual.lower(), table_qual)

        available = _find_available_columns(schema_text, real_table)
        if available:
            similar = _find_similar(col_name, available, top_k=3)
            hint = f"Column '{missing_col}' not found."
            if similar:
                hint += f" Similar columns: {', '.join(similar)}."
            if table_qual:
                hint += f" Available in {table_qual}: {', '.join(available[:8])}."
        record["error_message"] = hint
        return record

    # ambiguous_column → list which tables have it
    m = re.match(r"ambiguous column name:?\s*(.+)", error_msg, re.IGNORECASE)
    if m:
        col_name = m.group(1).strip()
        tables_with_col = _find_tables_with_column(schema_text, col_name)
        if tables_with_col:
            hint = f"Column '{col_name}' exists in multiple tables: {', '.join(tables_with_col)}. Qualify with table name or alias."
        record["error_message"] = hint
        return record

    # table_not_found → suggest similar tables
    m = re.match(r"no such table:?\s*(.+)", error_msg, re.IGNORECASE)
    if m:
        missing_table = m.group(1).strip()
        all_tables = _extract_table_names_from_ddl(schema_text)
        if all_tables:
            similar = _find_similar(missing_table, all_tables, top_k=3)
            hint = f"Table '{missing_table}' not found."
            if similar:
                hint += f" Similar tables: {', '.join(similar)}."
        record["error_message"] = hint
        return record

    return record


# ── Record transform helpers ─────────────────────────────────────────


def _resolve_aliases(sql: str) -> Dict[str, str]:
    """Extract alias → table name mapping from SQL.

    Handles: FROM table AS alias, JOIN table alias, FROM table T1
    Returns {alias_lower: table_name}.
    """
    alias_map: Dict[str, str] = {}
    try:
        import sqlglot
        tree = sqlglot.parse_one(sql, dialect="sqlite")
        for table in tree.find_all(sqlglot.exp.Table):
            name = table.name
            alias = table.alias
            if name and alias:
                alias_map[alias.lower()] = name
    except Exception:
        # Regex fallback: FROM/JOIN table (AS)? alias
        for m in re.finditer(
            r'(?:FROM|JOIN)\s+(\w+)\s+(?:AS\s+)?(\w+)',
            sql, re.IGNORECASE,
        ):
            table, alias = m.group(1), m.group(2)
            if alias.upper() not in ('ON', 'WHERE', 'SET', 'INNER', 'LEFT', 'RIGHT', 'OUTER', 'CROSS', 'JOIN', 'AS', 'AND', 'OR'):
                alias_map[alias.lower()] = table
    return alias_map


def _extract_tables_from_sql(sql: str) -> set:
    """Extract table names from SQL using sqlglot, regex fallback."""
    try:
        import sqlglot
        tree = sqlglot.parse_one(sql, dialect="sqlite")
        return {t.name for t in tree.find_all(sqlglot.exp.Table) if t.name}
    except Exception:
        tables = set()
        for m in re.finditer(r'\bFROM\s+(\w+)|\bJOIN\s+(\w+)', sql, re.IGNORECASE):
            tables.add(m.group(1) or m.group(2))
        return tables


def _parse_ddl_blocks(ddl: str) -> List[tuple]:
    """Parse DDL text into (table_name, block_text) pairs."""
    blocks = []

    # CREATE TABLE pattern
    for m in re.finditer(
        r'(CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`"\[]?(\w+)[`"\]]?\s*\([^;]*?\))',
        ddl, re.IGNORECASE | re.DOTALL,
    ):
        blocks.append((m.group(2), m.group(1).strip()))
    if blocks:
        return blocks

    # "# Table: X" block format
    current_name = None
    current_lines: List[str] = []
    for line in ddl.split("\n"):
        m = re.match(r'^#\s*Table:\s*(\w+)|^\[?\s*Table\s*\]?\s*:?\s*(\w+)', line, re.IGNORECASE)
        if m:
            if current_name and current_lines:
                blocks.append((current_name, "\n".join(current_lines)))
            current_name = m.group(1) or m.group(2)
            current_lines = [line]
        elif current_name:
            current_lines.append(line)
    if current_name and current_lines:
        blocks.append((current_name, "\n".join(current_lines)))

    return blocks


def _find_fk_neighbors(ddl: str, tables: set) -> set:
    """Find tables connected via FK to any table in the set (one hop)."""
    neighbors = set()
    lower_tables = {t.lower() for t in tables}

    for m in re.finditer(r'REFERENCES\s+[`"\[]?(\w+)[`"\]]?', ddl, re.IGNORECASE):
        ref_table = m.group(1)
        pos = m.start()
        create_m = None
        for cm in re.finditer(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`"\[]?(\w+)', ddl[:pos], re.IGNORECASE):
            create_m = cm
        if create_m:
            source_table = create_m.group(1)
            if source_table.lower() in lower_tables:
                neighbors.add(ref_table)
            elif ref_table.lower() in lower_tables:
                neighbors.add(source_table)

    return neighbors


def _find_available_columns(schema_text: str, table_name: str | None) -> List[str]:
    """Find column names from schema DDL, optionally filtered by table.

    Handles both formats:
        CREATE TABLE X (col_name TYPE, ...)      — standard DDL
        # Table: X\\n[(col_name:TYPE, ...)]       — prompt_profiler format
    """
    columns: List[str] = []

    # Column patterns for both formats
    _COL_PATTERNS = [
        r'[`"\[]?(\w+)[`"\]]?\s+(?:TEXT|INTEGER|REAL|NUMERIC|BLOB|VARCHAR|INT|FLOAT|DATE|BOOL)',  # DDL: col TYPE
        r'\((\w+):(?:TEXT|INTEGER|REAL|NUMERIC|BLOB|VARCHAR|INT|FLOAT|DATE|BOOL)',  # prompt format: (col:TYPE
    ]

    def _extract_cols(body: str) -> List[str]:
        cols = []
        for pat in _COL_PATTERNS:
            for m in re.finditer(pat, body, re.IGNORECASE):
                if m.group(1) not in cols:
                    cols.append(m.group(1))
        return cols

    if table_name:
        # Try CREATE TABLE format
        pattern = re.compile(
            r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`"\[]?' + re.escape(table_name) +
            r'[`"\]]?\s*\(([^;]*?)\)',
            re.IGNORECASE | re.DOTALL,
        )
        m = pattern.search(schema_text)
        if m:
            columns = _extract_cols(m.group(1))

        # Try "# Table: X" block format
        if not columns:
            block_pat = re.compile(
                r'#\s*Table:\s*' + re.escape(table_name) + r'\s*\n(.*?)(?=\n#\s*Table:|\Z)',
                re.IGNORECASE | re.DOTALL,
            )
            m = block_pat.search(schema_text)
            if m:
                columns = _extract_cols(m.group(1))

    if not columns:
        # Fallback: all columns from all tables
        columns = _extract_cols(schema_text)

    return columns


def _find_tables_with_column(schema_text: str, col_name: str) -> List[str]:
    """Find all tables that contain a given column name.

    Handles both CREATE TABLE and # Table: block formats.
    """
    tables = []
    col_pat = re.compile(r'\b' + re.escape(col_name) + r'\b', re.IGNORECASE)

    # CREATE TABLE format
    for m in re.finditer(
        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`"\[]?(\w+)[`"\]]?\s*\(([^;]*?)\)',
        schema_text, re.IGNORECASE | re.DOTALL,
    ):
        if col_pat.search(m.group(2)):
            tables.append(m.group(1))

    if tables:
        return tables

    # "# Table: X" block format
    current_name = None
    current_body: List[str] = []
    for line in schema_text.split("\n"):
        tm = re.match(r'^#\s*Table:\s*(\w+)', line, re.IGNORECASE)
        if tm:
            if current_name and current_body:
                body = "\n".join(current_body)
                if col_pat.search(body):
                    tables.append(current_name)
            current_name = tm.group(1)
            current_body = []
        elif current_name:
            current_body.append(line)
    if current_name and current_body:
        body = "\n".join(current_body)
        if col_pat.search(body):
            tables.append(current_name)

    return tables


def _extract_table_names_from_ddl(ddl: str) -> List[str]:
    """Extract all table names from DDL."""
    return [m.group(1) for m in re.finditer(
        r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`"\[]?(\w+)',
        ddl, re.IGNORECASE,
    )]


def _find_similar(target: str, candidates: List[str], top_k: int = 3) -> List[str]:
    """Find most similar strings by trigram Jaccard."""
    target_lower = target.lower()
    t_tri = {target_lower[i:i+3] for i in range(max(1, len(target_lower) - 2))}

    scored = []
    for c in candidates:
        c_lower = c.lower()
        if c_lower == target_lower:
            continue
        c_tri = {c_lower[i:i+3] for i in range(max(1, len(c_lower) - 2))}
        if not t_tri or not c_tri:
            continue
        jacc = len(t_tri & c_tri) / len(t_tri | c_tri)
        scored.append((jacc, c))

    scored.sort(key=lambda x: -x[0])
    return [c for s, c in scored[:top_k] if s > 0.1]
