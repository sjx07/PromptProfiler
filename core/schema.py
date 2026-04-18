"""Unified cube schema — function-based prompt configuration.

Core abstractions:
    func       = a named operation on prompt state (add_rule, set_format, enable_cot, define_section, ...)
    config     = a set of func_ids applied to build a prompt
    execution  = model × config × query → response
    evaluation = scorer(execution.prediction, query.gold) → score

Design principles:
    - Prompt is a state. Functions operate on state → produce prompt.
    - Rules, format, CoT, sections — everything is a func. No hardcoded config fields.
    - Each function's state lives in its own `params` (isolated store).
    - Sections are define_section funcs — same table, same ID space as rules.
    - Adding new experimental variables = new func_type + @register, no schema change.
"""
from __future__ import annotations

import hashlib

SCHEMA_VERSION = 7

TABLES: list[str] = [
    # ── reference tables ──────────────────────────────────────────────

    # Functions — universal building blocks for prompt construction.
    # Each func has a type (the "function pointer") and params (its state).
    # func_type maps to a Python handler via the func registry.
    #
    # Examples:
    #   func_id="s0",       func_type="define_section", params={"title":"role","ordinal":0,"is_system":true,...}
    #   func_id="s2/r0",    func_type="add_rule",       params={"section_id":"s2","content":"Use ORDER BY..."}
    #   func_id="fmt_json", func_type="set_format",     params={"style":"json"}
    #   func_id="cot_on",   func_type="enable_cot",     params={}
    """
    CREATE TABLE IF NOT EXISTS func (
        func_id     TEXT PRIMARY KEY,
        func_type   TEXT NOT NULL,
        params      TEXT NOT NULL DEFAULT '{}',
        meta        TEXT DEFAULT '{}'
    )
    """,

    # Queries — generic input instances.
    # Task-specific fields (gold_sql, evidence, difficulty, split, db_id, etc.)
    # all live in `meta`.
    """
    CREATE TABLE IF NOT EXISTS query (
        query_id    TEXT PRIMARY KEY,
        dataset     TEXT NOT NULL,
        content     TEXT NOT NULL,
        meta        TEXT DEFAULT '{}'
    )
    """,

    # ── experiment tables ─────────────────────────────────────────────

    # Config = a set of functions applied to build a prompt.
    # func_ids is a sorted JSON array; UNIQUE constraint handles dedup.
    # Query funcs via: SELECT config_id FROM config, json_each(func_ids) WHERE value = ?
    """
    CREATE TABLE IF NOT EXISTS config (
        config_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        func_ids    TEXT UNIQUE NOT NULL DEFAULT '[]',
        meta        TEXT DEFAULT '{}'
    )
    """,

    # ── execution = model × config × query → response ────────────────

    """
    CREATE TABLE IF NOT EXISTS execution (
        execution_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        config_id           INTEGER NOT NULL REFERENCES config(config_id),
        query_id            TEXT    NOT NULL REFERENCES query(query_id),
        model               TEXT    NOT NULL,

        -- the LLM interaction
        system_prompt       TEXT DEFAULT '',
        user_content        TEXT DEFAULT '',
        raw_response        TEXT DEFAULT '',
        prediction          TEXT DEFAULT '',

        -- cost / diagnostics
        latency_ms          REAL,
        prompt_tokens       INTEGER,
        completion_tokens   INTEGER,
        error               TEXT,

        -- experiment phase tracking
        phase_ids           TEXT DEFAULT '[]',

        created_at          TEXT NOT NULL DEFAULT (datetime('now')),
        meta                TEXT DEFAULT '{}'
    )
    """,

    # ── predicate = precomputed C-predicate per query ────────────────
    # Seeded once via seed_predicates(). Cohort builder reads from here.
    """
    CREATE TABLE IF NOT EXISTS predicate (
        query_id    TEXT NOT NULL REFERENCES query(query_id),
        name        TEXT NOT NULL,
        value       TEXT NOT NULL,
        PRIMARY KEY (query_id, name)
    )
    """,

    # ── evaluation = scorer(prediction, gold) → score ─────────────────
    # Lazy-computed. Same execution can be scored by multiple scorers.
    # Re-scoring does not require re-running the LLM.

    """
    CREATE TABLE IF NOT EXISTS evaluation (
        eval_id         INTEGER PRIMARY KEY AUTOINCREMENT,
        execution_id    INTEGER NOT NULL REFERENCES execution(execution_id),
        scorer          TEXT NOT NULL,
        score           REAL,
        metrics         TEXT DEFAULT '{}',
        created_at      TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,

    # ── feature = registry of named primitive-edit bundles ────────────
    # feature_id   = content-addressed hash of primitive_edits (12-char hex)
    # canonical_id = human-facing label (e.g. "enable_cot"); stable across
    #                tasks and versions; cross-task aggregation key.
    # primitive_spec = JSON array of primitive_edit dicts (insert_node, etc.)
    # source_path  = absolute path to the feature JSON file on disk (NULL for
    #                in-memory / programmatically constructed features).
    #
    # NOTE: only configs created after schema v7 store feature_ids as content
    # hashes in config.meta.feature_ids.  Pre-v7 configs stored canonical_id
    # strings and are NOT joinable via the feature_effect view.
    """
    CREATE TABLE IF NOT EXISTS feature (
        feature_id       TEXT PRIMARY KEY,
        canonical_id     TEXT NOT NULL,
        task             TEXT NOT NULL,
        requires_json    TEXT NOT NULL DEFAULT '[]',
        conflicts_json   TEXT NOT NULL DEFAULT '[]',
        primitive_spec   TEXT NOT NULL,
        rationale        TEXT,
        source_path      TEXT,
        synced_at        TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,

]

VIEWS: list[str] = [
    # Sections projected from insert_node(section) funcs.
    """
    CREATE VIEW IF NOT EXISTS section_view AS
    SELECT func_id                                               AS section_id,
           json_extract(params, '$.payload.title')               AS title,
           json_extract(params, '$.payload.ordinal')             AS ordinal,
           json_extract(params, '$.payload.is_system')           AS is_system,
           json_extract(params, '$.payload.min_rules')           AS min_rules,
           json_extract(params, '$.payload.max_rules')           AS max_rules,
           json_extract(params, '$.parent_id')                   AS parent_id,
           meta
    FROM func
    WHERE func_type = 'insert_node'
      AND json_extract(params, '$.node_type') = 'section'
    """,

    # Rules projected from insert_node(rule) funcs.
    """
    CREATE VIEW IF NOT EXISTS rule_view AS
    SELECT func_id                                               AS rule_id,
           json_extract(params, '$.parent_id')                   AS section_id,
           json_extract(params, '$.payload.content')             AS content,
           meta
    FROM func
    WHERE func_type = 'insert_node'
      AND json_extract(params, '$.node_type') = 'rule'
    """,

    # Input fields projected from insert_node(input_field) funcs.
    """
    CREATE VIEW IF NOT EXISTS input_field_view AS
    SELECT func_id                                               AS field_id,
           json_extract(params, '$.payload.name')                AS name,
           json_extract(params, '$.payload.description')         AS description,
           meta
    FROM func
    WHERE func_type = 'insert_node'
      AND json_extract(params, '$.node_type') = 'input_field'
    """,

    # Output fields projected from insert_node(output_field) funcs.
    """
    CREATE VIEW IF NOT EXISTS output_field_view AS
    SELECT func_id                                               AS field_id,
           json_extract(params, '$.payload.name')                AS name,
           json_extract(params, '$.payload.description')         AS description,
           meta
    FROM func
    WHERE func_type = 'insert_node'
      AND json_extract(params, '$.node_type') = 'output_field'
    """,

    # Feature effect — joins evaluation scores back to feature metadata.
    #
    # NOTE: only configs from schema v7+ store feature_ids as content-hash
    # strings in config.meta -> $.feature_ids.  Pre-v7 configs stored
    # canonical_id strings and will NOT join correctly via this view.
    #
    # Usage:
    #   SELECT canonical_id, AVG(score) FROM feature_effect
    #   WHERE task = 'table_qa' GROUP BY canonical_id ORDER BY AVG(score) DESC;
    """
    CREATE VIEW IF NOT EXISTS feature_effect AS
    SELECT
        f.feature_id,
        f.canonical_id,
        f.task,
        e.model,
        q.query_id,
        ev.score,
        ev.scorer
    FROM evaluation ev
    JOIN execution  e   USING (execution_id)
    JOIN config     c   USING (config_id)
    JOIN query      q   ON q.query_id = e.query_id
    JOIN json_each(json_extract(c.meta, '$.feature_ids')) AS fid
    JOIN feature    f   ON f.feature_id = fid.value
    """,
]

INDEXES: list[str] = [
    # func
    "CREATE        INDEX IF NOT EXISTS idx_func_type     ON func(func_type)",
    # execution
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_exec_cache     ON execution(config_id, query_id, model)",
    "CREATE        INDEX IF NOT EXISTS idx_exec_query    ON execution(query_id)",
    "CREATE        INDEX IF NOT EXISTS idx_exec_model    ON execution(model)",
    # evaluation
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_exec_scorer ON evaluation(execution_id, scorer)",
    "CREATE        INDEX IF NOT EXISTS idx_eval_exec     ON evaluation(execution_id)",
    "CREATE        INDEX IF NOT EXISTS idx_eval_scorer    ON evaluation(scorer)",
    # predicate
    "CREATE        INDEX IF NOT EXISTS idx_pred_name     ON predicate(name)",
    # query
    "CREATE        INDEX IF NOT EXISTS idx_query_dataset ON query(dataset)",
    # feature
    "CREATE        INDEX IF NOT EXISTS idx_feature_canonical ON feature(canonical_id)",
    "CREATE        INDEX IF NOT EXISTS idx_feature_task      ON feature(task)",
]

META_TABLE = """
CREATE TABLE IF NOT EXISTS _cube_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
)
"""


# ── hash helpers ──────────────────────────────────────────────────────

def make_query_id(dataset: str, content: str, context: str = "") -> str:
    """Content-addressed query ID.

    `context` is an optional disambiguator (e.g. "dev:mondial_geo")
    for cases where identical questions appear in different contexts.
    """
    payload = f"{dataset}:{context}:{content.strip()}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


