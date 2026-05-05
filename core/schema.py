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

SCHEMA_VERSION = 9   # v9: + semantic feature-label metadata and joins

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
    # semantic_labels_json = raw optional semantic_labels metadata from the
    #                        feature JSON. Used for auditability; normalized
    #                        joins live in feature_label_membership.
    # scope_json = optional component-level applicability metadata. It belongs
    #              to the concrete materialized component, not to semantic
    #              label descriptions.
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
        semantic_labels_json TEXT NOT NULL DEFAULT '[]',
        scope_json       TEXT NOT NULL DEFAULT '{}',
        rationale        TEXT,
        source_path      TEXT,
        synced_at        TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,

    # ── config_feature = many-to-many join (schema v8) ────────────────
    #
    # Promotes the (config × feature) relationship from JSON inside
    # config.meta to a proper indexed join table. Resolves the ambiguity
    # where two configs sharing a canonical_id under different bases
    # look identical at the meta level. Also lets the analysis layer
    # use indexed JOINs instead of json_extract(meta, '$.feature_ids').
    #
    # ``role`` is reserved for future use (e.g. "base" vs "experiment"
    # vs "coalition") — today it's always 'feature' and nobody reads it.
    # The PK (config_id, feature_id) implies a covering index on
    # config_id; we add idx_cf_feature separately for feature→configs
    # direction.
    """
    CREATE TABLE IF NOT EXISTS config_feature (
        config_id   INTEGER NOT NULL REFERENCES config(config_id) ON DELETE CASCADE,
        feature_id  TEXT    NOT NULL REFERENCES feature(feature_id),
        role        TEXT    NOT NULL DEFAULT 'feature',
        PRIMARY KEY (config_id, feature_id)
    )
    """,

    # ── semantic feature labels (schema v9) ──────────────────────────
    #
    # Concrete feature/component rows remain the materialization truth.
    # Labels are an analysis layer: one component can have many labels,
    # and many task-specific components can share one label.
    """
    CREATE TABLE IF NOT EXISTS feature_label (
        label_id      TEXT PRIMARY KEY,
        description   TEXT,
        created_at    TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,

    """
    CREATE TABLE IF NOT EXISTS feature_label_membership (
        feature_id    TEXT NOT NULL REFERENCES feature(feature_id) ON DELETE CASCADE,
        label_id      TEXT NOT NULL REFERENCES feature_label(label_id),
        role          TEXT NOT NULL DEFAULT 'implements',
        created_at    TEXT NOT NULL DEFAULT (datetime('now')),
        PRIMARY KEY (feature_id, label_id, role)
    )
    """,

]

ADDITIVE_COLUMNS: dict[str, list[tuple[str, str]]] = {
    # Existing v8 feature tables need these columns added on open. Fresh v9
    # cubes get them from the CREATE TABLE statement above.
    "feature": [
        ("semantic_labels_json", "TEXT NOT NULL DEFAULT '[]'"),
        ("scope_json", "TEXT NOT NULL DEFAULT '{}'"),
    ],
}

VIEWS: list[str] = [
    # Sections projected from insert_node(section) funcs.
    """
    CREATE VIEW IF NOT EXISTS section_view AS
    SELECT func_id                                               AS section_id,
           json_extract(params, '$.payload.title')               AS title,
           json_extract(params, '$.payload.content')             AS content,
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

    # Semantic label effect — same concrete execution/evaluation rows, but
    # grouped through feature_label_membership instead of only component IDs.
    """
    CREATE VIEW IF NOT EXISTS feature_label_effect AS
    SELECT
        flm.label_id,
        fl.description AS label_description,
        flm.role,
        f.feature_id,
        f.canonical_id,
        f.task AS feature_task,
        f.scope_json AS component_scope_json,
        q.dataset,
        json_extract(q.meta, '$.qtype') AS qtype,
        json_extract(q.meta, '$.qsubtype') AS qsubtype,
        e.config_id,
        e.model,
        q.query_id,
        ev.score,
        ev.scorer
    FROM evaluation ev
    JOIN execution  e   USING (execution_id)
    JOIN query      q   ON q.query_id = e.query_id
    JOIN config_feature cf ON cf.config_id = e.config_id
    JOIN feature    f   ON f.feature_id = cf.feature_id
    JOIN feature_label_membership flm ON flm.feature_id = f.feature_id
    JOIN feature_label fl ON fl.label_id = flm.label_id
    """,

    # Predicate fan-out for slice analysis by any seeded predicate
    # (qtype/qsubtype, answer type, table shape, etc.).
    """
    CREATE VIEW IF NOT EXISTS feature_label_predicate_effect AS
    SELECT
        fle.*,
        p.name AS predicate_name,
        p.value AS predicate_value
    FROM feature_label_effect fle
    JOIN predicate p ON p.query_id = fle.query_id
    """,

    # ── analysis-fork R5 views ───────────────────────────────────────
    #
    # Promotes the canonical 3-way join used throughout `analyze/` into
    # a reusable view. Callers that don't need predicate tags use
    # ``v_query_scores`` (execution × evaluation only, no predicate
    # fan-out). Callers that need per-predicate-value splits use
    # ``v_scored_executions``.
    #
    # Rows exclude errored executions (`error IS NULL OR error = ''`)
    # and NULL scores — matching the filters the Python layer has
    # always applied.
    """
    CREATE VIEW IF NOT EXISTS v_query_scores AS
    SELECT
        e.config_id  AS config_id,
        e.query_id   AS query_id,
        e.model      AS model,
        ev.scorer    AS scorer,
        ev.score     AS score
    FROM execution e
    JOIN evaluation ev ON ev.execution_id = e.execution_id
    WHERE (e.error IS NULL OR e.error = '')
      AND ev.score IS NOT NULL
    """,

    """
    CREATE VIEW IF NOT EXISTS v_scored_executions AS
    SELECT
        e.config_id  AS config_id,
        e.query_id   AS query_id,
        e.model      AS model,
        ev.scorer    AS scorer,
        ev.score     AS score,
        p.name       AS predicate_name,
        p.value      AS predicate_value
    FROM execution e
    JOIN evaluation ev ON ev.execution_id = e.execution_id
    JOIN predicate  p  ON p.query_id = e.query_id
    WHERE (e.error IS NULL OR e.error = '')
      AND ev.score IS NOT NULL
    """,

    # Per-(config × predicate-value) means. Layered over
    # ``v_scored_executions`` so the filter conditions stay in one
    # place. Callers narrow by (model, scorer, predicate_name) via
    # WHERE clauses.
    """
    CREATE VIEW IF NOT EXISTS v_per_config_predicate_means AS
    SELECT
        config_id,
        predicate_name,
        predicate_value,
        model,
        scorer,
        AVG(score)  AS mean_score,
        COUNT(*)    AS n
    FROM v_scored_executions
    GROUP BY config_id, predicate_name, predicate_value, model, scorer
    """,
]

INDEXES: list[str] = [
    # func
    "CREATE        INDEX IF NOT EXISTS idx_func_type     ON func(func_type)",
    # execution
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_exec_cache     ON execution(config_id, query_id, model)",
    "CREATE        INDEX IF NOT EXISTS idx_exec_query    ON execution(query_id)",
    "CREATE        INDEX IF NOT EXISTS idx_exec_model    ON execution(model)",
    "CREATE        INDEX IF NOT EXISTS idx_exec_model_config_query ON execution(model, config_id, query_id)",
    # evaluation
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_eval_exec_scorer ON evaluation(execution_id, scorer)",
    "CREATE        INDEX IF NOT EXISTS idx_eval_exec     ON evaluation(execution_id)",
    "CREATE        INDEX IF NOT EXISTS idx_eval_scorer    ON evaluation(scorer)",
    "CREATE        INDEX IF NOT EXISTS idx_eval_scorer_exec_score ON evaluation(scorer, execution_id, score)",
    # predicate
    "CREATE        INDEX IF NOT EXISTS idx_pred_name     ON predicate(name)",
    # query
    "CREATE        INDEX IF NOT EXISTS idx_query_dataset ON query(dataset)",
    "CREATE        INDEX IF NOT EXISTS idx_query_dataset_split ON query(dataset, json_extract(meta, '$.split'))",
    # feature
    "CREATE        INDEX IF NOT EXISTS idx_feature_canonical ON feature(canonical_id)",
    "CREATE        INDEX IF NOT EXISTS idx_feature_task      ON feature(task)",
    # config_feature (v8) — PK covers config_id lookups; add feature side.
    "CREATE        INDEX IF NOT EXISTS idx_cf_feature        ON config_feature(feature_id)",
    # semantic labels (v9)
    "CREATE        INDEX IF NOT EXISTS idx_flm_label         ON feature_label_membership(label_id)",
    "CREATE        INDEX IF NOT EXISTS idx_flm_feature       ON feature_label_membership(feature_id)",
]


# ── idempotent migrations applied on every _ensure_schema ────────────
# Each entry is a SQL statement that must be safe to re-run (INSERT OR
# IGNORE, etc.). Used to backfill new tables from pre-existing data
# so cubes at an earlier schema version come up to date automatically.

MIGRATIONS: list[str] = [
    # v7 → v8: populate config_feature from the existing
    # config.meta.feature_ids JSON arrays. Idempotent: INSERT OR IGNORE
    # against the (config_id, feature_id) PK.
    """
    INSERT OR IGNORE INTO config_feature (config_id, feature_id, role)
    SELECT c.config_id, fid.value AS feature_id, 'feature' AS role
    FROM config c,
         json_each(json_extract(c.meta, '$.feature_ids')) AS fid
    WHERE json_extract(c.meta, '$.feature_ids') IS NOT NULL
      AND EXISTS (SELECT 1 FROM feature f WHERE f.feature_id = fid.value)
    """,
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
