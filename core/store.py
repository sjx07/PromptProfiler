"""Unified CubeStore — function-based persistence layer.

Prompt is a state. Functions (func table) operate on it.
Config = a set of func_ids. No hardcoded experimental variables.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class OnConflict(Enum):
    """Policy for handling INSERT conflicts."""
    ERROR = "error"       # raise if exists (safe default)
    REPLACE = "replace"   # overwrite silently
    SKIP = "skip"         # ignore, keep existing
    WARN = "warn"         # log warning + skip


_SQL_CLAUSE = {
    OnConflict.ERROR: "",             # plain INSERT — will raise on conflict
    OnConflict.REPLACE: "OR REPLACE",
    OnConflict.SKIP: "OR IGNORE",
    OnConflict.WARN: "OR IGNORE",     # skip in SQL, warn in Python
}

from core.schema import (
    ADDITIVE_COLUMNS,
    INDEXES,
    META_TABLE,
    MIGRATIONS,
    SCHEMA_VERSION,
    TABLES,
    VIEWS,
)

logger = logging.getLogger(__name__)


class CubeStore:

    def __init__(self, db_path: str | Path, *, read_only: bool = False) -> None:
        self._db_path = str(db_path)
        self._read_only = read_only
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._ensure_schema()

    # ── connection management ──────────────────────────────────────────

    @contextmanager
    def _cursor(self):
        with self._lock:
            conn = self._get_conn()
            cur = conn.cursor()
            try:
                yield cur
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, timeout=30, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    # Additive upgrades that can be applied silently on open — no data
    # transformation, just new tables + backfills. Ranges inclusive.
    _AUTO_UPGRADE_FROM = (7, 8)

    def _check_schema_version(self, conn: sqlite3.Connection) -> None:
        """Raise RuntimeError if cube is at an older schema version and not read_only.

        Exception: additive upgrades listed in ``_AUTO_UPGRADE_FROM`` are
        allowed to proceed without raising — ``_ensure_schema`` will
        create any missing tables/indexes and run ``MIGRATIONS`` to
        backfill from existing data.
        """
        row = conn.execute(
            "SELECT value FROM _cube_meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            return  # fresh database — no version yet
        stored = int(row[0])
        if stored >= SCHEMA_VERSION:
            return
        if self._read_only:
            return
        if stored in self._AUTO_UPGRADE_FROM:
            logger.info(
                "cube schema_version=%s → %s: auto-applying additive upgrade",
                stored, SCHEMA_VERSION,
            )
            return
        raise RuntimeError(
            f"cube schema_version={stored} is older than required {SCHEMA_VERSION}. "
            f"Open with read_only=True for migration scripting, or create a fresh cube. "
            f"See docs/refactor_phase1.md."
        )

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        conn.execute(META_TABLE)
        # Check version before applying any new DDL
        self._check_schema_version(conn)
        for ddl in TABLES:
            conn.execute(ddl)
        self._ensure_additive_columns(conn)
        for view in VIEWS:
            conn.execute(view)
        for idx in INDEXES:
            conn.execute(idx)
        # Idempotent migrations — backfill new tables from pre-existing
        # data (e.g. v7 cubes get config_feature populated from
        # config.meta.feature_ids on first open).
        if not self._read_only:
            for migration in MIGRATIONS:
                try:
                    conn.execute(migration)
                except sqlite3.Error as e:
                    logger.warning("migration step failed (non-fatal): %s", e)
        conn.execute(
            "INSERT OR REPLACE INTO _cube_meta (key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        conn.commit()
        logger.info("CubeStore ready: %s", self._db_path)

    def _ensure_additive_columns(self, conn: sqlite3.Connection) -> None:
        """Add columns needed by additive schema upgrades.

        SQLite's CREATE TABLE IF NOT EXISTS does not add columns to existing
        tables, and SQLite has no ADD COLUMN IF NOT EXISTS syntax. Keep this
        small and explicit so old cubes can be opened without destructive
        migrations.
        """
        for table, columns in ADDITIVE_COLUMNS.items():
            existing = {
                str(row["name"])
                for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for name, ddl in columns:
                if name not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── funcs ──────────────────────────────────────────────────────────

    def upsert_funcs(
        self,
        funcs: List[Dict[str, Any]],
        on_conflict: OnConflict = OnConflict.ERROR,
    ) -> int:
        clause = _SQL_CLAUSE[on_conflict]
        if on_conflict == OnConflict.WARN:
            existing = {r["func_id"] for r in self.list_funcs()}
            incoming = {f["func_id"] for f in funcs}
            conflicts = existing & incoming
            if conflicts:
                logger.warning("func conflicts (skipping): %s", conflicts)

        with self._cursor() as cur:
            cur.executemany(
                f"""INSERT {clause} INTO func
                   (func_id, func_type, params, meta)
                   VALUES (?, ?, ?, ?)""",
                [
                    (
                        f["func_id"],
                        f["func_type"],
                        json.dumps(f.get("params", {})),
                        json.dumps(f.get("meta", {})),
                    )
                    for f in funcs
                ],
            )
            return cur.rowcount

    def get_func(self, func_id: str) -> Optional[Dict[str, Any]]:
        row = self._get_conn().execute(
            "SELECT * FROM func WHERE func_id = ?", (func_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_funcs(self, func_type: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        if func_type:
            rows = conn.execute(
                "SELECT * FROM func WHERE func_type = ?", (func_type,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM func").fetchall()
        return [dict(r) for r in rows]

    # ── queries ────────────────────────────────────────────────────────

    def upsert_queries(
        self,
        queries: List[Dict[str, Any]],
        on_conflict: OnConflict = OnConflict.ERROR,
    ) -> int:
        clause = _SQL_CLAUSE[on_conflict]
        if on_conflict == OnConflict.WARN:
            existing = {r[0] for r in self._get_conn().execute(
                "SELECT query_id FROM query"
            ).fetchall()}
            incoming = {q["query_id"] for q in queries}
            conflicts = existing & incoming
            if conflicts:
                logger.warning("query conflicts (skipping): %d rows", len(conflicts))

        with self._cursor() as cur:
            cur.executemany(
                f"""INSERT {clause} INTO query
                   (query_id, dataset, content, meta)
                   VALUES (?, ?, ?, ?)""",
                [
                    (
                        q["query_id"],
                        q["dataset"],
                        q["content"],
                        json.dumps(q.get("meta", {})),
                    )
                    for q in queries
                ],
            )
            return cur.rowcount

    # ── config management ──────────────────────────────────────────────

    def get_or_create_config(
        self,
        func_ids: List[str],
        *,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        canonical = json.dumps(sorted(func_ids))
        conn = self._get_conn()
        row = conn.execute(
            "SELECT config_id FROM config WHERE func_ids = ?", (canonical,)
        ).fetchone()
        if row:
            config_id = row[0]
        else:
            with self._cursor() as cur:
                cur.execute(
                    "INSERT INTO config (func_ids, meta) VALUES (?, ?)",
                    (canonical, json.dumps(meta or {})),
                )
                config_id = cur.lastrowid

        # v8: mirror meta.feature_ids into the config_feature join
        # table. Idempotent via INSERT OR IGNORE (PK covers (config_id,
        # feature_id)). Rows with a feature_id not in the feature table
        # are silently skipped by the FK; we filter them here too so
        # mis-named cubes don't hard-error.
        if meta:
            feat_ids = meta.get("feature_ids") or []
            if feat_ids:
                with self._cursor() as cur:
                    # Keep only feature_ids that exist in the feature table.
                    known = {
                        r[0] for r in cur.execute(
                            f"SELECT feature_id FROM feature WHERE feature_id IN ({','.join('?' * len(feat_ids))})",
                            tuple(feat_ids),
                        ).fetchall()
                    }
                    rows = [(config_id, fid, "feature")
                            for fid in feat_ids if fid in known]
                    if rows:
                        cur.executemany(
                            "INSERT OR IGNORE INTO config_feature "
                            "(config_id, feature_id, role) VALUES (?, ?, ?)",
                            rows,
                        )
        return config_id

    def get_config_func_ids(self, config_id: int) -> List[str]:
        row = self._get_conn().execute(
            "SELECT func_ids FROM config WHERE config_id = ?",
            (config_id,),
        ).fetchone()
        return json.loads(row[0]) if row else []

    def list_configs(self) -> List[Dict[str, Any]]:
        rows = self._get_conn().execute(
            "SELECT * FROM config ORDER BY config_id"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── execution (LLM call cache) ─────────────────────────────────────

    def get_cached_execution(
        self, config_id: int, query_id: str, model: str
    ) -> Optional[Dict[str, Any]]:
        row = self._get_conn().execute(
            """SELECT * FROM execution
               WHERE config_id = ? AND query_id = ? AND model = ?""",
            (config_id, query_id, model),
        ).fetchone()
        return dict(row) if row else None

    def get_cached_query_ids(self, config_id: int, model: str) -> Set[str]:
        rows = self._get_conn().execute(
            "SELECT query_id FROM execution WHERE config_id = ? AND model = ?",
            (config_id, model),
        ).fetchall()
        return {r[0] for r in rows}

    def insert_execution(
        self,
        config_id: int,
        query_id: str,
        model: str,
        *,
        system_prompt: str = "",
        user_content: str = "",
        raw_response: str = "",
        prediction: str = "",
        latency_ms: Optional[float] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        error: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        phase: Optional[str] = None,
        on_conflict: OnConflict = OnConflict.ERROR,
    ) -> int:
        clause = _SQL_CLAUSE[on_conflict]
        phase_ids = json.dumps([phase]) if phase else "[]"
        if on_conflict == OnConflict.WARN:
            existing = self.get_cached_execution(config_id, query_id, model)
            if existing:
                logger.warning(
                    "execution exists (skipping): config=%d query=%s model=%s",
                    config_id, query_id, model,
                )

        with self._cursor() as cur:
            cur.execute(
                f"""INSERT {clause} INTO execution
                   (config_id, query_id, model,
                    system_prompt, user_content, raw_response, prediction,
                    latency_ms, prompt_tokens, completion_tokens,
                    error, phase_ids, meta)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    config_id, query_id, model,
                    system_prompt, user_content, raw_response, prediction,
                    latency_ms, prompt_tokens, completion_tokens,
                    error, phase_ids, json.dumps(meta or {}),
                ),
            )
            return cur.lastrowid

    def tag_phase(self, execution_id: int, phase: str) -> None:
        """Append a phase ID to an execution's phase_ids list (idempotent)."""
        row = self._get_conn().execute(
            "SELECT phase_ids FROM execution WHERE execution_id = ?",
            (execution_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"execution_id {execution_id} not found")
        current = json.loads(row[0] or "[]")
        if phase not in current:
            current.append(phase)
            with self._cursor() as cur:
                cur.execute(
                    "UPDATE execution SET phase_ids = ? WHERE execution_id = ?",
                    (json.dumps(current), execution_id),
                )

    def get_executions_by_phase(
        self, phase: str, model: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return all executions tagged with a given phase."""
        conn = self._get_conn()
        if model:
            rows = conn.execute(
                """SELECT * FROM execution
                   WHERE phase_ids LIKE ? AND model = ?""",
                (f'%"{phase}"%', model),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM execution WHERE phase_ids LIKE ?",
                (f'%"{phase}"%',),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── evaluation (lazy scoring) ──────────────────────────────────────

    def get_evaluation(
        self, execution_id: int, scorer: str
    ) -> Optional[Dict[str, Any]]:
        row = self._get_conn().execute(
            "SELECT * FROM evaluation WHERE execution_id = ? AND scorer = ?",
            (execution_id, scorer),
        ).fetchone()
        return dict(row) if row else None

    def upsert_evaluation(
        self,
        execution_id: int,
        scorer: str,
        score: float,
        metrics: Optional[Dict[str, Any]] = None,
        on_conflict: OnConflict = OnConflict.ERROR,
    ) -> int:
        clause = _SQL_CLAUSE[on_conflict]
        with self._cursor() as cur:
            cur.execute(
                f"""INSERT {clause} INTO evaluation
                   (execution_id, scorer, score, metrics)
                   VALUES (?, ?, ?, ?)""",
                (execution_id, scorer, score, json.dumps(metrics or {})),
            )
            return cur.lastrowid

    def evaluate_batch(
        self,
        rows: List[Dict[str, Any]],
        on_conflict: OnConflict = OnConflict.ERROR,
    ) -> int:
        clause = _SQL_CLAUSE[on_conflict]
        with self._cursor() as cur:
            cur.executemany(
                f"""INSERT {clause} INTO evaluation
                   (execution_id, scorer, score, metrics)
                   VALUES (?, ?, ?, ?)""",
                [
                    (
                        r["execution_id"],
                        r["scorer"],
                        r["score"],
                        json.dumps(r.get("metrics", {})),
                    )
                    for r in rows
                ],
            )
            return cur.rowcount

    # ── query helpers ──────────────────────────────────────────────────

    def config_progress(
        self, config_id: int, model: str, total_queries: int
    ) -> Dict[str, Any]:
        done = len(self.get_cached_query_ids(config_id, model))
        return {
            "config_id": config_id,
            "done": done,
            "total": total_queries,
            "remaining": total_queries - done,
        }

    def scores_by_config(
        self,
        model: str,
        scorer: str,
        *,
        dataset: str = "",
        query_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: list[Any] = [model, scorer]
        if dataset:
            clauses.append("q.dataset = ?")
            params.append(dataset)
        if query_ids is not None:
            if not query_ids:
                return []
            placeholders = ",".join("?" for _ in query_ids)
            clauses.append(f"e.query_id IN ({placeholders})")
            params.extend(query_ids)
        extra_clause = ""
        if clauses:
            extra_clause = " AND " + " AND ".join(clauses)
        rows = self._get_conn().execute(
            """SELECT c.config_id,
                      COUNT(ev.score) as n,
                      AVG(ev.score) as avg_score,
                      MIN(ev.score) as min_score,
                      MAX(ev.score) as max_score
               FROM config c
               JOIN execution e ON c.config_id = e.config_id
               JOIN query q ON q.query_id = e.query_id
               JOIN evaluation ev ON e.execution_id = ev.execution_id
               WHERE e.model = ? AND ev.scorer = ?"""
            + extra_clause
            + """
               GROUP BY c.config_id
               ORDER BY avg_score DESC""",
            tuple(params),
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> Dict[str, int]:
        conn = self._get_conn()
        tables = ["func", "query", "config", "execution", "evaluation", "feature"]
        return {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in tables
        }

    # ── feature registry sync ──────────────────────────────────────────

    def sync_features(self, feature_rows: List[Dict[str, Any]]) -> Dict[str, int]:
        """Upsert feature rows into the feature table.

        Uses INSERT OR REPLACE so re-syncing identical content is a no-op at
        the DB level (same primary key, same columns).

        Args:
            feature_rows: List of dicts with keys:
                feature_id, canonical_id, task, requires_json, conflicts_json,
                primitive_spec, semantic_labels_json (optional), scope_json
                (optional), rationale (optional), source_path (optional).
                Rows may also include label_rows and label_memberships to sync
                feature_label / feature_label_membership exactly for that feature_id.

        Returns:
            {"synced": <count of rows processed>}
        """
        with self._cursor() as cur:
            cur.executemany(
                """INSERT INTO feature
                   (feature_id, canonical_id, task,
                    requires_json, conflicts_json, primitive_spec,
                    semantic_labels_json, scope_json,
                    rationale, source_path, synced_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                   ON CONFLICT(feature_id) DO UPDATE SET
                       canonical_id = excluded.canonical_id,
                       task = excluded.task,
                       requires_json = excluded.requires_json,
                       conflicts_json = excluded.conflicts_json,
                       primitive_spec = excluded.primitive_spec,
                       semantic_labels_json = excluded.semantic_labels_json,
                       scope_json = excluded.scope_json,
                       rationale = excluded.rationale,
                       source_path = excluded.source_path,
                       synced_at = datetime('now')""",
                [
                    (
                        r["feature_id"],
                        r["canonical_id"],
                        r["task"],
                        r.get("requires_json", "[]"),
                        r.get("conflicts_json", "[]"),
                        r["primitive_spec"],
                        r.get("semantic_labels_json", "[]"),
                        r.get("scope_json", "{}"),
                        r.get("rationale"),
                        r.get("source_path"),
                    )
                    for r in feature_rows
                ],
            )

            labeled_rows = [r for r in feature_rows if "label_memberships" in r]
            if labeled_rows:
                label_rows: Dict[str, Dict[str, Any]] = {}
                membership_rows: List[Dict[str, Any]] = []
                for row in labeled_rows:
                    cur.execute(
                        "DELETE FROM feature_label_membership WHERE feature_id = ?",
                        (row["feature_id"],),
                    )
                    for label in row.get("label_rows", []):
                        label_rows[label["label_id"]] = label
                    membership_rows.extend(row.get("label_memberships", []))

                if label_rows:
                    cur.executemany(
                        """INSERT INTO feature_label
                           (label_id, description)
                           VALUES (?, ?)
                           ON CONFLICT(label_id) DO UPDATE SET
                               description = excluded.description""",
                        [
                            (
                                r["label_id"],
                                r.get("description"),
                            )
                            for r in label_rows.values()
                        ],
                    )

                if membership_rows:
                    cur.executemany(
                        """INSERT OR REPLACE INTO feature_label_membership
                           (feature_id, label_id, role)
                           VALUES (?, ?, ?)""",
                        [
                            (
                                r["feature_id"],
                                r["label_id"],
                                r.get("role", "implements"),
                            )
                            for r in membership_rows
                        ],
                    )
        return {"synced": len(feature_rows)}

    def feature_effect_df(self):
        """Query the feature_effect view and return a pandas DataFrame.

        NOTE: Only configs created after schema v7 (with content-hash feature_ids
        stored in config.meta.feature_ids) are joinable via this view.
        Pre-v7 configs stored canonical_id strings and will not match.

        Returns:
            pd.DataFrame with columns: feature_id, canonical_id, task, model,
            query_id, score, scorer.  Empty DataFrame if no joined rows exist.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for feature_effect_df(). "
                "Install it with: pip install pandas"
            ) from exc

        rows = self._get_conn().execute(
            "SELECT * FROM feature_effect"
        ).fetchall()
        if not rows:
            return pd.DataFrame(columns=[
                "feature_id", "canonical_id", "task", "model",
                "query_id", "score", "scorer",
            ])
        return pd.DataFrame([dict(r) for r in rows])
