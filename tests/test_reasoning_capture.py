"""Reasoning capture persistence."""
from __future__ import annotations

import os
import sqlite3
import tempfile

from core.store import CubeStore, OnConflict


def test_execution_store_persists_raw_reasoning_and_finish_reason():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmpf:
        db_path = tmpf.name
    try:
        store = CubeStore(db_path)
        store.upsert_queries(
            [{"query_id": "q1", "dataset": "synthetic", "content": "What?", "meta": {}}],
            on_conflict=OnConflict.ERROR,
        )
        config_id = store.get_or_create_config([])

        store.insert_execution(
            config_id,
            "q1",
            "fake-model",
            raw_response="visible answer",
            raw_reasoning="hidden reasoning",
            finish_reason="stop",
            prediction="visible answer",
        )

        row = store.get_cached_execution(config_id, "q1", "fake-model")
        assert row["raw_response"] == "visible answer"
        assert row["raw_reasoning"] == "hidden reasoning"
        assert row["finish_reason"] == "stop"
        store.close()
    finally:
        os.unlink(db_path)


def test_v9_cube_auto_adds_reasoning_capture_columns():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmpf:
        db_path = tmpf.name
    try:
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE _cube_meta (key TEXT PRIMARY KEY, value TEXT);
            INSERT INTO _cube_meta (key, value) VALUES ('schema_version', '9');
            CREATE TABLE execution (
                execution_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                config_id           INTEGER NOT NULL,
                query_id            TEXT    NOT NULL,
                model               TEXT    NOT NULL,
                system_prompt       TEXT DEFAULT '',
                user_content        TEXT DEFAULT '',
                raw_response        TEXT DEFAULT '',
                prediction          TEXT DEFAULT '',
                latency_ms          REAL,
                prompt_tokens       INTEGER,
                completion_tokens   INTEGER,
                error               TEXT,
                phase_ids           TEXT DEFAULT '[]',
                created_at          TEXT NOT NULL DEFAULT (datetime('now')),
                meta                TEXT DEFAULT '{}'
            );
            """
        )
        conn.commit()
        conn.close()

        store = CubeStore(db_path)
        columns = {
            row["name"]
            for row in store._get_conn().execute("PRAGMA table_info(execution)").fetchall()
        }
        version = store._get_conn().execute(
            "SELECT value FROM _cube_meta WHERE key = 'schema_version'"
        ).fetchone()[0]

        assert "raw_reasoning" in columns
        assert "finish_reason" in columns
        assert version == "10"
        store.close()
    finally:
        os.unlink(db_path)
