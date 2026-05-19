#!/usr/bin/env python3
"""Build a clean surface-feature cube from an existing mixed experiment cube.

The migration is intentionally config/function based:

* clean config names are re-derived from canonical feature IDs
* config ``func_ids`` are materialized with the current FeatureRegistry
* executions/evaluations are copied only when the source cube has the exact
  same sorted ``func_ids`` array

This keeps the analysis join stable across old and new runs while giving the
new cube a smaller, cleaner feature namespace.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.feature_registry import FeatureRegistry  # noqa: E402
from core.store import CubeStore  # noqa: E402


DEFAULT_SOURCE_DB = Path("/data/users/jsu323/facet/wikitable_transfer_cube.db")
DEFAULT_DEST_DB = Path("/data/users/jsu323/facet/wikitable_clean_surface_v1.db")
DEFAULT_SUMMARY = ROOT / "study_layer" / "artifacts" / "wikitable_clean_surface_v1_summary.json"

DEFAULT_CONFIGS = [
    # Fresh WTQ/SQA family configs. These currently contain reasoning variants
    # too; the cleaner below keeps only format/serialization/context/base.
    "Obsidian/facet_exp/wtq/configs/systematic/qwen2_5_14b/family_coalitions/wikitable_family_v1.full.json",
    "Obsidian/facet_exp/sqa/configs/systematic/qwen2_5_14b/family_coalitions/wikitable_family_v1.full.json",
    # Older WTQ/SQA systematic panels; useful because many executions already
    # live under these exact func_id sets in the mixed cube.
    "Obsidian/facet_exp/wikitable_transfer/configs/systematic/qwen25_14b/feature_coalitions/wtq.format_axes.full.json",
    "Obsidian/facet_exp/wikitable_transfer/configs/systematic/qwen25_14b/feature_coalitions/wtq.context_format_interactions.full.json",
    "Obsidian/facet_exp/wikitable_transfer/configs/systematic/qwen25_14b/feature_coalitions/wtq.explicit_coalitions.full.json",
    "Obsidian/facet_exp/wikitable_transfer/configs/systematic/qwen25_14b/feature_coalitions/wtq.best_stack_interactions.full.json",
    "Obsidian/facet_exp/wikitable_transfer/configs/systematic/qwen25_14b/feature_coalitions/wtq.family_gapfill.n1024.json",
    "Obsidian/facet_exp/wikitable_transfer/configs/systematic/qwen25_14b/feature_coalitions/sqa.format_axes.n1024.json",
    "Obsidian/facet_exp/wikitable_transfer/configs/systematic/qwen25_14b/feature_coalitions/sqa.context_format_interactions.n1024.json",
    "Obsidian/facet_exp/wikitable_transfer/configs/systematic/qwen25_14b/feature_coalitions/sqa.explicit_coalitions.n1024.json",
    "Obsidian/facet_exp/wikitable_transfer/configs/systematic/qwen25_14b/feature_coalitions/sqa.best_stack_interactions.n1024.json",
    "Obsidian/facet_exp/wikitable_transfer/configs/systematic/qwen25_14b/feature_coalitions/sqa.family_gapfill.n1024.json",
    # Other table benchmarks.
    "Obsidian/facet_exp/wikitable_transfer/configs/systematic/qwen25_14b/feature_coalitions/tablebench.aligned_stack.n1024.json",
    "Obsidian/facet_exp/wikitable_transfer/configs/systematic/qwen25_14b/feature_coalitions/tablebench.family_gapfill.n1024.json",
    "Obsidian/facet_exp/wikitable_transfer/configs/systematic/qwen25_14b/feature_coalitions/tabfact.aligned_stack.n1024.json",
    "Obsidian/facet_exp/wikitable_transfer/configs/systematic/qwen25_14b/feature_coalitions/tabfact.family_gapfill.n1024.json",
    "Obsidian/facet_exp/wikitable_transfer/configs/systematic/qwen25_14b/feature_coalitions/hitab.aligned_stack.n1024.json",
    "Obsidian/facet_exp/wikitable_transfer/configs/systematic/qwen25_14b/feature_coalitions/hitab.family_gapfill.n1024.json",
]

DATASET_BY_TASK = {
    "wtq": "wtq",
    "sqa": "sqa",
    "tablebench": "tablebench",
    "tabfact": "tab_fact",
    "hitab": "hitab",
}

CLEAN_ALLOWED_PREFIXES = (
    "_section_",
    "facet_dp_scaffold",
    "sqa_dialog_binding_base",
    "prompt_format_",
    "table_serialization_",
    "input_context_",
    "output_contract_",
)

CLEAN_EXACT: set[str] = set()

TASK_REQUIRED_BASE_FEATURES = {
    # Older SQA configs predate the redesigned conversational binding. Keep the
    # clean cube on one SQA task spec rather than mixing two baselines under the
    # same surface label.
    "sqa": {"sqa_dialog_binding_base"},
}

CLEAN_EXCLUDE_SUBSTRINGS = (
    "reasoning",
    "scaffold_visible",
    "domain_",
    "retrieval",
    "agent",
    "critique",
    "repair",
)

FORMAT_BLOCKS = {
    "prompt_format_plain": "fmt.plain",
    "prompt_format_json": "fmt.json",
    "prompt_format_markdown": "fmt.markdown",
    "prompt_format_yaml": "fmt.yaml",
    "prompt_format_code_block": "fmt.code_block",
}

SER_BLOCKS = {
    "table_serialization_json_columns_data": "ser.json_columns",
    "table_serialization_json_records": "ser.records",
    "table_serialization_html": "ser.html",
    "table_serialization_markdown": "ser.markdown",
    "table_serialization_csv": "ser.csv",
}

CTX_ATOMS = {
    "input_context_type_annotation": "type",
    "input_context_column_statistics": "stats",
    "input_context_column_selection_relevance_12": "cols12",
    "input_context_row_selection_relevance_50": "rows50",
}

CONTRACT_BLOCKS = {
    "output_contract_json_answer_list": "contract.json_answer_list",
    "output_contract_json_verdict": "contract.json_verdict",
    "output_contract_tablebench_answer_string": "contract.tablebench_answer_string",
    "contract_wikitable_answer_surface_pack": "contract.wikitable_answer_surface",
    "contract_table_answer_surface_pack": "contract.table_answer_surface",
}

FAMILY_ORDER = {
    "fmt": 10,
    "ser": 20,
    "ctx": 30,
    "contract": 40,
}

DEFAULT_BLOCKS = {"fmt.plain", "ser.json_columns"}


@dataclass
class PlannedConfig:
    task: str
    dataset: str
    config_path: str
    source_label: str
    clean_label: str
    surface_label: str
    canonical_ids: list[str]
    feature_ids: list[str]
    func_ids: list[str]
    func_ids_json: str
    func_specs: dict[str, dict[str, Any]]
    source_config_ids: set[int] = field(default_factory=set)
    clean_config_id: int | None = None
    copied_executions: int = 0
    copied_evaluations: int = 0


def ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def parse_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def is_clean_feature(canonical_id: str) -> bool:
    if canonical_id.startswith("_section_"):
        return True
    if canonical_id in {"facet_dp_scaffold", "sqa_dialog_binding_base"}:
        return True
    if any(token in canonical_id for token in CLEAN_EXCLUDE_SUBSTRINGS):
        return False
    if canonical_id in CLEAN_EXACT:
        return True
    return canonical_id.startswith(CLEAN_ALLOWED_PREFIXES)


def is_clean_coalition(base_features: list[str], atoms: list[str]) -> bool:
    return all(is_clean_feature(cid) for cid in base_features + atoms)


def has_required_task_base(task: str, canonical_ids: list[str]) -> bool:
    required = TASK_REQUIRED_BASE_FEATURES.get(task, set())
    return required.issubset(set(canonical_ids))


def context_block(canonical_ids: Iterable[str]) -> str | None:
    atoms = {CTX_ATOMS[cid] for cid in canonical_ids if cid in CTX_ATOMS}
    if atoms == {"type", "stats", "cols12", "rows50"}:
        return "ctx.full"
    if atoms == {"type", "stats"}:
        return "ctx.type_stats"
    if atoms == {"cols12", "rows50"}:
        return "ctx.relevant_table"

    blocks: list[str] = []
    if {"type", "stats"}.issubset(atoms):
        blocks.append("ctx.type_stats")
        atoms -= {"type", "stats"}
    if {"cols12", "rows50"}.issubset(atoms):
        blocks.append("ctx.relevant_table")
        atoms -= {"cols12", "rows50"}
    for atom in ("type", "stats", "cols12", "rows50"):
        if atom in atoms:
            blocks.append(f"ctx.{atom}")
    return "__".join(blocks) if blocks else None


def label_blocks(canonical_ids: list[str]) -> list[str]:
    blocks: list[str] = []
    for cid in canonical_ids:
        if cid in FORMAT_BLOCKS:
            blocks.append(FORMAT_BLOCKS[cid])
        elif cid in SER_BLOCKS:
            blocks.append(SER_BLOCKS[cid])

    ctx = context_block(canonical_ids)
    if ctx:
        blocks.extend(ctx.split("__"))

    for cid in canonical_ids:
        if cid in CONTRACT_BLOCKS:
            blocks.append(CONTRACT_BLOCKS[cid])

    def key(block: str) -> tuple[int, str]:
        return (FAMILY_ORDER.get(block.split(".", 1)[0], 99), block)

    return sorted(ordered_unique(blocks), key=key)


def clean_label(canonical_ids: list[str]) -> tuple[str, str]:
    blocks = label_blocks(canonical_ids)
    surface_label = "__".join(blocks) if blocks else "base"

    display_blocks = [
        block
        for block in blocks
        if block not in DEFAULT_BLOCKS and not block.startswith("contract.")
    ]
    label = "__".join(display_blocks) if display_blocks else "base"
    return label, surface_label


def materialize_config(
    registry: FeatureRegistry,
    base_features: list[str],
    atoms: list[str],
) -> tuple[list[str], list[str], list[str], dict[str, dict[str, Any]]]:
    """Mirror experiment.config_generators.explicit_coalitions."""
    base_specs, _ = registry.materialize(base_features)
    base_func_ids = [spec["func_id"] for spec in base_specs]
    func_specs = {spec["func_id"]: spec for spec in base_specs}
    base_set = set(base_func_ids)
    func_ids = list(base_func_ids)
    seen = set(base_func_ids)
    feature_ids = [registry.feature_id_for(cid) for cid in base_features]

    for cid in atoms:
        specs, feature_to_funcs = registry.materialize(base_features + [cid])
        for spec in specs:
            func_specs[spec["func_id"]] = spec
        fid = registry.feature_id_for(cid)
        feature_ids.append(fid)
        for func_id in feature_to_funcs[fid]:
            if func_id not in base_set and func_id not in seen:
                seen.add(func_id)
                func_ids.append(func_id)

    canonical_ids = list(base_features) + list(atoms)
    return sorted(func_ids), canonical_ids, feature_ids, func_specs


def source_connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60)
    conn.row_factory = sqlite3.Row
    return conn


def dest_connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row["name"] for row in conn.execute(f"PRAGMA table_info({table})")]


def copy_rows_by_pk(
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    *,
    table: str,
    pk_col: str,
    ids: Iterable[Any],
) -> int:
    ids = list(dict.fromkeys(ids))
    if not ids:
        return 0
    columns = table_columns(src, table)
    placeholders = ",".join("?" for _ in columns)
    insert_sql = (
        f"INSERT OR IGNORE INTO {table} "
        f"({','.join(columns)}) VALUES ({placeholders})"
    )
    copied = 0
    for start in range(0, len(ids), 900):
        chunk = ids[start:start + 900]
        qmarks = ",".join("?" for _ in chunk)
        rows = src.execute(
            f"SELECT {','.join(columns)} FROM {table} WHERE {pk_col} IN ({qmarks})",
            chunk,
        ).fetchall()
        before = dst.total_changes
        dst.executemany(insert_sql, [tuple(row[col] for col in columns) for row in rows])
        copied += dst.total_changes - before
    return copied


def copy_predicates_for_queries(
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    query_ids: Iterable[str],
) -> int:
    query_ids = list(dict.fromkeys(query_ids))
    if not query_ids:
        return 0
    copied = 0
    for start in range(0, len(query_ids), 900):
        chunk = query_ids[start:start + 900]
        qmarks = ",".join("?" for _ in chunk)
        rows = src.execute(
            f"SELECT query_id, name, value FROM predicate WHERE query_id IN ({qmarks})",
            chunk,
        ).fetchall()
        before = dst.total_changes
        dst.executemany(
            "INSERT OR IGNORE INTO predicate (query_id, name, value) VALUES (?, ?, ?)",
            [(row["query_id"], row["name"], row["value"]) for row in rows],
        )
        copied += dst.total_changes - before
    return copied


def copy_feature_side_tables(
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    feature_ids: Iterable[str],
) -> dict[str, int]:
    feature_ids = list(dict.fromkeys(feature_ids))
    copied = {
        "feature": copy_rows_by_pk(src, dst, table="feature", pk_col="feature_id", ids=feature_ids),
        "feature_label": 0,
        "feature_label_membership": 0,
    }

    if not feature_ids:
        return copied

    memberships: list[sqlite3.Row] = []
    for start in range(0, len(feature_ids), 900):
        chunk = feature_ids[start:start + 900]
        qmarks = ",".join("?" for _ in chunk)
        memberships.extend(src.execute(
            "SELECT feature_id, label_id, role, created_at "
            f"FROM feature_label_membership WHERE feature_id IN ({qmarks})",
            chunk,
        ).fetchall())

    label_ids = sorted({row["label_id"] for row in memberships})
    copied["feature_label"] = copy_rows_by_pk(
        src, dst, table="feature_label", pk_col="label_id", ids=label_ids
    )

    before = dst.total_changes
    dst.executemany(
        "INSERT OR IGNORE INTO feature_label_membership "
        "(feature_id, label_id, role, created_at) VALUES (?, ?, ?, ?)",
        [
            (row["feature_id"], row["label_id"], row["role"], row["created_at"])
            for row in memberships
        ],
    )
    copied["feature_label_membership"] = dst.total_changes - before
    return copied


def upsert_clean_config(dst: sqlite3.Connection, plan: PlannedConfig) -> int:
    meta = {
        "kind": "clean_surface_coalition",
        "label": plan.clean_label,
        "clean_label": plan.clean_label,
        "surface_label": plan.surface_label,
        "source_label": plan.source_label,
        "source_config_path": plan.config_path,
        "source_config_ids": sorted(plan.source_config_ids),
        "task": plan.task,
        "tasks": [plan.task],
        "dataset": plan.dataset,
        "datasets": [plan.dataset],
        "canonical_ids": plan.canonical_ids,
        "feature_ids": plan.feature_ids,
        "naming_version": "clean_surface_v1",
    }
    before = dst.total_changes
    dst.execute(
        "INSERT OR IGNORE INTO config (func_ids, meta) VALUES (?, ?)",
        (plan.func_ids_json, json_dumps(meta)),
    )
    row = dst.execute(
        "SELECT config_id, meta FROM config WHERE func_ids = ?",
        (plan.func_ids_json,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"failed to insert config {plan.clean_label}")
    config_id = int(row["config_id"])
    if dst.total_changes == before:
        current = parse_json(row["meta"], {})
        aliases = current.get("clean_aliases") or []
        alias = {
            "task": plan.task,
            "dataset": plan.dataset,
            "clean_label": plan.clean_label,
            "surface_label": plan.surface_label,
            "source_label": plan.source_label,
            "source_config_path": plan.config_path,
            "source_config_ids": sorted(plan.source_config_ids),
        }
        if alias not in aliases:
            aliases.append(alias)
            current["clean_aliases"] = aliases
            current["tasks"] = sorted(set((current.get("tasks") or [current.get("task")]) + [plan.task]))
            current["datasets"] = sorted(set((current.get("datasets") or [current.get("dataset")]) + [plan.dataset]))
            dst.execute(
                "UPDATE config SET meta = ? WHERE config_id = ?",
                (json_dumps(current), config_id),
            )
    return config_id


def insert_config_feature_rows(
    dst: sqlite3.Connection,
    config_id: int,
    feature_ids: Iterable[str],
) -> int:
    before = dst.total_changes
    dst.executemany(
        "INSERT OR IGNORE INTO config_feature (config_id, feature_id, role) VALUES (?, ?, 'feature')",
        [(config_id, fid) for fid in ordered_unique(feature_ids)],
    )
    return dst.total_changes - before


def upsert_func_specs(dst: sqlite3.Connection, func_specs: Iterable[dict[str, Any]]) -> int:
    rows = []
    for spec in func_specs:
        rows.append((
            spec["func_id"],
            spec["func_type"],
            json.dumps(spec.get("params", {})),
            json.dumps(spec.get("meta", {})),
        ))
    before = dst.total_changes
    dst.executemany(
        "INSERT OR IGNORE INTO func (func_id, func_type, params, meta) VALUES (?, ?, ?, ?)",
        rows,
    )
    return dst.total_changes - before


def copy_executions_and_evals(
    src: sqlite3.Connection,
    dst: sqlite3.Connection,
    source_config_ids: Iterable[int],
    clean_config_id: int,
    *,
    model_filter: str | None,
) -> tuple[int, int, dict[int, int]]:
    copied_exec = 0
    copied_eval = 0
    execution_map: dict[int, int] = {}
    exec_cols = table_columns(src, "execution")
    eval_cols = table_columns(src, "evaluation")
    exec_insert_cols = [col for col in exec_cols if col != "execution_id"]
    eval_insert_cols = [col for col in eval_cols if col != "eval_id"]

    model_clause = ""
    params_extra: list[Any] = []
    if model_filter:
        model_clause = " AND model = ?"
        params_extra.append(model_filter)

    for source_config_id in sorted(source_config_ids):
        rows = src.execute(
            "SELECT * FROM execution WHERE config_id = ?" + model_clause,
            [source_config_id, *params_extra],
        ).fetchall()
        for row in rows:
            values = []
            for col in exec_insert_cols:
                values.append(clean_config_id if col == "config_id" else row[col])
            before = dst.total_changes
            dst.execute(
                "INSERT OR IGNORE INTO execution "
                f"({','.join(exec_insert_cols)}) VALUES ({','.join('?' for _ in exec_insert_cols)})",
                values,
            )
            copied_exec += dst.total_changes - before
            mapped = dst.execute(
                "SELECT execution_id FROM execution WHERE config_id = ? AND query_id = ? AND model = ?",
                (clean_config_id, row["query_id"], row["model"]),
            ).fetchone()
            if mapped is None:
                continue
            new_execution_id = int(mapped["execution_id"])
            execution_map[int(row["execution_id"])] = new_execution_id

            eval_rows = src.execute(
                "SELECT * FROM evaluation WHERE execution_id = ?",
                (row["execution_id"],),
            ).fetchall()
            for ev in eval_rows:
                ev_values = []
                for col in eval_insert_cols:
                    ev_values.append(new_execution_id if col == "execution_id" else ev[col])
                before = dst.total_changes
                dst.execute(
                    "INSERT OR IGNORE INTO evaluation "
                    f"({','.join(eval_insert_cols)}) VALUES ({','.join('?' for _ in eval_insert_cols)})",
                    ev_values,
                )
                copied_eval += dst.total_changes - before
    return copied_exec, copied_eval, execution_map


def ensure_mapping_tables(dst: sqlite3.Connection) -> None:
    dst.execute(
        """
        CREATE TABLE IF NOT EXISTS clean_config_map (
            map_id INTEGER PRIMARY KEY AUTOINCREMENT,
            clean_config_id INTEGER NOT NULL,
            source_config_id INTEGER,
            task TEXT NOT NULL,
            dataset TEXT NOT NULL,
            source_config_path TEXT NOT NULL,
            source_label TEXT NOT NULL,
            clean_label TEXT NOT NULL,
            surface_label TEXT NOT NULL,
            canonical_ids TEXT NOT NULL,
            feature_ids TEXT NOT NULL,
            func_ids TEXT NOT NULL,
            copied_executions INTEGER NOT NULL DEFAULT 0,
            copied_evaluations INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(clean_config_id, source_config_id, source_config_path, source_label)
        )
        """
    )
    dst.execute(
        """
        CREATE TABLE IF NOT EXISTS clean_execution_map (
            source_execution_id INTEGER PRIMARY KEY,
            clean_execution_id INTEGER NOT NULL,
            source_config_id INTEGER NOT NULL,
            clean_config_id INTEGER NOT NULL
        )
        """
    )


def insert_mapping_rows(
    dst: sqlite3.Connection,
    plan: PlannedConfig,
    execution_map: dict[int, int],
) -> None:
    source_ids = sorted(plan.source_config_ids) or [None]
    for source_config_id in source_ids:
        dst.execute(
            "INSERT OR IGNORE INTO clean_config_map "
            "(clean_config_id, source_config_id, task, dataset, source_config_path, "
            "source_label, clean_label, surface_label, canonical_ids, feature_ids, "
            "func_ids, copied_executions, copied_evaluations) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                plan.clean_config_id,
                source_config_id,
                plan.task,
                plan.dataset,
                plan.config_path,
                plan.source_label,
                plan.clean_label,
                plan.surface_label,
                json_dumps(plan.canonical_ids),
                json_dumps(plan.feature_ids),
                plan.func_ids_json,
                plan.copied_executions,
                plan.copied_evaluations,
            ),
        )
    if execution_map:
        src_cfg = min(plan.source_config_ids) if plan.source_config_ids else 0
        dst.executemany(
            "INSERT OR IGNORE INTO clean_execution_map "
            "(source_execution_id, clean_execution_id, source_config_id, clean_config_id) "
            "VALUES (?, ?, ?, ?)",
            [
                (old_id, new_id, src_cfg, plan.clean_config_id)
                for old_id, new_id in execution_map.items()
            ],
        )


def load_plans(config_paths: list[Path], source: sqlite3.Connection) -> list[PlannedConfig]:
    plans_by_key: dict[tuple[str, str], PlannedConfig] = {}
    registries: dict[str, FeatureRegistry] = {}

    for path in config_paths:
        cfg = json.loads(path.read_text())
        task = cfg["task"]
        dataset = DATASET_BY_TASK.get(task, task)
        registry = registries.setdefault(task, FeatureRegistry.load(task))
        base_features = list(cfg.get("base_features") or [])
        coalitions = cfg.get("coalitions") or {}

        for source_label, atoms in coalitions.items():
            atoms = list(atoms or [])
            if not is_clean_coalition(base_features, atoms):
                continue
            full_canonical = ordered_unique(base_features + atoms)
            if not has_required_task_base(task, full_canonical):
                continue
            label, surface = clean_label(full_canonical)
            try:
                func_ids, canonical_ids, feature_ids, func_specs = materialize_config(
                    registry, base_features, atoms
                )
            except ValueError as exc:
                print(
                    f"[skip] {path}:{source_label}: cannot materialize under current "
                    f"features/{task}: {exc}",
                    file=sys.stderr,
                )
                continue
            func_ids_json = json.dumps(func_ids)
            source_rows = source.execute(
                "SELECT config_id FROM config WHERE func_ids = ?",
                (func_ids_json,),
            ).fetchall()
            key = (task, func_ids_json)
            if key not in plans_by_key:
                plans_by_key[key] = PlannedConfig(
                    task=task,
                    dataset=dataset,
                    config_path=str(path),
                    source_label=str(source_label),
                    clean_label=label,
                    surface_label=surface,
                    canonical_ids=canonical_ids,
                    feature_ids=feature_ids,
                    func_ids=func_ids,
                    func_ids_json=func_ids_json,
                    func_specs=func_specs,
                )
            else:
                plans_by_key[key].func_specs.update(func_specs)
            plan = plans_by_key[key]
            plan.source_config_ids.update(int(row["config_id"]) for row in source_rows)
            if str(source_label) not in plan.source_label.split(" | "):
                plan.source_label = f"{plan.source_label} | {source_label}"
            if str(path) not in plan.config_path.split(" | "):
                plan.config_path = f"{plan.config_path} | {path}"

    return sorted(plans_by_key.values(), key=lambda p: (p.task, p.clean_label, p.surface_label))


def summarize_db(conn: sqlite3.Connection) -> dict[str, Any]:
    tables = ["func", "feature", "query", "predicate", "config", "config_feature", "execution", "evaluation"]
    out: dict[str, Any] = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in tables
    }
    has_clean_map = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'clean_config_map'"
    ).fetchone()
    if has_clean_map:
        out["configs_by_task"] = [
            dict(row) for row in conn.execute(
                "SELECT task, COUNT(DISTINCT clean_config_id) AS n "
                "FROM clean_config_map GROUP BY task ORDER BY task"
            ).fetchall()
        ]
        out["planned_config_aliases"] = conn.execute(
            "SELECT COUNT(*) FROM clean_config_map"
        ).fetchone()[0]
    else:
        out["configs_by_task"] = [
            dict(row) for row in conn.execute(
                "SELECT json_extract(meta, '$.task') AS task, COUNT(*) AS n "
                "FROM config GROUP BY task ORDER BY task"
            ).fetchall()
        ]
    out["executions_by_dataset"] = [
        dict(row) for row in conn.execute(
            "SELECT q.dataset, COUNT(*) AS n "
            "FROM execution e JOIN query q ON q.query_id = e.query_id "
            "GROUP BY q.dataset ORDER BY q.dataset"
        ).fetchall()
    ]
    return out


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, default=DEFAULT_SOURCE_DB)
    parser.add_argument("--dest-db", type=Path, default=DEFAULT_DEST_DB)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--config", type=Path, action="append", default=[])
    parser.add_argument("--model", default=None, help="Optional model filter for copied executions.")
    parser.add_argument("--replace", action="store_true", help="Replace destination DB if it exists.")
    parser.add_argument("--dry-run", action="store_true", help="Plan only; do not create/copy destination DB.")
    args = parser.parse_args()

    config_paths = args.config or [ROOT / p for p in DEFAULT_CONFIGS]
    config_paths = [p if p.is_absolute() else ROOT / p for p in config_paths]
    missing = [str(p) for p in config_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("missing config files:\n" + "\n".join(missing))

    src = source_connect(args.source_db)
    plans = load_plans(config_paths, src)
    datasets = sorted({p.dataset for p in plans})
    source_config_ids = sorted({cid for p in plans for cid in p.source_config_ids})

    dry_summary = {
        "source_db": str(args.source_db),
        "dest_db": str(args.dest_db),
        "naming_version": "clean_surface_v1",
        "dry_run": args.dry_run,
        "planned_configs": len(plans),
        "planned_configs_with_source_match": sum(1 for p in plans if p.source_config_ids),
        "source_config_ids": source_config_ids,
        "datasets": datasets,
        "configs": [
            {
                "task": p.task,
                "dataset": p.dataset,
                "clean_label": p.clean_label,
                "surface_label": p.surface_label,
                "source_label": p.source_label,
                "source_config_ids": sorted(p.source_config_ids),
                "canonical_ids": p.canonical_ids,
                "func_ids": p.func_ids,
            }
            for p in plans
        ],
    }

    if args.dry_run:
        write_summary(args.summary, dry_summary)
        print(json.dumps({
            "dry_run": True,
            "planned_configs": dry_summary["planned_configs"],
            "planned_configs_with_source_match": dry_summary["planned_configs_with_source_match"],
            "datasets": datasets,
            "summary": str(args.summary),
        }, indent=2))
        return 0

    if args.dest_db.exists():
        if not args.replace:
            raise FileExistsError(f"destination exists; pass --replace to overwrite: {args.dest_db}")
        for suffix in ("", "-wal", "-shm"):
            path = Path(str(args.dest_db) + suffix)
            if path.exists():
                path.unlink()

    args.dest_db.parent.mkdir(parents=True, exist_ok=True)
    # Initialize schema and feature labels with the current registry state.
    store = CubeStore(args.dest_db)
    for task in sorted({p.task for p in plans}):
        FeatureRegistry.load(task).sync_to_cube(store)
    store.close()

    dst = dest_connect(args.dest_db)
    ensure_mapping_tables(dst)

    feature_ids = sorted({fid for p in plans for fid in p.feature_ids})
    copied_feature_side = copy_feature_side_tables(src, dst, feature_ids)

    # Copy all queries/predicates for planned datasets. This makes the clean DB
    # usable for analysis and future runs even when a planned config has no
    # copied executions yet.
    query_ids: list[str] = []
    for dataset in datasets:
        query_ids.extend(
            row["query_id"]
            for row in src.execute(
                "SELECT query_id FROM query WHERE dataset = ?",
                (dataset,),
            ).fetchall()
        )
    copied_queries = copy_rows_by_pk(src, dst, table="query", pk_col="query_id", ids=query_ids)
    copied_predicates = copy_predicates_for_queries(src, dst, query_ids)

    # Insert current registry func rows first so planned configs are executable
    # even if there was no exact source config. Copy historical source func rows
    # afterward as a guard for exact old matches.
    current_func_specs = {
        func_id: spec
        for p in plans
        for func_id, spec in p.func_specs.items()
    }
    inserted_current_funcs = upsert_func_specs(dst, current_func_specs.values())
    func_ids = sorted({fid for p in plans for fid in p.func_ids})
    copied_source_funcs = copy_rows_by_pk(src, dst, table="func", pk_col="func_id", ids=func_ids)

    copied_config_features = 0
    total_exec = 0
    total_eval = 0
    for plan in plans:
        plan.clean_config_id = upsert_clean_config(dst, plan)
        copied_config_features += insert_config_feature_rows(
            dst, plan.clean_config_id, plan.feature_ids
        )
        copied_exec, copied_eval, execution_map = copy_executions_and_evals(
            src,
            dst,
            plan.source_config_ids,
            plan.clean_config_id,
            model_filter=args.model,
        )
        plan.copied_executions = copied_exec
        plan.copied_evaluations = copied_eval
        total_exec += copied_exec
        total_eval += copied_eval
        insert_mapping_rows(dst, plan, execution_map)

    dst.commit()

    final_summary = {
        **dry_summary,
        "dry_run": False,
        "model_filter": args.model,
        "copied": {
            "func_current_registry": inserted_current_funcs,
            "func_source_guard": copied_source_funcs,
            "feature_side": copied_feature_side,
            "query": copied_queries,
            "predicate": copied_predicates,
            "config_feature": copied_config_features,
            "execution": total_exec,
            "evaluation": total_eval,
        },
        "dest_counts": summarize_db(dst),
        "configs": [
            {
                "task": p.task,
                "dataset": p.dataset,
                "clean_config_id": p.clean_config_id,
                "clean_label": p.clean_label,
                "surface_label": p.surface_label,
                "source_label": p.source_label,
                "source_config_ids": sorted(p.source_config_ids),
                "copied_executions": p.copied_executions,
                "copied_evaluations": p.copied_evaluations,
                "canonical_ids": p.canonical_ids,
                "func_ids": p.func_ids,
            }
            for p in plans
        ],
    }
    write_summary(args.summary, final_summary)
    dst.close()
    src.close()

    print(json.dumps({
        "dest_db": str(args.dest_db),
        "summary": str(args.summary),
        "planned_configs": len(plans),
        "planned_configs_with_source_match": dry_summary["planned_configs_with_source_match"],
        "copied": final_summary["copied"],
        "dest_counts": final_summary["dest_counts"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
