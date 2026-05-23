#!/usr/bin/env python3
"""Interactive prompt renderer for FACET feature coalitions.

This serves a local HTML UI that lets you choose arbitrary feature canonical IDs
and renders the resulting system/user prompts for WTQ, SQA, TableBench, TabFact,
and HiTab. It is offline: no LLM calls, no experiment executions.

The renderer does not require an existing experiment cube. By default it uses
the task loaders to seed sample queries into a temporary local CubeStore, with a
small built-in fallback fixture if the dataset is unavailable. Pass
``--source-db`` only when you explicitly want examples from an existing cube.

Example:
    python3 tools/prompt_render_interface.py --port 8765
    python3 tools/prompt_render_interface.py --source-db /data/users/jsu323/facet/wikitable_clean_surface_v1.db --port 8765
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import seed_funcs
from core.feature_registry import FeatureRegistry
from core.func_registry import apply_config, apply_config_modules
from core.store import CubeStore
from task_registry import get_registry
from tools.render_prompts_from_config import TASK_DEFAULT_SPLIT, _default_render_base_features

TASKS = ["wtq", "sqa", "tablebench", "tabfact", "hitab"]
TASK_DATASET = {
    "wtq": "wtq",
    "sqa": "sqa",
    "tablebench": "tablebench",
    "tabfact": "tab_fact",
    "hitab": "hitab",
}
HIDDEN_FEATURES = {"facet_dp_scaffold", "sqa_dialog_binding_base"}
_SEEDED_SAMPLE_CACHE: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve an interactive prompt renderer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--source-db",
        default=None,
        help="Optional existing cube DB used only for sample query rows. Defaults to task-loader samples.",
    )
    args = parser.parse_args()

    server = PromptRenderServer(
        (args.host, args.port),
        source_db=Path(args.source_db) if args.source_db else None,
    )
    url = f"http://{args.host}:{args.port}"
    print(f"Prompt render interface: {url}")
    if args.source_db:
        print(f"Sample source cube: {args.source_db}")
    else:
        print("Sample source: task loaders with built-in fixture fallback")
    server.serve_forever()


class PromptRenderServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], *, source_db: Path | None) -> None:
        super().__init__(server_address, PromptRenderHandler)
        self.source_db = source_db
        self.registry = get_registry()


class PromptRenderHandler(BaseHTTPRequestHandler):
    server: PromptRenderServer

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), fmt % args))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(INDEX_HTML)
            return
        if parsed.path == "/api/features":
            self._send_json(feature_inventory())
            return
        if parsed.path == "/api/samples":
            params = parse_qs(parsed.query)
            task = params.get("task", [""])[0]
            limit = _int_param(params, "limit", 40)
            self._send_json({
                "task": task,
                "samples": list_sample_queries(
                    self.server.source_db,
                    task,
                    limit=limit,
                    registry=self.server.registry,
                ),
            })
            return
        self.send_error(404, "not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/render":
            self.send_error(404, "not found")
            return
        try:
            payload = self._read_json()
            selected = _string_list(payload.get("features"))
            requested_tasks = _task_list(payload.get("tasks")) or TASKS
            query_ids = payload.get("query_ids") if isinstance(payload.get("query_ids"), dict) else {}
            results = {}
            for task in requested_tasks:
                results[task] = render_task_prompt(
                    task,
                    selected_features=selected,
                    query_id=str(query_ids.get(task) or "").strip() or None,
                    source_db=self.server.source_db,
                    registry=self.server.registry,
                )
            self._send_json({"ok": True, "results": results})
        except Exception as exc:  # UI endpoint: return a structured error.
            self._send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status=500)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("content-length", "0") or "0")
        data = self.rfile.read(length) if length else b"{}"
        return json.loads(data.decode("utf-8"))

    def _send_json(self, data: Any, *, status: int = 200) -> None:
        raw = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_html(self, text: str) -> None:
        raw = text.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def feature_inventory() -> Dict[str, Any]:
    by_feature: Dict[str, Dict[str, Any]] = {}
    by_task: Dict[str, List[Dict[str, Any]]] = {}
    for task in TASKS:
        reg = FeatureRegistry.load(task=task)
        task_rows = []
        for canonical_id, spec in sorted(reg._by_canonical.items()):
            if _is_hidden_feature(canonical_id):
                continue
            row = feature_row(task, canonical_id, spec)
            task_rows.append(row)
            merged = by_feature.setdefault(
                canonical_id,
                {
                    "canonical_id": canonical_id,
                    "family": row["family"],
                    "label": canonical_id,
                    "tasks": {},
                    "semantic_labels": [],
                },
            )
            merged["tasks"][task] = {
                "family": row["family"],
                "requires": row["requires"],
                "conflicts_with": row["conflicts_with"],
                "rationale": row["rationale"],
            }
            for label in row["semantic_labels"]:
                if label not in merged["semantic_labels"]:
                    merged["semantic_labels"].append(label)
        by_task[task] = task_rows
    return {
        "tasks": TASKS,
        "features": sorted(by_feature.values(), key=lambda r: (r["family"], r["canonical_id"])),
        "features_by_task": by_task,
    }


def feature_row(task: str, canonical_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    source = str(spec.get("source_path") or "")
    family = source_family(source)
    if family == "other":
        family = canonical_family(canonical_id)
    semantic_labels = []
    for item in spec.get("semantic_labels") or []:
        if isinstance(item, str):
            semantic_labels.append(item)
        elif isinstance(item, dict) and item.get("label"):
            semantic_labels.append(str(item["label"]))
    return {
        "task": task,
        "canonical_id": canonical_id,
        "family": family,
        "requires": list(spec.get("requires") or []),
        "conflicts_with": list(spec.get("conflicts_with") or []),
        "semantic_labels": semantic_labels,
        "scope": spec.get("scope") or {},
        "rationale": spec.get("rationale") or "",
    }


def render_task_prompt(
    task: str,
    *,
    selected_features: List[str],
    query_id: str | None,
    source_db: Path | None,
    registry: Dict[str, Any],
) -> Dict[str, Any]:
    if task not in TASKS:
        return {"ok": False, "error": f"unsupported task {task!r}"}
    if task not in registry:
        return {"ok": False, "error": f"task {task!r} is not registered"}

    feature_registry = FeatureRegistry.load(task=task)
    available = set(feature_registry._by_canonical)
    active = [f for f in selected_features if f in available]
    missing = [f for f in selected_features if f not in available]
    base = _default_render_base_features(task, active, feature_registry)

    try:
        feature_registry.validate_feature_set(base + active)
        query = load_query(source_db, task, query_id=query_id, registry=registry)
        system_prompt, user_content = build_prompt_for_features(
            task,
            base_features=base,
            active_features=active,
            query=query,
            registry=registry,
        )
        return {
            "ok": True,
            "task": task,
            "dataset": TASK_DATASET[task],
            "query_id": query.get("query_id"),
            "question": query.get("content", ""),
            "base_features": base,
            "active_features": active,
            "missing_features": missing,
            "system_prompt": system_prompt,
            "user_content": user_content,
        }
    except Exception as exc:
        return {
            "ok": False,
            "task": task,
            "dataset": TASK_DATASET[task],
            "base_features": base,
            "active_features": active,
            "missing_features": missing,
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_prompt_for_features(
    task: str,
    *,
    base_features: List[str],
    active_features: List[str],
    query: Dict[str, Any],
    registry: Dict[str, Any],
) -> Tuple[str, str]:
    feature_registry = FeatureRegistry.load(task=task)
    canonical_ids = base_features + active_features
    specs, _feature_to_funcs = feature_registry.materialize(canonical_ids)
    with tempfile.TemporaryDirectory(prefix="prompt_render_ui_") as tmp:
        store = CubeStore(Path(tmp) / "render.db")
        try:
            seed_funcs(store, specs)
            func_ids = [spec["func_id"] for spec in specs]
            task_cls = registry[task].task_cls
            task_obj = task_cls()
            if hasattr(task_obj, "bind_modules"):
                states = apply_config_modules(
                    func_ids,
                    store,
                    module_names=task_obj.module_names(),
                )
                task_obj.bind_modules(states, example_pool=None)
            else:
                state = apply_config(func_ids, store)
                task_obj.bind(state, example_pool=None)
            return task_obj.build_prompt(query)
        finally:
            store.close()


def list_sample_queries(
    source_db: Path | None,
    task: str,
    *,
    limit: int = 40,
    registry: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if task not in TASK_DATASET:
        return []
    rows = _load_sample_queries(source_db, task, limit=int(limit), registry=registry)
    out = []
    for row in rows:
        meta = _json_loads(row["meta"], {})
        out.append({
            "query_id": row["query_id"],
            "content": row["content"],
            "split": meta.get("split", ""),
            "summary": query_summary(row["content"], meta),
        })
    return out


def load_query(
    source_db: Path | None,
    task: str,
    *,
    query_id: str | None = None,
    registry: Dict[str, Any],
) -> Dict[str, Any]:
    rows = _load_sample_queries(
        source_db,
        task,
        limit=80 if query_id else 1,
        registry=registry,
        query_id=query_id,
    )
    if not rows:
        dataset = TASK_DATASET[task]
        raise ValueError(f"no query found for dataset={dataset!r} query_id={query_id!r}")
    row = rows[0]
    return {
        "query_id": row["query_id"],
        "dataset": row["dataset"],
        "content": row["content"],
        "meta": row["meta"],
    }


def _load_sample_queries(
    source_db: Path | None,
    task: str,
    *,
    limit: int,
    registry: Dict[str, Any],
    query_id: str | None = None,
) -> List[Dict[str, Any]]:
    if source_db is not None and source_db.exists():
        rows = _load_queries_from_source_db(source_db, task, limit=limit, query_id=query_id)
        if rows:
            return rows
    rows = _load_queries_from_task_loader(task, limit=limit, registry=registry, query_id=query_id)
    return rows or _fixture_queries(task, limit=limit, query_id=query_id)


def _load_queries_from_source_db(
    source_db: Path,
    task: str,
    *,
    limit: int,
    query_id: str | None = None,
) -> List[Dict[str, Any]]:
    dataset = TASK_DATASET[task]
    conn = sqlite3.connect(str(source_db))
    conn.row_factory = sqlite3.Row
    try:
        if query_id:
            rows = conn.execute(
                "SELECT query_id, dataset, content, meta FROM query WHERE dataset = ? AND query_id = ?",
                (dataset, query_id),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT query_id, dataset, content, meta
                FROM query
                WHERE dataset = ?
                ORDER BY query_id
                LIMIT ?
                """,
                (dataset, int(limit)),
            ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def _load_queries_from_task_loader(
    task: str,
    *,
    limit: int,
    registry: Dict[str, Any],
    query_id: str | None = None,
) -> List[Dict[str, Any]]:
    cache_limit = max(int(limit), 80 if query_id else int(limit))
    cache_key = (task, cache_limit)
    if cache_key not in _SEEDED_SAMPLE_CACHE:
        with tempfile.TemporaryDirectory(prefix="prompt_render_samples_") as tmp:
            store = CubeStore(Path(tmp) / "samples.db")
            try:
                entry = registry[task]
                cfg = {
                    "max_queries": cache_limit,
                    "sample_seed": 0,
                }
                split = TASK_DEFAULT_SPLIT.get(task, "test")
                entry.seeder_fn(store, cfg, split)
                dataset = TASK_DATASET[task]
                rows = store._get_conn().execute(
                    """
                    SELECT query_id, dataset, content, meta
                    FROM query
                    WHERE dataset = ?
                    ORDER BY rowid
                    LIMIT ?
                    """,
                    (dataset, cache_limit),
                ).fetchall()
                _SEEDED_SAMPLE_CACHE[cache_key] = [dict(row) for row in rows]
            except Exception:
                _SEEDED_SAMPLE_CACHE[cache_key] = []
            finally:
                store.close()
    rows = _SEEDED_SAMPLE_CACHE[cache_key]
    if query_id:
        return [row for row in rows if row["query_id"] == query_id]
    return rows[:limit]


def _fixture_queries(task: str, *, limit: int, query_id: str | None = None) -> List[Dict[str, Any]]:
    rows = _FIXTURE_QUERIES.get(task, [])
    if query_id:
        rows = [row for row in rows if row["query_id"] == query_id]
    return rows[:limit]


_FIXTURE_QUERIES: Dict[str, List[Dict[str, Any]]] = {
    "wtq": [
        {
            "query_id": "fixture_wtq_1",
            "dataset": "wtq",
            "content": "How many players are from Japan?",
            "meta": json.dumps({
                "split": "fixture",
                "gold_answers": ["2"],
                "_raw": {
                    "question": "How many players are from Japan?",
                    "answers": ["2"],
                    "table": {
                        "name": "golf players",
                        "header": ["player", "country"],
                        "rows": [
                            ["Juli Inkster", "United States"],
                            ["Momoko Ueda", "Japan"],
                            ["Yuri Fudoh", "Japan"],
                        ],
                    },
                },
            }),
        }
    ],
    "sqa": [
        {
            "query_id": "fixture_sqa_1",
            "dataset": "sqa",
            "content": "Which city hosted the 2004 event?",
            "meta": json.dumps({
                "split": "fixture",
                "gold_answer": ["Athens"],
                "_raw": {
                    "question": "Which city hosted the 2004 event?",
                    "answer_text": ["Athens"],
                    "history": [],
                    "table_file": "events",
                    "table": {
                        "headers": ["year", "host city", "country"],
                        "rows": [["2000", "Sydney", "Australia"], ["2004", "Athens", "Greece"]],
                    },
                },
            }),
        }
    ],
    "tablebench": [
        {
            "query_id": "fixture_tablebench_1",
            "dataset": "tablebench",
            "content": "Which team has the highest score?",
            "meta": json.dumps({
                "split": "fixture",
                "gold_answer": "Carlton",
                "qtype": "DataAnalysis",
                "qsubtype": "superlative",
                "_raw": {
                    "question": "Which team has the highest score?",
                    "answer": "Carlton",
                    "qtype": "DataAnalysis",
                    "qsubtype": "superlative",
                    "table": {
                        "header": ["team", "score"],
                        "rows": [["Melbourne", "89"], ["Carlton", "149"], ["Essendon", "87"]],
                        "name": "scores",
                    },
                },
            }),
        }
    ],
    "tabfact": [
        {
            "query_id": "fixture_tabfact_1",
            "dataset": "tab_fact",
            "content": "Carlton has the highest score.",
            "meta": json.dumps({
                "split": "fixture",
                "gold_label": 1,
                "table_caption": "scores",
                "_raw": {
                    "statement": "Carlton has the highest score.",
                    "label": 1,
                    "table_text": "team#score\nMelbourne#89\nCarlton#149\nEssendon#87",
                    "table_caption": "scores",
                },
            }),
        }
    ],
    "hitab": [
        {
            "query_id": "fixture_hitab_1",
            "dataset": "hitab",
            "content": "What is the population of Beta?",
            "meta": json.dumps({
                "split": "fixture",
                "gold_answer": "[\"2000\"]",
                "_raw": {
                    "question": "What is the population of Beta?",
                    "answer": "[\"2000\"]",
                    "table_id": "fixture_hitab_table",
                    "table_source": "fixture",
                    "aggregation": "none",
                    "table_content": {
                        "title": "cities",
                        "top_header_rows_num": 1,
                        "texts": [["city", "population"], ["Alpha", "1000"], ["Beta", "2000"]],
                        "merged_regions": [],
                    },
                },
            }),
        }
    ],
}


def query_summary(content: str, meta: Dict[str, Any]) -> str:
    raw = meta.get("_raw") if isinstance(meta.get("_raw"), dict) else {}
    bits = []
    for key in ("qtype", "qsubtype", "statement"):
        value = meta.get(key) or raw.get(key)
        if value:
            bits.append(f"{key}: {value}")
    prefix = " | ".join(bits)
    text = str(content or raw.get("question") or raw.get("statement") or "")
    if len(text) > 120:
        text = text[:117] + "..."
    return f"{prefix} | {text}" if prefix else text


def source_family(source_path: str) -> str:
    if not source_path:
        return "other"
    parts = Path(source_path).parts
    try:
        idx = parts.index("features")
        # features/<task>/<family>/<file>.json
        if len(parts) > idx + 2:
            return parts[idx + 2]
    except ValueError:
        pass
    return "other"


def canonical_family(canonical_id: str) -> str:
    if canonical_id.startswith("input_context_"):
        return "input_context"
    if canonical_id.startswith("reasoning_scaffold_"):
        return "visible_reasoning_scaffold"
    if canonical_id.startswith("reasoning_"):
        return "private_reasoning_rule"
    if canonical_id.startswith("prompt_format_"):
        return "prompt_format"
    if canonical_id.startswith("table_serialization_"):
        return "table_serialization"
    if canonical_id.startswith("output_contract_") or canonical_id.startswith("contract_"):
        return "response_contract"
    if canonical_id.startswith("response_mode_"):
        return "response_mode"
    if canonical_id.startswith("dataset_domain_heuristics") or canonical_id.startswith("domain_heuristics"):
        return "domain_heuristics"
    if canonical_id.startswith("private_reasoning_rule"):
        return "private_reasoning_rule"
    return "other"


def _is_hidden_feature(canonical_id: str) -> bool:
    return canonical_id in HIDDEN_FEATURES or canonical_id.startswith("_section_")


def _json_loads(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value if value is not None else default


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def _task_list(value: Any) -> List[str]:
    return [task for task in _string_list(value) if task in TASKS]


def _int_param(params: Dict[str, List[str]], key: str, default: int) -> int:
    try:
        return int(params.get(key, [default])[0])
    except Exception:
        return default


INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FACET Prompt Renderer</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f5f6;
      --surface: #ffffff;
      --surface-soft: #f8fafb;
      --line: #d8dde3;
      --line-strong: #b6c0ca;
      --text: #20242a;
      --muted: #68707d;
      --accent: #12615c;
      --accent-soft: #e7f2f0;
      --warn-bg: #fff5d6;
      --warn-text: #765200;
      --bad: #9b2c2c;
      --good: #146c43;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font-family: var(--sans); font-size: 14px; line-height: 1.45; }
    button, select, input, textarea { font: inherit; }
    button { min-height: 34px; border: 1px solid var(--line-strong); border-radius: 6px; background: var(--surface); color: var(--text); padding: 7px 10px; cursor: pointer; }
    button:hover { border-color: var(--accent); }
    button.primary { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 650; }
    button.primary:disabled { opacity: .65; cursor: wait; }
    input[type="search"], select { width: 100%; min-height: 34px; border: 1px solid var(--line); border-radius: 6px; background: #fff; color: var(--text); padding: 6px 8px; }
    h1 { margin: 0; font-size: 18px; font-weight: 700; letter-spacing: 0; }
    h2 { margin: 0; font-size: 14px; font-weight: 700; letter-spacing: 0; }
    h3 { margin: 0; font-size: 12px; font-weight: 700; color: var(--muted); letter-spacing: .04em; text-transform: uppercase; }
    .small { color: var(--muted); font-size: 12px; }
    .mono { font-family: var(--mono); }
    .app { min-height: 100vh; display: grid; grid-template-columns: minmax(320px, 380px) minmax(0, 1fr); }
    .sidebar { min-height: 100vh; border-right: 1px solid var(--line); background: var(--surface); display: grid; grid-template-rows: auto auto minmax(0, 1fr) auto; }
    .brand { padding: 14px 16px; border-bottom: 1px solid var(--line); }
    .brand .small { margin-top: 4px; }
    .control-block { padding: 12px 16px; border-bottom: 1px solid var(--line); display: grid; gap: 10px; }
    .field { display: grid; gap: 5px; }
    .field-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    details.presets { border-top: 1px solid var(--line); padding-top: 8px; }
    details.presets summary { cursor: pointer; color: var(--muted); font-size: 12px; }
    .preset-list { display: grid; gap: 6px; margin-top: 8px; }
    .preset-list button { text-align: left; min-height: auto; padding: 7px 8px; font-size: 12px; }
    .feature-panel { min-height: 0; display: grid; grid-template-rows: auto minmax(0, 1fr); }
    .feature-head { padding: 10px 16px; border-bottom: 1px solid var(--line); display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    .feature-list { min-height: 0; overflow: auto; padding: 8px 10px 12px; }
    .family { margin-bottom: 10px; }
    .family-title { display: flex; align-items: center; justify-content: space-between; padding: 8px 6px 5px; color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; }
    label.feature { display: grid; grid-template-columns: 20px 1fr; gap: 7px; padding: 7px 6px; border-radius: 6px; cursor: pointer; }
    label.feature:hover { background: var(--surface-soft); }
    label.feature input { margin-top: 2px; }
    .feature-id { display: block; font-family: var(--mono); font-size: 12px; overflow-wrap: anywhere; }
    .feature-note { display: block; margin-top: 2px; color: var(--muted); font-size: 11px; overflow-wrap: anywhere; }
    .selected-dock { border-top: 1px solid var(--line); padding: 10px 16px 12px; display: grid; gap: 8px; background: var(--surface-soft); }
    .selected-head { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
    .chip-row { display: flex; flex-wrap: wrap; gap: 5px; max-height: 92px; overflow: auto; }
    .chip { display: inline-flex; align-items: center; max-width: 100%; border-radius: 999px; padding: 2px 7px; background: var(--accent-soft); color: #134f4b; font-family: var(--mono); font-size: 11px; overflow-wrap: anywhere; }
    .chip.base { background: #eef0f3; color: #4f5661; }
    .chip.missing { background: #fbeaea; color: var(--bad); }
    .chip.warn { background: var(--warn-bg); color: var(--warn-text); }
    .content { min-width: 0; height: 100vh; display: grid; grid-template-rows: auto minmax(0, 1fr); }
    .prompt-toolbar { background: var(--surface); border-bottom: 1px solid var(--line); padding: 12px 16px; display: grid; gap: 10px; }
    .toolbar-top { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .toolbar-title { display: grid; gap: 2px; min-width: 0; }
    .toolbar-title .small { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .sample-row { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; align-items: end; }
    .prompt-area { min-height: 0; overflow: auto; padding: 12px 16px 16px; display: grid; gap: 12px; align-content: start; }
    .status-line { display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 8px 10px; border: 1px solid var(--line); border-radius: 7px; background: var(--surface); }
    .status-line.good { border-color: #a7d3bf; }
    .status-line.bad { border-color: #e8b2b2; color: var(--bad); }
    .meta-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
    .meta-card { border: 1px solid var(--line); border-radius: 7px; background: var(--surface); padding: 9px 10px; min-width: 0; }
    .meta-card h3 { margin-bottom: 6px; }
    .prompt-card { border: 1px solid var(--line); border-radius: 8px; background: var(--surface); overflow: hidden; }
    .prompt-card-head { padding: 10px 12px; border-bottom: 1px solid var(--line); display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    .prompt-card-head span:last-child { color: var(--muted); font-size: 12px; }
    .prompt-grid { display: grid; grid-template-columns: minmax(0, .85fr) minmax(0, 1.15fr); gap: 0; }
    .prompt-section { min-width: 0; padding: 10px 12px 12px; }
    .prompt-section + .prompt-section { border-left: 1px solid var(--line); }
    .prompt-label { display: flex; justify-content: space-between; gap: 8px; color: var(--muted); font-size: 12px; margin-bottom: 5px; }
    textarea { width: 100%; min-height: 520px; resize: vertical; border: 1px solid var(--line); border-radius: 6px; background: #fcfcfd; color: var(--text); padding: 9px; font-family: var(--mono); font-size: 12px; line-height: 1.45; }
    .system-box { min-height: 320px; }
    .error { white-space: pre-wrap; color: var(--bad); font-family: var(--mono); font-size: 12px; }
    @media (max-width: 1100px) {
      .app { grid-template-columns: 1fr; }
      .sidebar, .content { min-height: auto; height: auto; }
      .sidebar { border-right: 0; border-bottom: 1px solid var(--line); }
      .feature-list { max-height: 360px; }
      .prompt-grid, .meta-grid { grid-template-columns: 1fr; }
      .prompt-section + .prompt-section { border-left: 0; border-top: 1px solid var(--line); }
      textarea { min-height: 340px; }
    }
    @media (max-width: 640px) {
      .field-row, .actions, .sample-row { grid-template-columns: 1fr; }
      .toolbar-top { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <main class="app">
    <aside class="sidebar">
      <section class="brand">
        <h1>FACET Prompt Renderer</h1>
        <div class="small">Select one benchmark and a feature vector. The rendered system and user prompts stay on the right.</div>
      </section>

      <section class="control-block">
        <label class="field small">Benchmark
          <select id="taskView">
            <option value="wtq">WTQ</option>
            <option value="sqa">SQA</option>
            <option value="tablebench">TableBench</option>
            <option value="tabfact">TabFact</option>
            <option value="hitab">HiTab</option>
          </select>
        </label>
        <div class="field-row">
          <label class="field small">Feature Family
            <select id="familyFilter"><option value="">All families</option></select>
          </label>
          <label class="field small">Search
            <input id="search" type="search" placeholder="feature id or label">
          </label>
        </div>
        <div class="actions">
          <button id="clearBtn" type="button">Clear</button>
          <button id="renderBtn" class="primary" type="button">Render</button>
        </div>
        <details class="presets">
          <summary>Preset vectors</summary>
          <div class="preset-list">
            <button type="button" data-vector="table_serialization_html,input_context_column_statistics,reasoning_scaffold_visible_cot,reasoning_extract_then_compute,reasoning_evidence_localization">HTML + stats + extract/filter</button>
            <button type="button" data-vector="table_serialization_json_records,input_context_type_annotation,reasoning_scaffold_visible_cot,reasoning_candidate_enumeration">Records + type + enumerate</button>
            <button type="button" data-vector="prompt_format_json,table_serialization_json_records,reasoning_scaffold_visible_cot">JSON prompt + records + trace</button>
          </div>
        </details>
      </section>

      <section class="feature-panel">
        <div class="feature-head">
          <h2>Features</h2>
          <span id="featureCount" class="small">loading</span>
        </div>
        <div id="featureList" class="feature-list"></div>
      </section>

      <section class="selected-dock">
        <div class="selected-head">
          <h2>Selected Vector</h2>
          <span id="selectedCount" class="small">0 selected</span>
        </div>
        <div id="vectorBox" class="chip-row"><span class="small">No optional features selected.</span></div>
        <textarea id="featureText" readonly style="min-height:64px; max-height:110px"></textarea>
      </section>
    </aside>

    <section class="content">
      <section class="prompt-toolbar">
        <div class="toolbar-top">
          <div class="toolbar-title">
            <h2 id="promptTitle">WTQ Prompt</h2>
            <div id="promptSubtitle" class="small">Fixed base features are applied automatically.</div>
          </div>
          <div class="small mono" id="renderState">idle</div>
        </div>
        <div class="sample-row">
          <label class="field small">Sample query
            <select id="sampleSelect"></select>
          </label>
          <button id="refreshBtn" type="button">Refresh Sample</button>
        </div>
      </section>

      <section id="promptArea" class="prompt-area">
        <div class="status-line">Loading feature inventory...</div>
      </section>
    </section>
  </main>

<script>
const TASKS = ["wtq", "sqa", "tablebench", "tabfact", "hitab"];
const TASK_LABEL = {wtq: "WTQ", sqa: "SQA", tablebench: "TableBench", tabfact: "TabFact", hitab: "HiTab"};
let FEATURES = [];
let FEATURE_BY_ID = new Map();
let SELECTED = new Set();
let SAMPLES = {};
let LAST_RESULTS = {};

function esc(s) { return String(s ?? '').replace(/[&<>\"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c])); }
function currentTask() { return document.getElementById('taskView').value; }
function familyOf(f) { return f.family || 'other'; }
function chip(text, cls='') { return `<span class="chip ${cls}">${esc(text)}</span>`; }
function taskHasFeature(feature, task) { return Boolean(feature?.tasks?.[task]); }
function featureHaystack(f) { return [f.canonical_id, f.family, ...(f.semantic_labels || [])].join(' ').toLowerCase(); }
function taskFeatureRows(task) { return FEATURES.filter(f => taskHasFeature(f, task)); }

async function init() {
  const inv = await fetch('/api/features').then(r => r.json());
  FEATURES = inv.features;
  FEATURE_BY_ID = new Map(FEATURES.map(f => [f.canonical_id, f]));
  populateFamilies();
  await loadSamples(currentTask());
  renderFeatureList();
  renderVector();
  renderSampleSelect();
  await renderSelectedTask();
}

async function loadSamples(task) {
  if (SAMPLES[task]) return;
  SAMPLES[task] = (await fetch(`/api/samples?task=${task}&limit=80`).then(r => r.json())).samples || [];
}

function populateFamilies() {
  const families = [...new Set(FEATURES.map(familyOf))].sort();
  const select = document.getElementById('familyFilter');
  for (const family of families) {
    const opt = document.createElement('option');
    opt.value = family;
    opt.textContent = family;
    select.appendChild(opt);
  }
}

function conflictsFor(featureId) {
  const f = FEATURE_BY_ID.get(featureId);
  const out = new Set();
  if (!f) return out;
  for (const task of TASKS) {
    const row = f.tasks?.[task];
    for (const c of (row?.conflicts_with || [])) out.add(c);
  }
  return out;
}

function conflictsWithSelected(featureId) {
  const addingConflicts = conflictsFor(featureId);
  const out = new Set([...addingConflicts].filter(id => SELECTED.has(id)));
  for (const selected of SELECTED) {
    if (conflictsFor(selected).has(featureId)) out.add(selected);
  }
  return out;
}

function selectFeature(featureId, checked, rerender=true) {
  if (!checked) {
    SELECTED.delete(featureId);
  } else {
    for (const conflict of conflictsWithSelected(featureId)) SELECTED.delete(conflict);
    SELECTED.add(featureId);
  }
  renderFeatureList();
  renderVector();
  if (rerender) renderSelectedTask();
}

function setVector(ids) {
  SELECTED.clear();
  for (const id of ids) {
    if (FEATURE_BY_ID.has(id)) selectFeature(id, true, false);
  }
  renderFeatureList();
  renderVector();
  renderSelectedTask();
}

function renderFeatureList() {
  const task = currentTask();
  const family = document.getElementById('familyFilter').value;
  const q = document.getElementById('search').value.toLowerCase().trim();
  const grouped = new Map();
  for (const f of taskFeatureRows(task)) {
    if (family && familyOf(f) !== family) continue;
    if (q && !featureHaystack(f).includes(q)) continue;
    const fam = familyOf(f);
    if (!grouped.has(fam)) grouped.set(fam, []);
    grouped.get(fam).push(f);
  }
  const count = [...grouped.values()].reduce((n, rows) => n + rows.length, 0);
  document.getElementById('featureCount').textContent = `${count} shown`;
  const html = [...grouped.entries()].map(([fam, rows]) => `
    <div class="family">
      <div class="family-title"><span>${esc(fam)}</span><span>${rows.length}</span></div>
      ${rows.map(f => featureToggleHtml(f)).join('')}
    </div>`).join('');
  document.getElementById('featureList').innerHTML = html || '<div class="small" style="padding:8px">No features match.</div>';
  document.querySelectorAll('input[data-feature]').forEach(cb => {
    cb.addEventListener('change', e => selectFeature(e.target.dataset.feature, e.target.checked));
  });
  document.getElementById('selectedCount').textContent = `${SELECTED.size} selected`;
}

function featureToggleHtml(f) {
  const selected = SELECTED.has(f.canonical_id) ? 'checked' : '';
  const labels = (f.semantic_labels || []).slice(0, 2).join(' / ');
  const note = labels || familyOf(f);
  return `<label class="feature">
    <input type="checkbox" data-feature="${esc(f.canonical_id)}" ${selected}>
    <span>
      <span class="feature-id">${esc(f.canonical_id)}</span>
      <span class="feature-note">${esc(note)}</span>
    </span>
  </label>`;
}

function renderVector() {
  const ids = [...SELECTED].sort();
  document.getElementById('selectedCount').textContent = `${ids.length} selected`;
  document.getElementById('featureText').value = ids.join(',');
  document.getElementById('vectorBox').innerHTML = ids.length ? ids.map(id => chip(id)).join('') : '<span class="small">No optional features selected.</span>';
}

function renderSampleSelect() {
  const task = currentTask();
  const select = document.getElementById('sampleSelect');
  const rows = SAMPLES[task] || [];
  select.innerHTML = rows.length
    ? rows.map(r => `<option value="${esc(r.query_id)}">${esc(r.summary || r.query_id)}</option>`).join('')
    : '<option value="">fixture/default sample</option>';
}

function selectedQueryId() {
  return document.getElementById('sampleSelect').value || '';
}

async function renderSelectedTask() {
  const task = currentTask();
  document.getElementById('renderBtn').disabled = true;
  document.getElementById('renderBtn').textContent = 'Rendering...';
  document.getElementById('renderState').textContent = 'rendering';
  try {
    const query_ids = {};
    query_ids[task] = selectedQueryId();
    const res = await fetch('/api/render', {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify({features: [...SELECTED], tasks: [task], query_ids})
    }).then(r => r.json());
    if (!res.ok) throw new Error(res.error || 'render failed');
    LAST_RESULTS[task] = res.results[task];
    renderPrompt(task, LAST_RESULTS[task]);
  } catch (err) {
    document.getElementById('promptArea').innerHTML = `<div class="status-line bad"><span class="error">${esc(err.message || String(err))}</span></div>`;
  } finally {
    document.getElementById('renderBtn').disabled = false;
    document.getElementById('renderBtn').textContent = 'Render';
    document.getElementById('renderState').textContent = 'idle';
  }
}

function renderPrompt(task, result) {
  document.getElementById('promptTitle').textContent = `${TASK_LABEL[task]} Prompt`;
  const availableSelected = [...SELECTED].filter(id => taskHasFeature(FEATURE_BY_ID.get(id), task));
  const unavailableSelected = [...SELECTED].filter(id => !taskHasFeature(FEATURE_BY_ID.get(id), task));
  document.getElementById('promptSubtitle').textContent = `${availableSelected.length} selected features available for this benchmark`;
  if (!result) {
    document.getElementById('promptArea').innerHTML = '<div class="status-line">No render yet.</div>';
    return;
  }
  if (!result.ok) {
    document.getElementById('promptArea').innerHTML = `<div class="status-line bad"><span class="error">${esc(result.error)}</span></div>`;
    return;
  }
  const base = result.base_features || [];
  const active = result.active_features || [];
  const missing = result.missing_features || unavailableSelected;
  document.getElementById('promptArea').innerHTML = `
    <div class="status-line good">
      <span><strong>${esc(result.query_id || '')}</strong></span>
      <span class="small">${active.length} toggled / ${base.length} base</span>
    </div>
    <div class="meta-grid">
      <section class="meta-card"><h3>Fixed Base</h3><div class="chip-row">${base.map(x => chip(x, 'base')).join('')}</div></section>
      <section class="meta-card"><h3>Active Toggles</h3><div class="chip-row">${active.length ? active.map(x => chip(x)).join('') : '<span class="small">none</span>'}</div></section>
      <section class="meta-card"><h3>Unavailable Here</h3><div class="chip-row">${missing.length ? missing.map(x => chip(x, 'missing')).join('') : '<span class="small">none</span>'}</div></section>
    </div>
    <article class="prompt-card">
      <div class="prompt-card-head"><h2>Rendered Prompt</h2><span>${esc((result.question || '').slice(0, 120))}</span></div>
      <div class="prompt-grid">
        <section class="prompt-section">
          <div class="prompt-label"><span>System prompt</span><span>${(result.system_prompt || '').length} chars</span></div>
          <textarea class="system-box" readonly>${esc(result.system_prompt || '')}</textarea>
        </section>
        <section class="prompt-section">
          <div class="prompt-label"><span>User prompt</span><span>${(result.user_content || '').length} chars</span></div>
          <textarea readonly>${esc(result.user_content || '')}</textarea>
        </section>
      </div>
    </article>`;
}

document.getElementById('taskView').addEventListener('change', async () => {
  await loadSamples(currentTask());
  renderSampleSelect();
  renderFeatureList();
  await renderSelectedTask();
});
document.getElementById('familyFilter').addEventListener('change', renderFeatureList);
document.getElementById('search').addEventListener('input', renderFeatureList);
document.getElementById('clearBtn').addEventListener('click', () => { SELECTED.clear(); renderFeatureList(); renderVector(); renderSelectedTask(); });
document.getElementById('renderBtn').addEventListener('click', renderSelectedTask);
document.getElementById('refreshBtn').addEventListener('click', renderSelectedTask);
document.getElementById('sampleSelect').addEventListener('change', renderSelectedTask);
document.querySelectorAll('button[data-vector]').forEach(btn => {
  btn.addEventListener('click', () => setVector(btn.dataset.vector.split(',').map(x => x.trim()).filter(Boolean)));
});

init().catch(err => {
  document.getElementById('promptArea').innerHTML = `<div class="status-line bad"><span class="error">${esc(err.message || String(err))}</span></div>`;
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
