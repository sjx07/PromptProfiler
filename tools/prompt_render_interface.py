#!/usr/bin/env python3
"""Interactive prompt renderer for FACET feature coalitions.

This serves a local HTML UI that lets you choose arbitrary feature canonical IDs
and renders the resulting system/user prompts for WTQ, SQA, TableBench, TabFact,
and HiTab. It is offline: no LLM calls, no experiment executions.

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
from tools.render_prompts_from_config import _default_render_base_features

TASKS = ["wtq", "sqa", "tablebench", "tabfact", "hitab"]
TASK_DATASET = {
    "wtq": "wtq",
    "sqa": "sqa",
    "tablebench": "tablebench",
    "tabfact": "tab_fact",
    "hitab": "hitab",
}
DEFAULT_SOURCE_DB = "/data/users/jsu323/facet/wikitable_clean_surface_v1.db"
HIDDEN_FEATURES = {"facet_dp_scaffold", "sqa_dialog_binding_base"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve an interactive prompt renderer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--source-db",
        default=DEFAULT_SOURCE_DB,
        help="Cube DB used only for sample query rows. Defaults to wikitable_clean_surface_v1.db.",
    )
    args = parser.parse_args()

    server = PromptRenderServer((args.host, args.port), source_db=Path(args.source_db))
    url = f"http://{args.host}:{args.port}"
    print(f"Prompt render interface: {url}")
    print(f"Source cube: {args.source_db}")
    server.serve_forever()


class PromptRenderServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], *, source_db: Path) -> None:
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
            self._send_json({"task": task, "samples": list_sample_queries(self.server.source_db, task, limit=limit)})
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
            query_ids = payload.get("query_ids") if isinstance(payload.get("query_ids"), dict) else {}
            results = {}
            for task in TASKS:
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
    source_db: Path,
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
        query = load_query(source_db, task, query_id=query_id)
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


def list_sample_queries(source_db: Path, task: str, *, limit: int = 40) -> List[Dict[str, Any]]:
    if task not in TASK_DATASET:
        return []
    if not source_db.exists():
        return []
    dataset = TASK_DATASET[task]
    conn = sqlite3.connect(str(source_db))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT query_id, content, meta
            FROM query
            WHERE dataset = ?
            ORDER BY query_id
            LIMIT ?
            """,
            (dataset, int(limit)),
        ).fetchall()
    finally:
        conn.close()
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


def load_query(source_db: Path, task: str, *, query_id: str | None = None) -> Dict[str, Any]:
    if not source_db.exists():
        raise FileNotFoundError(f"source cube not found: {source_db}")
    dataset = TASK_DATASET[task]
    conn = sqlite3.connect(str(source_db))
    conn.row_factory = sqlite3.Row
    try:
        if query_id:
            row = conn.execute(
                "SELECT query_id, dataset, content, meta FROM query WHERE dataset = ? AND query_id = ?",
                (dataset, query_id),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT query_id, dataset, content, meta FROM query WHERE dataset = ? ORDER BY query_id LIMIT 1",
                (dataset,),
            ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise ValueError(f"no query found for dataset={dataset!r} query_id={query_id!r}")
    return {
        "query_id": row["query_id"],
        "dataset": row["dataset"],
        "content": row["content"],
        "meta": row["meta"],
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
      --bg: #f7f7f4;
      --panel: #ffffff;
      --line: #d7d8d0;
      --text: #20231f;
      --muted: #666c63;
      --accent: #1f6f68;
      --accent-soft: #dbece8;
      --bad: #9b2c2c;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: var(--sans); color: var(--text); background: var(--bg); }
    header { padding: 14px 18px; border-bottom: 1px solid var(--line); background: #fff; display: flex; gap: 12px; align-items: center; justify-content: space-between; }
    h1 { font-size: 18px; margin: 0; font-weight: 650; }
    button { border: 1px solid #155e57; background: var(--accent); color: #fff; border-radius: 6px; padding: 8px 12px; cursor: pointer; font-weight: 600; }
    button.secondary { background: #fff; color: var(--text); border-color: var(--line); }
    main { display: grid; grid-template-columns: 390px 1fr; gap: 12px; padding: 12px; min-height: calc(100vh - 58px); }
    aside, section.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; min-width: 0; }
    aside { display: flex; flex-direction: column; max-height: calc(100vh - 82px); }
    .controls { padding: 12px; border-bottom: 1px solid var(--line); display: grid; gap: 8px; }
    input[type="search"], select { width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 8px; font: inherit; background: #fff; }
    .feature-list { overflow: auto; padding: 8px; }
    .family { margin-bottom: 10px; }
    .family-title { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; padding: 8px 6px 4px; }
    label.feature { display: grid; grid-template-columns: 20px 1fr; gap: 6px; padding: 6px; border-radius: 6px; cursor: pointer; }
    label.feature:hover { background: #f0f3ef; }
    .feature-name { font-family: var(--mono); font-size: 12px; overflow-wrap: anywhere; }
    .task-dots { margin-top: 4px; display: flex; gap: 4px; flex-wrap: wrap; }
    .dot { font-size: 10px; border: 1px solid var(--line); border-radius: 999px; padding: 1px 5px; color: var(--muted); }
    .selected-bar { display: flex; gap: 8px; align-items: center; color: var(--muted); font-size: 13px; }
    .results { display: grid; grid-template-columns: repeat(5, minmax(260px, 1fr)); gap: 12px; align-items: start; }
    .task-card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; min-width: 0; overflow: hidden; }
    .task-head { padding: 10px; border-bottom: 1px solid var(--line); display: grid; gap: 8px; }
    .task-title { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
    .task-title h2 { margin: 0; font-size: 15px; text-transform: uppercase; letter-spacing: .02em; }
    .status { color: var(--muted); font-size: 12px; }
    .status.bad { color: var(--bad); }
    .prompt-block { padding: 10px; display: grid; gap: 10px; }
    .prompt-label { font-size: 12px; color: var(--muted); display: flex; justify-content: space-between; }
    textarea { width: 100%; min-height: 180px; resize: vertical; border: 1px solid var(--line); border-radius: 6px; padding: 8px; font-family: var(--mono); font-size: 12px; line-height: 1.45; background: #fcfcfb; color: var(--text); }
    .user textarea { min-height: 240px; }
    .chips { display: flex; gap: 4px; flex-wrap: wrap; }
    .chip { background: var(--accent-soft); color: #164e49; border-radius: 999px; padding: 2px 6px; font-size: 11px; font-family: var(--mono); }
    .error { color: var(--bad); font-family: var(--mono); white-space: pre-wrap; font-size: 12px; padding: 10px; }
    @media (max-width: 1400px) { .results { grid-template-columns: repeat(2, minmax(320px, 1fr)); } }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } aside { max-height: none; } .results { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>
    <h1>FACET Prompt Renderer</h1>
    <div class="selected-bar"><span id="selectedCount">0 selected</span><button id="renderBtn">Render All Five</button></div>
  </header>
  <main>
    <aside>
      <div class="controls">
        <input id="search" type="search" placeholder="Filter features">
        <div style="display:flex; gap:8px;"><button class="secondary" id="clearBtn" type="button">Clear</button><button class="secondary" id="selectReasonBtn" type="button">Reasoning Trace</button></div>
      </div>
      <div id="featureList" class="feature-list"></div>
    </aside>
    <section class="panel" style="padding:12px; overflow:auto;">
      <div class="results" id="results"></div>
    </section>
  </main>
<script>
const TASKS = ["wtq", "sqa", "tablebench", "tabfact", "hitab"];
let FEATURES = [];
let SELECTED = new Set();
let SAMPLES = {};

function esc(s) { return String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function familyOf(f) { return f.family || 'other'; }

async function init() {
  const inv = await fetch('/api/features').then(r => r.json());
  FEATURES = inv.features;
  for (const task of TASKS) {
    SAMPLES[task] = (await fetch(`/api/samples?task=${task}&limit=80`).then(r => r.json())).samples || [];
  }
  renderFeatureList();
  renderEmptyCards();
  updateSelectedCount();
}

function renderFeatureList() {
  const q = document.getElementById('search').value.toLowerCase().trim();
  const grouped = new Map();
  for (const f of FEATURES) {
    const hay = [f.canonical_id, f.family, ...(f.semantic_labels || [])].join(' ').toLowerCase();
    if (q && !hay.includes(q)) continue;
    const fam = familyOf(f);
    if (!grouped.has(fam)) grouped.set(fam, []);
    grouped.get(fam).push(f);
  }
  const html = [...grouped.entries()].map(([fam, rows]) => `
    <div class="family">
      <div class="family-title">${esc(fam)}</div>
      ${rows.map(f => `
        <label class="feature">
          <input type="checkbox" data-feature="${esc(f.canonical_id)}" ${SELECTED.has(f.canonical_id) ? 'checked' : ''}>
          <span>
            <span class="feature-name">${esc(f.canonical_id)}</span>
            <span class="task-dots">${TASKS.map(t => `<span class="dot" style="opacity:${f.tasks[t] ? 1 : .25}">${t}</span>`).join('')}</span>
          </span>
        </label>`).join('')}
    </div>`).join('');
  document.getElementById('featureList').innerHTML = html || '<div class="status">No features match.</div>';
  document.querySelectorAll('input[data-feature]').forEach(cb => cb.addEventListener('change', e => {
    const id = e.target.dataset.feature;
    if (e.target.checked) SELECTED.add(id); else SELECTED.delete(id);
    updateSelectedCount();
  }));
}

function updateSelectedCount() {
  document.getElementById('selectedCount').textContent = `${SELECTED.size} selected`;
}

function renderEmptyCards() {
  document.getElementById('results').innerHTML = TASKS.map(task => cardHtml(task, null)).join('');
  wireSampleSelectors();
}

function sampleOptions(task, selectedId) {
  const rows = SAMPLES[task] || [];
  if (!rows.length) return '<option value="">No samples in source cube</option>';
  return rows.map(r => `<option value="${esc(r.query_id)}" ${r.query_id === selectedId ? 'selected' : ''}>${esc(r.summary || r.query_id)}</option>`).join('');
}

function cardHtml(task, result) {
  const ok = result && result.ok;
  const selectedId = result?.query_id || (SAMPLES[task]?.[0]?.query_id || '');
  const status = result ? (ok ? `${result.active_features.length} active, ${result.missing_features.length} unavailable` : 'error') : 'ready';
  return `<div class="task-card" id="card-${task}">
    <div class="task-head">
      <div class="task-title"><h2>${task}</h2><span class="status ${ok === false ? 'bad' : ''}">${esc(status)}</span></div>
      <select data-task-sample="${task}">${sampleOptions(task, selectedId)}</select>
      ${ok ? `<div class="chips">${result.active_features.map(x => `<span class="chip">${esc(x)}</span>`).join('')}</div>` : ''}
    </div>
    ${result && !ok ? `<div class="error">${esc(result.error)}\n\nMissing here: ${(result.missing_features || []).map(esc).join(', ')}</div>` : `
    <div class="prompt-block">
      <div><div class="prompt-label"><span>System Prompt</span><span>${esc(result?.query_id || '')}</span></div><textarea readonly>${esc(result?.system_prompt || '')}</textarea></div>
      <div class="user"><div class="prompt-label"><span>User Prompt</span></div><textarea readonly>${esc(result?.user_content || '')}</textarea></div>
    </div>`}
  </div>`;
}

function wireSampleSelectors() {
  document.querySelectorAll('select[data-task-sample]').forEach(sel => {
    sel.addEventListener('change', () => {});
  });
}

async function renderAll() {
  const query_ids = {};
  document.querySelectorAll('select[data-task-sample]').forEach(sel => { query_ids[sel.dataset.taskSample] = sel.value; });
  document.getElementById('renderBtn').disabled = true;
  document.getElementById('renderBtn').textContent = 'Rendering...';
  try {
    const res = await fetch('/api/render', {
      method: 'POST',
      headers: {'content-type': 'application/json'},
      body: JSON.stringify({features: [...SELECTED], query_ids})
    }).then(r => r.json());
    if (!res.ok) throw new Error(res.error || 'render failed');
    document.getElementById('results').innerHTML = TASKS.map(task => cardHtml(task, res.results[task])).join('');
    wireSampleSelectors();
  } catch (err) {
    alert(err.message || String(err));
  } finally {
    document.getElementById('renderBtn').disabled = false;
    document.getElementById('renderBtn').textContent = 'Render All Five';
  }
}

document.getElementById('search').addEventListener('input', renderFeatureList);
document.getElementById('clearBtn').addEventListener('click', () => { SELECTED.clear(); renderFeatureList(); updateSelectedCount(); });
document.getElementById('selectReasonBtn').addEventListener('click', () => { SELECTED.add('reasoning_scaffold_visible_cot'); renderFeatureList(); updateSelectedCount(); });
document.getElementById('renderBtn').addEventListener('click', renderAll);
init().catch(err => alert(err.message || String(err)));
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
