#!/usr/bin/env python3
"""Small read-only web UI for inspecting prompt-profiler cubes.

The server has no third-party dependencies. It serves a single static HTML
workspace plus JSON endpoints backed by ``analyze.cube_ops``.

Example:
    python cube_visualizer.py \\
        --cube /data/users/jsu323/facet/tablebench_facet_round5_official_json_full_qwen25_coder32b.db \\
        --port 8765
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.store import CubeStore  # noqa: E402
from analyze import cube_ops  # noqa: E402


class CubeApp:
    def __init__(self, cube_path: str) -> None:
        path = Path(cube_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"cube does not exist: {path}")
        self.cube_path = str(path)
        self.store = CubeStore(self.cube_path, read_only=True)
        self.lock = threading.Lock()

    def close(self) -> None:
        self.store.close()


def make_handler(app: CubeApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "CubeVisualizer/0.1"

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(INDEX_HTML)
                return
            if parsed.path == "/api/summary":
                self._api(lambda _body, _qs: {
                    "cubePath": app.cube_path,
                    "summary": cube_ops.cube_summary(app.store),
                })
                return
            if parsed.path == "/api/configs":
                self._api(lambda _body, qs: cube_ops.list_configs_detailed(
                    app.store,
                    model=_first_qs(qs, "model"),
                    scorer=_first_qs(qs, "scorer"),
                ))
                return
            if parsed.path == "/api/meta-fields":
                self._api(lambda _body, _qs: cube_ops.list_query_meta_fields(app.store))
                return
            if parsed.path == "/api/artifact":
                self._api(lambda _body, qs: _not_none(
                    cube_ops.execution_artifact(
                        app.store,
                        execution_id=int(_first_qs(qs, "executionId") or "0"),
                    ),
                    "execution not found",
                ))
                return
            self.send_error(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            routes: Dict[str, Callable[[Dict[str, Any], Dict[str, list]], Any]] = {
                "/api/slices": _post_slices,
                "/api/examples": _post_examples,
                "/api/compare": _post_compare,
                "/api/compare-examples": _post_compare_examples,
                "/api/diagnostics": _post_diagnostics,
                "/api/plan-delete": _post_plan_delete,
            }
            route = routes.get(parsed.path)
            if route is None:
                self.send_error(HTTPStatus.NOT_FOUND, "not found")
                return
            self._api(route)

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("%s - - [%s] %s\n" % (
                self.client_address[0],
                self.log_date_time_string(),
                fmt % args,
            ))

        def _api(self, fn: Callable[[Dict[str, Any], Dict[str, list]], Any]) -> None:
            parsed = urlparse(self.path)
            try:
                body = self._read_json_body()
                qs = parse_qs(parsed.query)
                with app.lock:
                    payload = fn(body, qs)
                self._send_json({"ok": True, "data": payload})
            except ValueError as e:
                self._send_json(
                    {"ok": False, "error": {"code": "BAD_REQUEST", "message": str(e)}},
                    status=HTTPStatus.BAD_REQUEST,
                )
            except Exception as e:  # pragma: no cover - operational safety
                traceback.print_exc()
                self._send_json(
                    {"ok": False, "error": {"code": "SERVER_ERROR", "message": str(e)}},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

        def _read_json_body(self) -> Dict[str, Any]:
            if self.command != "POST":
                return {}
            n = int(self.headers.get("Content-Length", "0") or "0")
            if n == 0:
                return {}
            raw = self.rfile.read(n).decode("utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("JSON body must be an object")
            return data

        def _send_json(self, payload: Dict[str, Any], *, status: int = HTTPStatus.OK) -> None:
            data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_html(self, html: str) -> None:
            data = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def _post_slices(body: Dict[str, Any], _qs: Dict[str, list]) -> Any:
        return cube_ops.slice_scores(
            app.store,
            model=_required(body, "model"),
            scorer=_required(body, "scorer"),
            config_ids=_int_list(body.get("configIds")),
            group_by=body.get("groupBy") or [],
            filters=body.get("filters") or [],
            base_config_id=_maybe_int(body.get("baseConfigId")),
            limit=int(body.get("limit") or 500),
        )

    def _post_examples(body: Dict[str, Any], _qs: Dict[str, list]) -> Any:
        return cube_ops.examples(
            app.store,
            model=_required(body, "model"),
            scorer=_required(body, "scorer"),
            config_ids=_int_list(body.get("configIds")),
            filters=body.get("filters") or [],
            score_order=str(body.get("scoreOrder") or "asc"),
            limit=int(body.get("limit") or 100),
        )

    def _post_compare(body: Dict[str, Any], _qs: Dict[str, list]) -> Any:
        return cube_ops.compare_configs(
            app.store,
            model=_required(body, "model"),
            scorer=_required(body, "scorer"),
            base_config_id=int(_required(body, "baseConfigId")),
            target_config_id=int(_required(body, "targetConfigId")),
            filters=body.get("filters") or [],
        )

    def _post_compare_examples(body: Dict[str, Any], _qs: Dict[str, list]) -> Any:
        return cube_ops.comparison_examples(
            app.store,
            model=_required(body, "model"),
            scorer=_required(body, "scorer"),
            base_config_id=int(_required(body, "baseConfigId")),
            target_config_id=int(_required(body, "targetConfigId")),
            direction=str(body.get("direction") or "both"),
            filters=body.get("filters") or [],
            limit=int(body.get("limit") or 100),
        )

    def _post_diagnostics(body: Dict[str, Any], _qs: Dict[str, list]) -> Any:
        return cube_ops.diagnostics(
            app.store,
            model=_required(body, "model"),
            scorer=_required(body, "scorer"),
            config_ids=_int_list(body.get("configIds")),
            filters=body.get("filters") or [],
            limit=int(body.get("limit") or 5000),
        )

    def _post_plan_delete(body: Dict[str, Any], _qs: Dict[str, list]) -> Any:
        return cube_ops.plan_delete(
            app.store,
            model=body.get("model"),
            config_ids=_int_list(body.get("configIds")),
            filters=body.get("filters") or [],
        )

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cube", required=True, help="Path to cube SQLite DB.")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind host. Default 0.0.0.0 works best with VS Code port forwarding.",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the UI in a browser.")
    args = parser.parse_args()

    app = CubeApp(args.cube)
    counts = app.store.stats()
    handler = make_handler(app)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    bind_url = f"http://{args.host}:{args.port}/"
    browser_host = "localhost" if args.host in {"0.0.0.0", "::"} else args.host
    browser_url = f"http://{browser_host}:{args.port}/"
    print(f"Cube visualizer serving {app.cube_path}", file=sys.stderr)
    print(
        "Cube counts: "
        f"configs={counts.get('config', 0)} "
        f"queries={counts.get('query', 0)} "
        f"executions={counts.get('execution', 0)} "
        f"evaluations={counts.get('evaluation', 0)}",
        file=sys.stderr,
    )
    if not counts.get("config") or not counts.get("query"):
        print(
            "WARNING: cube appears empty. Check that --cube points to the populated .db file.",
            file=sys.stderr,
        )
    print(f"Bind URL: {bind_url}", file=sys.stderr)
    print(f"Browser URL: {browser_url}", file=sys.stderr)
    print("VS Code: forward this port from the Ports panel if it is not detected automatically.", file=sys.stderr)
    if args.open:
        webbrowser.open(browser_url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.close()
    return 0


def _required(body: Dict[str, Any], key: str) -> Any:
    value = body.get(key)
    if value is None or value == "":
        raise ValueError(f"missing required field: {key}")
    return value


def _maybe_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _int_list(value: Any) -> Optional[list[int]]:
    if value is None:
        return None
    if isinstance(value, str):
        value = [v.strip() for v in value.split(",") if v.strip()]
    return [int(v) for v in value]


def _first_qs(qs: Dict[str, list], key: str) -> Optional[str]:
    values = qs.get(key)
    if not values:
        return None
    return values[0] or None


def _not_none(value: Any, message: str) -> Any:
    if value is None:
        raise ValueError(message)
    return value


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Cube Visualizer</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --surface: #ffffff;
      --surface-2: #eef1f5;
      --text: #18202a;
      --muted: #667085;
      --border: #d6dbe3;
      --accent: #1f6f5b;
      --accent-2: #164c63;
      --bad: #b42318;
      --good: #087443;
      --warn: #986f0b;
      --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      font-size: 13px;
      letter-spacing: 0;
    }
    button, input, select, textarea {
      font: inherit;
    }
    button {
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--text);
      border-radius: 6px;
      padding: 6px 10px;
      cursor: pointer;
      min-height: 32px;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    button:disabled { opacity: .55; cursor: not-allowed; }
    select, input {
      width: 100%;
      min-height: 32px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
      padding: 5px 7px;
      color: var(--text);
    }
    select[multiple] {
      height: 132px;
      padding: 4px;
    }
    .app {
      display: grid;
      grid-template-columns: 286px minmax(780px, 1fr);
      min-height: 100vh;
    }
    .sidebar {
      border-right: 1px solid var(--border);
      background: var(--surface);
      padding: 12px;
      overflow: auto;
    }
    .main {
      display: grid;
      grid-template-rows: auto minmax(280px, 43vh) minmax(300px, 1fr);
      gap: 10px;
      padding: 12px;
      overflow: hidden;
    }
    h1 {
      font-size: 16px;
      line-height: 1.2;
      margin: 0 0 10px;
      font-weight: 650;
    }
    h2 {
      font-size: 13px;
      margin: 0 0 8px;
      font-weight: 650;
    }
    label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }
    .field { margin-bottom: 10px; }
    .row {
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .row > * { flex: 1; }
    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 7px;
      overflow: hidden;
      min-height: 0;
    }
    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      min-height: 38px;
      padding: 8px 10px;
      border-bottom: 1px solid var(--border);
      background: #fbfcfd;
    }
    .panel-body {
      height: calc(100% - 39px);
      overflow: auto;
    }
    .top-grid {
      display: grid;
      grid-template-columns: minmax(420px, 1.3fr) minmax(260px, .7fr);
      gap: 10px;
      min-height: 205px;
    }
    .mid-grid {
      display: grid;
      grid-template-columns: minmax(520px, 1.2fr) minmax(360px, .8fr);
      gap: 10px;
      min-height: 0;
    }
    .bottom-grid {
      display: grid;
      grid-template-columns: minmax(430px, .95fr) minmax(520px, 1.05fr);
      gap: 10px;
      min-height: 0;
    }
    table {
      border-collapse: collapse;
      width: 100%;
      table-layout: fixed;
    }
    th, td {
      border-bottom: 1px solid var(--surface-2);
      padding: 6px 8px;
      vertical-align: top;
      text-align: left;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f9fafb;
      color: #475467;
      font-weight: 650;
    }
    tr.selectable { cursor: pointer; }
    tr.selectable:hover td { background: #eef7f4; }
    tr.selected td { background: #dff3ed; }
    .metric-strip {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      padding: 10px;
    }
    .metric {
      border: 1px solid var(--border);
      border-radius: 7px;
      padding: 8px;
      background: #fff;
      min-height: 54px;
    }
    .metric .k { color: var(--muted); font-size: 11px; }
    .metric .v { font-size: 18px; font-weight: 700; margin-top: 2px; }
    .muted { color: var(--muted); }
    .mono { font-family: var(--mono); }
    .good { color: var(--good); }
    .bad { color: var(--bad); }
    .warn { color: var(--warn); }
    .tabs {
      display: flex;
      gap: 4px;
      padding: 6px;
      border-bottom: 1px solid var(--border);
      background: #fbfcfd;
    }
    .tabs button {
      min-height: 28px;
      padding: 4px 8px;
    }
    .tabs button.active {
      background: var(--accent-2);
      border-color: var(--accent-2);
      color: #fff;
    }
    pre {
      margin: 0;
      padding: 10px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.45;
    }
    .status {
      min-height: 28px;
      color: var(--muted);
      font-size: 12px;
      padding-top: 6px;
    }
    .pill {
      display: inline-block;
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 2px 7px;
      margin: 1px 3px 1px 0;
      background: #fff;
      font-size: 11px;
      color: #344054;
    }
    .split-actions {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }
    @media (max-width: 1100px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { border-right: 0; border-bottom: 1px solid var(--border); }
      .main { grid-template-rows: auto auto auto; overflow: visible; }
      .top-grid, .mid-grid, .bottom-grid { grid-template-columns: 1fr; }
      .panel-body { max-height: 420px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <h1>Cube Visualizer</h1>
      <div class="field">
        <label>Cube</label>
        <div id="cubePath" class="mono muted"></div>
      </div>
      <div class="field">
        <label>Model</label>
        <select id="modelSelect"></select>
      </div>
      <div class="field">
        <label>Scorer</label>
        <select id="scorerSelect"></select>
      </div>
      <div class="field">
        <label>Configs</label>
        <select id="configSelect" multiple></select>
      </div>
      <div class="row">
        <div class="field">
          <label>Base</label>
          <select id="baseSelect"></select>
        </div>
        <div class="field">
          <label>Target</label>
          <select id="targetSelect"></select>
        </div>
      </div>
      <div class="field">
        <label>Group by</label>
        <select id="groupBySelect" multiple></select>
      </div>
      <div class="split-actions">
        <button class="primary" id="refreshBtn">Refresh</button>
        <button id="sliceBtn">Slice</button>
        <button id="compareBtn">Compare</button>
        <button id="diagBtn">Diagnostics</button>
        <button id="planBtn">Plan Delete</button>
      </div>
      <div id="status" class="status"></div>
    </aside>

    <main class="main">
      <section class="top-grid">
        <div class="panel">
          <div class="panel-head"><h2>Configs</h2><span id="configCount" class="muted"></span></div>
          <div class="panel-body"><table id="configsTable"></table></div>
        </div>
        <div class="panel">
          <div class="panel-head"><h2>Summary</h2></div>
          <div class="panel-body">
            <div id="summaryMetrics" class="metric-strip"></div>
            <pre id="summaryText"></pre>
          </div>
        </div>
      </section>

      <section class="mid-grid">
        <div class="panel">
          <div class="panel-head"><h2>Slices</h2><span id="sliceCount" class="muted"></span></div>
          <div class="panel-body"><table id="slicesTable"></table></div>
        </div>
        <div class="panel">
          <div class="panel-head"><h2>Compare / Diagnostics</h2></div>
          <div class="panel-body" id="analysisPane"><pre></pre></div>
        </div>
      </section>

      <section class="bottom-grid">
        <div class="panel">
          <div class="panel-head"><h2>Examples</h2><span id="exampleCount" class="muted"></span></div>
          <div class="panel-body"><table id="examplesTable"></table></div>
        </div>
        <div class="panel">
          <div class="panel-head"><h2>Artifact</h2><span id="artifactTitle" class="muted"></span></div>
          <div class="tabs" id="artifactTabs"></div>
          <div class="panel-body"><pre id="artifactPane"></pre></div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const state = {
      summary: null,
      configs: [],
      metaFields: [],
      selectedSlice: null,
      selectedExampleId: null,
      artifact: null,
      artifactTab: 'rawResponse'
    };

    const $ = (id) => document.getElementById(id);

    function setStatus(text, kind) {
      const el = $('status');
      el.textContent = text || '';
      el.className = 'status ' + (kind || '');
    }

    function fmtScore(v) {
      return v === null || v === undefined || Number.isNaN(v) ? '' : Number(v).toFixed(3);
    }

    function fmtDelta(v) {
      if (v === null || v === undefined || Number.isNaN(v)) return '';
      const n = Number(v);
      return (n > 0 ? '+' : '') + n.toFixed(3);
    }

    function esc(value) {
      return String(value ?? '').replace(/[&<>"']/g, ch => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
      }[ch]));
    }

    async function api(path, options = {}) {
      const response = await fetch(path, {
        method: options.body ? 'POST' : 'GET',
        headers: options.body ? {'Content-Type': 'application/json'} : {},
        body: options.body ? JSON.stringify(options.body) : undefined
      });
      const payload = await response.json();
      if (!payload.ok) {
        throw new Error(payload.error?.message || 'request failed');
      }
      return payload.data;
    }

    function selectedValues(selectId) {
      return Array.from($(selectId).selectedOptions).map(o => o.value).filter(Boolean);
    }

    function selectedConfigIds() {
      return selectedValues('configSelect').map(Number);
    }

    function currentScope() {
      return {
        model: $('modelSelect').value,
        scorer: $('scorerSelect').value,
        configIds: selectedConfigIds(),
        baseConfigId: Number($('baseSelect').value || 0) || null,
        targetConfigId: Number($('targetSelect').value || 0) || null,
        groupBy: selectedValues('groupBySelect')
      };
    }

    function option(label, value, selected = false) {
      return `<option value="${esc(value)}"${selected ? ' selected' : ''}>${esc(label)}</option>`;
    }

    async function loadBoot() {
      setStatus('Loading cube...');
      const data = await api('/api/summary');
      state.summary = data.summary;
      $('cubePath').textContent = data.cubePath;
      populateScopeControls();
      state.metaFields = await api('/api/meta-fields');
      populateGroupBy();
      await refreshConfigs();
      renderSummary();
      setStatus('Ready');
    }

    function populateScopeControls() {
      const models = state.summary.models || [];
      const scorers = state.summary.scorers || [];
      $('modelSelect').innerHTML = models.map((r, i) => option(`${r.model} (${r.n_executions})`, r.model, i === 0)).join('');
      $('scorerSelect').innerHTML = scorers.map((r, i) => option(`${r.scorer} (${r.n_evaluations})`, r.scorer, i === 0)).join('');
    }

    function populateGroupBy() {
      const preferred = ['query.meta.qtype', 'query.meta.qsubtype'];
      const fields = [
        {field: 'dataset', key: 'dataset'},
        ...state.metaFields
      ];
      $('groupBySelect').innerHTML = fields.map(f =>
        option(f.field, f.field, preferred.includes(f.field))
      ).join('');
    }

    async function refreshConfigs() {
      const scope = currentScope();
      const q = new URLSearchParams({model: scope.model || '', scorer: scope.scorer || ''});
      state.configs = await api('/api/configs?' + q.toString());
      renderConfigControls();
      renderConfigsTable();
    }

    function renderConfigControls() {
      const rows = state.configs;
      const selectedDefaults = new Set(rows.map(r => r.configId));
      $('configSelect').innerHTML = rows.map(r => {
        const label = `${r.configId} ${r.canonicalId || r.kind || ''} score=${fmtScore(r.avgScore)}`;
        return option(label, r.configId, selectedDefaults.has(r.configId));
      }).join('');
      $('baseSelect').innerHTML = rows.map((r, i) => option(`${r.configId} ${r.canonicalId || ''}`, r.configId, i === 0)).join('');
      $('targetSelect').innerHTML = rows.map((r, i) => option(`${r.configId} ${r.canonicalId || ''}`, r.configId, i === Math.min(1, rows.length - 1))).join('');
      $('configCount').textContent = `${rows.length} configs`;
    }

    function renderSummary() {
      const counts = state.summary.counts || {};
      const metrics = [
        ['Configs', counts.config],
        ['Queries', counts.query],
        ['Executions', counts.execution],
        ['Evaluations', counts.evaluation]
      ];
      $('summaryMetrics').innerHTML = metrics.map(([k, v]) =>
        `<div class="metric"><div class="k">${esc(k)}</div><div class="v">${esc(v ?? 0)}</div></div>`
      ).join('');
      const brief = {
        datasets: state.summary.datasets,
        phases: state.summary.phases?.slice(0, 8),
        tasks: state.summary.tasks
      };
      $('summaryText').textContent = JSON.stringify(brief, null, 2);
    }

    function renderConfigsTable() {
      const rows = state.configs;
      $('configsTable').innerHTML = `
        <thead><tr>
          <th style="width:64px">id</th><th>canonical</th><th>kind</th>
          <th style="width:72px">exec</th><th style="width:72px">eval</th>
          <th style="width:82px">score</th>
        </tr></thead>
        <tbody>
          ${rows.map(r => `<tr>
            <td class="mono">${r.configId}</td>
            <td title="${esc((r.resolvedCanonicalIds || []).join(', '))}">${esc(r.canonicalId || (r.resolvedCanonicalIds || []).join(', '))}</td>
            <td>${esc(r.kind || '')}</td>
            <td>${r.nExecutions}</td>
            <td>${r.nEvaluations}</td>
            <td>${fmtScore(r.avgScore)}</td>
          </tr>`).join('')}
        </tbody>`;
    }

    async function runSlices() {
      const scope = currentScope();
      if (!scope.model || !scope.scorer) return;
      setStatus('Aggregating slices...');
      const rows = await api('/api/slices', {
        body: {
          model: scope.model,
          scorer: scope.scorer,
          configIds: scope.configIds,
          baseConfigId: scope.baseConfigId,
          groupBy: scope.groupBy,
          limit: 1000
        }
      });
      renderSlices(rows);
      setStatus(`Loaded ${rows.length} slice rows`);
    }

    function groupLabel(group) {
      const parts = Object.entries(group || {}).map(([k, v]) => `${k.replace('query.meta.', '')}=${v ?? '(null)'}`);
      return parts.join(' | ') || '(overall)';
    }

    function renderSlices(rows) {
      $('sliceCount').textContent = `${rows.length} rows`;
      $('slicesTable').innerHTML = `
        <thead><tr>
          <th>slice</th><th style="width:64px">cfg</th><th style="width:64px">n</th>
          <th style="width:82px">score</th><th style="width:82px">delta</th>
        </tr></thead>
        <tbody>
          ${rows.map((r, idx) => {
            const delta = r.deltaVsBase;
            const cls = delta > 0 ? 'good' : delta < 0 ? 'bad' : '';
            return `<tr class="selectable" data-slice="${idx}">
              <td title="${esc(groupLabel(r.group))}">${esc(groupLabel(r.group))}</td>
              <td class="mono">${r.configId}</td>
              <td>${r.n}</td>
              <td>${fmtScore(r.avgScore)}</td>
              <td class="${cls}">${fmtDelta(delta)}</td>
            </tr>`;
          }).join('')}
        </tbody>`;
      $('slicesTable').querySelectorAll('tr[data-slice]').forEach(tr => {
        tr.addEventListener('click', () => {
          const idx = Number(tr.dataset.slice);
          state.selectedSlice = rows[idx];
          $('slicesTable').querySelectorAll('tr').forEach(x => x.classList.remove('selected'));
          tr.classList.add('selected');
          runExamplesForSlice(rows[idx]);
        });
      });
    }

    function filtersFromGroup(group) {
      return Object.entries(group || {})
        .filter(([field, value]) => field && value !== null && value !== undefined)
        .map(([field, value]) => ({field, op: '=', value}));
    }

    async function runExamplesForSlice(slice) {
      const scope = currentScope();
      setStatus('Loading examples...');
      const rows = await api('/api/examples', {
        body: {
          model: scope.model,
          scorer: scope.scorer,
          configIds: [slice.configId],
          filters: filtersFromGroup(slice.group),
          scoreOrder: 'asc',
          limit: 100
        }
      });
      renderExamples(rows);
      setStatus(`Loaded ${rows.length} examples`);
    }

    function renderExamples(rows) {
      $('exampleCount').textContent = `${rows.length} rows`;
      $('examplesTable').innerHTML = `
        <thead><tr>
          <th style="width:86px">exec</th><th style="width:64px">score</th>
          <th>question</th><th>prediction</th>
        </tr></thead>
        <tbody>
          ${rows.map(r => `<tr class="selectable" data-execution="${r.executionId}">
            <td class="mono">${r.executionId}</td>
            <td>${fmtScore(r.score)}</td>
            <td title="${esc(r.question)}">${esc(r.question)}</td>
            <td title="${esc(r.prediction)}">${esc(r.prediction)}</td>
          </tr>`).join('')}
        </tbody>`;
      $('examplesTable').querySelectorAll('tr[data-execution]').forEach(tr => {
        tr.addEventListener('click', () => {
          $('examplesTable').querySelectorAll('tr').forEach(x => x.classList.remove('selected'));
          tr.classList.add('selected');
          loadArtifact(Number(tr.dataset.execution));
        });
      });
    }

    async function loadArtifact(executionId) {
      const artifact = await api('/api/artifact?executionId=' + encodeURIComponent(executionId));
      state.artifact = artifact;
      state.artifactTab = 'rawResponse';
      renderArtifact();
    }

    function renderArtifact() {
      const a = state.artifact;
      if (!a) {
        $('artifactTitle').textContent = '';
        $('artifactTabs').innerHTML = '';
        $('artifactPane').textContent = '';
        return;
      }
      $('artifactTitle').textContent = `exec ${a.executionId} · cfg ${a.configId} · score ${fmtScore(a.score)}`;
      const tabs = [
        ['rawResponse', 'Raw'],
        ['prediction', 'Prediction'],
        ['systemPrompt', 'System'],
        ['userContent', 'User'],
        ['queryMeta', 'Query'],
        ['metrics', 'Metrics']
      ];
      $('artifactTabs').innerHTML = tabs.map(([key, label]) =>
        `<button data-tab="${key}" class="${state.artifactTab === key ? 'active' : ''}">${label}</button>`
      ).join('');
      $('artifactTabs').querySelectorAll('button[data-tab]').forEach(btn => {
        btn.addEventListener('click', () => {
          state.artifactTab = btn.dataset.tab;
          renderArtifact();
        });
      });
      const key = state.artifactTab;
      let value = a[key];
      if (key === 'queryMeta') value = {question: a.question, gold: a.gold, queryMeta: a.queryMeta};
      if (key === 'metrics') value = {metrics: a.metrics, error: a.error, latencyMs: a.latencyMs, promptTokens: a.promptTokens, completionTokens: a.completionTokens};
      $('artifactPane').textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
    }

    async function runCompare() {
      const scope = currentScope();
      if (!scope.baseConfigId || !scope.targetConfigId) return;
      setStatus('Comparing configs...');
      const summary = await api('/api/compare', {
        body: {
          model: scope.model,
          scorer: scope.scorer,
          baseConfigId: scope.baseConfigId,
          targetConfigId: scope.targetConfigId
        }
      });
      const rows = await api('/api/compare-examples', {
        body: {
          model: scope.model,
          scorer: scope.scorer,
          baseConfigId: scope.baseConfigId,
          targetConfigId: scope.targetConfigId,
          direction: 'both',
          limit: 80
        }
      });
      $('analysisPane').innerHTML = `
        <div class="metric-strip">
          <div class="metric"><div class="k">shared</div><div class="v">${summary.nShared}</div></div>
          <div class="metric"><div class="k">base</div><div class="v">${fmtScore(summary.avgBase)}</div></div>
          <div class="metric"><div class="k">target</div><div class="v">${fmtScore(summary.avgTarget)}</div></div>
          <div class="metric"><div class="k">delta</div><div class="v">${fmtDelta(summary.avgDelta)}</div></div>
        </div>
        <table>
          <thead><tr><th style="width:68px">dir</th><th style="width:82px">base</th><th style="width:82px">target</th><th>question</th><th>prediction</th></tr></thead>
          <tbody>${rows.map(r => `<tr class="selectable" data-execution="${r.targetExecutionId}">
            <td class="${r.direction === 'up' ? 'good' : r.direction === 'down' ? 'bad' : ''}">${r.direction}</td>
            <td>${fmtScore(r.baseScore)}</td>
            <td>${fmtScore(r.targetScore)}</td>
            <td title="${esc(r.question)}">${esc(r.question)}</td>
            <td title="${esc(r.targetPrediction)}">${esc(r.targetPrediction)}</td>
          </tr>`).join('')}</tbody>
        </table>`;
      $('analysisPane').querySelectorAll('tr[data-execution]').forEach(tr => {
        tr.addEventListener('click', () => loadArtifact(Number(tr.dataset.execution)));
      });
      setStatus('Compare loaded');
    }

    async function runDiagnostics() {
      const scope = currentScope();
      setStatus('Loading diagnostics...');
      const data = await api('/api/diagnostics', {
        body: {
          model: scope.model,
          scorer: scope.scorer,
          configIds: scope.configIds,
          limit: 5000
        }
      });
      const sections = Object.entries(data.buckets || {}).map(([name, rows]) => `
        <h2 style="padding:10px 10px 0">${esc(name)}</h2>
        <table><thead><tr><th>value</th><th style="width:72px">n</th><th style="width:90px">score</th></tr></thead>
        <tbody>${rows.map(r => `<tr><td>${esc(r.value)}</td><td>${r.n}</td><td>${fmtScore(r.avgScore)}</td></tr>`).join('')}</tbody></table>
      `).join('');
      $('analysisPane').innerHTML = `<div class="metric-strip"><div class="metric"><div class="k">rows</div><div class="v">${data.n}</div></div></div>${sections}`;
      setStatus('Diagnostics loaded');
    }

    async function runPlanDelete() {
      const scope = currentScope();
      setStatus('Building dry-run delete plan...');
      const data = await api('/api/plan-delete', {
        body: {
          model: scope.model,
          configIds: scope.configIds
        }
      });
      $('analysisPane').innerHTML = `<pre>${esc(JSON.stringify(data, null, 2))}</pre>`;
      setStatus('Dry-run plan loaded');
    }

    $('refreshBtn').addEventListener('click', async () => {
      try {
        await refreshConfigs();
        renderSummary();
        setStatus('Refreshed');
      } catch (err) {
        setStatus(err.message, 'bad');
      }
    });
    $('sliceBtn').addEventListener('click', () => runSlices().catch(err => setStatus(err.message, 'bad')));
    $('compareBtn').addEventListener('click', () => runCompare().catch(err => setStatus(err.message, 'bad')));
    $('diagBtn').addEventListener('click', () => runDiagnostics().catch(err => setStatus(err.message, 'bad')));
    $('planBtn').addEventListener('click', () => runPlanDelete().catch(err => setStatus(err.message, 'bad')));
    $('modelSelect').addEventListener('change', () => refreshConfigs().catch(err => setStatus(err.message, 'bad')));
    $('scorerSelect').addEventListener('change', () => refreshConfigs().catch(err => setStatus(err.message, 'bad')));

    loadBoot().catch(err => setStatus(err.message, 'bad'));
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
