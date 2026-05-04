"""Feature editor — FastAPI backend.

Browser frontend lets you list / view / add / remove / edit feature JSONs
under features/<task>/, with validation via FeatureRegistry.materialize and
optional render preview against a sample query.

Run:
    cd prompt_profiler
    python3 -m uvicorn tools.feature_editor.server:app --port 8765 --reload
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FEATURES_DIR = REPO / "features"
INDEX_HTML = Path(__file__).resolve().parent / "index.html"

app = FastAPI(title="Feature editor")


# ── helpers ──────────────────────────────────────────────────────────

def _task_dir(task: str) -> Path:
    p = FEATURES_DIR / task
    if not p.is_dir():
        raise HTTPException(404, f"task not found: {task}")
    return p


def _feature_path(task: str, name: str) -> Path:
    if "/" in name or name.startswith(".") or not name.endswith(".json"):
        raise HTTPException(400, "invalid feature name")
    return _task_dir(task) / name


def _sample_for_task(task: str) -> dict | None:
    """A single representative input record per task for render preview."""
    if task in ("tablebench", "tablebench_repro"):
        return {
            "content": "How many companies have profits greater than 10 billion?",
            "meta": {
                "_raw": {
                    "question": "How many companies have profits greater than 10 billion?",
                    "answer": "5",
                    "qtype": "NumericalReasoning",
                    "qsubtype": "Counting",
                    "table": {
                        "header": ["rank", "company", "profits"],
                        "rows": [
                            ["1", "citigroup", "17.85"],
                            ["2", "general electric", "15.59"],
                            ["3", "aig", "6.46"],
                            ["4", "exxonmobil", "20.96"],
                            ["5", "bp", "10.27"],
                            ["6", "bofa", "10.81"],
                            ["7", "hsbc", "6.66"],
                        ],
                    },
                },
            },
        }
    return None


# ── tasks / features listing ─────────────────────────────────────────

@app.get("/api/tasks")
def list_tasks() -> list[str]:
    return sorted(p.name for p in FEATURES_DIR.iterdir() if p.is_dir())


@app.get("/api/tasks/{task}/features")
def list_features(task: str) -> list[dict]:
    out: list[dict] = []
    for p in sorted(_task_dir(task).glob("*.json")):
        try:
            spec = json.loads(p.read_text())
        except json.JSONDecodeError as exc:
            out.append({"name": p.name, "canonical_id": p.stem, "error": str(exc)})
            continue
        out.append({
            "name": p.name,
            "canonical_id": spec.get("canonical_id") or spec.get("feature_id") or p.stem,
            "is_section": p.name.startswith("_section_"),
            "n_edits": len(spec.get("primitive_edits", [])),
        })
    return out


@app.get("/api/tasks/{task}/features/{name}")
def get_feature(task: str, name: str) -> dict:
    path = _feature_path(task, name)
    if not path.exists():
        raise HTTPException(404, f"feature not found: {name}")
    return json.loads(path.read_text())


class WriteFeatureBody(BaseModel):
    spec: dict
    skip_validation: bool = False


def _validate(task: str, spec: dict) -> None:
    """Round-trip via FeatureRegistry: load + materialize the spec.

    Loads the on-disk task dir (other features) and materialises this spec
    plus its `requires`. Surfaces errors raised by validate_feature_set or
    materialize so the user sees them before the file is written.
    """
    from core.feature_registry import FeatureRegistry

    reg = FeatureRegistry.load(task=task)
    canonical_id = spec.get("canonical_id") or spec.get("feature_id") or "<unsaved>"
    reg._by_canonical[canonical_id] = {**spec, "canonical_id": canonical_id}
    needed = [canonical_id, *spec.get("requires", [])]
    reg.materialize(needed)


@app.put("/api/tasks/{task}/features/{name}")
def write_feature(task: str, name: str, body: WriteFeatureBody) -> dict:
    path = _feature_path(task, name)
    if not body.skip_validation:
        try:
            _validate(task, body.spec)
        except Exception as exc:
            raise HTTPException(400, f"validation failed: {exc}")
    path.write_text(json.dumps(body.spec, indent=2) + "\n")
    return {"ok": True, "path": str(path.relative_to(REPO))}


@app.post("/api/tasks/{task}/features")
def create_feature(task: str, body: WriteFeatureBody) -> dict:
    canonical = body.spec.get("canonical_id") or body.spec.get("feature_id")
    if not canonical:
        raise HTTPException(400, "spec.canonical_id is required")
    name = f"{canonical}.json"
    path = _feature_path(task, name)
    if path.exists():
        raise HTTPException(409, f"already exists: {name}")
    body.spec.setdefault("task", task)
    if not body.skip_validation:
        try:
            _validate(task, body.spec)
        except Exception as exc:
            raise HTTPException(400, f"validation failed: {exc}")
    path.write_text(json.dumps(body.spec, indent=2) + "\n")
    return {"ok": True, "name": name, "path": str(path.relative_to(REPO))}


@app.delete("/api/tasks/{task}/features/{name}")
def delete_feature(task: str, name: str) -> dict:
    path = _feature_path(task, name)
    if not path.exists():
        raise HTTPException(404, f"feature not found: {name}")
    path.unlink()
    return {"ok": True}


# ── render preview ───────────────────────────────────────────────────

class RenderBody(BaseModel):
    canonical_id: str
    extra_features: list[str] = []  # additional canonical_ids to materialize alongside
    apply_runtime_hook: bool = False  # default: render the AUTHORED prompt, not the per-record-mutated one


@app.post("/api/tasks/{task}/render")
def render(task: str, body: RenderBody) -> dict:
    sample = _sample_for_task(task)
    if sample is None:
        raise HTTPException(400, f"no render sample registered for task={task}")

    try:
        from core.feature_registry import FeatureRegistry
        from core.func_registry import PromptBuildState, REGISTRY, _func_sort_key
        from task_registry import get_registry as get_task_registry
    except Exception as exc:
        raise HTTPException(500, f"import error: {exc}")

    reg = FeatureRegistry.load(task=task)
    # Auto-include all _section_* features so rules have parent sections to land on.
    section_ids = [cid for cid in reg._by_canonical if cid.startswith("_section_")]
    canonical_ids = list(dict.fromkeys([
        *section_ids,
        body.canonical_id,
        *body.extra_features,
    ]))
    try:
        specs, _ = reg.materialize(canonical_ids)
    except Exception as exc:
        raise HTTPException(400, f"materialize failed: {exc}")

    # Build the prompt state.
    state = PromptBuildState()
    for spec in sorted(
        specs,
        key=lambda s: _func_sort_key(s["func_id"], s["func_type"], s["params"]),
    ):
        params = dict(spec["params"])
        params["_func_id"] = spec["func_id"]
        try:
            REGISTRY[spec["func_type"]](state, params)
        except Exception as exc:
            raise HTTPException(500, f"primitive {spec['func_type']} failed: {exc}")

    # Bind a task instance and render.
    task_registry = get_task_registry()
    if task not in task_registry:
        raise HTTPException(400, f"task {task} not registered (cannot render)")
    task_cls = task_registry[task].task_cls

    try:
        task_obj = task_cls()
        task_obj.bind(state)
        if body.apply_runtime_hook:
            sys_prompt, user_content = task_obj.build_prompt(sample)
        else:
            # Render the AUTHORED prompt — skip per-record mutation hooks so
            # rules the author just wrote are visible even when a runtime
            # filter (e.g. tablebench's _remove_generic_dataanalysis_contracts)
            # would normally strip them for the chosen sample qsubtype.
            from task import _collapse_messages_to_system_user  # type: ignore

            meta = sample.get("meta", {})
            raw = meta.get("_raw", {}) if isinstance(meta, dict) else {}
            record = task_obj.build_record(sample, meta, raw)
            record = task_obj._apply_record_transforms(record)
            messages = task_obj._prompt_state.build_messages(record)
            sys_prompt, user_content = _collapse_messages_to_system_user(messages)
    except Exception as exc:
        raise HTTPException(500, f"render failed: {exc}\n{traceback.format_exc()}")

    return {
        "system_prompt": sys_prompt,
        "user_content": user_content,
        "n_specs": len(specs),
        "sample_question": sample.get("content"),
    }


# ── static html ──────────────────────────────────────────────────────

@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)
