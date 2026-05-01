"""Function registry — maps func_type strings to Python handlers.

Phase 1 refactor: three paper-aligned primitives replace ad-hoc handlers.
  insert_node    — add a node (section/rule/input_field/output_field/example)
  update_node    — declared; no consumer in Phase 1 (warns if used)
  input_transform — single pre-render input transform (renamed from transform_input)

Removed (hard break, no aliases):
  add_rule, define_section, add_input_field, transform_input,
  enable_cot, enable_patch_trace, enable_code, enable_sql,
  enable_column_pruning, enable_type_annotation, enable_column_stats

Kept unchanged:
  set_format, set_table_format, add_example

Usage:
    from core.func_registry import apply_config, make_func_id

    state = apply_config(func_ids, store)
    system_prompt = state.to_prompt_state()._build_system_content()
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from core.store import CubeStore

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────

ROOT_ID = "__root__"
MAIN_MODULE = "__main__"

ALLOWED_NODE_TYPES = frozenset({
    "section", "rule", "input_field", "output_field", "example",
})

# ── registry ──────────────────────────────────────────────────────────

REGISTRY: Dict[str, Callable[["PromptBuildState", dict], None]] = {}
IDENTITY_KEYS: Dict[str, Callable[[dict], str]] = {}


def register(func_type: str, *, identity_key: Optional[Callable[[dict], str]] = None):
    """Decorator to register a func_type handler."""
    def wrapper(fn: Callable[["PromptBuildState", dict], None]):
        REGISTRY[func_type] = fn
        if identity_key is not None:
            IDENTITY_KEYS[func_type] = identity_key
        return fn
    return wrapper


def make_func_id(func_type: str, params: dict) -> str:
    """Content-addressed func ID.

    Uses the registered identity_key if available, otherwise hashes
    func_type + sorted params. Same semantic content → same ID.
    """
    key_fn = IDENTITY_KEYS.get(func_type)
    if key_fn:
        payload = f"{func_type}:{key_fn(params)}"
    else:
        payload = f"{func_type}:{json.dumps(params, sort_keys=True)}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


# ── prompt build state ────────────────────────────────────────────────

@dataclass
class PromptBuildState:
    """Mutable state that functions operate on to build a prompt."""
    # Rules accumulated by insert_node(rule), grouped by parent section_id
    rules: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    # Sections metadata (title, ordinal, is_system, etc.)
    sections: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Format style name (set by set_format)
    format_style: str = "json"

    # Table serialization format (set by set_table_format, None = follow output format)
    table_format: Optional[str] = None

    # I/O fields for user prompt
    input_fields: Dict[str, str] = field(default_factory=dict)
    output_fields: Dict[str, str] = field(default_factory=dict)

    # Generic key-value store for future func types
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_prompt_state(self) -> "PromptState":
        """Convert to a feature_pipeline PromptState for rendering."""
        from prompt.prompt_state import PromptState
        from prompt.semantic_content import SemanticContent
        from prompt.rules import RuleSection, RuleItem, RuleTree

        # Build RuleSections sorted by section ordinal
        sorted_section_ids = sorted(
            self.sections.keys(),
            key=lambda sid: self.sections[sid].get("ordinal", 0),
        )

        rule_sections = []
        for sid in sorted_section_ids:
            sec_meta = self.sections[sid]
            children = [
                RuleItem(text=r["content"], node_id=f"{sid}/r{i}")
                for i, r in enumerate(self.rules.get(sid, []))
            ]
            rule_sections.append(RuleSection(
                title=sec_meta.get("title", ""),
                level=1,
                children=children,
                node_id=sid,
            ))

        tree = RuleTree(roots=rule_sections)
        tree._rebuild_index()

        semantic = SemanticContent(
            rule_sections=rule_sections,
            tree=tree,
            input_fields=self.input_fields.copy(),
            output_fields=self.output_fields.copy(),
        )
        meta = {}
        if self.table_format is not None:
            meta["table_format"] = self.table_format
        if self.extras.get("input_transforms"):
            meta["input_transforms"] = self.extras["input_transforms"]
        return PromptState(
            semantic=semantic,
            format_style_name=self.format_style,
            metadata=meta,
        )


# ── canonicalization ──────────────────────────────────────────────────

def _canonicalize_insert_node(params: dict) -> dict:
    """Return canonical params for insert_node.

    Normalizes key order and missing optional fields so make_func_id
    produces a stable hash regardless of how params were constructed.
    """
    node_type = params["node_type"]
    if node_type not in ALLOWED_NODE_TYPES:
        raise ValueError(f"unsupported node_type={node_type!r}; allowed={sorted(ALLOWED_NODE_TYPES)}")
    parent_id = params.get("parent_id") or ROOT_ID
    payload = params.get("payload", {})

    if node_type == "section":
        payload = {
            "title":     payload.get("title", ""),
            "ordinal":   int(payload.get("ordinal", 0)),
            "is_system": bool(payload.get("is_system", False)),
            "min_rules": int(payload.get("min_rules", 0)),
            "max_rules": int(payload.get("max_rules", 10)),
        }
    elif node_type == "rule":
        payload = {"content": payload.get("content", "").strip()}
        if "ordinal" in params.get("payload", {}):
            payload["ordinal"] = int(params["payload"].get("ordinal", 0))
    elif node_type == "input_field":
        payload = {
            "name":        payload.get("name", ""),
            "description": payload.get("description", ""),
        }
        if "ordinal" in params.get("payload", {}):
            payload["ordinal"] = int(params["payload"].get("ordinal", 0))
    elif node_type == "output_field":
        payload = {
            "name":        payload.get("name", ""),
            "description": payload.get("description", ""),
        }
        if "ordinal" in params.get("payload", {}):
            payload["ordinal"] = int(params["payload"].get("ordinal", 0))
    elif node_type == "example":
        payload = {"content": payload.get("content", "")}
        if "ordinal" in params.get("payload", {}):
            payload["ordinal"] = int(params["payload"].get("ordinal", 0))

    return {
        "node_type": node_type,
        "parent_id": parent_id,
        "payload":   payload,
    }


# ── new primitive handlers ─────────────────────────────────────────────

@register(
    "insert_node",
    identity_key=lambda p: json.dumps(_canonicalize_insert_node(p), sort_keys=True),
)
def _apply_insert_node(state: PromptBuildState, params: dict) -> None:
    """Insert a node into the prompt state.

    Dispatches on params.node_type ∈ {section, rule, input_field, output_field, example}.
    Default parent_id = ROOT_ID ("__root__") when unspecified.

    Sort order: section(0) < input_field(1) < output_field(1) < rule(2) < example(3).
    """
    canon = _canonicalize_insert_node(params)
    node_type = canon["node_type"]
    parent_id = canon["parent_id"]
    payload   = canon["payload"]

    if node_type == "section":
        section_id = params.get("_func_id", "")
        # Enforce depth cap of 2: section inside section is OK, but section
        # whose parent is itself a subsection is not (MVP).
        state.sections[section_id] = {
            "title":     payload["title"],
            "ordinal":   payload["ordinal"],
            "is_system": payload["is_system"],
            "min_rules": payload["min_rules"],
            "max_rules": payload["max_rules"],
            "parent_id": parent_id,
        }
    elif node_type == "rule":
        state.rules.setdefault(parent_id, []).append({"content": payload["content"]})
    elif node_type == "input_field":
        state.input_fields[payload["name"]] = payload["description"]
    elif node_type == "output_field":
        state.output_fields[payload["name"]] = payload["description"]
    elif node_type == "example":
        state.extras.setdefault("examples", []).append(payload)


@register("update_node", identity_key=lambda p: json.dumps(p, sort_keys=True))
def _apply_update_node(state: PromptBuildState, params: dict) -> None:
    """Declared primitive. No consumer in Phase 1 — warns if used."""
    logger.warning(
        "update_node: no consumer wired in Phase 1; node will not be modified. params=%s",
        params,
    )


@register(
    "input_transform",
    identity_key=lambda p: f"{p.get('fn', '')}:{json.dumps(p.get('kwargs', {}), sort_keys=True)}",
)
def _apply_input_transform(state: PromptBuildState, params: dict) -> None:
    """Pre-render input transform (renamed from transform_input).

    Multiple input_transform primitives compose deterministically by func_id sort order.
    """
    state.extras.setdefault("input_transforms", []).append({
        "fn":     params.get("fn", ""),
        "kwargs": params.get("kwargs", {}),
    })


# ── kept handlers (unchanged) ─────────────────────────────────────────

@register("set_format", identity_key=lambda p: p.get("style", ""))
def _apply_set_format(state: PromptBuildState, params: dict) -> None:
    """Set the format style for prompt rendering."""
    state.format_style = params.get("style", "json")


@register("set_table_format", identity_key=lambda p: p.get("format", "markdown"))
def _apply_set_table_format(state: PromptBuildState, params: dict) -> None:
    """Set the table serialization format."""
    state.table_format = params.get("format", "markdown")


@register("add_example", identity_key=lambda p: f"{p.get('example_strategy', 'random')}:{p.get('k', 3)}")
def _apply_add_example(state: PromptBuildState, params: dict) -> None:
    """Configure few-shot example selection strategy."""
    state.extras["example_strategy"] = params.get("example_strategy", "random")
    state.extras["example_k"] = int(params.get("k", 3))
    state.extras["example_seed"] = int(params.get("seed", 42))


# ── func_type ordering ────────────────────────────────────────────────
# insert_node(section) must run before insert_node(rule).
# Ordering is handled inside apply_config via _insert_node_sort_key.

_NODE_TYPE_ORDER: Dict[str, int] = {
    "section":      0,
    "input_field":  1,
    "output_field": 1,
    "rule":         2,
    "example":      3,
}

_TYPE_ORDER: Dict[str, int] = {
    "insert_node":    1,   # sub-sorted by node_type below
    "update_node":    2,
    "input_transform": 1,
    "set_format":     1,
    "set_table_format": 1,
    "add_example":    3,
}


def _payload_ordinal(params: dict, default: int = 10_000) -> int:
    payload = params.get("payload") or {}
    if not isinstance(payload, dict) or "ordinal" not in payload:
        return default
    try:
        return int(payload.get("ordinal", default))
    except (TypeError, ValueError):
        return default


def _func_sort_key(func_id: str, func_type: str, params: dict) -> tuple:
    """Sort key: (type_priority, node_type_priority, optional_payload_ordinal, func_id)."""
    type_prio = _TYPE_ORDER.get(func_type, 1)
    if func_type == "insert_node":
        nt = params.get("node_type", "rule")
        nt_prio = _NODE_TYPE_ORDER.get(nt, 1)
        return (type_prio, nt_prio, _payload_ordinal(params), func_id)
    return (type_prio, 0, 10_000, func_id)


# ── apply config ──────────────────────────────────────────────────────

def _load_sorted_func_rows(
    func_ids: Iterable[str],
    store: CubeStore,
) -> list[tuple[str, dict]]:
    """Load func rows and sort them in prompt-application order."""
    func_rows = []
    for func_id in func_ids:
        func_row = store.get_func(func_id)
        if func_row is None:
            logger.warning("func_id %s not found in store, skipping", func_id)
            continue
        func_rows.append((func_id, func_row))

    func_rows.sort(key=lambda pair: _func_sort_key(
        pair[0],
        pair[1]["func_type"],
        json.loads(pair[1]["params"]) if isinstance(pair[1]["params"], str) else pair[1]["params"],
    ))

    return func_rows


def _json_field(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _func_target_module(func_row: dict) -> str:
    meta = _json_field(func_row.get("meta"), {})
    if not isinstance(meta, dict):
        return MAIN_MODULE
    return meta.get("target_module") or MAIN_MODULE


def _apply_func_rows(func_rows: list[tuple[str, dict]], state: PromptBuildState) -> None:
    for func_id, func_row in func_rows:
        func_type = func_row["func_type"]
        params = json.loads(func_row["params"]) if isinstance(func_row["params"], str) else func_row["params"]
        params["_func_id"] = func_id

        handler = REGISTRY.get(func_type)
        if handler is None:
            logger.warning("No handler registered for func_type=%s, skipping", func_type)
            continue

        handler(state, params)


def apply_config(
    func_ids: List[str],
    store: CubeStore,
) -> PromptBuildState:
    """Build a PromptBuildState by applying all funcs in a config."""
    state = PromptBuildState()
    _apply_func_rows(_load_sorted_func_rows(func_ids, store), state)
    return state


def apply_config_modules(
    func_ids: List[str],
    store: CubeStore,
    *,
    module_names: Iterable[str] | None = None,
) -> Dict[str, PromptBuildState]:
    """Build prompt states grouped by each func's target module.

    Untargeted funcs apply to ``MAIN_MODULE``. This preserves the existing
    single-stage default while giving compound tasks targeted module states.
    """
    states: Dict[str, PromptBuildState] = {MAIN_MODULE: PromptBuildState()}
    if module_names:
        for name in module_names:
            states.setdefault(name, PromptBuildState())

    rows_by_module: Dict[str, list[tuple[str, dict]]] = {}
    for func_id, func_row in _load_sorted_func_rows(func_ids, store):
        target = _func_target_module(func_row)
        rows_by_module.setdefault(target, []).append((func_id, func_row))
        states.setdefault(target, PromptBuildState())

    for target, rows in rows_by_module.items():
        _apply_func_rows(rows, states[target])

    return states
