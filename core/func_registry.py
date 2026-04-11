"""Function registry — maps func_type strings to Python handlers.

Each registered handler receives a PromptBuildState and params dict,
and mutates the state. The state accumulates all function effects,
then produces the final prompt.

Usage:
    from prompt_profiler.func_registry import apply_config, make_func_id

    state = apply_config(func_ids, store)
    system_prompt, user_content = state.build_prompt(query)
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from prompt_profiler.core.store import CubeStore

logger = logging.getLogger(__name__)

# ── registry ──────────────────────────────────────────────────────────

REGISTRY: Dict[str, Callable[["PromptBuildState", dict], None]] = {}
IDENTITY_KEYS: Dict[str, Callable[[dict], str]] = {}


def register(func_type: str, *, identity_key: Optional[Callable[[dict], str]] = None):
    """Decorator to register a func_type handler.

    Args:
        func_type: String key for this handler.
        identity_key: Extracts the semantic identity from params for
                      content-addressed ID generation. If None, falls
                      back to hashing all params.
    """
    def wrapper(fn: Callable[["PromptBuildState", dict], None]):
        REGISTRY[func_type] = fn
        if identity_key is not None:
            IDENTITY_KEYS[func_type] = identity_key
        return fn
    return wrapper


def make_func_id(func_type: str, params: dict) -> str:
    """Content-addressed func ID.

    Uses the registered identity_key if available, otherwise hashes
    func_type + sorted params. Same semantic content → same ID
    regardless of import source.
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
    """Mutable state that functions operate on to build a prompt.

    Starts blank. Each function mutates it. After all functions are applied,
    call build_prompt(query) to produce the final (system_prompt, user_content).
    """
    # Rules accumulated by add_rule functions, grouped by section
    rules: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    # Sections metadata (title, ordinal, is_system, etc.)
    sections: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Format style name (set by set_format)
    format_style: str = "json"

    # Chain of thought (set by enable_cot)
    chain_of_thought: bool = False

    # Patch trace (set by enable_patch_trace)
    patch_trace: bool = False

    # Code execution (set by enable_code)
    code_execution: bool = False

    # SQL execution (set by enable_sql)
    sql_execution: bool = False

    # Table serialization format (set by set_table_format, None = follow output format)
    table_format: Optional[str] = None

    # Table input transforms
    column_pruning: bool = False
    type_annotation: bool = False
    column_stats: bool = False


    # I/O fields for user prompt (task sets defaults, funcs can add more)
    input_fields: Dict[str, str] = field(default_factory=dict)
    output_fields: Dict[str, str] = field(default_factory=dict)

    # Generic key-value store for future func types
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_prompt_state(self) -> "PromptState":
        """Convert to a feature_pipeline PromptState for rendering.

        Maps accumulated rules/sections → SemanticContent → PromptState,
        reusing the existing FormatStyle rendering pipeline.
        """
        from prompt_profiler.prompt.prompt_state import PromptState
        from prompt_profiler.prompt.semantic_content import SemanticContent
        from prompt_profiler.prompt.rules import RuleSection, RuleItem, RuleTree

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

        # Build a RuleTree so the format style can look up nodes via is_enabled()
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
        if self.column_pruning:
            meta["column_pruning"] = True
        if self.type_annotation:
            meta["type_annotation"] = True
        if self.column_stats:
            meta["column_stats"] = True
        # Pass transform_input specs through metadata so tasks can access them
        if self.extras.get("input_transforms"):
            meta["input_transforms"] = self.extras["input_transforms"]
        return PromptState(
            semantic=semantic,
            format_style_name=self.format_style,
            chain_of_thought=self.chain_of_thought,
            patch_trace=self.patch_trace,
            code_execution=self.code_execution,
            sql_execution=self.sql_execution,
            metadata=meta,
        )


# ── built-in handlers ─────────────────────────────────────────────────

@register("add_rule", identity_key=lambda p: p.get("content", "").strip())
def _apply_add_rule(state: PromptBuildState, params: dict) -> None:
    """Add a rule to the prompt state, grouped by section."""
    section_id = params.get("section_id", "")
    content = params.get("content", "")
    if not content.strip():
        return
    if section_id not in state.rules:
        state.rules[section_id] = []
    state.rules[section_id].append({
        "content": content,
    })


@register("set_format", identity_key=lambda p: p.get("style", ""))
def _apply_set_format(state: PromptBuildState, params: dict) -> None:
    """Set the format style for prompt rendering."""
    style = params.get("style", "json")
    state.format_style = style


@register("enable_cot")
def _apply_enable_cot(state: PromptBuildState, params: dict) -> None:
    """Enable chain-of-thought reasoning."""
    state.chain_of_thought = True


@register("enable_patch_trace")
def _apply_enable_patch_trace(state: PromptBuildState, params: dict) -> None:
    """Enable structured patch trace — model outputs error_analysis + patch_description before final SQL."""
    state.patch_trace = True


@register("enable_code")
def _apply_enable_code(state: PromptBuildState, params: dict) -> None:
    """Enable code execution output — model writes Python to compute the answer."""
    state.code_execution = True


@register("enable_sql")
def _apply_enable_sql(state: PromptBuildState, params: dict) -> None:
    """Enable SQL execution output — model writes SQL to query the table."""
    state.sql_execution = True


@register("set_table_format", identity_key=lambda p: p.get("format", "markdown"))
def _apply_set_table_format(state: PromptBuildState, params: dict) -> None:
    """Set the table serialization format (markdown, csv, html, json_records)."""
    state.table_format = params.get("format", "markdown")


@register("enable_column_pruning")
def _apply_enable_column_pruning(state: PromptBuildState, params: dict) -> None:
    """Enable column pruning — remove irrelevant columns based on question keywords."""
    state.column_pruning = True


@register("enable_type_annotation")
def _apply_enable_type_annotation(state: PromptBuildState, params: dict) -> None:
    """Enable type annotation — add (int)/(float)/(date) to column headers."""
    state.type_annotation = True


@register("enable_column_stats")
def _apply_enable_column_stats(state: PromptBuildState, params: dict) -> None:
    """Enable column statistics — prepend cardinality/range/type summary before table."""
    state.column_stats = True




@register("add_input_field", identity_key=lambda p: p.get("name", ""))
def _apply_add_input_field(state: PromptBuildState, params: dict) -> None:
    """Add an input field to the user prompt.

    Example: params={"name": "filtered_schema", "description": "Relevant tables and columns"}
    The field value comes from the query record at build_prompt time.
    """
    name = params.get("name", "")
    desc = params.get("description", "")
    if name:
        state.input_fields[name] = desc


@register("add_example", identity_key=lambda p: f"{p.get('example_strategy','random')}:{p.get('k',3)}")
def _apply_add_example(state: PromptBuildState, params: dict) -> None:
    """Configure few-shot example selection strategy."""
    state.extras["example_strategy"] = params.get("example_strategy", "random")
    state.extras["example_k"] = int(params.get("k", 3))
    state.extras["example_seed"] = int(params.get("seed", 42))


@register("transform_input", identity_key=lambda p: f"{p.get('fn', '')}:{json.dumps(p.get('kwargs', {}), sort_keys=True)}")
def _apply_transform_input(state: PromptBuildState, params: dict) -> None:
    """Register an input preprocess transform (resolved at build_record time).

    Spec/materialization decoupling: this handler stores the spec.
    The actual Python function lives in core.preprocess.REGISTRY
    and runs when the task builds the prompt.

    Config example:
        {"func_type": "transform_input", "params": {"fn": "filter_rows", "kwargs": {"max_rows": 50}}}
    """
    state.extras.setdefault("input_transforms", []).append({
        "fn": params.get("fn", ""),
        "kwargs": params.get("kwargs", {}),
    })


@register("define_section", identity_key=lambda p: p.get("title", "").strip())
def _apply_define_section(state: PromptBuildState, params: dict) -> None:
    """Register a section in the prompt state."""
    # The func_id IS the section_id — caller passes it via extras
    section_id = params.get("_func_id", "")
    state.sections[section_id] = {
        "title": params.get("title", ""),
        "ordinal": params.get("ordinal", 0),
        "is_system": params.get("is_system", False),
        "min_rules": params.get("min_rules", 0),
        "max_rules": params.get("max_rules", 10),
    }


# ── func_type ordering ────────────────────────────────────────────────
# define_section must run before add_rule so sections exist when rules reference them.

_TYPE_ORDER: Dict[str, int] = {
    "define_section": 0,
    "set_format": 1,
    "set_table_format": 1,
    "enable_cot": 1,
    "enable_patch_trace": 1,
    "enable_code": 1,
    "enable_sql": 1,
    "transform_input": 1,
    "add_rule": 2,
    "add_example": 3,
}


def _func_sort_key(func_id: str, func_type: str) -> tuple:
    return (_TYPE_ORDER.get(func_type, 1), func_id)


# ── apply config ──────────────────────────────────────────────────────

def apply_config(
    func_ids: List[str],
    store: CubeStore,
) -> PromptBuildState:
    """Build a PromptBuildState by applying all funcs in a config.

    Args:
        func_ids: List of func_ids from config.func_ids.
        store: CubeStore to look up func definitions.

    Returns:
        PromptBuildState with all functions applied.
    """
    state = PromptBuildState()

    # Load all func rows, then sort by type priority
    func_rows = []
    for func_id in func_ids:
        func_row = store.get_func(func_id)
        if func_row is None:
            logger.warning("func_id %s not found in store, skipping", func_id)
            continue
        func_rows.append((func_id, func_row))

    func_rows.sort(key=lambda pair: _func_sort_key(pair[0], pair[1]["func_type"]))

    # Apply each func in priority order
    for func_id, func_row in func_rows:
        func_type = func_row["func_type"]
        params = json.loads(func_row["params"]) if isinstance(func_row["params"], str) else func_row["params"]

        # Inject func_id so handlers can use it (e.g. define_section uses it as section_id)
        params["_func_id"] = func_id

        handler = REGISTRY.get(func_type)
        if handler is None:
            logger.warning("No handler registered for func_type=%s, skipping", func_type)
            continue

        handler(state, params)

    return state
