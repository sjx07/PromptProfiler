"""Task base class and protocol.

BaseTask provides shared bind() and build_prompt() logic.
Subclasses declare default_input_fields, default_output_fields,
and implement parse_response() and score().

Lifecycle:
    task.bind(state)                       # once per config
    sys, usr = task.build_prompt(query)    # per query
    pred = task.parse_response(response)   # per response
    s, m = task.score(pred, query_meta)    # lazy, per execution
"""
from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

from core.func_registry import PromptBuildState
from prompt.prompt_state import PromptState

logger = logging.getLogger(__name__)


@dataclass
class ModuleSpec:
    """Prompt defaults for one internal module of a compound task."""

    input_fields: Dict[str, str] = field(default_factory=dict)
    output_fields: Dict[str, str] = field(default_factory=dict)


@dataclass
class ModuleTrace:
    """Trace for one module-level LLM call."""

    module_name: str
    stage_index: int
    system_prompt: str
    user_content: str
    raw_response: str = ""
    parsed_output: Any = ""
    latency_ms: Optional[float] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_name": self.module_name,
            "stage_index": self.stage_index,
            "system_prompt": self.system_prompt,
            "user_content": self.user_content,
            "raw_response": self.raw_response,
            "parsed_output": _json_safe(self.parsed_output),
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "error": self.error,
            "meta": _json_safe(self.meta),
        }


@dataclass
class ModuleCallResult:
    """Return value from ModuleRuntime.call()."""

    raw_response: str
    parsed_output: Any
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class ModuleRuntime:
    """LLM call wrapper that records module-level traces."""

    def __init__(self, llm_call: Callable[[str, str], Dict[str, Any]]) -> None:
        self._llm_call = llm_call
        self.traces: list[ModuleTrace] = []

    def call(
        self,
        module_name: str,
        system_prompt: str,
        user_content: str,
        *,
        parse: Optional[Callable[[str], Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> ModuleCallResult:
        stage_index = len(self.traces)
        t0 = time.time()
        raw_response = ""
        parsed_output: Any = ""
        prompt_tokens = None
        completion_tokens = None
        error = None
        try:
            result = self._llm_call(system_prompt, user_content)
            raw_response = result.get("raw_response", "")
            if not isinstance(raw_response, str):
                raw_response = str(raw_response)
            parsed_output = parse(raw_response) if parse else raw_response
            prompt_tokens = result.get("prompt_tokens")
            completion_tokens = result.get("completion_tokens")
            return ModuleCallResult(
                raw_response=raw_response,
                parsed_output=parsed_output,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as exc:
            error = str(exc)[:500]
            raise
        finally:
            self.traces.append(ModuleTrace(
                module_name=module_name,
                stage_index=stage_index,
                system_prompt=system_prompt,
                user_content=user_content,
                raw_response=raw_response,
                parsed_output=parsed_output,
                latency_ms=(time.time() - t0) * 1000,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                error=error,
                meta=meta or {},
            ))

    def trace_dicts(self) -> list[Dict[str, Any]]:
        return [trace.to_dict() for trace in self.traces]

    def last_trace(self) -> Optional[ModuleTrace]:
        return self.traces[-1] if self.traces else None

    def total_prompt_tokens(self) -> Optional[int]:
        values = [t.prompt_tokens for t in self.traces if t.prompt_tokens is not None]
        return sum(values) if values else None

    def total_completion_tokens(self) -> Optional[int]:
        values = [t.completion_tokens for t in self.traces if t.completion_tokens is not None]
        return sum(values) if values else None


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, Mapping):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        return str(value)


class CompoundTask:
    """Base for tasks that execute multiple named prompt modules internally."""

    name: str = ""
    scorer: str = ""
    module_specs: Dict[str, ModuleSpec] = {}

    def __init__(self) -> None:
        self._module_prompt_states: Dict[str, PromptState] = {}

    @classmethod
    def module_names(cls) -> tuple[str, ...]:
        return tuple(cls.module_specs.keys())

    def bind_modules(
        self,
        module_states: Mapping[str, PromptBuildState],
        *,
        example_pool: Optional[list] = None,
    ) -> None:
        """Bind per-module prompt build states to renderable PromptStates."""
        _ = example_pool  # Reserved for future module-aware few-shot support.
        self._module_prompt_states = {}
        for module_name, spec in self.module_specs.items():
            state = copy.deepcopy(module_states.get(module_name, PromptBuildState()))
            for key, desc in spec.input_fields.items():
                state.input_fields.setdefault(key, desc)
            for key, desc in spec.output_fields.items():
                state.output_fields.setdefault(key, desc)
            self._module_prompt_states[module_name] = state.to_prompt_state()

    def build_module_prompt(self, module_name: str, record: dict) -> tuple[str, str]:
        if module_name not in self._module_prompt_states:
            raise RuntimeError(f"Module {module_name!r} not bound — call bind_modules() first")

        messages = self._module_prompt_states[module_name].build_messages(record)
        system_prompt = ""
        user_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                user_content = msg["content"]
        return system_prompt, user_content

    def run(self, query: dict, runtime: ModuleRuntime) -> Any:
        raise NotImplementedError


class BaseTask:
    """Shared base for all experiment tasks.

    Subclasses must define:
        name: str
        scorer: str
        default_input_fields: Dict[str, str]
        default_output_fields: Dict[str, str]
        parse_response(raw_response) -> str
        score(prediction, query_meta) -> (float, dict)
        build_record(query, meta, raw) -> dict   (optional override)
    """

    name: str = ""
    scorer: str = ""
    default_input_fields: Dict[str, str] = {}
    default_output_fields: Dict[str, str] = {}

    def __init__(self) -> None:
        self._prompt_state: Optional[PromptState] = None
        self._pending_stats: str = ""

    def bind(self, state: PromptBuildState, *, example_pool: Optional[list] = None) -> None:
        """Merge task defaults with func-added fields, then convert to PromptState.

        Args:
            example_pool: Train-split query dicts for few-shot selection.
                          Only used when state has example_strategy configured.
                          Caller is responsible for train/eval separation.

        Output-field merge rule (output_fields ONLY):
          Task defaults are merged into feature-declared output_fields ONLY
          when none of those declared fields is a parser-registered dispatch
          field. This lets:
            * enable_cot (declares `reasoning`, no parser) → merges in default
              `answer` so dispatch picks `answer`, `reasoning` becomes companion.
            * enable_code (declares `code`, registered) → NO merge; `code`
              takes over as the single dispatch field.
            * enable_cot + enable_code → `{reasoning, code}`, dispatch=code.
          This is the "fallback-unless-replaced" contract for output_fields.

        Input fields follow the simpler "replace when present, else default"
        contract: feature-declared inputs are authoritative.
        """
        if not state.input_fields:
            state.input_fields = self.default_input_fields.copy()

        # Output-field "fallback-unless-replaced": merge defaults only when
        # no declared output_field is a registered dispatch field.
        registry = self._get_parser_registry()
        registered_fields = set(registry.keys()) if registry else set()
        declared_fields = set(state.output_fields)
        has_dispatch_field = bool(declared_fields & registered_fields)
        if not has_dispatch_field:
            # Merge task defaults, without clobbering any feature-added field.
            for key, desc in self.default_output_fields.items():
                state.output_fields.setdefault(key, desc)

        self._prompt_state = state.to_prompt_state()

        # Validate dispatch field (exactly one parser-registered output_field).
        # Skipped when the task class has no PARSER_REGISTRY (backward compat).
        self._validate_dispatch_field()

        example_strategy = state.extras.get("example_strategy")
        if example_strategy and example_pool:
            from prompt.example_sampler import build_sampler

            pool, pool_texts = self._build_example_pool(example_pool)
            self._prompt_state.demo_selector = build_sampler(
                strategy=example_strategy,
                k=state.extras.get("example_k", 3),
                pool=pool,
                pool_texts=pool_texts,
                seed=state.extras.get("example_seed", 42),
                predicate_fn=self._predicate_fn(),
            )

    def _get_parser_registry(self) -> Optional[dict]:
        """Return the task's PARSER_REGISTRY, or None if not defined.

        Looks up the global PARSER_REGISTRY populated by autoload_parsers()
        (called from prompt_profiler/__init__.py).  Falls back to lazy import
        for test isolation / unusual import orders.
        """
        module_path = getattr(self, "_parser_module_path", None)
        from core.parser_registry import get_parser_registry
        return get_parser_registry(module_path)

    def _validate_dispatch_field(self) -> None:
        """Enforce exactly one parser-registered output_field in the config.

        Zero  → ValueError("no dispatch parser for this config")
        Two+  → ValueError("ambiguous dispatch: fields X, Y both registered")
        Exactly one → OK
        Skipped when task has no PARSER_REGISTRY (no _parser_module_path set).
        """
        registry = self._get_parser_registry()
        if registry is None:
            return
        if self._prompt_state is None:
            return

        output_fields = set(self._prompt_state.semantic.output_fields.keys())
        dispatch_candidates = output_fields & set(registry.keys())

        if len(dispatch_candidates) == 0:
            raise ValueError(
                f"Task '{self.name}': no dispatch parser registered for any output_field "
                f"in this config. output_fields={sorted(output_fields)}, "
                f"registered={sorted(registry.keys())}"
            )
        if len(dispatch_candidates) > 1:
            raise ValueError(
                f"Task '{self.name}': ambiguous dispatch — multiple parser-registered "
                f"output_fields present: {sorted(dispatch_candidates)}. "
                f"Declare conflicts_with between the corresponding features."
            )

    def _dispatch_field(self) -> Optional[str]:
        """Return the single dispatch output_field name, or None."""
        registry = self._get_parser_registry()
        if registry is None or self._prompt_state is None:
            return None
        output_fields = set(self._prompt_state.semantic.output_fields.keys())
        hits = output_fields & set(registry.keys())
        return next(iter(hits)) if len(hits) == 1 else None

    def _build_example_pool(self, pool_queries):
        """Convert query dicts to example pool."""
        pool = []
        pool_texts = []
        for q in pool_queries:
            meta = q.get("meta", {})
            if isinstance(meta, str):
                meta = json.loads(meta)
            raw = meta.get("_raw", {})
            record = self.build_record(q, meta, raw)
            input_keys = getattr(self, '_example_input_fields', None) or self.default_input_fields
            inputs = {k: record.get(k, "") for k in input_keys}
            outputs = self._gold_output(meta, raw)
            pool.append({"inputs": inputs, "outputs": outputs})
            pool_texts.append(q.get("content", ""))
        return pool, pool_texts

    def _gold_output(self, meta: dict, raw: dict) -> dict:
        """Extract gold output fields for a few-shot example. Override per task."""
        return {}

    def _predicate_fn(self):
        """Return a predicate extractor for predicate/hybrid example selection. Override per task."""
        return None

    def _apply_transforms(
        self,
        header: list[str],
        rows: list[list[str]],
        text: str,
    ) -> tuple[list[str], list[list[str]]]:
        """Apply registered input transforms from PromptBuildState.

        Handles both legacy flags (column_pruning, type_annotation, column_stats)
        and new transform_input specs from the preprocess registry.

        Returns (header, rows). Column stats string stored in self._pending_stats
        for the caller to prepend after table formatting.
        """
        from core.preprocess import apply_transforms

        self._pending_stats = ""
        if self._prompt_state is None:
            return header, rows

        transforms = []
        ps_meta = self._prompt_state.metadata
        compute_stats = False

        # Generic transforms from input_transform func_type
        if ps_meta.get("input_transforms"):
            for t in ps_meta["input_transforms"]:
                if t["fn"] == "prepend_stats":
                    compute_stats = True
                else:
                    transforms.append(t)

        # Apply structural transforms
        if transforms:
            header, rows = apply_transforms(header, rows, text, transforms)

        # Compute stats separately (needs the post-transform header/rows)
        if compute_stats:
            from tasks.wtq.table_transforms import compute_column_stats
            stats_str = compute_column_stats(header, rows)
            if stats_str:
                self._pending_stats = stats_str

        return header, rows

    def _apply_record_transforms(self, record: dict) -> dict:
        """Apply record-level transforms from PromptBuildState.

        Record transforms operate on the full record dict — they can
        read/modify any field (schema DDL, error messages, etc.).
        Dispatches to RECORD_REGISTRY in preprocess.py.
        """
        if self._prompt_state is None:
            return record
        ps_meta = self._prompt_state.metadata
        transforms = ps_meta.get("input_transforms", [])
        if not transforms:
            return record
        from core.preprocess import apply_record_transforms
        return apply_record_transforms(record, transforms)

    def build_record(self, query: dict, meta: dict, raw: dict) -> dict:
        """Build the record dict from query data.

        Override in subclasses to customize field extraction.
        Default: flattens content, evidence, and _raw fields.

        Record-level transforms (focus_schema, localize_error, etc.)
        are applied automatically after subclass builds the base record.
        """
        return {
            "question": query.get("content", ""),
            "schema": raw.get("schema", ""),
            "evidence": meta.get("evidence", ""),
            "db_id": meta.get("db_id", ""),
            **{k: v for k, v in raw.items() if k not in ("question", "schema")},
        }

    def build_prompt(self, query: dict) -> tuple[str, str]:
        """Build LLM prompt from bound state + query."""
        if self._prompt_state is None:
            raise RuntimeError("Task not bound — call bind(state) first")

        meta = query.get("meta", {})
        if isinstance(meta, str):
            meta = json.loads(meta)

        raw = meta.get("_raw", {})
        record = self.build_record(query, meta, raw)
        record = self._apply_record_transforms(record)

        messages = self._prompt_state.build_messages(record)
        system_prompt = ""
        user_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                user_content = msg["content"]
        return system_prompt, user_content

    def parse_response(self, raw_response: str) -> str:
        """Dispatch to the registered parser for the single dispatch output_field.

        Subclasses with a PARSER_REGISTRY (_parser_module_path set) get automatic
        dispatch. Subclasses that do not define _parser_module_path must override
        this method directly.
        """
        registry = self._get_parser_registry()
        if registry is not None:
            field = self._dispatch_field()
            if field is not None:
                return registry[field](raw_response, self)
        raise NotImplementedError(
            f"Task '{self.name}' has no parser registry and did not override parse_response()"
        )

    def score(self, prediction: str, query_meta: dict) -> tuple[float, dict]:
        raise NotImplementedError
