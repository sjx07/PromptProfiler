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

import json
import logging
from typing import Dict, Optional

from prompt_profiler.core.func_registry import PromptBuildState
from prompt_profiler.prompt.prompt_state import PromptState

logger = logging.getLogger(__name__)


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
        """
        # If config declared explicit field funcs, use them as authoritative.
        # Fall back to task class defaults only when config has no field funcs.
        if not state.input_fields:
            state.input_fields = self.default_input_fields.copy()
        if not state.output_fields:
            state.output_fields = self.default_output_fields.copy()
        self._prompt_state = state.to_prompt_state()

        # Validate dispatch field (exactly one parser-registered output_field).
        # Skipped when the task class has no PARSER_REGISTRY (backward compat).
        self._validate_dispatch_field()

        example_strategy = state.extras.get("example_strategy")
        if example_strategy and example_pool:
            from prompt_profiler.prompt.example_sampler import build_sampler

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
        """Return the task's PARSER_REGISTRY module, or None if not defined."""
        # Subclasses override _parser_module_path to point to their parsers module.
        module_path = getattr(self, "_parser_module_path", None)
        if module_path is None:
            return None
        import importlib
        try:
            mod = importlib.import_module(module_path)
            return getattr(mod, "PARSER_REGISTRY", None)
        except ImportError:
            return None

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
        from prompt_profiler.core.preprocess import apply_transforms

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
            from prompt_profiler.tasks.wtq.table_transforms import compute_column_stats
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
        from prompt_profiler.core.preprocess import apply_record_transforms
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
