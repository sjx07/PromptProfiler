"""SkVM structured generation L3 task — deeply nested JSON generation."""
from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from task import BaseTask
from tasks.SkVM.evaluators import eval_structured_l3, score_checkpoints


class StructuredGenL3(BaseTask):
    """gen.text.structured L3 — generate a 4-level-deep org hierarchy JSON.

    Conventions:
      - ``name`` matches the dataset string used by loaders.py and the
        ``task`` field on the feature JSONs.
      - Single input_field ``task`` carries the full generator prompt.
      - Single dispatch output_field ``answer`` — the generated JSON text.
      - ``scorer = "skvm_structured_l3_cp"`` keeps evaluation rows
        partitionable by primitive in the analysis views.
    """

    name = "skvm_structured_l3"
    scorer = "skvm_structured_l3_cp"

    # Parser module for output_field dispatch.
    _parser_module_path = "tasks.SkVM.parsers"

    default_input_fields: Dict[str, str] = {
        "task": "The structured JSON generation task to solve, as a natural-language spec.",
    }
    default_output_fields: Dict[str, str] = {
        "answer": "The generated JSON object. Return ONLY the JSON, no surrounding prose.",
    }

    # ── few-shot / eval hooks ─────────────────────────────────────────

    def _gold_output(self, meta: dict, raw: dict) -> dict:
        # Open-ended generation — no single gold string. Few-shot selection
        # cannot use reference outputs; return an empty marker so
        # BaseTask._build_example_pool remains valid if called.
        return {"answer": ""}

    def build_record(self, query: dict, meta: dict, raw: dict) -> dict:
        """Expose the generator prompt under the ``task`` input_field."""
        return {"task": query.get("content", "")}

    def score(self, prediction: str, query_meta: dict) -> Tuple[float, Dict[str, Any]]:
        """Run the patched structured-L3 evaluator and aggregate checkpoints."""
        if isinstance(query_meta, str):
            query_meta = json.loads(query_meta)

        raw = query_meta.get("_raw", {}) or {}
        params = raw.get("eval_params") or raw.get("predicates") or {}
        if not params:
            return 0.0, {
                "status": "missing_eval_params",
                "prediction_preview": prediction[:200],
            }

        checkpoints = eval_structured_l3(prediction, params)
        return score_checkpoints(checkpoints)
