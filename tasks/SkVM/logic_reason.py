"""SkVM logic reasoning L2 task — seating arrangement puzzle."""
from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from task import BaseTask
from tasks.SkVM.evaluators import eval_logic_l2, score_checkpoints


class LogicReasonL2(BaseTask):
    """reason.logic L2 — K-person seating arrangement with constraints."""

    name = "skvm_logic_l2"
    scorer = "skvm_logic_l2_cp"

    _parser_module_path = "tasks.SkVM.parsers"

    default_input_fields: Dict[str, str] = {
        "task": "The logic puzzle to solve.",
    }
    default_output_fields: Dict[str, str] = {
        "answer": "The name of the person at the target position.",
    }

    def _gold_output(self, meta: dict, raw: dict) -> dict:
        return {"answer": raw.get("eval_params", {}).get("answer", "")}

    def build_record(self, query: dict, meta: dict, raw: dict) -> dict:
        return {"task": query.get("content", "")}

    def score(self, prediction: str, query_meta: dict) -> Tuple[float, Dict[str, Any]]:
        if isinstance(query_meta, str):
            query_meta = json.loads(query_meta)
        raw = query_meta.get("_raw", {}) or {}
        params = raw.get("eval_params") or {}
        if not params:
            return 0.0, {
                "status": "missing_eval_params",
                "prediction_preview": prediction[:200],
            }
        checkpoints = eval_logic_l2(prediction, params)
        return score_checkpoints(checkpoints)
