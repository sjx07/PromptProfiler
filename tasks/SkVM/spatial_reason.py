"""SkVM spatial reasoning L3 task — great-circle distance between two cities."""
from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from task import BaseTask
from tasks.SkVM.evaluators import eval_spatial_l3, score_checkpoints


class SpatialReasonL3(BaseTask):
    """reason.spatial L3 — haversine distance between two of 8 fixed cities."""

    name = "skvm_spatial_l3"
    scorer = "skvm_spatial_l3_cp"

    _parser_module_path = "tasks.SkVM.parsers"

    default_input_fields: Dict[str, str] = {
        "task": "The spatial-reasoning task to solve.",
    }
    default_output_fields: Dict[str, str] = {
        "answer": "The integer great-circle distance in kilometers.",
    }

    def _gold_output(self, meta: dict, raw: dict) -> dict:
        expected = raw.get("eval_params", {}).get("expected_km", "")
        return {"answer": str(expected)}

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
        checkpoints = eval_spatial_l3(prediction, params)
        return score_checkpoints(checkpoints)
