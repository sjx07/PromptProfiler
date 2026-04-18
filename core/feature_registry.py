"""Feature registry — load features from JSON specs and enforce dependencies.

A feature is a named bundle of primitive func specs (insert_node, input_transform, etc.)
annotated with dependency metadata:
  requires       — list of feature_ids that must also be in the config (error if missing)
  conflicts_with — list of feature_ids that must NOT be in the config (error if present)

Cross-task requires are disallowed (error).

Usage:
    from prompt_profiler.core.feature_registry import FeatureRegistry

    reg = FeatureRegistry.load(task="table_qa")
    func_specs = reg.materialize(["enable_cot"])
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from prompt_profiler.core.func_registry import ROOT_ID, make_func_id

logger = logging.getLogger(__name__)

# Base path for feature files: features/<task>/<feature>.json
_FEATURES_BASE = Path(__file__).parent.parent / "features"


class FeatureRegistry:
    """Loads feature specs for a task and enforces dependency rules."""

    def __init__(self, task: str, features: Dict[str, dict], sections: Dict[str, dict]) -> None:
        self.task = task
        self._features = features        # feature_id -> feature spec
        self._sections = sections        # section key -> section params

    # ── loading ───────────────────────────────────────────────────────

    @classmethod
    def load(cls, task: str) -> "FeatureRegistry":
        """Load all feature specs for a task from features/<task>/*.json."""
        task_dir = _FEATURES_BASE / task
        if not task_dir.exists():
            raise FileNotFoundError(f"Feature directory not found: {task_dir}")

        # Load _sections.json catalog first
        sections_path = task_dir / "_sections.json"
        sections: Dict[str, dict] = {}
        if sections_path.exists():
            raw = json.loads(sections_path.read_text())
            sections = raw.get("sections", {})

        # Load all feature files (skip _sections.json)
        features: Dict[str, dict] = {}
        for path in sorted(task_dir.glob("*.json")):
            if path.name == "_sections.json":
                continue
            try:
                spec = json.loads(path.read_text())
                fid = spec.get("feature_id")
                if not fid:
                    logger.warning("Feature file %s missing feature_id, skipping", path)
                    continue
                features[fid] = spec
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse %s: %s", path, e)

        logger.info("FeatureRegistry loaded task=%s: %d features, %d sections",
                    task, len(features), len(sections))
        return cls(task=task, features=features, sections=sections)

    # ── dependency enforcement ────────────────────────────────────────

    def validate_feature_set(self, feature_ids: List[str]) -> None:
        """Enforce requires / conflicts_with / cross-task constraints.

        Raises ValueError on violation. Does NOT do transitive auto-include
        (per round 7 decision 1a) — the user must list all required features
        explicitly.
        """
        id_set = set(feature_ids)

        for fid in feature_ids:
            spec = self._features.get(fid)
            if spec is None:
                raise ValueError(
                    f"Feature '{fid}' not found in task='{self.task}'. "
                    f"Available: {sorted(self._features)}"
                )

            # Cross-task dependency check
            spec_task = spec.get("task")
            if spec_task and spec_task != self.task:
                raise ValueError(
                    f"Feature '{fid}' belongs to task='{spec_task}', "
                    f"but registry is for task='{self.task}'. "
                    f"cross-task feature references are disallowed."
                )

            # requires check: all listed features must be in id_set
            for req in spec.get("requires", []):
                if req not in id_set:
                    raise ValueError(
                        f"Feature '{fid}' requires '{req}', which is not in the config. "
                        f"Add '{req}' to feature_ids."
                    )
                # Cross-task requires check
                req_spec = self._features.get(req)
                if req_spec:
                    req_task = req_spec.get("task")
                    if req_task and req_task != self.task:
                        raise ValueError(
                            f"Feature '{fid}' requires '{req}' from task='{req_task}'. "
                            f"Cross-task requires are disallowed."
                        )

            # conflicts_with check
            for conflict in spec.get("conflicts_with", []):
                if conflict in id_set:
                    raise ValueError(
                        f"Features '{fid}' and '{conflict}' conflict with each other. "
                        f"Remove one from the config."
                    )

    # ── $ref resolution ───────────────────────────────────────────────

    def _resolve_parent_ref(self, params: dict) -> dict:
        """Resolve {"$ref": "_sections.X"} parent refs via _sections.json.

        Replaces params["parent"] = {"$ref": "_sections.reasoning"}
        with params["parent_id"] = <computed func_id for that section>.
        """
        parent_ref = params.get("parent")
        if not isinstance(parent_ref, dict):
            return params

        ref = parent_ref.get("$ref", "")
        if not ref.startswith("_sections."):
            return params

        section_key = ref[len("_sections."):]
        section_def = self._sections.get(section_key)
        if section_def is None:
            raise ValueError(
                f"$ref '{ref}' not found in _sections.json for task='{self.task}'. "
                f"Available section keys: {sorted(self._sections)}"
            )

        # Build the canonical section params to compute its func_id
        section_params = {
            "node_type": "section",
            "parent_id": ROOT_ID,
            "payload": {
                "title":     section_def.get("title", section_key),
                "ordinal":   int(section_def.get("ordinal", 0)),
                "is_system": bool(section_def.get("is_system", False)),
                "min_rules": int(section_def.get("min_rules", 0)),
                "max_rules": int(section_def.get("max_rules", 10)),
            },
        }
        section_func_id = make_func_id("insert_node", section_params)

        new_params = {k: v for k, v in params.items() if k != "parent"}
        new_params["parent_id"] = section_func_id
        return new_params

    # ── materialization ───────────────────────────────────────────────

    def materialize(self, feature_ids: List[str]) -> List[Dict[str, Any]]:
        """Validate features and expand to a list of primitive func specs.

        Returns list of {"func_type": ..., "params": ..., "func_id": ...} dicts
        ready for seed_funcs / apply_config.

        Raises ValueError on dependency violations.
        """
        self.validate_feature_set(feature_ids)

        func_specs: List[Dict[str, Any]] = []
        seen_func_ids: set = set()

        for fid in feature_ids:
            spec = self._features[fid]
            for edit in spec.get("primitive_edits", []):
                func_type = edit.get("func_type", "")
                params = dict(edit.get("params", {}))

                # Resolve $ref parent references
                params = self._resolve_parent_ref(params)

                func_id = make_func_id(func_type, params)
                if func_id in seen_func_ids:
                    continue  # dedup (multiple features can reference same section)
                seen_func_ids.add(func_id)

                func_specs.append({
                    "func_id":   func_id,
                    "func_type": func_type,
                    "params":    params,
                    "meta":      {"source_feature": fid},
                })

        return func_specs

    def list_features(self) -> List[str]:
        """Return sorted list of all loaded feature_ids."""
        return sorted(self._features)
