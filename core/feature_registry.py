"""Feature registry — load features from JSON specs and enforce dependencies.

A feature is a named bundle of primitive func specs (insert_node, input_transform, etc.)
annotated with dependency metadata:
  requires       — list of feature_ids that must also be in the config (error if missing)
  conflicts_with — list of feature_ids that must NOT be in the config (error if present)

Cross-task requires are disallowed (error).

Sections are ordinary features (feature_id="_section_<name>") with a single
insert_node(section) primitive_edit. No special-casing or $ref resolution.

Usage:
    from prompt_profiler.core.feature_registry import FeatureRegistry

    reg = FeatureRegistry.load(task="table_qa")
    specs, provenance = reg.materialize(["_section_reasoning", "enable_cot"])
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prompt_profiler.core.func_registry import make_func_id

logger = logging.getLogger(__name__)

# Package-default base path for feature files: features/<task>/<feature>.json
# Override via FeatureRegistry.load(features_base=...) or env PROMPTPROFILER_FEATURES_BASE.
_FEATURES_BASE = Path(__file__).parent.parent / "features"


def _resolve_features_base(features_base: Optional[Path]) -> Path:
    """Resolve the features base directory.

    Resolution order:
      1. Explicit ``features_base`` argument (if not None)
      2. Environment variable ``PROMPTPROFILER_FEATURES_BASE``
      3. Package-default ``_FEATURES_BASE``
    """
    if features_base is not None:
        return Path(features_base)
    env_val = os.environ.get("PROMPTPROFILER_FEATURES_BASE")
    if env_val:
        return Path(env_val)
    return _FEATURES_BASE


class FeatureRegistry:
    """Loads feature specs for a task and enforces dependency rules."""

    def __init__(self, task: str, features: Dict[str, dict]) -> None:
        self.task = task
        self._features = features        # feature_id -> feature spec

    # ── loading ───────────────────────────────────────────────────────

    @classmethod
    def load(cls, task: str, features_base: Optional[Path] = None) -> "FeatureRegistry":
        """Load all feature specs for a task from features/<task>/*.json.

        Args:
            task: Task name (subdirectory under features_base).
            features_base: Override base directory. If None, resolution falls
                through env PROMPTPROFILER_FEATURES_BASE then the package default.
        """
        base = _resolve_features_base(features_base)
        task_dir = base / task
        if not task_dir.exists():
            raise FileNotFoundError(f"Feature directory not found: {task_dir}")

        # Load all feature files
        features: Dict[str, dict] = {}
        for path in sorted(task_dir.glob("*.json")):
            try:
                spec = json.loads(path.read_text())
                fid = spec.get("feature_id")
                if not fid:
                    logger.warning("Feature file %s missing feature_id, skipping", path)
                    continue
                features[fid] = spec
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse %s: %s", path, e)

        logger.info("FeatureRegistry loaded task=%s: %d features", task, len(features))
        return cls(task=task, features=features)

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

    # ── materialization ───────────────────────────────────────────────

    def materialize(
        self, feature_ids: List[str]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
        """Validate features and expand to primitive func specs.

        Returns:
            specs: list of {"func_id", "func_type", "params", "meta"} dicts
                ready for seed_funcs / apply_config.
            feature_to_funcs: dict mapping feature_id -> [func_id, ...] for
                provenance tracking. A func_id shared by multiple features
                appears in each feature's list (many-to-many attribution).

        Raises ValueError on dependency violations.
        """
        self.validate_feature_set(feature_ids)

        func_specs: List[Dict[str, Any]] = []
        seen_func_ids: Dict[str, int] = {}   # func_id -> index in func_specs
        feature_to_funcs: Dict[str, List[str]] = {}

        for fid in feature_ids:
            spec = self._features[fid]
            fid_funcs: List[str] = []

            for edit in spec.get("primitive_edits", []):
                func_type = edit.get("func_type", "")
                params = dict(edit.get("params", {}))

                func_id = make_func_id(func_type, params)

                if func_id not in seen_func_ids:
                    seen_func_ids[func_id] = len(func_specs)
                    func_specs.append({
                        "func_id":   func_id,
                        "func_type": func_type,
                        "params":    params,
                        "meta":      {},
                    })

                fid_funcs.append(func_id)

            feature_to_funcs[fid] = fid_funcs

        return func_specs, feature_to_funcs

    def list_features(self) -> List[str]:
        """Return sorted list of all loaded feature_ids."""
        return sorted(self._features)
