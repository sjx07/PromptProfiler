"""Feature registry — load features from JSON specs and enforce dependencies.

A feature is a named bundle of primitive func specs (insert_node, input_transform, etc.)
annotated with dependency metadata:
  requires       — list of canonical_ids that must also be in the config (error if missing)
  conflicts_with — list of canonical_ids that must NOT be in the config (error if present)

Cross-task requires are disallowed (error).

Sections are ordinary features (canonical_id="_section_<name>") with a single
insert_node(section) primitive_edit. No special-casing or $ref resolution.

Content-addressed feature_id
─────────────────────────────
Each feature spec's ``feature_id`` is a 12-char SHA-256 hex derived exclusively
from its ``primitive_edits`` list (NOT from requires / conflicts_with / canonical_id
/ rationale).  Same primitive content → same feature_id.  Different content → different
feature_id (even if canonical_id is the same).

``canonical_id`` is the stable human-facing label.  It is the key used in
``validate_feature_set()`` and ``materialize()`` (user-facing API).

Cross-task: two features with canonical_id="enable_cot" but different section
parent_ids (because the sections live in different tasks) produce different
feature_ids.  Both share the same canonical_id.

Usage:
    from prompt_profiler.core.feature_registry import FeatureRegistry

    reg = FeatureRegistry.load(task="table_qa")
    specs, provenance = reg.materialize(["_section_reasoning", "enable_cot"])
    # provenance keys are content-hash feature_ids
    # use reg.feature_id_for("enable_cot") to resolve
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from prompt_profiler.core.func_registry import make_func_id, _canonicalize_insert_node

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


# ── content-addressed feature_id ──────────────────────────────────────

def _canonical_edit(edit: dict) -> str:
    """Canonicalize a single primitive_edit to a stable string for hashing.

    For ``insert_node`` edits, uses the same canonicalization as func_registry
    to produce a string that matches make_func_id's identity_key.
    For all other func_types, sorts params keys.
    """
    func_type = edit.get("func_type", "")
    params = edit.get("params", {})
    if func_type == "insert_node":
        canon = _canonicalize_insert_node(params)
        return f"{func_type}:{json.dumps(canon, sort_keys=True)}"
    return f"{func_type}:{json.dumps(params, sort_keys=True)}"


def compute_feature_id(primitive_edits: List[dict]) -> str:
    """Content-addressed 12-char hex ID for a feature.

    Hash input = canonicalized primitive_edits only.
    NOT requires / conflicts_with / canonical_id / rationale.
    """
    canonical = json.dumps(
        sorted(_canonical_edit(e) for e in primitive_edits),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


class FeatureRegistry:
    """Loads feature specs for a task and enforces dependency rules.

    Public API uses ``canonical_id`` (human-readable, e.g. "enable_cot").
    Internally, each feature has a content-addressed ``feature_id`` (12-char hash).

    Lookups:
        reg.feature_id_for("enable_cot")    -> "abc123def456"
        reg.canonical_id_for("abc123def456") -> "enable_cot"
    """

    def __init__(self, task: str, features: Dict[str, dict]) -> None:
        """
        Args:
            task: Task name.
            features: Dict keyed by canonical_id -> feature spec.
                      feature_id is computed from primitive_edits at init time.
        """
        self.task = task

        # canonical_id -> full spec (with feature_id injected)
        self._by_canonical: Dict[str, dict] = {}
        # feature_id (hash) -> full spec
        self._by_feature_id: Dict[str, dict] = {}
        # canonical_id -> feature_id
        self._cid_to_fid: Dict[str, str] = {}
        # feature_id -> canonical_id
        self._fid_to_cid: Dict[str, str] = {}

        for canonical_id, spec in features.items():
            primitive_edits = spec.get("primitive_edits", [])
            feature_id = compute_feature_id(primitive_edits)

            enriched = dict(spec)
            enriched["feature_id"] = feature_id
            enriched["canonical_id"] = enriched.get("canonical_id", canonical_id)

            self._by_canonical[canonical_id] = enriched
            self._by_feature_id[feature_id] = enriched
            self._cid_to_fid[canonical_id] = feature_id
            self._fid_to_cid[feature_id] = canonical_id

    # ── lookup helpers ────────────────────────────────────────────────

    def feature_id_for(self, canonical_id: str) -> str:
        """Return the content-hash feature_id for a given canonical_id.

        Raises KeyError if the canonical_id is not in this registry.
        """
        return self._cid_to_fid[canonical_id]

    def canonical_id_for(self, feature_id: str) -> str:
        """Return the canonical_id for a given content-hash feature_id.

        Raises KeyError if the feature_id is not in this registry.
        """
        return self._fid_to_cid[feature_id]

    # ── loading ───────────────────────────────────────────────────────

    @classmethod
    def load(cls, task: str, features_base: Optional[Path] = None) -> "FeatureRegistry":
        """Load all feature specs for a task from features/<task>/*.json.

        Args:
            task: Task name (subdirectory under features_base).
            features_base: Override base directory. If None, resolution falls
                through env PROMPTPROFILER_FEATURES_BASE then the package default.

        Each spec's ``canonical_id`` is read from the JSON ``canonical_id`` field;
        if absent, the filename stem is used as fallback.  ``_source_path`` is
        attached to each spec so sync_to_cube can record provenance.
        """
        base = _resolve_features_base(features_base)
        task_dir = base / task
        if not task_dir.exists():
            raise FileNotFoundError(f"Feature directory not found: {task_dir}")

        features: Dict[str, dict] = {}
        for path in sorted(task_dir.glob("*.json")):
            try:
                spec = json.loads(path.read_text())
                # canonical_id: explicit field > legacy feature_id field > filename stem
                canonical_id = spec.get("canonical_id") or spec.get("feature_id") or path.stem
                spec["canonical_id"] = canonical_id
                spec["_source_path"] = str(path)
                features[canonical_id] = spec
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse %s: %s", path, e)

        logger.info("FeatureRegistry loaded task=%s: %d features", task, len(features))
        return cls(task=task, features=features)

    # ── dependency enforcement ────────────────────────────────────────

    def validate_feature_set(self, canonical_ids: List[str]) -> None:
        """Enforce requires / conflicts_with / cross-task constraints.

        Args:
            canonical_ids: List of canonical_ids to validate.

        Raises ValueError on violation. Does NOT do transitive auto-include
        (per round 7 decision 1a) — the user must list all required features
        explicitly.
        """
        id_set = set(canonical_ids)

        for cid in canonical_ids:
            spec = self._by_canonical.get(cid)
            if spec is None:
                raise ValueError(
                    f"Feature '{cid}' not found in task='{self.task}'. "
                    f"Available: {sorted(self._by_canonical)}"
                )

            # Cross-task dependency check
            spec_task = spec.get("task")
            if spec_task and spec_task != self.task:
                raise ValueError(
                    f"Feature '{cid}' belongs to task='{spec_task}', "
                    f"but registry is for task='{self.task}'. "
                    f"cross-task feature references are disallowed."
                )

            # requires check: all listed canonical_ids must be in id_set
            for req in spec.get("requires", []):
                if req not in id_set:
                    raise ValueError(
                        f"Feature '{cid}' requires '{req}', which is not in the config. "
                        f"Add '{req}' to feature_ids."
                    )
                # Cross-task requires check
                req_spec = self._by_canonical.get(req)
                if req_spec:
                    req_task = req_spec.get("task")
                    if req_task and req_task != self.task:
                        raise ValueError(
                            f"Feature '{cid}' requires '{req}' from task='{req_task}'. "
                            f"Cross-task requires are disallowed."
                        )

            # conflicts_with check
            for conflict in spec.get("conflicts_with", []):
                if conflict in id_set:
                    raise ValueError(
                        f"Features '{cid}' and '{conflict}' conflict with each other. "
                        f"Remove one from the config."
                    )

    # ── materialization ───────────────────────────────────────────────

    def materialize(
        self, canonical_ids: List[str]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
        """Validate features and expand to primitive func specs.

        Args:
            canonical_ids: Human-readable feature identifiers (e.g. ["enable_cot"]).

        Returns:
            specs: list of {"func_id", "func_type", "params", "meta"} dicts
                ready for upsert_funcs / apply_config.
            feature_to_funcs: dict mapping **content-hash feature_id** -> [func_id, ...]
                for provenance tracking. Use canonical_id_for() / feature_id_for()
                to translate between canonical and hash keys.
                A func_id shared by multiple features appears in each feature's list
                (many-to-many attribution).

        Raises ValueError on dependency violations.
        """
        self.validate_feature_set(canonical_ids)

        func_specs: List[Dict[str, Any]] = []
        seen_func_ids: Dict[str, int] = {}   # func_id -> index in func_specs
        feature_to_funcs: Dict[str, List[str]] = {}  # keyed by content-hash feature_id

        for cid in canonical_ids:
            spec = self._by_canonical[cid]
            fid = self._cid_to_fid[cid]
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
        """Return sorted list of all loaded canonical_ids."""
        return sorted(self._by_canonical)

    def all_specs(self) -> List[dict]:
        """Return all feature specs (with feature_id and canonical_id populated)."""
        return list(self._by_feature_id.values())
