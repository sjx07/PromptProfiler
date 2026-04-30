"""Validator unit tests — exercise factory.validators.validate_feature against
existing round-1/round-2 features.

Round-1/round-2 features should all pass validation; a deliberately-malformed
spec should fail.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from factory.validators import validate_feature

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = REPO_ROOT / "features"


def _round_1_2_cases() -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for task_dir in sorted(FEATURES_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        for path in sorted(task_dir.glob("[!_]*.json")):
            if path.name.startswith(("base_", "gepa_")):
                continue
            try:
                spec = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if spec.get("provenance") in {"creative", "vista_synthesis", "gepa_decompose", "round_2_factory"}:
                out.append((task_dir.name, path))
    return out


_CASES = _round_1_2_cases()


@pytest.mark.skipif(not _CASES, reason="no authored features discovered")
@pytest.mark.parametrize("task,path", _CASES)
def test_authored_feature_validates(task: str, path: Path) -> None:
    spec = json.loads(path.read_text())
    result = validate_feature(spec, task)
    assert result.ok, f"{path.name}: {result.errors}"


def test_malformed_feature_fails() -> None:
    bad = {"canonical_id": "bogus_foo", "target_module": "summarize1", "primitive_edits": []}
    r = validate_feature(bad, "hover_context")
    assert not r.ok
    assert any("primitive_edits" in e for e in r.errors)


def test_no_primitive_edits_fails() -> None:
    """target_module is now optional (single-prompt tasks); the empty edits check
    is the next failure mode for an obviously-bad feature."""
    bad = {"canonical_id": "no_edits", "primitive_edits": []}
    r = validate_feature(bad, "hover_context")
    assert not r.ok
    assert any("primitive_edits" in e for e in r.errors)
