"""Round-MC run config consistency checks."""
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROUND_MC_DIR = REPO_ROOT / "runs" / "round_mc"
FEATURES_DIR = REPO_ROOT / "features"

CELLS = {
    "bird": {"split": "dev", "n_features": 35},
    "spider": {"split": "dev", "n_features": 34},
    "wtq": {"split": "test", "n_features": 32},
    "sqa": {"split": "dev", "n_features": 33},
    "tabfact": {"split": "validation", "n_features": 35},
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _feature_ids_for_cell(cell: str) -> list[str]:
    out: list[str] = []
    for path in sorted((FEATURES_DIR / cell).glob("*.json")):
        if path.name.startswith(("_section_", "base_")):
            continue
        spec = _load_json(path)
        canonical_id = spec.get("canonical_id")
        if canonical_id:
            out.append(canonical_id)
    return out


def test_round_mc_spec_matches_feature_dirs() -> None:
    spec = _load_json(ROUND_MC_DIR / "SPEC_TEMPLATE.json")
    per_cell = spec["feature_canonical_ids_per_cell"]

    for cell, expected in CELLS.items():
        feature_ids = _feature_ids_for_cell(cell)
        assert per_cell[cell] == feature_ids
        assert len(feature_ids) == expected["n_features"]


def test_round_mc_generated_configs_match_spec() -> None:
    spec = _load_json(ROUND_MC_DIR / "SPEC_TEMPLATE.json")
    per_cell = spec["feature_canonical_ids_per_cell"]
    base_per_cell = spec["base_features_per_cell"]

    for cell, expected in CELLS.items():
        for model_key in ("7b", "32b"):
            cfg = _load_json(ROUND_MC_DIR / f"round_mc_{cell}_{model_key}.json")
            assert cfg["experiment_type"] == "add_one_feature"
            assert cfg["task"] == cell
            assert cfg["split"] == expected["split"]
            assert cfg["max_queries"] == 1000
            assert cfg["sample_seed"] == 7
            assert cfg["base_features"] == base_per_cell[cell]
            assert cfg["experiment_features"] == per_cell[cell]
            assert len(cfg["experiment_features"]) == expected["n_features"]


def test_round_mc_index_and_pilot_are_current() -> None:
    index_entries = (ROUND_MC_DIR / "INDEX.txt").read_text().splitlines()
    expected_entries = [
        f"runs/round_mc/round_mc_{cell}_{model_key}.json"
        for cell in CELLS
        for model_key in ("7b", "32b")
    ]
    expected_entries.extend(
        f"runs/round_mc/round_mc_pilot_{cell}_7b.json"
        for cell in CELLS
    )

    assert index_entries == expected_entries

    spec = _load_json(ROUND_MC_DIR / "SPEC_TEMPLATE.json")
    for cell in CELLS:
        pilot = _load_json(ROUND_MC_DIR / f"round_mc_pilot_{cell}_7b.json")
        assert pilot["task"] == cell
        assert pilot["max_queries"] == 100
        assert pilot["experiment_features"] == spec["feature_canonical_ids_per_cell"][cell][:5]
