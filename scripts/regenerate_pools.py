#!/usr/bin/env python3
"""scripts/regenerate_pools.py — regenerate pool*.json files in the new shape.

New shape: a JSON array of primitive func specs:
  [{"func_type": "insert_node", "params": {...}, "meta": {...}}, ...]

This script reads the section feature files (_section_*.json) for each task
and emits a pool file containing the section insert_node specs.

Usage:
    python scripts/regenerate_pools.py [--dry-run]

Output:
    pools/table_qa_pool.json
    pools/sql_repair_pool.json

Since there are no old-shape pool files (they were never committed),
no archival is needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make prompt_profiler importable when run from the submodule root
_PKG_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PKG_DIR))

from prompt_profiler.core.func_registry import make_func_id

TASKS = ["table_qa", "sql_repair"]
FEATURES_BASE = Path(__file__).parent.parent / "features"
POOLS_DIR = Path(__file__).parent.parent / "pools"


def build_pool_for_task(task: str) -> list:
    """Read _section_*.json files for task and return list of insert_node specs."""
    task_dir = FEATURES_BASE / task
    specs = []
    for path in sorted(task_dir.glob("_section_*.json")):
        feat = json.loads(path.read_text())
        for edit in feat.get("primitive_edits", []):
            spec = {
                "func_type": edit["func_type"],
                "params": edit["params"],
                "meta": {"pool_id": feat["feature_id"]},
            }
            specs.append(spec)
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate pool*.json files in new shape.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written without writing files.")
    args = parser.parse_args()

    POOLS_DIR.mkdir(parents=True, exist_ok=True)

    for task in TASKS:
        specs = build_pool_for_task(task)
        out_path = POOLS_DIR / f"{task}_pool.json"

        if args.dry_run:
            print(f"[dry-run] Would write {len(specs)} specs to {out_path}")
            print(json.dumps(specs, indent=2))
        else:
            out_path.write_text(json.dumps(specs, indent=2) + "\n")
            print(f"Wrote {len(specs)} specs to {out_path}")


if __name__ == "__main__":
    main()
