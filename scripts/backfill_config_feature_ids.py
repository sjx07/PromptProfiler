"""backfill_config_feature_ids.py — retro-fill config.meta.feature_ids.

Before the feature-level generators were fixed to write the FULL active
feature set into config.meta, only ``coalition_feature`` populated
``meta.feature_ids`` (and even there, only for the varying subset — not
base). This broke ``analyze.has_feature()`` and the ``feature_effect`` view
on cubes produced by the earlier code.

This script walks every config, reconstructs its feature set from
``config.func_ids`` via the feature table's ``primitive_spec``, and writes
the full ``canonical_ids`` + ``feature_ids`` arrays into ``config.meta``
without touching anything else.

Safe to re-run: it's idempotent (same content → same arrays written).

Usage:
    python3 -m prompt_profiler.scripts.backfill_config_feature_ids <db_path>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

from prompt_profiler.core.store import CubeStore


def _feature_func_signatures(feature_rows) -> Dict[str, Set[str]]:
    """For each feature_id, compute the set of func_type:canonicalized-params
    strings contributed by its primitive_edits. We match these against the
    config's funcs to decide whether the feature is 'in' the config.
    """
    # Import lazily so this script works even when the registry isn't importable
    # from a clean shell.
    from prompt_profiler.core.feature_registry import _canonical_edit

    sigs: Dict[str, Set[str]] = {}
    for r in feature_rows:
        fid = r["feature_id"]
        try:
            edits = json.loads(r["primitive_spec"])
        except json.JSONDecodeError:
            continue
        sigs[fid] = {_canonical_edit(e) for e in edits}
    return sigs


def _config_func_signatures(func_rows, func_ids: List[str]) -> Set[str]:
    """Compute func signatures for a given config's func_ids."""
    from prompt_profiler.core.feature_registry import _canonical_edit

    by_id = {r["func_id"]: r for r in func_rows}
    out: Set[str] = set()
    for fid in func_ids:
        row = by_id.get(fid)
        if row is None:
            continue
        params = json.loads(row["params"]) if isinstance(row["params"], str) else row["params"]
        out.add(_canonical_edit({"func_type": row["func_type"], "params": params}))
    return out


def backfill(db_path: str) -> dict:
    store = CubeStore(db_path)
    conn = store._get_conn()

    feature_rows = conn.execute(
        "SELECT feature_id, canonical_id, primitive_spec FROM feature"
    ).fetchall()
    if not feature_rows:
        print("No features in cube — nothing to backfill.")
        return {"configs_examined": 0, "configs_updated": 0}

    fid_to_cid = {r["feature_id"]: r["canonical_id"] for r in feature_rows}
    feat_sigs = _feature_func_signatures(feature_rows)

    func_rows = conn.execute("SELECT func_id, func_type, params FROM func").fetchall()
    cfg_rows = conn.execute("SELECT config_id, func_ids, meta FROM config").fetchall()

    examined = 0
    updated = 0
    for cfg in cfg_rows:
        examined += 1
        try:
            meta = json.loads(cfg["meta"] or "{}")
        except json.JSONDecodeError:
            meta = {}
        try:
            func_ids = json.loads(cfg["func_ids"] or "[]")
        except json.JSONDecodeError:
            continue

        config_sigs = _config_func_signatures(func_rows, func_ids)

        # A feature is 'in' the config iff ALL its primitive_edit signatures
        # appear in the config's func signatures.
        active_fids: List[str] = []
        for fid, sigs in feat_sigs.items():
            if sigs and sigs.issubset(config_sigs):
                active_fids.append(fid)
        # Preserve determinism across runs.
        active_fids.sort()
        active_cids = [fid_to_cid[f] for f in active_fids]

        new_meta = dict(meta)
        changed = (
            new_meta.get("feature_ids") != active_fids
            or new_meta.get("canonical_ids") != active_cids
        )
        if changed:
            new_meta["feature_ids"] = active_fids
            new_meta["canonical_ids"] = active_cids
            with store._cursor() as cur:
                cur.execute(
                    "UPDATE config SET meta = ? WHERE config_id = ?",
                    (json.dumps(new_meta), cfg["config_id"]),
                )
            updated += 1
            print(f"[cfg {cfg['config_id']:>4}] {len(active_cids)} features: "
                  f"{', '.join(active_cids) or '(none)'}")
        else:
            print(f"[cfg {cfg['config_id']:>4}] already correct, skipping")

    store.close()
    return {"configs_examined": examined, "configs_updated": updated}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("db_path", help="Path to the CubeStore SQLite file")
    args = ap.parse_args()
    if not Path(args.db_path).exists():
        print(f"ERROR: {args.db_path} does not exist", file=sys.stderr)
        sys.exit(1)
    result = backfill(args.db_path)
    print(f"\nDone: examined {result['configs_examined']} configs, "
          f"updated {result['configs_updated']}.")


if __name__ == "__main__":
    main()
