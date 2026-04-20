#!/usr/bin/env python3
"""CLI wrapper around ``analyze.export.flipped_responses``.

Dump every base-vs-target score flip in a cube to JSONL (default) or CSV
for downstream review or LLM-driven feature discovery.

Example:
    python scripts/export_flips.py \\
        --cube runs/wtq_mvp.db \\
        --base 1 \\
        --model "Qwen/Qwen3.6-35B-A3B" \\
        --scorer denotation_acc \\
        --out runs/wtq_mvp/flipped.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cube", required=True,
                   help="Path to the SQLite cube (read-only).")
    p.add_argument("--base", required=True, type=int,
                   help="Baseline config_id.")
    p.add_argument("--model", required=True)
    p.add_argument("--scorer", required=True)
    p.add_argument("--targets", default=None,
                   help="Comma-separated config_ids to compare; default = all configs ≠ base.")
    p.add_argument("--features", default=None,
                   help="Comma-separated canonical_ids to filter to.")
    p.add_argument("--direction", default="both",
                   choices=("up", "down", "both"))
    p.add_argument("--no-predicates", action="store_true",
                   help="Skip the predicate-tag annotation column.")
    p.add_argument("--out", required=True,
                   help="Output path; suffix decides format unless --fmt is set.")
    p.add_argument("--fmt", default=None, choices=("jsonl", "csv"),
                   help="Override output format. Default: infer from --out suffix.")
    p.add_argument("--git-add", action="store_true",
                   help="Run `git add <out>` after writing. Quiet skip if not "
                        "in a git work tree or git missing.")
    args = p.parse_args()

    # Make `tool/prompt_profiler/` importable when invoked directly.
    here = Path(__file__).resolve().parent.parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    from core.store import CubeStore
    from analyze.export import flipped_responses

    fmt = args.fmt or ("csv" if args.out.lower().endswith(".csv") else "jsonl")
    targets = [int(x) for x in args.targets.split(",")] if args.targets else None
    features = [s.strip() for s in args.features.split(",")] if args.features else None

    store = CubeStore(args.cube, read_only=True)
    df = flipped_responses(
        store,
        base_config_id=args.base,
        model=args.model, scorer=args.scorer,
        target_configs=targets,
        feature_filter=features,
        direction=args.direction,
        include_predicates=not args.no_predicates,
        out_path=args.out,
        fmt=fmt,
        git_add=args.git_add,
    )
    print(f"wrote {len(df)} rows → {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
