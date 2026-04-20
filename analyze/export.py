"""Layer 7 — error-report exporters.

Builds on the existing ``analyze.compare.flip_rows`` primitive plus
the ``analyze.data`` configs/predicates fetchers to produce a single
flat table of "every query where any feature flipped the score
relative to a baseline".

Output is intended for two consumers:

  * Humans eyeballing failures (CSV).
  * Downstream LLM-driven feature-discovery loops (JSONL — multi-line
    raw responses + list-valued gold survive without escape-tax).

Public entry point:

  * ``flipped_responses(store, *, base_config_id, model, scorer, ...)``

Pure read — no writes to the cube.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from core.store import CubeStore
from analyze import data
from analyze.compare import flip_rows


def flipped_responses(
    store: CubeStore,
    *,
    base_config_id: int,
    model: str,
    scorer: str,
    target_configs: Optional[List[int]] = None,
    feature_filter: Optional[List[str]] = None,
    direction: str = "both",                 # "up" | "down" | "both"
    include_predicates: bool = True,
    out_path: Optional[Union[str, Path]] = None,
    fmt: str = "jsonl",                      # "jsonl" | "csv"
    git_add: bool = False,                   # `git add` the output after writing
):
    """Return + optionally write a flat table of base-vs-target score flips.

    For each target config in scope (default: all configs ≠ base), call
    ``flip_rows`` and concatenate, tagging each row with the target's
    canonical_id / feature_id (when resolvable from ``config.meta``).
    Optionally annotates each row with the predicate tags for its query
    + base/target ``execution.error`` fields.

    Args:
        base_config_id: the baseline to compare against.
        model, scorer:  pin the evaluation.
        target_configs: restrict to these config_ids. Default: every
                        config in the cube except the base.
        feature_filter: restrict output to these canonical_ids
                        (post-filter on the joined table).
        direction:      "up" (base wrong → target right),
                        "down" (base right → target wrong),
                        "both" (default).
        include_predicates: if True, attach a ``predicates`` dict
                        ({name: value}) per row (one DB hit total).
        out_path:       if set, write the DataFrame to disk in the
                        chosen format.
        fmt:            "jsonl" (default — preserves newlines + lists)
                        or "csv" (gold + predicates JSON-serialized).
        git_add:        if True (and ``out_path`` set), run
                        ``git add <out_path>`` after writing. Quiet
                        no-op when not in a git repo or git missing
                        (warns instead of crashing). Raises if
                        ``out_path`` is None.

    Returns:
        pandas.DataFrame with columns:
            target_config_id, feature_canonical_id, feature_id,
            query_id, question, direction,
            base_score, target_score,
            base_prediction, target_prediction,
            base_raw, target_raw,
            gold,
            predicates                (when include_predicates=True),
            error_base, error_target  (always; "" when no error)
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas required for flipped_responses") from e

    if direction not in ("up", "down", "both"):
        raise ValueError(f"direction must be 'up' | 'down' | 'both'; got {direction!r}")
    if fmt not in ("jsonl", "csv"):
        raise ValueError(f"fmt must be 'jsonl' | 'csv'; got {fmt!r}")
    if git_add and out_path is None:
        raise ValueError("git_add=True requires out_path to be set")

    # ── candidate configs ────────────────────────────────────────────
    cdf = data.configs_df(store)
    if cdf.empty:
        empty = pd.DataFrame(columns=_OUTPUT_COLUMNS)
        if out_path:
            _write(empty, out_path, fmt)
        return empty

    if target_configs is None:
        target_configs = [int(cid) for cid in cdf["config_id"].tolist()
                          if int(cid) != base_config_id]
    else:
        target_configs = [int(c) for c in target_configs if int(c) != base_config_id]

    # canonical_id / feature_id lookup from config.meta.
    cid_to_meta = {int(r["config_id"]): r["meta"] for _, r in cdf.iterrows()}

    # ── error map (per execution row) ────────────────────────────────
    relevant_cids = set(target_configs) | {base_config_id}
    err_sql = (
        "SELECT config_id, query_id, COALESCE(error, '') AS error "
        "FROM execution WHERE model = ? AND config_id IN ({})"
    ).format(",".join("?" * len(relevant_cids)))
    err_map: Dict[tuple, str] = {}
    if relevant_cids:
        for r in store._get_conn().execute(
            err_sql, (model, *relevant_cids),
        ).fetchall():
            err_map[(int(r["config_id"]), r["query_id"])] = r["error"] or ""

    # ── predicate tags per query (one fetch, stitched in memory) ─────
    pred_map: Dict[str, Dict[str, str]] = {}
    if include_predicates:
        for r in store._get_conn().execute(
            "SELECT query_id, name, value FROM predicate"
        ).fetchall():
            pred_map.setdefault(r["query_id"], {})[r["name"]] = r["value"]

    # ── iterate targets, harvest flips, tag ──────────────────────────
    records: List[Dict[str, Any]] = []
    for tcid in target_configs:
        rows = flip_rows(
            store,
            base_config=base_config_id, target_config=tcid,
            model=model, scorer=scorer, direction=direction,
        )
        meta = cid_to_meta.get(tcid, {}) or {}
        canonical = meta.get("canonical_id")
        feat_id = meta.get("feature_id")
        if feature_filter is not None and canonical not in feature_filter:
            continue
        for r in rows:
            qid = r["query_id"]
            rec: Dict[str, Any] = {
                "target_config_id":     tcid,
                "feature_canonical_id": canonical,
                "feature_id":           feat_id,
                "query_id":             qid,
                "question":             r.get("question", ""),
                "direction":            r["direction"],
                "base_score":           r["base_score"],
                "target_score":         r["target_score"],
                "base_prediction":      r.get("base_prediction", ""),
                "target_prediction":    r.get("target_prediction", ""),
                "base_raw":             r.get("base_raw", ""),
                "target_raw":           r.get("target_raw", ""),
                "gold":                 r.get("gold", []),
                "error_base":           err_map.get((base_config_id, qid), ""),
                "error_target":         err_map.get((tcid, qid), ""),
            }
            if include_predicates:
                rec["predicates"] = pred_map.get(qid, {})
            records.append(rec)

    df = pd.DataFrame(records, columns=_OUTPUT_COLUMNS if include_predicates
                      else [c for c in _OUTPUT_COLUMNS if c != "predicates"])
    if out_path:
        _write(df, out_path, fmt)
        if git_add:
            _git_add(out_path)
    return df


_OUTPUT_COLUMNS = [
    "target_config_id", "feature_canonical_id", "feature_id",
    "query_id", "question", "direction",
    "base_score", "target_score",
    "base_prediction", "target_prediction",
    "base_raw", "target_raw",
    "gold", "predicates",
    "error_base", "error_target",
]


def _write(df, out_path: Union[str, Path], fmt: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with out_path.open("w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                f.write(json.dumps(_jsonable(row.to_dict()),
                                   ensure_ascii=False) + "\n")
    else:  # csv
        # Serialize list/dict columns as JSON for spreadsheet safety.
        df2 = df.copy()
        for col in ("gold", "predicates"):
            if col in df2.columns:
                df2[col] = df2[col].map(
                    lambda v: json.dumps(v, ensure_ascii=False) if v is not None else ""
                )
        df2.to_csv(out_path, index=False)


def _git_add(out_path: Union[str, Path]) -> None:
    """Run ``git add <out_path>`` from the file's directory.

    Quiet on success. Emits a ``UserWarning`` and returns instead of
    raising when:
      * git is not installed
      * the file isn't inside a git working tree
      * git itself returns non-zero (e.g. .gitignored, fingerprint
        mismatch, etc.)

    The cube-export workflow doesn't run inside the same repo on every
    machine; we'd rather skip than abort the export.
    """
    import shutil
    import subprocess
    import warnings

    out_path = Path(out_path)
    if shutil.which("git") is None:
        warnings.warn("git_add=True but `git` not found on PATH — skipping",
                      stacklevel=3)
        return
    cwd = out_path.parent if out_path.parent.exists() else Path.cwd()
    # Inside a working tree?
    inside = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=cwd, capture_output=True, text=True,
    )
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        warnings.warn(
            f"git_add=True but {out_path} is not inside a git work tree — skipping",
            stacklevel=3,
        )
        return
    res = subprocess.run(
        ["git", "add", "--", str(out_path.resolve())],
        cwd=cwd, capture_output=True, text=True,
    )
    if res.returncode != 0:
        warnings.warn(
            f"git add {out_path} failed (exit {res.returncode}): {res.stderr.strip()}",
            stacklevel=3,
        )


def _jsonable(d: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce numpy scalars / pandas NaN to JSON-serializable values."""
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = None
            continue
        # numpy scalar?
        if hasattr(v, "item"):
            try:
                out[k] = v.item()
                continue
            except (ValueError, AttributeError):
                pass
        out[k] = v
    return out
