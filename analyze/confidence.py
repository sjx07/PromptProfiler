"""Layer 3 — bootstrap confidence.

Two engines:

  * ``paired_bootstrap``    — used when the same query_ids appear on
                              both arms (e.g. add-one design).
  * ``unpaired_bootstrap``  — used when each arm is a bag of independent
                              observations (e.g. per-config means under
                              ``method="marginal"``).

Both return ``(ci_lo, ci_hi, p_gt_zero)`` at the 95% level.
``p_gt_zero`` is the fraction of bootstrap resamples where lift > 0;
NOT a p-value.

Top-level ``attach_ci`` stitches the right engine onto a lift-shaped
DataFrame, given the underlying ``scores_df`` and the canonical mapping.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set


# ── engines ───────────────────────────────────────────────────────────

def paired_bootstrap(
    with_scores: Dict[str, float],
    without_scores: Dict[str, float],
    *, n_boot: int = 1000, seed: int = 42,
):
    """Paired bootstrap over shared query_ids.

    Returns (ci_lo, ci_hi, p_gt_zero). NaN×3 if fewer than 2 shared
    queries exist.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return (float("nan"), float("nan"), float("nan"))

    shared = sorted(set(with_scores) & set(without_scores))
    if len(shared) < 2:
        return (float("nan"), float("nan"), float("nan"))

    w = np.array([with_scores[q] for q in shared], dtype=float)
    o = np.array([without_scores[q] for q in shared], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(shared), size=(n_boot, len(shared)))
    diffs = w[idx].mean(axis=1) - o[idx].mean(axis=1)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p_gt = float((diffs > 0).mean())
    return (float(ci_lo), float(ci_hi), p_gt)


def unpaired_bootstrap(
    with_vals: List[float],
    without_vals: List[float],
    *, n_boot: int = 1000, seed: int = 42,
):
    """Unpaired bootstrap. Each side resampled independently.

    Returns (ci_lo, ci_hi, p_gt_zero). NaN×3 if either side has < 2
    observations.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return (float("nan"), float("nan"), float("nan"))

    w = np.asarray(with_vals, dtype=float)
    o = np.asarray(without_vals, dtype=float)
    if len(w) < 2 or len(o) < 2:
        return (float("nan"), float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    iw = rng.integers(0, len(w), size=(n_boot, len(w)))
    io_ = rng.integers(0, len(o), size=(n_boot, len(o)))
    diffs = w[iw].mean(axis=1) - o[io_].mean(axis=1)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p_gt = float((diffs > 0).mean())
    return (float(ci_lo), float(ci_hi), p_gt)


# ── df-level attacher ─────────────────────────────────────────────────

def _bootstrap_one(args):
    """Top-level worker so ``ProcessPoolExecutor.map`` can pickle it.

    ``args`` is ``(kind, w_data, o_data, n_boot, seed)`` where
    ``kind ∈ {'paired','unpaired','nan'}``.
    """
    kind, w, o, n_boot, seed = args
    if kind == "paired":
        return paired_bootstrap(w, o, n_boot=n_boot, seed=seed)
    if kind == "unpaired":
        return unpaired_bootstrap(w, o, n_boot=n_boot, seed=seed)
    return (float("nan"), float("nan"), float("nan"))


def attach_ci(
    lift_df_in,
    scores_df_in,
    *,
    method: str,                                       # "simple" | "marginal"
    base_config_id: Optional[int] = None,              # required for "simple"
    canonical_to_cid: Optional[Dict[str, int]] = None, # for "simple"
    canonical_to_with_cids: Optional[Dict[str, Set[int]]] = None,  # for "marginal"
    all_cids: Optional[List[int]] = None,              # for "marginal"
    n_boot: int = 1000,
    seed: int = 42,
    progress: bool = False,
    workers: int = 1,
):
    """Attach ci_lo, ci_hi, p_gt_zero, effect_lb columns to a lift DataFrame.

    Caller supplies the mappings; this layer doesn't touch SQL or the
    feature table directly.
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas required") from e

    df = lift_df_in.copy()
    if df.empty:
        for col in ("ci_lo", "ci_hi", "p_gt_zero", "effect_lb"):
            df[col] = []
        return df

    from analyze.data import per_query_scores

    all_cids_set = set(all_cids or [])
    # Cache per_query_scores results: (config_id, predicate_name) → {pv: {qid: score}}
    pqs_cache: Dict[tuple, Dict[str, Dict[str, float]]] = {}
    # Cache per-config means per (predicate_name, predicate_value) for marginal.
    marg_cache: Dict[tuple, Dict[int, float]] = {}

    def _pqs(cid, pname):
        key = (cid, pname)
        if key not in pqs_cache:
            pqs_cache[key] = per_query_scores(scores_df_in,
                                              config_id=cid, predicate_name=pname)
        return pqs_cache[key]

    def _marg(pname, pv):
        key = (pname, pv)
        if key not in marg_cache:
            sub = scores_df_in[
                (scores_df_in["predicate_name"] == pname) &
                (scores_df_in["predicate_value"] == pv)
            ]
            marg_cache[key] = sub.groupby("config_id")["score"].mean().to_dict()
        return marg_cache[key]

    # Build a flat list of bootstrap jobs (one per row). Each job is
    # a tuple (kind, w_data, o_data, n_boot, seed) — pickle-safe.
    jobs: List[tuple] = []
    for _, row in df.iterrows():
        canon = row["canonical_id"]
        pname = row["predicate_name"]
        pv = row["predicate_value"]
        if method == "simple":
            target_cid = (canonical_to_cid or {}).get(canon)
            if target_cid is None:
                jobs.append(("nan", None, None, n_boot, seed))
                continue
            w = _pqs(target_cid, pname).get(pv, {})
            o = _pqs(base_config_id, pname).get(pv, {})
            jobs.append(("paired", w, o, n_boot, seed))
        else:
            with_cfgs = (canonical_to_with_cids or {}).get(canon, set())
            without_cfgs = all_cids_set - with_cfgs
            per_cfg = _marg(pname, pv)
            w_vals = [per_cfg[c] for c in with_cfgs if c in per_cfg]
            o_vals = [per_cfg[c] for c in without_cfgs if c in per_cfg]
            jobs.append(("unpaired", w_vals, o_vals, n_boot, seed))

    # Dispatch — serial for workers<=1, ProcessPoolExecutor otherwise.
    from analyze._progress import progress_iter
    if workers <= 1 or len(jobs) <= 1:
        it = progress_iter(jobs, enable=progress,
                           total=len(jobs), desc="bootstrap")
        results = [_bootstrap_one(j) for j in it]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            # Use chunksize to amortize per-task overhead.
            chunksize = max(1, len(jobs) // (workers * 4))
            raw = pool.map(_bootstrap_one, jobs, chunksize=chunksize)
            results = list(progress_iter(
                raw, enable=progress, total=len(jobs), desc="bootstrap",
            ))

    ci_lo = [r[0] for r in results]
    ci_hi = [r[1] for r in results]
    p_col = [r[2] for r in results]

    df["ci_lo"] = ci_lo
    df["ci_hi"] = ci_hi
    df["p_gt_zero"] = p_col
    df["effect_lb"] = df["ci_lo"]
    return df
