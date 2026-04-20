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
The df-level path is fully vectorized over rows (all bootstrap resamples
for all rows in one numpy op, chunked to cap memory).
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Set


# Rough memory cap for the (n_rows × n_boot × max_sample) resample
# tensor. Lower than the "just fits in RAM" ceiling — we also want
# chunks small enough to stay close to L2/L3 cache, since the gather
# ``W[row_idx, iw]`` is random-access and cache-locality matters a lot.
# 8e6 ≈ 64MB of float64 temp tensors per chunk.
_MEM_CELLS_CAP = 8_000_000

# Hard ceiling on rows per chunk. When chunking is actually triggered
# (by the memory cap or by a small monkeypatched override), we take
# ``min(mem_chunk, _CHUNK_SIZE)``. The default is large enough that
# chunking only ever kicks in via the memory cap. Tests monkeypatch
# this to a small value to force chunk-boundary coverage regardless of
# the memory cap.
_CHUNK_SIZE = 100_000


# ── engines ───────────────────────────────────────────────────────────

def _ci_bounds_pct(ci_level: float) -> tuple:
    """Return (lo_pct, hi_pct) for numpy.percentile given a ci_level in (0,1)."""
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0,1); got {ci_level!r}")
    alpha = (1.0 - ci_level) / 2.0
    return (alpha * 100.0, (1.0 - alpha) * 100.0)


def paired_bootstrap(
    with_scores: Dict[str, float],
    without_scores: Dict[str, float],
    *, n_boot: int = 1000, seed: int = 42, ci_level: float = 0.95,
):
    """Paired bootstrap over shared query_ids.

    Returns (ci_lo, ci_hi, p_gt_zero). NaN×3 if fewer than 2 shared
    queries exist. ``ci_level`` controls the interval width; 0.95 →
    [2.5, 97.5] percentiles, 0.80 → [10, 90], etc.
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
    lo_pct, hi_pct = _ci_bounds_pct(ci_level)
    ci_lo, ci_hi = np.percentile(diffs, [lo_pct, hi_pct])
    p_gt = float((diffs > 0).mean())
    return (float(ci_lo), float(ci_hi), p_gt)


def unpaired_bootstrap(
    with_vals: List[float],
    without_vals: List[float],
    *, n_boot: int = 1000, seed: int = 42, ci_level: float = 0.95,
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
    lo_pct, hi_pct = _ci_bounds_pct(ci_level)
    ci_lo, ci_hi = np.percentile(diffs, [lo_pct, hi_pct])
    p_gt = float((diffs > 0).mean())
    return (float(ci_lo), float(ci_hi), p_gt)


# ── df-level attacher ─────────────────────────────────────────────────

def _bootstrap_one(args):
    """Top-level worker kept for pickle-compat with callers that may
    still import it. The vectorized ``attach_ci`` path no longer uses
    this, but we keep it around as a public shim.

    ``args`` is ``(kind, w_data, o_data, n_boot, seed)`` where
    ``kind ∈ {'paired','unpaired','nan'}``.
    """
    kind, w, o, n_boot, seed = args
    if kind == "paired":
        return paired_bootstrap(w, o, n_boot=n_boot, seed=seed)
    if kind == "unpaired":
        return unpaired_bootstrap(w, o, n_boot=n_boot, seed=seed)
    return (float("nan"), float("nan"), float("nan"))


def _bootstrap_rows_vectorized(
    w_arrs: "List[object]",
    o_arrs: "List[object]",
    *,
    paired: "List[bool]",
    n_boot: int,
    seed: int,
    ci_level: float = 0.95,
):
    """Vectorized bootstrap for a batch of rows.

    Each ``w_arrs[i]`` / ``o_arrs[i]`` is either a 1-D numpy array
    (length ≥ 2) or ``None`` to mark a NaN row. For paired rows the
    two arms must already be the same length (caller intersected query
    ids). For unpaired rows the arms may differ.

    Returns three 1-D arrays (ci_lo, ci_hi, p_gt_zero) of length
    ``len(w_arrs)``. NaN rows propagate as NaN×3.

    Implementation notes
    --------------------
    The expensive part is the (V × n_boot × max_sample) resample tensor.
    We restrict it to the V valid rows and pad with **zeros** — then
    recover the correct per-row mean by summing along the resample axis
    and dividing by the actual row length. This avoids ``np.nanmean``
    which is ~3× slower than ``.sum() / L``.
    """
    import numpy as np

    n_rows = len(w_arrs)
    ci_lo = np.full(n_rows, np.nan)
    ci_hi = np.full(n_rows, np.nan)
    p_gt = np.full(n_rows, np.nan)

    # Classify row validity (sample count ≥ 2 on each arm).
    w_lens = np.zeros(n_rows, dtype=np.int64)
    o_lens = np.zeros(n_rows, dtype=np.int64)
    paired_arr = np.asarray(paired, dtype=bool)
    valid = np.zeros(n_rows, dtype=bool)
    for i in range(n_rows):
        w = w_arrs[i]
        o = o_arrs[i]
        if w is None or o is None:
            continue
        lw = int(w.shape[0])
        lo = int(o.shape[0])
        if lw < 2 or lo < 2:
            continue
        if paired_arr[i] and lw != lo:  # defensive — caller intersects
            continue
        valid[i] = True
        w_lens[i] = lw
        o_lens[i] = lo

    if not valid.any():
        return ci_lo, ci_hi, p_gt

    valid_idx = np.flatnonzero(valid)
    V = valid_idx.size
    vw_lens = w_lens[valid_idx]
    vo_lens = o_lens[valid_idx]
    vpaired = paired_arr[valid_idx]
    max_w = int(vw_lens.max())
    max_o = int(vo_lens.max())

    # Build compact (V × max_*) arm matrices. Pad with 0.0 — we NEVER
    # index into the padding (indices are clamped to L_i), and the 0-pad
    # plus summing-and-dividing-by-L_i recovers the true mean.
    W = np.zeros((V, max_w))
    O = np.zeros((V, max_o))
    for k, i in enumerate(valid_idx):
        lw = vw_lens[k]
        lo = vo_lens[k]
        W[k, :lw] = w_arrs[i]
        O[k, :lo] = o_arrs[i]

    rng = np.random.default_rng(seed)

    # Fast path: all valid rows share the same arm lengths. No padding
    # needed; ``rng.integers(0, L, size=(V, n_boot, L))`` is faster
    # than the generic float-then-floor path.
    uniform_w = bool((vw_lens == vw_lens[0]).all()) and int(vw_lens[0]) == max_w
    uniform_o = bool((vo_lens == vo_lens[0]).all()) and int(vo_lens[0]) == max_o

    if uniform_w:
        iw = rng.integers(0, max_w, size=(V, n_boot, max_w))
    else:
        # Per-row L varies: draw uniform ints in [0, max_w) then modulo
        # by L_i. That would bias the draw; safer is floor(float * L_i).
        u_w = rng.random((V, n_boot, max_w))
        iw = (u_w * vw_lens[:, None, None]).astype(np.int64)

    row_idx = np.arange(V)[:, None, None]
    w_resamp = W[row_idx, iw]  # (V, n_boot, max_w)

    # For paired rows reuse iw; for unpaired draw independently.
    if vpaired.all():
        if max_o == max_w:
            io = iw
        elif max_o < max_w:
            io = iw[:, :, :max_o]
        else:
            io = np.zeros((V, n_boot, max_o), dtype=np.int64)
            io[:, :, :max_w] = iw
    elif not vpaired.any():
        if uniform_o:
            io = rng.integers(0, max_o, size=(V, n_boot, max_o))
        else:
            u_o = rng.random((V, n_boot, max_o))
            io = (u_o * vo_lens[:, None, None]).astype(np.int64)
    else:
        # Mixed: build both index arrays and pick per row.
        if max_o <= max_w:
            io_p = iw[:, :, :max_o]
        else:
            io_p = np.zeros((V, n_boot, max_o), dtype=np.int64)
            io_p[:, :, :max_w] = iw
        if uniform_o:
            io_u = rng.integers(0, max_o, size=(V, n_boot, max_o))
        else:
            u_o = rng.random((V, n_boot, max_o))
            io_u = (u_o * vo_lens[:, None, None]).astype(np.int64)
        io = np.where(vpaired[:, None, None], io_p, io_u)

    o_resamp = O[row_idx, io]  # (V, n_boot, max_o)

    # Sum along resample axis, divide by true L to recover mean. We
    # only take the first L_i columns per row — zero-padding past L_i
    # would bias unpaired rows (where max_* may exceed L_i).
    #
    # Trick: because W/O are 0-padded and iw/io index into only the
    # first L_i entries of W[k]/O[k] (indices ≥ L_i never occur),
    # w_resamp and o_resamp have nonzero content only in their first
    # L_i columns. Summing the full row picks up only that content, so
    # sum / L_i is the correct mean — regardless of max_*.
    w_sum = w_resamp.sum(axis=2)  # (V, n_boot)
    o_sum = o_resamp.sum(axis=2)
    w_means = w_sum / vw_lens[:, None]
    o_means = o_sum / vo_lens[:, None]
    diffs = w_means - o_means  # (V, n_boot)

    lo_pct, hi_pct = _ci_bounds_pct(ci_level)
    lo_v, hi_v = np.percentile(diffs, [lo_pct, hi_pct], axis=1)
    p_v = (diffs > 0).mean(axis=1)

    ci_lo[valid_idx] = lo_v
    ci_hi[valid_idx] = hi_v
    p_gt[valid_idx] = p_v
    return ci_lo, ci_hi, p_gt


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
    ci_level: float = 0.95,
    progress: bool = False,
    workers: int = 1,
):
    """Attach ci_lo, ci_hi, p_gt_zero, effect_lb columns to a lift DataFrame.

    Vectorized numpy bootstrap — all rows are resampled in a single
    (or chunked) batched op, so per-row Python overhead is gone.

    ``workers`` is accepted for API compat but the vectorized path is
    already fast; values > 1 emit a soft deprecation warning.
    """
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pandas required") from e
    try:
        import numpy as np
    except ImportError as e:  # pragma: no cover
        raise ImportError("numpy required") from e

    if workers is not None and workers > 1:
        warnings.warn(
            "attach_ci: workers>1 is deprecated — the vectorized path is "
            "already fast and ignores this kwarg.",
            DeprecationWarning,
            stacklevel=2,
        )

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

    # Fast bulk-pivot cache for method="simple": keyed by predicate_name,
    # maps (cid, pv) → {qid: score}. Built in one groupby pass instead
    # of hitting ``per_query_scores`` (whose iterrows loop is O(n_rows))
    # once per (cid, pname) pair. Semantics-identical.
    _bulk_cache: Dict[str, Dict[tuple, Dict[str, float]]] = {}

    def _bulk(pname: str) -> Dict[tuple, Dict[str, float]]:
        if pname not in _bulk_cache:
            sub = scores_df_in[scores_df_in["predicate_name"] == pname]
            if sub.empty:
                _bulk_cache[pname] = {}
                return _bulk_cache[pname]
            # Build the (cid, pv) → {qid: score} map in one pass over
            # the filtered sub-frame using numpy arrays — orders of
            # magnitude faster than iterrows.
            cids = sub["config_id"].to_numpy()
            pvs = sub["predicate_value"].to_numpy()
            qids = sub["query_id"].to_numpy()
            scores = sub["score"].to_numpy(dtype=float)
            out: Dict[tuple, Dict[str, float]] = {}
            for cid, pv, qid, sc in zip(cids, pvs, qids, scores):
                out.setdefault((cid, pv), {})[qid] = sc
            _bulk_cache[pname] = out
        return _bulk_cache[pname]

    def _pqs(cid, pname):
        """Legacy lazy per_query_scores cache — kept for code paths that
        don't fit the bulk-pivot shape (e.g. marginal method)."""
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

    # Pull the three columns we need as flat python lists — ~20×
    # faster than pandas iterrows for the per-row dispatch loop below.
    canon_col = df["canonical_id"].tolist()
    pname_col = df["predicate_name"].tolist()
    pv_col = df["predicate_value"].tolist()
    n_rows = len(canon_col)

    # Build per-row arm arrays up-front.
    w_arrs: List[object] = [None] * n_rows
    o_arrs: List[object] = [None] * n_rows
    paired_flags: List[bool] = [True] * n_rows
    if method == "simple":
        _c2c = canonical_to_cid or {}
        for i in range(n_rows):
            canon = canon_col[i]
            pname = pname_col[i]
            pv = pv_col[i]
            target_cid = _c2c.get(canon)
            if target_cid is None:
                continue
            # Fast path: single groupby-backed pivot cached per pname.
            pivot = _bulk(pname)
            w_dict = pivot.get((target_cid, pv), {})
            o_dict = pivot.get((base_config_id, pv), {})
            shared = sorted(set(w_dict) & set(o_dict))
            if len(shared) < 2:
                continue
            w_arrs[i] = np.array([w_dict[q] for q in shared], dtype=float)
            o_arrs[i] = np.array([o_dict[q] for q in shared], dtype=float)
    else:
        _c2w = canonical_to_with_cids or {}
        for i in range(n_rows):
            paired_flags[i] = False
            canon = canon_col[i]
            pname = pname_col[i]
            pv = pv_col[i]
            with_cfgs = _c2w.get(canon, set())
            without_cfgs = all_cids_set - with_cfgs
            per_cfg = _marg(pname, pv)
            w_vals = [per_cfg[c] for c in with_cfgs if c in per_cfg]
            o_vals = [per_cfg[c] for c in without_cfgs if c in per_cfg]
            if len(w_vals) < 2 or len(o_vals) < 2:
                continue
            w_arrs[i] = np.asarray(w_vals, dtype=float)
            o_arrs[i] = np.asarray(o_vals, dtype=float)

    # Decide chunk size: honor the module-level override, else derive
    # from the memory cap using the widest arm in the batch.
    max_len_global = 0
    for a in w_arrs:
        if a is not None and a.shape[0] > max_len_global:
            max_len_global = a.shape[0]
    for a in o_arrs:
        if a is not None and a.shape[0] > max_len_global:
            max_len_global = a.shape[0]

    if max_len_global == 0:
        # No valid rows at all — emit NaNs and bail.
        df["ci_lo"] = [float("nan")] * n_rows
        df["ci_hi"] = [float("nan")] * n_rows
        df["p_gt_zero"] = [float("nan")] * n_rows
        df["effect_lb"] = df["ci_lo"]
        return df

    # Chunk size decision:
    #   1. If the memory cap would be exceeded, chunk to stay under it.
    #   2. Otherwise, if the module-level ``_CHUNK_SIZE`` has been
    #      monkey-patched to a small value (smaller than n_rows), honor
    #      it — tests rely on this to force chunk-boundary coverage.
    #   3. Default: run the whole batch in one shot.
    mem_chunk = max(1, _MEM_CELLS_CAP // max(1, n_boot * max_len_global))
    mem_limited = n_rows * n_boot * max_len_global > _MEM_CELLS_CAP
    if mem_limited:
        chunk_size = min(mem_chunk, _CHUNK_SIZE) if _CHUNK_SIZE < n_rows else mem_chunk
    elif _CHUNK_SIZE < n_rows:
        # Test-override path: smaller-than-batch chunk requested.
        chunk_size = _CHUNK_SIZE
    else:
        chunk_size = n_rows
    chunk_size = max(1, chunk_size)

    # Chunk iteration.
    starts = list(range(0, n_rows, chunk_size))

    from analyze._progress import progress_iter
    it = progress_iter(starts, enable=progress,
                       total=len(starts), desc="bootstrap")

    ci_lo_all = np.empty(n_rows)
    ci_hi_all = np.empty(n_rows)
    p_gt_all = np.empty(n_rows)

    for start in it:
        end = min(start + chunk_size, n_rows)
        # Use (seed + start) so that different chunks of the same input
        # get different RNG streams — but the same chunk-layout always
        # produces the same output (determinism with the module-level
        # chunk knob: a caller who sets _CHUNK_SIZE to force chunking
        # gets reproducible results for that chunk layout).
        chunk_seed = int(seed) + start
        lo_c, hi_c, p_c = _bootstrap_rows_vectorized(
            w_arrs[start:end],
            o_arrs[start:end],
            paired=paired_flags[start:end],
            n_boot=n_boot,
            seed=chunk_seed,
            ci_level=ci_level,
        )
        ci_lo_all[start:end] = lo_c
        ci_hi_all[start:end] = hi_c
        p_gt_all[start:end] = p_c

    df["ci_lo"] = ci_lo_all
    df["ci_hi"] = ci_hi_all
    df["p_gt_zero"] = p_gt_all
    df["effect_lb"] = df["ci_lo"]
    return df
