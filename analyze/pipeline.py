"""Layer 7 — chain-of-ops Pipeline for feature × predicate analysis.

The ``feature_predicate_table`` function accumulated 17+ kwargs across
six concerns (source, scope, effect, confidence, filter, rank, render).
This module exposes the same machinery as a builder where each stage
is independently settable and cached.

Usage
-----

    from analyze import Pipeline

    p = (Pipeline(store)
          .source(model="...", scorer="...", task="table_qa")
          .scope(base_config_id=2, skip_numeric=True)
          .effect(method="simple", metric="did")
          .confidence(n_bootstrap=1000, seed=42)
          .filter(min_lift_in_pair=0.03)
          .rank(sort_by="did")
          .render(fmt="markdown"))
    df, report = p.run()

    # Tweak only the filter — source / scope / effect / confidence reuse cache.
    df2, report2 = p.filter(min_lift_in_pair=0.05).render(fmt="markdown").run()

Design
------

* **Immutable chain** — each setter returns a new ``Pipeline`` instance;
  the cache dict is shared across sibling instances via reference, so
  mutating a late stage reuses early-stage results.
* **Lazy** — nothing runs until ``.run()``.
* **Stage-keyed cache** — each stage's result is keyed by a hash of
  ``(stage_name, stage_params, upstream_hash)``. Changing an early
  stage naturally invalidates downstream cache entries because their
  upstream_hash changes.
* **In-memory only** — no disk persistence this round.

``feature_predicate_table`` remains as a thin shim that builds a
Pipeline under the hood; it emits a ``DeprecationWarning`` pointing
users here.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.store import CubeStore
from analyze import data, resolve, effect as effect_mod, rank as rank_mod
from analyze import confidence as confidence_mod
from analyze import report as report_mod


# ── stage order (fixed) ──────────────────────────────────────────────
_STAGE_ORDER = ("source", "scope", "effect", "confidence", "filter", "rank", "render")


def _hash_params(name: str, params: Dict[str, Any], upstream: Optional[str]) -> str:
    """Content hash of (stage, params, upstream) — stable across runs."""
    payload = {
        "stage":    name,
        "params":   _canon(params),
        "upstream": upstream or "",
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def _canon(obj: Any) -> Any:
    """Canonicalize into JSON-serializable form. Sets → sorted lists; frozensets too."""
    if isinstance(obj, (set, frozenset)):
        return sorted(_canon(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _canon(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_canon(x) for x in obj]
    return obj


# ── the Pipeline ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class _Infra:
    progress: bool = False
    workers: int = 1


class Pipeline:
    """Immutable chainable pipeline over a CubeStore.

    See module docstring for usage. Each stage method returns a new
    Pipeline sharing the cache; mutations invalidate downstream stages
    by recomputing their hash.
    """

    def __init__(
        self,
        store: CubeStore,
        *,
        _stages: Optional[Dict[str, Dict[str, Any]]] = None,
        _cache: Optional[Dict[str, Any]] = None,
        _infra: Optional[_Infra] = None,
    ):
        self._store = store
        self._stages: Dict[str, Dict[str, Any]] = dict(_stages) if _stages else {}
        # Cache is SHARED across sibling pipelines — reference, not copy.
        self._cache: Dict[str, Any] = _cache if _cache is not None else {}
        self._infra: _Infra = _infra or _Infra()

    # ── stage setters (return new Pipeline sharing cache) ─────────────

    def _with_stage(self, name: str, params: Dict[str, Any]) -> "Pipeline":
        new_stages = dict(self._stages)
        new_stages[name] = dict(params)
        return Pipeline(
            self._store,
            _stages=new_stages,
            _cache=self._cache,
            _infra=self._infra,
        )

    def source(self, *, model: str, scorer: str, task: Optional[str] = None) -> "Pipeline":
        """Cube identity. Must be called once before .run()."""
        return self._with_stage("source", {"model": model, "scorer": scorer, "task": task})

    def scope(
        self,
        *,
        base_config_id: Optional[int] = None,
        predicate_names: Optional[List[str]] = None,
        feature_include: Optional[List[str]] = None,
        feature_exclude: Optional[List[str]] = None,
        skip_numeric: bool = True,
        include_unmatched: bool = False,
    ) -> "Pipeline":
        """Universe filters: which features × predicates enter."""
        return self._with_stage("scope", {
            "base_config_id":    base_config_id,
            "predicate_names":   predicate_names,
            "feature_include":   feature_include,
            "feature_exclude":   feature_exclude,
            "skip_numeric":      skip_numeric,
            "include_unmatched": include_unmatched,
        })

    def effect(
        self,
        *,
        method: str = "simple",                # "simple" | "marginal"
        metric: str = "lift",                  # "lift" | "did"
        reference_values: Optional[Dict[str, str]] = None,
    ) -> "Pipeline":
        """What's measured — lift or DiD, simple vs marginal."""
        if method not in ("simple", "marginal"):
            raise ValueError(f"method must be 'simple' | 'marginal'; got {method!r}")
        if metric not in ("lift", "did"):
            raise ValueError(f"metric must be 'lift' | 'did'; got {metric!r}")
        return self._with_stage("effect", {
            "method":           method,
            "metric":           metric,
            "reference_values": reference_values,
        })

    def confidence(
        self,
        *,
        n_bootstrap: int = 1000,
        seed: int = 42,
        ci_level: float = 0.95,
        skip: bool = False,
    ) -> "Pipeline":
        """Bootstrap CI + P(lift>0). Set skip=True to disable entirely."""
        if not (0.0 < ci_level < 1.0):
            raise ValueError(f"ci_level must be in (0,1); got {ci_level!r}")
        return self._with_stage("confidence", {
            "n_bootstrap": n_bootstrap,
            "seed":        seed,
            "ci_level":    ci_level,
            "skip":        skip,
        })

    def filter(
        self,
        *,
        confidence_min: Optional[float] = None,
        min_effect: Optional[float] = None,
        min_lift_in_pair: Optional[float] = None,
        require_sign: Optional[str] = None,   # "positive" | "negative" | "any"
    ) -> "Pipeline":
        """Row/pair gates — see rank.rank for semantics."""
        return self._with_stage("filter", {
            "confidence_min":   confidence_min,
            "min_effect":       min_effect,
            "min_lift_in_pair": min_lift_in_pair,
            "require_sign":     require_sign,
        })

    def rank(
        self,
        *,
        sort_by: Optional[str] = None,
        top_k: Optional[int] = None,
        sort_secondary: Optional[object] = None,  # str | list[str] | list[(col,asc)]
    ) -> "Pipeline":
        """Sort + top_k truncation."""
        return self._with_stage("rank", {
            "sort_by":        sort_by,
            "top_k":          top_k,
            "sort_secondary": sort_secondary,
        })

    def render(
        self,
        *,
        fmt: str = "markdown",                # "markdown" | "text" | "html" | "both"
        verbose: bool = True,
    ) -> "Pipeline":
        """Render output format. If never called, .run() returns only the df."""
        return self._with_stage("render", {"fmt": fmt, "verbose": verbose})

    # ── infra knobs ──────────────────────────────────────────────────

    def with_progress(self, flag: bool = True) -> "Pipeline":
        return Pipeline(
            self._store,
            _stages=self._stages,
            _cache=self._cache,
            _infra=_Infra(progress=flag, workers=self._infra.workers),
        )

    def with_workers(self, n: int) -> "Pipeline":
        return Pipeline(
            self._store,
            _stages=self._stages,
            _cache=self._cache,
            _infra=_Infra(progress=self._infra.progress, workers=n),
        )

    # ── run ──────────────────────────────────────────────────────────

    def run(self):
        """Walk stages in order, computing or cache-hitting each.

        Returns:
            df                  — when no .render() stage was set
            (df, rendered_str)  — when .render() was set (markdown | text | html)
            (df, dict)          — when .render(fmt="both")
        """
        if "source" not in self._stages:
            raise ValueError("Pipeline.run(): .source(model=..., scorer=...) is required")
        # Defaults for optional stages.
        stages = dict(self._stages)
        stages.setdefault("scope", {
            "base_config_id": None, "predicate_names": None,
            "feature_include": None, "feature_exclude": None,
            "skip_numeric": True, "include_unmatched": False,
        })
        stages.setdefault("effect", {
            "method": "simple", "metric": "lift", "reference_values": None,
        })
        stages.setdefault("confidence", {
            "n_bootstrap": 1000, "seed": 42, "ci_level": 0.95, "skip": True,
        })
        stages.setdefault("filter", {
            "confidence_min": None, "min_effect": None,
            "min_lift_in_pair": None, "require_sign": None,
        })
        stages.setdefault("rank", {"sort_by": None, "top_k": None, "sort_secondary": None})

        # Chain hashes as we go.
        upstream = None
        source_hash = _hash_params("source", stages["source"], upstream)
        source_out  = self._run_source(stages["source"], source_hash)

        scope_hash  = _hash_params("scope", stages["scope"], source_hash)
        scope_out   = self._run_scope(stages["scope"], source_out, scope_hash)

        effect_hash = _hash_params("effect", stages["effect"], scope_hash)
        effect_out  = self._run_effect(stages["effect"], scope_out, effect_hash)

        conf_hash   = _hash_params("confidence", stages["confidence"], effect_hash)
        conf_out    = self._run_confidence(
            stages["confidence"], effect_out, source_out, scope_out, conf_hash,
        )

        filt_hash   = _hash_params("filter", stages["filter"], conf_hash)
        filt_out    = self._run_filter(
            stages["filter"], conf_out, stages["effect"]["metric"],
            not stages["confidence"]["skip"], filt_hash,
        )

        rank_hash   = _hash_params("rank", stages["rank"], filt_hash)
        rank_out    = self._run_rank(
            stages["rank"], filt_out, stages["effect"]["metric"],
            not stages["confidence"]["skip"], rank_hash,
        )

        if "render" not in stages:
            return rank_out
        render_hash = _hash_params("render", stages["render"], rank_hash)
        rendered    = self._run_render(
            stages["render"], rank_out, stages, render_hash,
        )
        return rank_out, rendered

    # ── stage executors (cache-aware) ────────────────────────────────

    def _cached(self, key: str, fn):
        if key in self._cache:
            return self._cache[key]
        val = fn()
        self._cache[key] = val
        return val

    def _run_source(self, p: Dict[str, Any], key: str):
        def _compute():
            return {
                "scores":   data.scores_df(self._store, model=p["model"], scorer=p["scorer"]),
                "configs":  data.configs_df(self._store),
                "features": data.features_df(self._store, task=p["task"]),
            }
        return self._cached(key, _compute)

    def _run_scope(self, p: Dict[str, Any], src, key: str):
        def _compute():
            cdf = src["configs"]
            fdf = src["features"]
            sdf = src["scores"]

            # Simple mode needs an explicit base_config_id; marginal-only
            # leaves it None. The lookup is dtype-tolerant (see
            # resolve.base_func_ids) so a cube with config_id stored as
            # int64 / float / string all behaves the same.
            base_fids: frozenset = frozenset()
            if p["base_config_id"] is not None:
                base_fids = resolve.base_func_ids(cdf, p["base_config_id"])

            # Drop base-like features (all primitives already in base).
            if not fdf.empty:
                fdf = fdf[fdf["func_ids"].map(
                    lambda fids: not resolve.is_base_feature(fids, base_fids)
                )].reset_index(drop=True)

            # Feature include / exclude filters.
            if p["feature_include"] is not None and not fdf.empty:
                inc = set(p["feature_include"])
                fdf = fdf[fdf["canonical_id"].isin(inc)].reset_index(drop=True)
            if p["feature_exclude"] is not None and not fdf.empty:
                exc = set(p["feature_exclude"])
                fdf = fdf[~fdf["canonical_id"].isin(exc)].reset_index(drop=True)

            canonicals = fdf["canonical_id"].tolist() if not fdf.empty else []

            # Predicate universe.
            if p["predicate_names"] is None:
                predicate_names = (sorted(sdf["predicate_name"].unique().tolist())
                                   if not sdf.empty else [])
            else:
                predicate_names = list(p["predicate_names"])

            # Numeric-predicate guard (default on).
            if p["skip_numeric"] and predicate_names:
                kinds = data.predicate_kinds(self._store)
                numeric = [n for n in predicate_names if kinds.get(n) == "numeric"]
                if numeric:
                    warnings.warn(
                        "Pipeline.scope: dropping numeric predicate(s) "
                        f"{numeric} — group-by-value is meaningless for "
                        "continuous data. Pass skip_numeric=False to force-include.",
                        stacklevel=3,
                    )
                    predicate_names = [n for n in predicate_names if kinds.get(n) != "numeric"]

            # Build mappings for simple/marginal here so downstream reuses cache.
            canonical_to_cid = (resolve.simple_effect_configs(cdf, fdf, p["base_config_id"])
                                if p["base_config_id"] is not None else {})
            canonical_to_with_cids = resolve.configs_containing_feature(cdf, fdf)

            return {
                "scores":                 sdf,
                "features":               fdf,
                "configs":                cdf,
                "canonicals":             canonicals,
                "predicate_names":        predicate_names,
                "canonical_to_cid":       canonical_to_cid,
                "canonical_to_with_cids": canonical_to_with_cids,
                "base_config_id":         p["base_config_id"],
                "include_unmatched":      p["include_unmatched"],
                "all_cids":               cdf["config_id"].astype(int).tolist()
                                          if not cdf.empty else [],
            }
        return self._cached(key, _compute)

    def _run_effect(self, p: Dict[str, Any], scope_out, key: str):
        def _compute():
            try:
                import pandas as pd
            except ImportError as e:  # pragma: no cover
                raise ImportError("pandas required") from e

            method = p["method"]
            metric = p["metric"]
            ref_values = p["reference_values"]
            include_unmatched = scope_out["include_unmatched"]

            empty_cols = [
                "canonical_id", "predicate_name", "predicate_value",
                "n_with", "mean_with", "n_without", "mean_without", "lift",
            ]
            if not scope_out["canonicals"] or not scope_out["predicate_names"]:
                return pd.DataFrame(columns=empty_cols)

            if method == "simple":
                if scope_out["base_config_id"] is None:
                    raise ValueError("effect(method='simple') requires scope(base_config_id=...)")
                df = effect_mod.lift_simple(
                    scope_out["scores"],
                    base_cid=scope_out["base_config_id"],
                    canonicals=scope_out["canonicals"],
                    canonical_to_cid=scope_out["canonical_to_cid"],
                    predicate_names=scope_out["predicate_names"],
                    reference_values=ref_values,
                    progress=self._infra.progress,
                )
            else:
                df = effect_mod.lift_marginal(
                    scope_out["scores"],
                    canonicals=scope_out["canonicals"],
                    canonical_to_with_cids=scope_out["canonical_to_with_cids"],
                    all_cids=scope_out["all_cids"],
                    predicate_names=scope_out["predicate_names"],
                    reference_values=ref_values,
                    progress=self._infra.progress,
                )

            if df.empty:
                return df.drop(columns=["_ref"], errors="ignore")

            # Unmatched-row drop.
            if not include_unmatched:
                df = df[(df["n_with"] > 0) & (df["n_without"] > 0)].reset_index(drop=True)
                if df.empty:
                    return df.drop(columns=["_ref"], errors="ignore")

            if metric == "did":
                df = effect_mod.did(df, drop_ref_col=True)
            else:
                df = df.drop(columns=["_ref"], errors="ignore")
            return df
        return self._cached(key, _compute)

    def _run_confidence(self, p: Dict[str, Any], eff_df, src, scope_out, key: str):
        def _compute():
            if p["skip"] or eff_df.empty:
                return eff_df
            return confidence_mod.attach_ci(
                eff_df, scope_out["scores"],
                method=self._stages.get("effect", {}).get("method", "simple"),
                base_config_id=scope_out["base_config_id"],
                canonical_to_cid=scope_out["canonical_to_cid"],
                canonical_to_with_cids=scope_out["canonical_to_with_cids"],
                all_cids=scope_out["all_cids"],
                n_boot=p["n_bootstrap"],
                seed=p["seed"],
                ci_level=p["ci_level"],
                progress=self._infra.progress,
                workers=self._infra.workers,
            )
        return self._cached(key, _compute)

    def _run_filter(self, p: Dict[str, Any], df_in, metric: str, confidence: bool, key: str):
        def _compute():
            if df_in.empty:
                return df_in
            # rank.rank does both filter + sort; here we call it with
            # only the filter kwargs and leave sort untouched.
            return rank_mod.rank(
                df_in,
                sort_by=None, top_k=None,
                confidence_min=p["confidence_min"],
                min_effect=p["min_effect"],
                min_lift_in_pair=p["min_lift_in_pair"],
                require_sign=p["require_sign"],
                metric=metric, confidence=confidence,
            )
        return self._cached(key, _compute)

    def _run_rank(self, p: Dict[str, Any], df_in, metric: str, confidence: bool, key: str):
        def _compute():
            if df_in.empty:
                return df_in
            return rank_mod.rank(
                df_in,
                sort_by=p["sort_by"],
                top_k=p["top_k"],
                sort_secondary=p["sort_secondary"],
                metric=metric, confidence=confidence,
            )
        return self._cached(key, _compute)

    def _run_render(self, p: Dict[str, Any], df_in, stages: Dict[str, Dict[str, Any]], key: str):
        def _compute():
            run_meta = dict(
                model=stages["source"]["model"],
                scorer=stages["source"]["scorer"],
                method=stages["effect"]["method"],
                metric=stages["effect"]["metric"],
                confidence=not stages["confidence"]["skip"],
                n_bootstrap=stages["confidence"]["n_bootstrap"],
                sort_by=stages["rank"]["sort_by"],
                top_k=stages["rank"]["top_k"],
                confidence_min=stages["filter"]["confidence_min"],
                min_effect=stages["filter"]["min_effect"],
                min_lift_in_pair=stages["filter"]["min_lift_in_pair"],
                require_sign=stages["filter"]["require_sign"],
                sort_secondary=stages["rank"]["sort_secondary"],
                ci_level=stages["confidence"]["ci_level"],
            )
            return report_mod.render(
                df_in, fmt=p["fmt"], run_meta=run_meta, verbose=p["verbose"],
            )
        return self._cached(key, _compute)

    # ── introspection ────────────────────────────────────────────────

    def stages(self) -> Dict[str, Dict[str, Any]]:
        """Return a copy of the current stage param dict (for debugging)."""
        return {k: dict(v) for k, v in self._stages.items()}

    def cache_size(self) -> int:
        """Number of cached stage results currently resident."""
        return len(self._cache)
