"""Layer 4 — progress monitor.

Live view over a cube during (or after) an experiment run. Built on
``analyze.query.ExecutionQuery`` and ``analyze.meta`` — no private state;
just a thin, stateless ``ProgressMonitor`` facade that packages common
lookups.

Intended usage
──────────────

.. code-block:: python

    from analyze import ProgressMonitor
    mon = ProgressMonitor(store, model="meta-llama/Llama-3.1-8B-Instruct")
    print(mon.overall())
    for row in mon.by_config(total_queries_expected=200):
        print(row)
    print(mon.errors(limit=10))
    for row in mon.recent(10):
        print(row["created_at"], row["config_id"], row["query_id"])

Nothing writes. Call it any time — during a run it shows progress so far,
after a run it's a read-only summary.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from core.store import CubeStore
from analyze.query import ExecutionQuery


class ProgressMonitor:

    def __init__(
        self,
        store: CubeStore,
        *,
        model: Optional[str] = None,
        scorer: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> None:
        """
        Args:
            model:  Optional model filter scoping ALL monitor views.
            scorer: Optional scorer filter (makes score columns available).
            phase:  Optional phase filter.
        """
        self.store = store
        self.model = model
        self.scorer = scorer
        self.phase = phase

    # ── internal ──────────────────────────────────────────────────────

    def _q(self) -> ExecutionQuery:
        q = ExecutionQuery(self.store)
        if self.model:
            q = q.model(self.model)
        if self.scorer:
            q = q.scorer(self.scorer)
        if self.phase:
            q = q.phase(self.phase)
        return q

    # ── overall ───────────────────────────────────────────────────────

    def overall(self) -> Dict[str, Any]:
        """Aggregate counts over the selected scope.

        Returns:
            {
              "model":          str | None,
              "scorer":         str | None,
              "phase":          str | None,
              "n_executions":   int,
              "n_errors":       int,
              "n_evaluations":  int,   # only populated when scorer is set
              "n_configs":      int,   # distinct config_ids in scope
              "n_queries":      int,   # distinct query_ids in scope
              "mean_score":     float, # only populated when scorer is set
            }
        """
        conn = self.store._get_conn()

        n_executions = self._q().count()
        n_errors = self._q().with_error().count()

        rows = self._q().agg(by=["config_id"], fn="count", metric="*")
        n_configs = len(rows)
        rows_q = self._q().agg(by=["query_id"], fn="count", metric="*")
        n_queries = len(rows_q)

        result: Dict[str, Any] = {
            "model":        self.model,
            "scorer":       self.scorer,
            "phase":        self.phase,
            "n_executions": n_executions,
            "n_errors":     n_errors,
            "n_configs":    n_configs,
            "n_queries":    n_queries,
        }

        if self.scorer:
            score_rows = self._q().agg(by=[], fn="avg", metric="score")
            # Empty groupby → SQL returns 1 row with mean.
            mean = score_rows[0].get("avg_score") if score_rows else None
            evals = self._q().agg(by=[], fn="count", metric="eval_id")
            result["mean_score"] = mean
            result["n_evaluations"] = (evals[0].get("count_eval_id") if evals else 0) or 0

        return result

    # ── per-config progress ───────────────────────────────────────────

    def by_config(
        self,
        *,
        total_queries_expected: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """One row per config in scope.

        Columns:
            config_id, canonical_id (if in config.meta), n_done, n_expected,
            pct, n_errors, mean_score (if scorer set).
        """
        conn = self.store._get_conn()
        q_base = self._q()

        # n_done per config
        done_rows = q_base.agg(by=["config_id"], fn="count", metric="*")
        done_map = {r["config_id"]: r["count_all"] for r in done_rows}

        err_rows = q_base.with_error().agg(by=["config_id"], fn="count", metric="*")
        err_map = {r["config_id"]: r["count_all"] for r in err_rows}

        mean_map: Dict[int, Optional[float]] = {}
        if self.scorer:
            s_rows = q_base.agg(by=["config_id"], fn="avg", metric="score")
            for r in s_rows:
                mean_map[r["config_id"]] = r.get("avg_score")

        # Canonical-id resolution from config.meta
        cfg_rows = conn.execute(
            "SELECT config_id, meta FROM config WHERE config_id IN ({})".format(
                ",".join("?" * len(done_map))
            ) if done_map else "SELECT config_id, meta FROM config WHERE 0",
            tuple(done_map.keys()),
        ).fetchall()
        canon_map: Dict[int, Optional[str]] = {}
        for r in cfg_rows:
            try:
                meta = json.loads(r["meta"] or "{}")
            except json.JSONDecodeError:
                meta = {}
            canon_map[r["config_id"]] = (
                meta.get("canonical_id")
                or (",".join(meta["canonical_ids"]) if meta.get("canonical_ids") else None)
                or meta.get("kind")
            )

        out: List[Dict[str, Any]] = []
        for cid, n_done in sorted(done_map.items()):
            row: Dict[str, Any] = {
                "config_id":    cid,
                "canonical_id": canon_map.get(cid),
                "n_done":       n_done,
                "n_errors":     err_map.get(cid, 0),
            }
            if total_queries_expected is not None:
                row["n_expected"] = total_queries_expected
                row["pct"] = 100.0 * n_done / total_queries_expected
            if self.scorer:
                row["mean_score"] = mean_map.get(cid)
            out.append(row)
        return out

    # ── error inspection ──────────────────────────────────────────────

    def errors(self, *, limit: int = 20) -> List[Dict[str, Any]]:
        """Most-recent executions with a non-empty error field."""
        return (self._q()
                .with_error()
                .order_by("e.created_at DESC, e.execution_id DESC")
                .limit(limit)
                .columns(["execution_id", "config_id", "query_id",
                          "model", "error", "created_at"])
                .rows())

    # ── recent stream ─────────────────────────────────────────────────

    def recent(self, n: int = 20) -> List[Dict[str, Any]]:
        """Last ``n`` executions in scope, most-recent first.

        Ordering is ``(created_at, execution_id)`` DESC so that bulk-inserted
        rows with identical timestamps still come back in insertion order
        (newest first).
        """
        cols = ["execution_id", "config_id", "query_id", "model",
                "latency_ms", "prompt_tokens", "completion_tokens",
                "error", "created_at"]
        if self.scorer:
            cols += ["score", "scorer"]
        return (self._q()
                .order_by("e.created_at DESC, e.execution_id DESC")
                .limit(n)
                .columns(cols)
                .rows())
