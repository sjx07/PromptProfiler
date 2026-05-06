"""Read-only cube explorer operations.

These functions are the backend contract for lightweight cube inspection
UIs.  They intentionally return plain JSON-serializable dict/list shapes
and avoid pandas so they can be served by a stdlib HTTP script.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.store import CubeStore
from analyze import meta as analyze_meta


FilterSpec = Dict[str, Any]

_SIMPLE_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def cube_summary(store: CubeStore) -> Dict[str, Any]:
    """Return high-level cube inventory for UI bootstrapping."""
    return analyze_meta.summary(store)


def list_configs_detailed(
    store: CubeStore,
    *,
    model: Optional[str] = None,
    scorer: Optional[str] = None,
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    only_with_results: bool = False,
) -> List[Dict[str, Any]]:
    """Configs with parsed metadata plus execution/evaluation score counts."""
    conn = store._get_conn()
    exec_where, exec_params = _query_scope_where("q", dataset=dataset, split=split)
    if model:
        exec_where.append("e.model = ?")
        exec_params.append(model)

    eval_join = "LEFT JOIN evaluation ev ON ev.execution_id = e.execution_id"
    eval_params: List[Any] = []
    if scorer:
        eval_join += " AND ev.scorer = ?"
        eval_params.append(scorer)

    rows = conn.execute(
        f"""
        WITH filtered_execution AS (
            SELECT e.*
            FROM execution e
            JOIN query q ON q.query_id = e.query_id
            WHERE {' AND '.join(exec_where)}
        ),
        feature_names AS (
            SELECT cf.config_id,
                   GROUP_CONCAT(f.canonical_id, '|') AS resolved_canonical_ids
            FROM config_feature cf
            JOIN feature f ON f.feature_id = cf.feature_id
            GROUP BY cf.config_id
        )
        SELECT c.config_id,
               c.func_ids,
               c.meta,
               COALESCE(fn.resolved_canonical_ids, '') AS resolved_canonical_ids,
               COUNT(DISTINCT e.execution_id) AS n_executions,
               COUNT(DISTINCT ev.eval_id) AS n_evaluations,
               AVG(ev.score) AS avg_score,
               MIN(e.created_at) AS first_execution_at,
               MAX(e.created_at) AS last_execution_at
        FROM config c
        LEFT JOIN filtered_execution e ON e.config_id = c.config_id
        {eval_join}
        LEFT JOIN feature_names fn ON fn.config_id = c.config_id
        GROUP BY c.config_id
        ORDER BY c.config_id
        """,
        tuple(exec_params + eval_params),
    ).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        meta = _json_loads(r["meta"], {})
        func_ids = _json_loads(r["func_ids"], [])
        resolved = [x for x in (r["resolved_canonical_ids"] or "").split("|") if x]
        meta_canonical_ids = _string_list(meta.get("canonical_ids"))
        all_canonical_ids = meta_canonical_ids or resolved
        canonical = (
            meta.get("label")
            or meta.get("canonical_id")
            or _display_feature_set(all_canonical_ids, kind=meta.get("kind"), has_funcs=bool(func_ids))
            or ("base" if not func_ids else None)
        )
        n_executions = int(r["n_executions"] or 0)
        n_evaluations = int(r["n_evaluations"] or 0)
        if only_with_results:
            n_results = n_evaluations if scorer else n_executions
            if n_results <= 0:
                continue
        out.append({
            "configId": int(r["config_id"]),
            "canonicalId": canonical,
            "kind": meta.get("kind"),
            "funcIds": func_ids,
            "nFuncs": len(func_ids),
            "featureIds": meta.get("feature_ids", []),
            "canonicalIds": all_canonical_ids,
            "resolvedCanonicalIds": resolved,
            "nExecutions": n_executions,
            "nEvaluations": n_evaluations,
            "avgScore": _float_or_none(r["avg_score"]),
            "firstExecutionAt": r["first_execution_at"],
            "lastExecutionAt": r["last_execution_at"],
            "meta": meta,
        })
    return out


def list_query_meta_fields(
    store: CubeStore,
    *,
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    model: Optional[str] = None,
    scorer: Optional[str] = None,
    only_with_results: bool = False,
    limit: int = 2000,
) -> List[Dict[str, Any]]:
    """Top-level query.meta keys with coverage and sample values."""
    where, params = _query_scope_where("q", dataset=dataset, split=split)
    result_join, result_params = _query_result_join(
        "q",
        model=model,
        scorer=scorer,
        only_with_results=only_with_results,
    )
    rows = store._get_conn().execute(
        f"""
        SELECT DISTINCT q.query_id, q.meta
        FROM query q
        {result_join}
        WHERE {' AND '.join(where)}
        LIMIT ?
        """,
        tuple(result_params + params + [int(limit)]),
    ).fetchall()
    counts: Counter[str] = Counter()
    samples: Dict[str, List[str]] = defaultdict(list)
    for r in rows:
        meta = _json_loads(r["meta"], {})
        if not isinstance(meta, dict):
            continue
        for key, value in meta.items():
            counts[key] += 1
            if len(samples[key]) < 8:
                text = _sample_value(value)
                if text not in samples[key]:
                    samples[key].append(text)
    return [
        {
            "field": f"query.meta.{key}",
            "key": key,
            "nQueries": n,
            "samples": samples[key],
        }
        for key, n in counts.most_common()
    ]


def list_predicate_fields(
    store: CubeStore,
    *,
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    model: Optional[str] = None,
    scorer: Optional[str] = None,
    only_with_results: bool = False,
) -> List[Dict[str, Any]]:
    """Predicate names available for slicing, with value counts and samples."""
    where, params = _query_scope_where("q", dataset=dataset, split=split)
    result_join, result_params = _query_result_join(
        "q",
        model=model,
        scorer=scorer,
        only_with_results=only_with_results,
    )
    rows = store._get_conn().execute(
        f"""
        SELECT p.name,
               COUNT(*) AS n_rows,
               COUNT(DISTINCT p.query_id) AS n_queries,
               COUNT(DISTINCT p.value) AS n_values,
               GROUP_CONCAT(DISTINCT p.value) AS sample_values
        FROM predicate p
        JOIN query q ON q.query_id = p.query_id
        {result_join}
        WHERE {' AND '.join(where)}
        GROUP BY p.name
        ORDER BY p.name
        """,
        tuple(result_params + params),
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        values = [v for v in str(r["sample_values"] or "").split(",") if v]
        out.append({
            "field": f"predicate.{r['name']}",
            "name": r["name"],
            "nRows": int(r["n_rows"] or 0),
            "nQueries": int(r["n_queries"] or 0),
            "nValues": int(r["n_values"] or 0),
            "samples": values[:8],
        })
    return out


def slice_scores(
    store: CubeStore,
    *,
    model: str,
    scorer: str,
    config_ids: Optional[Sequence[int]] = None,
    group_by: Optional[Sequence[str]] = None,
    filters: Optional[Sequence[FilterSpec]] = None,
    base_config_id: Optional[int] = None,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """Aggregate score by config and requested slice dimensions."""
    group_by = [g for g in (group_by or []) if g and g != "config_id"]
    compiler = _QueryCompiler()
    group_exprs: List[str] = []
    group_aliases: List[str] = []
    params: List[Any] = [model, scorer]

    for idx, field in enumerate(group_by):
        expr = compiler.dimension_expr(field)
        alias = f"g{idx}"
        group_exprs.append(f"{expr} AS {alias}")
        group_aliases.append(alias)

    where, filter_params = compiler.compile_filters(filters or [])
    params.extend(filter_params)
    if config_ids is not None:
        ids = [int(c) for c in config_ids]
        if not ids:
            return []
        where.append(f"s.config_id IN ({','.join('?' * len(ids))})")
        params.extend(ids)

    select_groups = (", " + ", ".join(group_exprs)) if group_exprs else ""
    group_cols = ["s.config_id", *group_aliases]
    sql = f"""
        SELECT s.config_id AS config_id{select_groups},
               COUNT(*) AS n,
               AVG(s.score) AS avg_score,
               SUM(CASE WHEN s.score = 1 THEN 1 ELSE 0 END) AS n_score_one,
               SUM(CASE WHEN s.score = 0 THEN 1 ELSE 0 END) AS n_score_zero
        FROM v_query_scores s
        JOIN query q ON q.query_id = s.query_id
        {compiler.join_sql()}
        WHERE s.model = ? AND s.scorer = ?
          {('AND ' + ' AND '.join(where)) if where else ''}
        GROUP BY {', '.join(group_cols)}
        ORDER BY {', '.join(group_aliases) + ',' if group_aliases else ''} s.config_id
        LIMIT ?
    """
    params.append(int(limit))

    rows = store._get_conn().execute(sql, tuple(params)).fetchall()
    result: List[Dict[str, Any]] = []
    for r in rows:
        group = {
            field: _normalize_sql_value(r[group_aliases[i]])
            for i, field in enumerate(group_by)
        }
        result.append({
            "configId": int(r["config_id"]),
            "group": group,
            "n": int(r["n"] or 0),
            "avgScore": _float_or_none(r["avg_score"]),
            "nScoreOne": int(r["n_score_one"] or 0),
            "nScoreZero": int(r["n_score_zero"] or 0),
        })

    if base_config_id is not None:
        base_by_group = {
            _group_key(row["group"]): row["avgScore"]
            for row in result
            if row["configId"] == int(base_config_id)
        }
        for row in result:
            base_score = base_by_group.get(_group_key(row["group"]))
            row["baseScore"] = base_score
            row["deltaVsBase"] = (
                None
                if base_score is None or row["avgScore"] is None
                else row["avgScore"] - base_score
            )
    return result


def feature_label_analysis(
    store: CubeStore,
    *,
    model: str,
    scorer: str,
    config_ids: Optional[Sequence[int]] = None,
    predicate_name: Optional[str] = None,
    base_config_id: Optional[int] = None,
    filters: Optional[Sequence[FilterSpec]] = None,
    limit: int = 500,
) -> List[Dict[str, Any]]:
    """Aggregate evaluated rows by semantic feature label.

    The label tables are component metadata, but the measured unit remains a
    rendered config/query score. If multiple components with the same label are
    present in one config, this query counts that config/query score once for
    the label to avoid duplicate fan-out.
    """
    compiler = _QueryCompiler()
    filter_where, filter_params = compiler.compile_filters(filters or [])
    where_params: List[Any] = [model, scorer]
    where = [
        "e.model = ?",
        "ev.scorer = ?",
        "(e.error IS NULL OR e.error = '')",
        "ev.score IS NOT NULL",
    ]
    if config_ids is not None:
        ids = [int(c) for c in config_ids]
        if not ids:
            return []
        where.append(f"e.config_id IN ({','.join('?' * len(ids))})")
        where_params.extend(ids)
    where.extend(filter_where)
    where_params.extend(filter_params)

    pred_select = "NULL AS predicate_value"
    pred_join = ""
    pred_params: List[Any] = []
    pred_group = ""
    pred_order = ""
    if predicate_name:
        pred_select = "p.value AS predicate_value"
        pred_join = "JOIN predicate p ON p.query_id = e.query_id AND p.name = ?"
        pred_params.append(predicate_name)
        pred_group = ", predicate_value"
        pred_order = ", predicate_value"

    sql = f"""
        WITH scored_labels AS (
            SELECT DISTINCT
                   flm.label_id,
                   flm.role,
                   e.config_id,
                   e.query_id,
                   ev.score,
                   {pred_select}
            FROM evaluation ev
            JOIN execution e ON e.execution_id = ev.execution_id
            JOIN query q ON q.query_id = e.query_id
            JOIN config_feature cf ON cf.config_id = e.config_id
            JOIN feature f ON f.feature_id = cf.feature_id
            JOIN feature_label_membership flm ON flm.feature_id = f.feature_id
            {compiler.join_sql()}
            {pred_join}
            WHERE {' AND '.join(where)}
        )
        SELECT label_id,
               role,
               predicate_value,
               COUNT(DISTINCT config_id) AS n_configs,
               COUNT(DISTINCT query_id) AS n_queries,
               COUNT(*) AS n,
               AVG(score) AS avg_score
        FROM scored_labels
        GROUP BY label_id, role{pred_group}
        ORDER BY label_id, role{pred_order}
        LIMIT ?
    """
    rows = store._get_conn().execute(
        sql,
        tuple(pred_params + where_params + [int(limit)]),
    ).fetchall()

    component_map = _label_components(store)
    base_scores = (
        _base_scores_by_predicate(
            store,
            model=model,
            scorer=scorer,
            base_config_id=int(base_config_id),
            predicate_name=predicate_name,
            filters=filters,
        )
        if base_config_id is not None
        else {}
    )

    out: List[Dict[str, Any]] = []
    for r in rows:
        pred_value = _normalize_sql_value(r["predicate_value"])
        base_score = base_scores.get(pred_value if predicate_name else None)
        avg_score = _float_or_none(r["avg_score"])
        components = component_map.get((r["label_id"], r["role"]), [])
        out.append({
            "labelId": r["label_id"],
            "role": r["role"],
            "predicateName": predicate_name,
            "predicateValue": pred_value,
            "nConfigs": int(r["n_configs"] or 0),
            "nQueries": int(r["n_queries"] or 0),
            "n": int(r["n"] or 0),
            "avgScore": avg_score,
            "baseScore": base_score,
            "deltaVsBase": (
                None
                if base_score is None or avg_score is None
                else avg_score - base_score
            ),
            "components": components,
        })
    return out


def compare_configs(
    store: CubeStore,
    *,
    model: str,
    scorer: str,
    base_config_id: int,
    target_config_id: int,
    filters: Optional[Sequence[FilterSpec]] = None,
) -> Dict[str, Any]:
    """Compare two configs on shared query ids, optionally sliced/filtered."""
    rows = _comparison_rows(
        store,
        model=model,
        scorer=scorer,
        base_config_id=base_config_id,
        target_config_id=target_config_id,
        filters=filters,
        limit=None,
    )
    n = len(rows)
    if n == 0:
        return {
            "baseConfigId": int(base_config_id),
            "targetConfigId": int(target_config_id),
            "nShared": 0,
            "avgBase": None,
            "avgTarget": None,
            "avgDelta": None,
            "flippedUp": 0,
            "flippedDown": 0,
            "agree": 0,
        }
    sum_base = sum(float(r["baseScore"]) for r in rows)
    sum_target = sum(float(r["targetScore"]) for r in rows)
    up = sum(1 for r in rows if r["targetScore"] > r["baseScore"])
    down = sum(1 for r in rows if r["targetScore"] < r["baseScore"])
    return {
        "baseConfigId": int(base_config_id),
        "targetConfigId": int(target_config_id),
        "nShared": n,
        "avgBase": sum_base / n,
        "avgTarget": sum_target / n,
        "avgDelta": (sum_target - sum_base) / n,
        "flippedUp": up,
        "flippedDown": down,
        "agree": n - up - down,
    }


def comparison_examples(
    store: CubeStore,
    *,
    model: str,
    scorer: str,
    base_config_id: int,
    target_config_id: int,
    direction: str = "both",
    filters: Optional[Sequence[FilterSpec]] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Rows where base and target agree or disagree, with raw previews."""
    if direction not in {"both", "up", "down", "agree"}:
        raise ValueError("direction must be one of: both, up, down, agree")
    rows = _comparison_rows(
        store,
        model=model,
        scorer=scorer,
        base_config_id=base_config_id,
        target_config_id=target_config_id,
        filters=filters,
        limit=limit,
    )
    out: List[Dict[str, Any]] = []
    for r in rows:
        if r["targetScore"] > r["baseScore"]:
            d = "up"
        elif r["targetScore"] < r["baseScore"]:
            d = "down"
        else:
            d = "agree"
        if direction != "both" and d != direction:
            continue
        rec = dict(r)
        rec["direction"] = d
        out.append(rec)
        if len(out) >= limit:
            break
    return out


def examples(
    store: CubeStore,
    *,
    model: str,
    scorer: str,
    config_ids: Optional[Sequence[int]] = None,
    filters: Optional[Sequence[FilterSpec]] = None,
    score_order: str = "asc",
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """Execution examples for a selected config/slice."""
    compiler = _QueryCompiler()
    where, params = compiler.compile_filters(filters or [])
    base_params: List[Any] = [scorer, model]
    if config_ids is not None:
        ids = [int(c) for c in config_ids]
        if not ids:
            return []
        where.append(f"e.config_id IN ({','.join('?' * len(ids))})")
        params.extend(ids)
    direction = "DESC" if score_order.lower() == "desc" else "ASC"
    sql = f"""
        SELECT e.execution_id,
               e.config_id,
               e.query_id,
               q.dataset,
               q.content AS question,
               q.meta AS query_meta,
               ev.score,
               ev.metrics,
               e.prediction,
               e.raw_response,
               e.error,
               e.latency_ms,
               e.prompt_tokens,
               e.completion_tokens,
               e.created_at
        FROM execution e
        JOIN evaluation ev ON ev.execution_id = e.execution_id AND ev.scorer = ?
        JOIN query q ON q.query_id = e.query_id
        {compiler.join_sql()}
        WHERE e.model = ?
          {('AND ' + ' AND '.join(where)) if where else ''}
        ORDER BY ev.score {direction}, e.config_id, e.query_id
        LIMIT ?
    """
    rows = store._get_conn().execute(
        sql, tuple(base_params + params + [int(limit)])
    ).fetchall()
    return [_example_from_row(r) for r in rows]


def execution_artifact(store: CubeStore, *, execution_id: int) -> Optional[Dict[str, Any]]:
    """Full prompt/response artifact for one execution row."""
    row = store._get_conn().execute(
        """
        SELECT e.*,
               q.dataset,
               q.content AS question,
               q.meta AS query_meta,
               c.meta AS config_meta,
               ev.scorer,
               ev.score,
               ev.metrics,
               ev.created_at AS evaluated_at
        FROM execution e
        JOIN query q ON q.query_id = e.query_id
        JOIN config c ON c.config_id = e.config_id
        LEFT JOIN evaluation ev ON ev.execution_id = e.execution_id
        WHERE e.execution_id = ?
        ORDER BY ev.eval_id
        """,
        (int(execution_id),),
    ).fetchone()
    if row is None:
        return None
    query_meta = _json_loads(row["query_meta"], {})
    config_meta = _json_loads(row["config_meta"], {})
    return {
        "executionId": int(row["execution_id"]),
        "configId": int(row["config_id"]),
        "queryId": row["query_id"],
        "model": row["model"],
        "dataset": row["dataset"],
        "question": row["question"],
        "queryMeta": query_meta,
        "gold": _gold_from_meta(query_meta),
        "configMeta": config_meta,
        "systemPrompt": row["system_prompt"] or "",
        "userContent": row["user_content"] or "",
        "rawResponse": row["raw_response"] or "",
        "prediction": row["prediction"] or "",
        "latencyMs": _float_or_none(row["latency_ms"]),
        "promptTokens": row["prompt_tokens"],
        "completionTokens": row["completion_tokens"],
        "error": row["error"] or "",
        "phaseIds": _json_loads(row["phase_ids"], []),
        "createdAt": row["created_at"],
        "scorer": row["scorer"],
        "score": _float_or_none(row["score"]),
        "metrics": _json_loads(row["metrics"], {}),
        "evaluatedAt": row["evaluated_at"],
    }


def diagnostics(
    store: CubeStore,
    *,
    model: str,
    scorer: str,
    config_ids: Optional[Sequence[int]] = None,
    filters: Optional[Sequence[FilterSpec]] = None,
    limit: int = 5000,
) -> Dict[str, Any]:
    """Generic execution diagnostics bucketed from metrics/errors/responses."""
    rows = examples(
        store,
        model=model,
        scorer=scorer,
        config_ids=config_ids,
        filters=filters,
        score_order="asc",
        limit=limit,
    )
    buckets: Dict[str, Counter[str]] = {
        "ecr": Counter(),
        "outputMode": Counter(),
        "errorState": Counter(),
        "responsePattern": Counter(),
    }
    score_sums: Dict[Tuple[str, str], float] = defaultdict(float)
    score_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    for row in rows:
        metrics = row.get("metrics") or {}
        ecr = _metric_value(metrics, "ECR@1", "ECR")
        output_mode = _metric_value(metrics, "output_mode", "outputMode", "method")
        values = {
            "ecr": _bucket_value(ecr),
            "outputMode": _bucket_value(output_mode),
            "errorState": "error" if row.get("error") else "ok",
            "responsePattern": _response_pattern(
                row.get("rawResponsePreview") or "",
                row.get("prediction") or "",
                row.get("error") or "",
            ),
        }
        score = row.get("score")
        for name, value in values.items():
            buckets[name][value] += 1
            if score is not None:
                score_sums[(name, value)] += float(score)
                score_counts[(name, value)] += 1

    return {
        "n": len(rows),
        "buckets": {
            name: [
                {
                    "value": value,
                    "n": count,
                    "avgScore": (
                        score_sums[(name, value)] / score_counts[(name, value)]
                        if score_counts[(name, value)]
                        else None
                    ),
                }
                for value, count in counter.most_common()
            ]
            for name, counter in buckets.items()
        },
    }


def plan_delete(
    store: CubeStore,
    *,
    model: Optional[str] = None,
    config_ids: Optional[Sequence[int]] = None,
    filters: Optional[Sequence[FilterSpec]] = None,
    limit_preview: int = 20,
) -> Dict[str, Any]:
    """Dry-run deletion plan for matching execution/evaluation rows."""
    compiler = _QueryCompiler()
    where, params = compiler.compile_filters(filters or [])
    if model:
        where.append("e.model = ?")
        params.append(model)
    if config_ids is not None:
        ids = [int(c) for c in config_ids]
        if not ids:
            return _empty_delete_plan()
        where.append(f"e.config_id IN ({','.join('?' * len(ids))})")
        params.extend(ids)
    if not where:
        raise ValueError("plan_delete requires at least one filter, model, or config id")

    where_sql = " AND ".join(where)
    conn = store._get_conn()
    count_row = conn.execute(
        f"""
        SELECT COUNT(DISTINCT e.execution_id) AS n_executions,
               COUNT(DISTINCT ev.eval_id) AS n_evaluations
        FROM execution e
        LEFT JOIN query q ON q.query_id = e.query_id
        LEFT JOIN evaluation ev ON ev.execution_id = e.execution_id
        {compiler.join_sql()}
        WHERE {where_sql}
        """,
        tuple(params),
    ).fetchone()
    preview_rows = conn.execute(
        f"""
        SELECT DISTINCT e.execution_id
        FROM execution e
        LEFT JOIN query q ON q.query_id = e.query_id
        {compiler.join_sql()}
        WHERE {where_sql}
        ORDER BY e.execution_id
        LIMIT ?
        """,
        tuple(params + [int(limit_preview)]),
    ).fetchall()
    sql_preview = _delete_sql_preview(model=model, config_ids=config_ids)
    return {
        "nExecutions": int(count_row["n_executions"] or 0),
        "nEvaluations": int(count_row["n_evaluations"] or 0),
        "executionIdPreview": [int(r["execution_id"]) for r in preview_rows],
        "sqlPreview": sql_preview,
        "note": "Dry run only. Back up the cube before executing any delete.",
    }


class _QueryCompiler:
    """Compile UI field specs into SQL over aliases e/s/q."""

    def __init__(self) -> None:
        self._predicate_aliases: Dict[str, str] = {}
        self._joins: List[str] = []
        self._params: List[Any] = []

    def join_sql(self) -> str:
        return "\n".join(self._joins)

    def dimension_expr(self, field: str) -> str:
        kind, value = _parse_field(field)
        if kind == "dataset":
            return "q.dataset"
        if kind == "query_split":
            return "COALESCE(json_extract(q.meta, '$.split'), '(no split)')"
        if kind == "query_id":
            return "q.query_id"
        if kind == "query_meta":
            return f"json_extract(q.meta, '{_json_path(value)}')"
        if kind == "predicate":
            alias = self._predicate_alias(value)
            return f"{alias}.value"
        if kind == "score":
            return "s.score"
        raise ValueError(f"unsupported group field: {field!r}")

    def compile_filters(self, filters: Sequence[FilterSpec]) -> Tuple[List[str], List[Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        for spec in filters:
            field = str(spec.get("field") or "")
            if not field:
                continue
            op = str(spec.get("op") or "=").lower()
            value = spec.get("value")
            expr = self._filter_expr(field)
            if op in {"=", "eq"}:
                clauses.append(f"{expr} = ?")
                params.append(value)
            elif op in {"!=", "ne"}:
                clauses.append(f"{expr} != ?")
                params.append(value)
            elif op == "in":
                values = list(value or [])
                if not values:
                    clauses.append("1=0")
                else:
                    clauses.append(f"{expr} IN ({','.join('?' * len(values))})")
                    params.extend(values)
            elif op in {"<", "<=", ">", ">="}:
                clauses.append(f"{expr} {op} ?")
                params.append(value)
            elif op == "is_not_empty":
                clauses.append(f"({expr} IS NOT NULL AND {expr} != '')")
            elif op == "is_empty":
                clauses.append(f"({expr} IS NULL OR {expr} = '')")
            else:
                raise ValueError(f"unsupported filter op: {op!r}")
        return clauses, params

    def _filter_expr(self, field: str) -> str:
        kind, value = _parse_field(field)
        if kind == "dataset":
            return "q.dataset"
        if kind == "query_split":
            return "json_extract(q.meta, '$.split')"
        if kind == "query_id":
            return "q.query_id"
        if kind == "config_id":
            return "e.config_id"
        if kind == "score":
            return "ev.score"
        if kind == "error":
            return "e.error"
        if kind == "query_meta":
            return f"json_extract(q.meta, '{_json_path(value)}')"
        if kind == "predicate":
            alias = self._predicate_alias(value)
            return f"{alias}.value"
        raise ValueError(f"unsupported filter field: {field!r}")

    def _predicate_alias(self, name: str) -> str:
        if name not in self._predicate_aliases:
            alias = f"p{len(self._predicate_aliases)}"
            self._predicate_aliases[name] = alias
            escaped = name.replace("'", "''")
            self._joins.append(
                "LEFT JOIN predicate "
                f"{alias} ON {alias}.query_id = q.query_id AND {alias}.name = '{escaped}'"
            )
        return self._predicate_aliases[name]


def _comparison_rows(
    store: CubeStore,
    *,
    model: str,
    scorer: str,
    base_config_id: int,
    target_config_id: int,
    filters: Optional[Sequence[FilterSpec]],
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    compiler = _QueryCompiler()
    where, params = compiler.compile_filters(filters or [])
    limit_sql = "" if limit is None else "LIMIT ?"
    limit_params: List[Any] = [] if limit is None else [int(limit)]
    sql = f"""
        WITH base AS (
            SELECT e.execution_id, e.query_id, e.prediction, e.raw_response,
                   e.error, ev.score
            FROM execution e
            JOIN evaluation ev ON ev.execution_id = e.execution_id
            WHERE e.config_id = ? AND e.model = ? AND ev.scorer = ?
              AND (e.error IS NULL OR e.error = '')
              AND ev.score IS NOT NULL
        ),
        target AS (
            SELECT e.execution_id, e.query_id, e.prediction, e.raw_response,
                   e.error, ev.score
            FROM execution e
            JOIN evaluation ev ON ev.execution_id = e.execution_id
            WHERE e.config_id = ? AND e.model = ? AND ev.scorer = ?
              AND (e.error IS NULL OR e.error = '')
              AND ev.score IS NOT NULL
        )
        SELECT base.query_id,
               q.dataset,
               q.content AS question,
               q.meta AS query_meta,
               base.execution_id AS base_execution_id,
               target.execution_id AS target_execution_id,
               base.score AS base_score,
               target.score AS target_score,
               base.prediction AS base_prediction,
               target.prediction AS target_prediction,
               base.raw_response AS base_raw_response,
               target.raw_response AS target_raw_response
        FROM base
        JOIN target USING (query_id)
        JOIN query q ON q.query_id = base.query_id
        {compiler.join_sql()}
        {('WHERE ' + ' AND '.join(where)) if where else ''}
        ORDER BY base.query_id
        {limit_sql}
    """
    base_params = [
        int(base_config_id), model, scorer,
        int(target_config_id), model, scorer,
    ]
    rows = store._get_conn().execute(
        sql, tuple(base_params + params + limit_params)
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        qmeta = _json_loads(r["query_meta"], {})
        out.append({
            "queryId": r["query_id"],
            "dataset": r["dataset"],
            "question": r["question"],
            "queryMeta": _compact_meta(qmeta),
            "gold": _gold_from_meta(qmeta),
            "baseExecutionId": int(r["base_execution_id"]),
            "targetExecutionId": int(r["target_execution_id"]),
            "baseScore": _float_or_none(r["base_score"]),
            "targetScore": _float_or_none(r["target_score"]),
            "basePrediction": r["base_prediction"] or "",
            "targetPrediction": r["target_prediction"] or "",
            "baseRawPreview": _preview(r["base_raw_response"]),
            "targetRawPreview": _preview(r["target_raw_response"]),
        })
    return out


def _example_from_row(row: sqlite3.Row) -> Dict[str, Any]:
    qmeta = _json_loads(row["query_meta"], {})
    metrics = _json_loads(row["metrics"], {})
    return {
        "executionId": int(row["execution_id"]),
        "configId": int(row["config_id"]),
        "queryId": row["query_id"],
        "dataset": row["dataset"],
        "question": row["question"],
        "queryMeta": _compact_meta(qmeta),
        "gold": _gold_from_meta(qmeta),
        "score": _float_or_none(row["score"]),
        "metrics": metrics,
        "prediction": row["prediction"] or "",
        "rawResponsePreview": _preview(row["raw_response"]),
        "error": row["error"] or "",
        "latencyMs": _float_or_none(row["latency_ms"]),
        "promptTokens": row["prompt_tokens"],
        "completionTokens": row["completion_tokens"],
        "createdAt": row["created_at"],
    }


def _label_components(store: CubeStore) -> Dict[Tuple[str, str], List[str]]:
    rows = store._get_conn().execute(
        """
        SELECT flm.label_id,
               flm.role,
               f.canonical_id
        FROM feature_label_membership flm
        JOIN feature f ON f.feature_id = flm.feature_id
        ORDER BY flm.label_id, flm.role, f.canonical_id
        """
    ).fetchall()
    out: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for r in rows:
        key = (r["label_id"], r["role"])
        canonical_id = r["canonical_id"]
        if canonical_id not in out[key]:
            out[key].append(canonical_id)
    return out


def _base_scores_by_predicate(
    store: CubeStore,
    *,
    model: str,
    scorer: str,
    base_config_id: int,
    predicate_name: Optional[str],
    filters: Optional[Sequence[FilterSpec]] = None,
) -> Dict[Any, Optional[float]]:
    compiler = _QueryCompiler()
    where, params = compiler.compile_filters(filters or [])
    extra_where = f" AND {' AND '.join(where)}" if where else ""
    if predicate_name:
        rows = store._get_conn().execute(
            f"""
            SELECT p.value AS predicate_value,
                   AVG(ev.score) AS avg_score
            FROM execution e
            JOIN evaluation ev ON ev.execution_id = e.execution_id
            JOIN query q ON q.query_id = e.query_id
            JOIN predicate p ON p.query_id = e.query_id AND p.name = ?
            {compiler.join_sql()}
            WHERE e.config_id = ?
              AND e.model = ?
              AND ev.scorer = ?
              AND (e.error IS NULL OR e.error = '')
              AND ev.score IS NOT NULL
              {extra_where}
            GROUP BY p.value
            """,
            tuple([predicate_name, int(base_config_id), model, scorer] + params),
        ).fetchall()
        return {
            _normalize_sql_value(r["predicate_value"]): _float_or_none(r["avg_score"])
            for r in rows
        }

    row = store._get_conn().execute(
        f"""
        SELECT AVG(ev.score) AS avg_score
        FROM execution e
        JOIN evaluation ev ON ev.execution_id = e.execution_id
        JOIN query q ON q.query_id = e.query_id
        {compiler.join_sql()}
        WHERE e.config_id = ?
          AND e.model = ?
          AND ev.scorer = ?
          AND (e.error IS NULL OR e.error = '')
          AND ev.score IS NOT NULL
          {extra_where}
        """,
        tuple([int(base_config_id), model, scorer] + params),
    ).fetchone()
    return {None: _float_or_none(row["avg_score"]) if row else None}


def _parse_field(field: str) -> Tuple[str, str]:
    if field in {"dataset", "query.dataset"}:
        return "dataset", ""
    if field in {"split", "query.split", "benchmark.split"}:
        return "query_split", ""
    if field in {"query_id", "query.query_id"}:
        return "query_id", ""
    if field in {"config_id", "config.config_id"}:
        return "config_id", ""
    if field in {"score", "ev.score", "evaluation.score"}:
        return "score", ""
    if field in {"error", "execution.error"}:
        return "error", ""
    if field.startswith("query.meta."):
        return "query_meta", field[len("query.meta."):]
    if field.startswith("predicate."):
        return "predicate", field[len("predicate."):]
    raise ValueError(f"unsupported field: {field!r}")


def _query_scope_where(
    query_alias: str,
    *,
    dataset: Optional[str] = None,
    split: Optional[str] = None,
) -> Tuple[List[str], List[Any]]:
    where = ["1=1"]
    params: List[Any] = []
    if dataset:
        where.append(f"{query_alias}.dataset = ?")
        params.append(dataset)
    if split:
        split_clause, split_params = _split_filter_clause(query_alias, split)
        where.append(split_clause)
        params.extend(split_params)
    return where, params


def _query_result_join(
    query_alias: str,
    *,
    model: Optional[str] = None,
    scorer: Optional[str] = None,
    only_with_results: bool = False,
) -> Tuple[str, List[Any]]:
    if not only_with_results or not model or not scorer:
        return "", []
    return (
        "JOIN v_query_scores s_scope "
        f"ON s_scope.query_id = {query_alias}.query_id "
        "AND s_scope.model = ? AND s_scope.scorer = ?",
        [model, scorer],
    )


def _split_filter_clause(query_alias: str, split: str) -> Tuple[str, List[Any]]:
    expr = f"json_extract({query_alias}.meta, '$.split')"
    if split == "(no split)":
        return f"({expr} IS NULL OR {expr} = '')", []
    return f"{expr} = ?", [split]


def _json_path(key: str) -> str:
    if not key:
        raise ValueError("empty JSON key")
    parts = key.split(".")
    out = "$"
    for part in parts:
        if _SIMPLE_KEY_RE.match(part):
            out += f".{part}"
        else:
            out += f'["{part.replace(chr(34), chr(92) + chr(34))}"]'
    return out


def _group_key(group: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted(group.items()))


def _json_loads(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _first(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[0]
    return value


def _string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(v) for v in value if v is not None and str(v)]


def _display_feature_set(
    canonical_ids: Sequence[str],
    *,
    kind: Optional[str],
    has_funcs: bool,
) -> Optional[str]:
    ids = [str(v) for v in canonical_ids if str(v)]
    if not ids:
        return None
    if kind == "base":
        non_sections = [v for v in ids if not v.startswith("_section_")]
        shown = non_sections or ids
        return "base: " + ", ".join(shown)
    if len(ids) <= 3:
        return ", ".join(ids)
    if not has_funcs:
        return ", ".join(ids)
    return f"{len(ids)} features: " + ", ".join(ids[:3]) + ", ..."


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _normalize_sql_value(value: Any) -> Any:
    if value is None:
        return None
    return value


def _sample_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True)
    else:
        text = str(value)
    return text if len(text) <= 80 else text[:77] + "..."


def _gold_from_meta(meta: Dict[str, Any]) -> Any:
    if not isinstance(meta, dict):
        return None
    raw = meta.get("_raw")
    if isinstance(raw, dict):
        for key in ("answers", "answer", "gold", "gold_answer"):
            if key in raw:
                return raw[key]
    for key in ("gold_answer", "gold_answers", "answer", "answers", "label"):
        if key in meta:
            return meta[key]
    return None


def _compact_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    return {k: v for k, v in meta.items() if k != "_raw"}


def _preview(value: Any, *, n: int = 500) -> str:
    text = value or ""
    return text if len(text) <= n else text[:n] + "..."


def _metric_value(metrics: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metrics:
            return metrics[key]
    return None


def _bucket_value(value: Any) -> str:
    if value is None or value == "":
        return "(missing)"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _response_pattern(raw_preview: str, prediction: str, error: str) -> str:
    raw = raw_preview or ""
    if error:
        return "execution_error"
    if not raw.strip():
        return "empty_response"
    if re.search(r"(?m)^\s*Final Answer\s*:", raw):
        return "bare_final_answer_line"
    if "print(" in raw and "Final Answer:" in raw:
        return "prints_final_answer"
    if "```python" in raw or "```py" in raw:
        return "python_fenced_block"
    if prediction.strip():
        return "has_prediction"
    return "unclassified"


def _empty_delete_plan() -> Dict[str, Any]:
    return {
        "nExecutions": 0,
        "nEvaluations": 0,
        "executionIdPreview": [],
        "sqlPreview": [],
        "note": "Dry run only. No config ids selected.",
    }


def _delete_sql_preview(
    *,
    model: Optional[str],
    config_ids: Optional[Sequence[int]],
) -> List[str]:
    clauses: List[str] = []
    if model:
        clauses.append(f"model = {_sql_quote(model)}")
    if config_ids is not None:
        ids = ",".join(str(int(c)) for c in config_ids)
        clauses.append(f"config_id IN ({ids})")
    where = " AND ".join(clauses) if clauses else "<same filters as dry run>"
    return [
        "DELETE FROM evaluation WHERE execution_id IN "
        f"(SELECT execution_id FROM execution WHERE {where});",
        f"DELETE FROM execution WHERE {where};",
    ]


def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
