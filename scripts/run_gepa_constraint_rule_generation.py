#!/usr/bin/env python3
"""Constraint-style GEPA rule generation batches over FACET cube failures.

This is a discovery tool, not an experiment runner. It samples compact failure
batches from an existing cube, asks a local instruction model to convert those
failures into concrete constraint-style prompt rules, and writes structured
candidate constraints for later FACET validation.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DB = Path("/data/users/jsu323/facet/wikitable_clean_surface_v1.db")
DEFAULT_OUT = Path("study_layer/artifacts/gepa_constraint_rule_generation_v1")
DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"

BASE_CONFIGS = {
    "wtq": 1,
    "sqa": 5,
    "tablebench": 34,
    "hitab": 1,
}


@dataclass(frozen=True)
class BatchSpec:
    batch_id: str
    dataset: str
    base_config_id: int
    description: str
    focus: str
    max_failures: int = 6
    max_successes: int = 2


BATCHES = [
    BatchSpec(
        batch_id="wtq_surface_role_failures",
        dataset="wtq",
        base_config_id=BASE_CONFIGS["wtq"],
        description="WTQ failures involving surface roles such as score, rank, title, code, row order, or named entity vs identifier.",
        focus="Discover small source/table rules for atomic values, row-order navigation, rank-vs-value, and identifier-vs-name confusions.",
    ),
    BatchSpec(
        batch_id="sqa_followup_active_set_failures",
        dataset="sqa",
        base_config_id=BASE_CONFIGS["sqa"],
        description="SQA follow-up failures where the current question depends on prior answer sets or prior row binding.",
        focus="Discover task-local follow-up rules for active-set filtering, one-answer-per-active-item, projection after history, and anchor exclusion.",
    ),
    BatchSpec(
        batch_id="tablebench_numeric_operation_failures",
        dataset="tablebench",
        base_config_id=BASE_CONFIGS["tablebench"],
        description="TableBench numerical-reasoning failures involving ranking, aggregation, time filters, units, or formatted numeric values.",
        focus="Discover operation rules for bind-then-compute, unit-aware comparison, ranking direction, and aggregate scope.",
    ),
    BatchSpec(
        batch_id="hitab_hierarchy_source_failures",
        dataset="hitab",
        base_config_id=BASE_CONFIGS["hitab"],
        description="HiTab hierarchy/source failures from StatCan and hierarchical tables with header paths, units, and pair comparisons.",
        focus="Discover hierarchy/source rules for header-path inheritance, source units, total/subcategory roles, and pair argmax/argmin cells.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ports", default="8000,8001,8002,8003")
    parser.add_argument("--cases-per-batch", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--dry-run", action="store_true", help="Write prompts but do not call the local model.")
    return parser.parse_args()


def load_rows(conn: sqlite3.Connection, spec: BatchSpec, score: float, limit: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT q.query_id, q.dataset, q.content, q.meta,
               e.prediction, e.raw_response, e.error, ev.score
        FROM query q
        JOIN execution e ON e.query_id = q.query_id
        JOIN evaluation ev ON ev.execution_id = e.execution_id
        WHERE q.dataset = ?
          AND e.config_id = ?
          AND e.model = ?
          AND ev.score = ?
          AND (e.error IS NULL OR e.error = '')
        ORDER BY q.query_id
        """,
        (spec.dataset, spec.base_config_id, DEFAULT_MODEL, score),
    ).fetchall()
    filtered = [row_to_case(row) for row in rows]
    filtered = [case for case in filtered if case_matches_batch(case, spec)]
    return filtered[:limit]


def row_to_case(row: sqlite3.Row) -> dict[str, Any]:
    meta = parse_json(row["meta"], {})
    return {
        "query_id": row["query_id"],
        "dataset": row["dataset"],
        "question": row["content"],
        "gold": extract_gold(meta),
        "prediction": row["prediction"],
        "score": row["score"],
        "metadata": compact_meta(meta),
        "table_preview": table_preview(meta),
        "history": compact_history(meta),
    }


def parse_json(text: str | None, default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def extract_gold(meta: dict[str, Any]) -> Any:
    for key in ("gold_answers", "gold_answer", "answer", "gold_label"):
        if key in meta:
            return meta[key]
    raw = meta.get("_raw", {}) if isinstance(meta.get("_raw"), dict) else {}
    for key in ("answers", "answer", "answer_text"):
        if key in raw:
            return raw[key]
    return None


def compact_meta(meta: dict[str, Any]) -> dict[str, Any]:
    keep = {}
    for key in [
        "split",
        "table_name",
        "table_file",
        "position",
        "sequence_id",
        "qtype",
        "qsubtype",
        "instruction_type",
        "source_dataset",
        "table_id",
        "table_source",
        "aggregation",
    ]:
        if key in meta:
            keep[key] = meta[key]
    raw = meta.get("_raw", {}) if isinstance(meta.get("_raw"), dict) else {}
    for key in ["id", "source_line", "sub_sentence", "answer_formulas", "reference_cells_map"]:
        if key in raw:
            keep[key] = raw[key]
    return keep


def compact_history(meta: dict[str, Any]) -> list[dict[str, Any]]:
    raw = meta.get("_raw", {}) if isinstance(meta.get("_raw"), dict) else {}
    history = raw.get("history", [])
    if not isinstance(history, list):
        return []
    return history[-3:]


def table_preview(meta: dict[str, Any]) -> str:
    raw = meta.get("_raw", {}) if isinstance(meta.get("_raw"), dict) else {}

    if "table" in raw and isinstance(raw["table"], dict):
        table = raw["table"]
        header = table.get("header") or table.get("headers") or []
        rows = table.get("rows") or []
        title = table.get("name") or meta.get("table_name") or meta.get("table_file") or ""
        return render_table(title, header, rows)

    if "table_content" in raw and isinstance(raw["table_content"], dict):
        table = raw["table_content"]
        texts = table.get("texts") or []
        title = table.get("title") or ""
        if texts and isinstance(texts, list):
            header = texts[0]
            rows = texts[1:]
            return render_table(title, header, rows, max_rows=10, max_cols=10)

    return ""


def render_table(title: str, header: Any, rows: Any, *, max_rows: int = 7, max_cols: int = 9) -> str:
    if not isinstance(header, list):
        return ""
    row_list = rows if isinstance(rows, list) else []
    header = [clean_cell(x) for x in header[:max_cols]]
    lines = []
    if title:
        lines.append(f"title: {clean_cell(title, 180)}")
    lines.append("columns: " + " | ".join(header))
    for idx, row in enumerate(row_list[:max_rows]):
        if not isinstance(row, list):
            continue
        cells = [clean_cell(x) for x in row[:max_cols]]
        lines.append(f"row {idx}: " + " | ".join(cells))
    if len(row_list) > max_rows:
        lines.append(f"... {len(row_list) - max_rows} more rows")
    return "\n".join(lines)


def clean_cell(value: Any, limit: int = 80) -> str:
    text = str(value).replace("\n", " ").replace("\t", " ").strip()
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def case_matches_batch(case: dict[str, Any], spec: BatchSpec) -> bool:
    q = case["question"].lower()
    meta = case.get("metadata", {})
    if spec.batch_id == "wtq_surface_role_failures":
        cues = [
            "score", "result", "rank", "place", "position", "seed", "points", "medal",
            "title", "code", "number", "episode", "route", "symbol", "previous", "next",
            "first", "last", "before", "after", "country", "chart",
        ]
        return any(cue in q for cue in cues)
    if spec.batch_id == "sqa_followup_active_set_failures":
        return int(meta.get("position") or 0) > 0
    if spec.batch_id == "tablebench_numeric_operation_failures":
        return meta.get("qtype") == "NumericalReasoning" or meta.get("qsubtype") in {
            "Ranking", "Aggregation", "ArithmeticCalculation", "Time-basedCalculation", "Comparison",
        }
    if spec.batch_id == "hitab_hierarchy_source_failures":
        agg = str(meta.get("aggregation", "")).lower()
        table_source = str(meta.get("table_source", "")).lower()
        return bool(table_source in {"statcan", "nsf", "totto"} or "arg" in agg or "opposite" in agg)
    return True


def build_batches(db_path: Path, cases_per_batch: int) -> list[dict[str, Any]]:
    failures = max(1, min(6, cases_per_batch - 2))
    successes = max(0, cases_per_batch - failures)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    batches = []
    try:
        for spec in BATCHES:
            failure_cases = load_rows(conn, spec, 0.0, failures)
            success_cases = load_rows(conn, spec, 1.0, successes)
            cases = []
            for case in failure_cases:
                case = dict(case)
                case["outcome"] = "failure"
                cases.append(case)
            for case in success_cases:
                case = dict(case)
                case["outcome"] = "success_contrast"
                cases.append(case)
            batches.append({
                "batch_id": spec.batch_id,
                "dataset": spec.dataset,
                "base_config_id": spec.base_config_id,
                "description": spec.description,
                "focus": spec.focus,
                "cases": cases,
            })
    finally:
        conn.close()
    return batches


def system_prompt() -> str:
    return (
        "You imitate GEPA-style prompt reflection, but your output must be constraint-like, "
        "not broad guidance. Given compact failed trajectories, infer reusable constraints that "
        "could be inserted into a prompt and later evaluated as FACET features. Each rule must "
        "have a narrow trigger, an explicit required action, and an explicit forbidden action. "
        "Avoid vague verbs like handle, ensure, correctly, consider, understand, or pay attention. "
        "Do not propose generic 'think step by step'. Do not propose output formatting fixes unless "
        "the cases are purely parser failures. Generalize one level beyond exact entities, but do "
        "not invent unsupported benchmark facts. Return strict JSON only."
    )


def user_prompt(batch: dict[str, Any]) -> str:
    compact_cases = []
    for idx, case in enumerate(batch["cases"], 1):
        compact_cases.append({
            "case_id": f"{batch['batch_id']}:{idx}",
            "outcome": case["outcome"],
            "query_id": case["query_id"],
            "question": case["question"],
            "gold": case["gold"],
            "prediction": case["prediction"],
            "metadata": case["metadata"],
            "history": case["history"],
            "table_preview": case["table_preview"],
        })
    return (
        f"Batch: {batch['batch_id']}\n"
        f"Dataset: {batch['dataset']}\n"
        f"Description: {batch['description']}\n"
        f"Focus: {batch['focus']}\n\n"
        "Cases include failures and a few success contrasts from the same slice.\n"
        "Propose 3-6 candidate prompt constraints that could rescue the failures with minimal off-slice harm.\n"
        "Each candidate must be specific enough to test as a binary intervention. Prefer if/then, must, only, never, or do-not wording.\n"
        "A good constraint names the exact operation boundary, role binding, comparison direction, active set, or cell-surface preservation rule.\n\n"
        "Return JSON with this schema:\n"
        "{\n"
        "  \"batch_id\": str,\n"
        "  \"failure_mode_summary\": str,\n"
        "  \"candidate_rules\": [\n"
        "    {\n"
        "      \"rule_id\": short_snake_case,\n"
        "      \"constraint_text\": prompt-ready constraint using if/then, must, only, never, or do-not wording,\n"
        "      \"trigger_predicate\": deterministic trigger using question/header/history/meta cues,\n"
        "      \"required_action\": exact action the model must take,\n"
        "      \"forbidden_action\": exact behavior the model must avoid,\n"
        "      \"feature_family\": one of [input_context, private_reasoning_rule, response_contract, table_serialization, domain_heuristic, task_heuristic],\n"
        "      \"off_slice_risk\": low|medium|high,\n"
        "      \"validation_slice\": deterministic slice to test later,\n"
        "      \"source_case_ids\": list of case_id strings\n"
        "    }\n"
        "  ],\n"
        "  \"rules_to_reject\": [{\"idea\": str, \"reason\": str}]\n"
        "}\n\n"
        "Cases:\n"
        f"{json.dumps(compact_cases, indent=2, ensure_ascii=True)}"
    )


def call_local_chat(port: int, model: str, system: str, user: str, max_tokens: int, timeout: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "top_p": 0.95,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    start = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        choice = body["choices"][0]
        message = choice.get("message", {})
        return {
            "ok": True,
            "port": port,
            "latency_s": round(time.time() - start, 3),
            "raw_response": (message.get("content") or "").strip(),
            "raw_reasoning": (message.get("reasoning_content") or message.get("reasoning") or "").strip(),
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage", {}),
        }
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
        return {
            "ok": False,
            "port": port,
            "latency_s": round(time.time() - start, 3),
            "error": repr(exc),
            "raw_response": "",
            "raw_reasoning": "",
        }


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        parsed = json.loads(snippet)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_report(out_dir: Path, batches: list[dict[str, Any]], responses: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> None:
    md_lines = [
        "# GEPA-Style Constraint Rule Generation v1",
        "",
        "This is a small failure-driven constraint discovery run over the current cube. Candidate constraints are hypotheses for later FACET validation, not accepted features.",
        "",
        f"Batches: `{len(batches)}`",
        f"Candidate constraints parsed: `{len(candidates)}`",
        "",
        "## Batches",
        "",
    ]
    for batch in batches:
        n_fail = sum(1 for c in batch["cases"] if c["outcome"] == "failure")
        n_success = sum(1 for c in batch["cases"] if c["outcome"] == "success_contrast")
        md_lines.append(f"- `{batch['batch_id']}` ({batch['dataset']}): {n_fail} failures, {n_success} success contrasts. {batch['focus']}")
    md_lines.extend(["", "## Candidate Constraints", ""])
    for rule in candidates:
        md_lines.append(
            f"- `{rule.get('batch_id')}.{rule.get('rule_id')}` [{rule.get('feature_family')}, risk={rule.get('off_slice_risk')}]: "
            f"{rule.get('constraint_text') or rule.get('rule_text')} Trigger: {rule.get('trigger_predicate') or rule.get('context_trigger')}"
        )
    md_lines.extend(["", "## Raw Response Status", ""])
    for response in responses:
        status = "ok" if response.get("ok") else "error"
        md_lines.append(f"- `{response['batch_id']}` via port `{response.get('port')}`: {status}, latency={response.get('latency_s')}s, finish={response.get('finish_reason')}")
    (out_dir / "report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    rows_html = []
    for rule in candidates:
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(rule.get('batch_id', '')))}</td>"
            f"<td>{html.escape(str(rule.get('rule_id', '')))}</td>"
            f"<td>{html.escape(str(rule.get('feature_family', '')))}</td>"
            f"<td>{html.escape(str(rule.get('off_slice_risk', '')))}</td>"
            f"<td>{html.escape(str(rule.get('trigger_predicate') or rule.get('context_trigger', '')))}</td>"
            f"<td>{html.escape(str(rule.get('constraint_text') or rule.get('rule_text', '')))}</td>"
            f"<td>{html.escape(str(rule.get('validation_slice', '')))}</td>"
            "</tr>"
        )
    html_doc = f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>GEPA-style Constraint Rule Generation v1</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #202124; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
th {{ background: #f6f8fa; position: sticky; top: 0; }}
code {{ background: #f6f8fa; padding: 1px 4px; border-radius: 3px; }}
</style></head><body>
<h1>GEPA-style Constraint Rule Generation v1</h1>
<p>Candidate constraints are failure-driven hypotheses for later FACET validation.</p>
<p><b>Batches:</b> {len(batches)} &nbsp; <b>Candidate rules:</b> {len(candidates)}</p>
<table><thead><tr><th>Batch</th><th>Rule</th><th>Family</th><th>Risk</th><th>Trigger</th><th>Constraint Text</th><th>Validation Slice</th></tr></thead>
<tbody>{''.join(rows_html)}</tbody></table>
</body></html>
"""
    (out_dir / "report.html").write_text(html_doc, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ports = [int(p.strip()) for p in args.ports.split(",") if p.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    batches = build_batches(args.db, args.cases_per_batch)
    prompts = []
    for batch in batches:
        prompts.append({
            "batch_id": batch["batch_id"],
            "system_prompt": system_prompt(),
            "user_prompt": user_prompt(batch),
        })

    (args.out_dir / "case_batches.json").write_text(json.dumps(batches, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    write_jsonl(args.out_dir / "llm_prompts.jsonl", prompts)

    responses: list[dict[str, Any]] = []
    if args.dry_run:
        for prompt in prompts:
            responses.append({"batch_id": prompt["batch_id"], "ok": False, "error": "dry_run", "raw_response": ""})
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(ports), len(prompts))) as pool:
            futures = []
            for idx, prompt in enumerate(prompts):
                port = ports[idx % len(ports)]
                futures.append(
                    pool.submit(
                        call_local_chat,
                        port,
                        args.model,
                        prompt["system_prompt"],
                        prompt["user_prompt"],
                        args.max_tokens,
                        args.timeout,
                    )
                )
            for prompt, future in zip(prompts, futures):
                result = future.result()
                result["batch_id"] = prompt["batch_id"]
                responses.append(result)

    write_jsonl(args.out_dir / "raw_responses.jsonl", responses)

    candidates: list[dict[str, Any]] = []
    for response in responses:
        parsed = extract_json_object(response.get("raw_response", ""))
        response["parsed"] = parsed
        if not parsed:
            continue
        for idx, rule in enumerate(parsed.get("candidate_rules", []) or [], 1):
            if not isinstance(rule, dict):
                continue
            row = dict(rule)
            row.setdefault("rule_id", f"rule_{idx}")
            row["batch_id"] = response["batch_id"]
            row["failure_mode_summary"] = parsed.get("failure_mode_summary", "")
            candidates.append(row)

    write_jsonl(args.out_dir / "parsed_responses.jsonl", responses)
    write_jsonl(args.out_dir / "candidate_rules.jsonl", candidates)
    write_report(args.out_dir, batches, responses, candidates)

    print(f"wrote {args.out_dir}")
    print(f"batches={len(batches)} responses={len(responses)} candidates={len(candidates)}")
    for response in responses:
        status = "ok" if response.get("ok") else f"error={response.get('error')}"
        print(f"{response['batch_id']}: {status} port={response.get('port')} latency={response.get('latency_s')}")


if __name__ == "__main__":
    main()
