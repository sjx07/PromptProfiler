#!/usr/bin/env python3
"""Generate GEPA source-benchmark domain heuristic bundles and configs.

The design is target_benchmark x source_benchmark x rule_style:
  target benchmarks: wtq, sqa, tabfact, tablebench, hitab, hover_context
  source benchmarks: wtq, sqa, tablebench, hitab
  rule styles: guidance, constraint

The same canonical feature ids are written into each target task registry because
FeatureRegistry is task-local at runtime. Prompt text is taken from the curated
GEPA guidance/constraint artifacts, grouped by the benchmark that produced the
failure slice.
"""

from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

GUIDANCE_PATH = ROOT / "study_layer/artifacts/gepa_rule_generation_v0/example_enriched_guidance_v1.jsonl"
CONSTRAINT_PATH = ROOT / "study_layer/artifacts/gepa_constraint_rule_generation_v1/curated_constraints_v1.jsonl"
OUT_DIR = (
    ROOT
    / "Obsidian/facet_exp/wikitable_transfer/configs/systematic/qwen25_14b"
    / "gepa_source_benchmark_domain_heuristics"
)
MANIFEST_PATH = ROOT / "study_layer/coalition_manifests/wikitable_gepa_source_benchmark_domain_v1.json"
PREVIEW_PATH = ROOT / "study_layer/artifacts/wikitable_gepa_source_benchmark_domain_v1_preview.html"

CUBE_DB = "/data/users/jsu323/facet/wikitable_clean_surface_v1.db"
VLLM_DB = "/data/users/jsu323/facet/wikitable_clean_surface_v1_vllm.db"

TABLE_REQUIRES = ["facet_dp_scaffold", "_section_table_handling", "_section_reasoning"]
HOVER_REQUIRES = [
    "facet_retrieval_scaffold",
    "_section_retrieval_strategy",
    "_section_evidence_summary",
]

TABLE_HANDLING_SECTION_ID = "1c2498fdcc32"
TABLE_REASONING_SECTION_ID = "d51d3f3e65a8"
HOVER_RETRIEVAL_SECTION_ID = "d25e9162c43e"
HOVER_EVIDENCE_SECTION_ID = "90b700e0f2aa"

SOURCE_BATCH_TO_BENCHMARK = {
    "wtq_surface_role_failures": "wtq",
    "sqa_followup_active_set_failures": "sqa",
    "tablebench_numeric_operation_failures": "tablebench",
    "hitab_hierarchy_source_failures": "hitab",
}
SOURCE_BENCHMARKS = ["wtq", "sqa", "tablebench", "hitab"]
STYLES = ["guidance", "constraint"]

TASKS: dict[str, dict[str, Any]] = {
    "wtq": {
        "config_stem": "wtq.wikitable_gepa_source_benchmark_domain_v1.full",
        "phase": "wtq_wikitable_gepa_source_benchmark_domain_v1_qwen25_14b_full",
        "split": "test",
        "max_queries": 0,
        "max_tokens": 1024,
        "request_timeout": 240,
        "sample_seed": 42,
        "base_features": [
            "_section_role",
            "_section_task",
            "_section_table_handling",
            "_section_reasoning",
            "_section_format_fix",
            "_section_rules",
            "facet_dp_scaffold",
        ],
        "fixed_coalition": [
            "prompt_format_plain",
            "table_serialization_json_columns_data",
            "output_contract_json_answer_list",
        ],
    },
    "sqa": {
        "config_stem": "sqa.wikitable_gepa_source_benchmark_domain_v1.full",
        "phase": "sqa_wikitable_gepa_source_benchmark_domain_v1_qwen25_14b_full",
        "split": "test",
        "max_queries": 0,
        "max_tokens": 1024,
        "request_timeout": 240,
        "sample_seed": 42,
        "sqa_data_dir": "/data/users/jsu323/sqa/SQA Release 1.0",
        "base_features": [
            "_section_role",
            "_section_task",
            "_section_table_handling",
            "_section_reasoning",
            "_section_format_fix",
            "_section_rules",
            "facet_dp_scaffold",
            "sqa_dialog_binding_base",
        ],
        "fixed_coalition": [
            "prompt_format_plain",
            "table_serialization_json_columns_data",
            "output_contract_json_answer_list",
        ],
    },
    "tabfact": {
        "config_stem": "tabfact.wikitable_gepa_source_benchmark_domain_v1.n1024",
        "phase": "tabfact_wikitable_gepa_source_benchmark_domain_v1_qwen25_14b_1024",
        "split": "validation",
        "max_queries": 1024,
        "max_tokens": 1024,
        "request_timeout": 240,
        "sample_seed": 42,
        "base_features": [
            "_section_role",
            "_section_task",
            "_section_rules",
            "_section_table_handling",
            "_section_reasoning",
            "_section_format_fix",
            "facet_dp_scaffold",
            "output_contract_json_verdict",
        ],
        "fixed_coalition": ["prompt_format_plain", "table_serialization_json_columns_data"],
    },
    "tablebench": {
        "config_stem": "tablebench.wikitable_gepa_source_benchmark_domain_v1.n1024",
        "phase": "tablebench_wikitable_gepa_source_benchmark_domain_v1_qwen25_14b_1024",
        "split": "test",
        "max_queries": 1024,
        "max_tokens": 1024,
        "request_timeout": 240,
        "sample_seed": 42,
        "base_features": [
            "_section_role",
            "_section_task",
            "_section_rules",
            "_section_table_handling",
            "_section_reasoning",
            "_section_format_fix",
            "facet_dp_scaffold",
            "output_contract_tablebench_answer_string",
        ],
        "fixed_coalition": ["prompt_format_plain", "table_serialization_json_columns_data"],
    },
    "hitab": {
        "config_stem": "hitab.wikitable_gepa_source_benchmark_domain_v1.n1024",
        "phase": "hitab_wikitable_gepa_source_benchmark_domain_v1_qwen25_14b_1024",
        "split": "test",
        "max_queries": 1024,
        "max_tokens": 1024,
        "request_timeout": 240,
        "sample_seed": 42,
        "base_features": [
            "_section_role",
            "_section_task",
            "_section_rules",
            "_section_table_handling",
            "_section_reasoning",
            "_section_format_fix",
            "facet_dp_scaffold",
            "output_contract_json_answer_list",
        ],
        "fixed_coalition": ["prompt_format_plain", "table_serialization_json_columns_data"],
    },
    "hover_context": {
        "config_stem": "hover.wikitable_gepa_source_benchmark_domain_v1.n300",
        "phase": "hover_wikitable_gepa_source_benchmark_domain_v1_qwen25_14b_300",
        "split": "test",
        "data_path": "/data/users/jsu323/datasets/hover/hover_dev_release_v1.1.json",
        "wiki_abstracts_path": "/data/users/jsu323/datasets/hotpotqa/wiki.abstracts.2017.jsonl",
        "bm25_index_dir": "/data/users/jsu323/datasets/hotpotqa/bm25_index",
        "retrieval_k": 15,
        "max_queries": 300,
        "max_tokens": 512,
        "request_timeout": 180,
        "sample_seed": 1,
        "retry_errors": True,
        "base_features": [
            "_section_role",
            "_section_task",
            "_section_retrieval_strategy",
            "_section_evidence_summary",
            "_section_format_fix",
            "prompt_format_json",
            "facet_retrieval_scaffold",
        ],
        "fixed_coalition": [],
    },
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
    return rows


def canonical_id(source: str, style: str) -> str:
    return f"domain_heuristics.source_benchmark.{source}.{style}"


def file_stem(source: str, style: str) -> str:
    return f"domain_heuristics_source_benchmark_{source}_{style}"


def axis_label(source: str, style: str) -> str:
    return f"source.{source}.{style}"


def group_rules() -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in load_jsonl(GUIDANCE_PATH):
        source = SOURCE_BATCH_TO_BENCHMARK[row["source_batch"]]
        grouped[(source, "guidance")].append({
            "rule_id": row.get("rule_id") or row.get("guidance_id"),
            "text": row["guidance_text"],
            "family": row.get("feature_family", "domain_heuristic"),
            "source_batch": row["source_batch"],
            "validation_slice": row.get("validation_slice", ""),
        })
    for row in load_jsonl(CONSTRAINT_PATH):
        source = SOURCE_BATCH_TO_BENCHMARK[row["source_batch"]]
        grouped[(source, "constraint")].append({
            "rule_id": row["constraint_id"],
            "text": row["constraint_text"],
            "family": row.get("feature_family", "domain_heuristic"),
            "source_batch": row["source_batch"],
            "required_action": row.get("required_action", ""),
            "forbidden_action": row.get("forbidden_action", ""),
            "validation_slice": row.get("validation_slice", ""),
        })
    missing = [(source, style) for source in SOURCE_BENCHMARKS for style in STYLES if (source, style) not in grouped]
    if missing:
        raise ValueError(f"Missing rule bundles: {missing}")
    return grouped


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def prompt_content(source: str, style: str, rules: list[dict[str, str]]) -> str:
    style_name = "guidance rules" if style == "guidance" else "hard constraints"
    intro = (
        f"Use these GEPA-derived {style_name} from {source.upper()} failures only when the current table, question, "
        "history, headers, or cell surfaces match. The examples inside each rule are part of the instruction: "
        "they anchor what kind of evidence the rule is talking about."
    )
    bullets = []
    for rule in rules:
        bullets.append(f"- [{rule['rule_id']}] {rule['text']}")
    return intro + "\n" + "\n".join(bullets)


def guard_content(source: str, style: str) -> str:
    return (
        f"Apply the {source.upper()} {style} bundle conditionally. If a rule does not match the table/question surface, "
        "ignore it. If a rule conflicts with explicit evidence, task instructions, or the required answer format, follow "
        "the explicit evidence/instructions/format. Do not invent benchmark-specific facts beyond the listed examples."
    )


def base_feature_spec(task: str, source: str, style: str, rules: list[dict[str, str]], requires: list[str]) -> dict[str, Any]:
    cid = canonical_id(source, style)
    rule_ids = [rule["rule_id"] for rule in rules]
    source_batches = sorted({rule["source_batch"] for rule in rules})
    return {
        "canonical_id": cid,
        "task": task,
        "semantic_labels": [
            {"label": "domain_heuristic.source_benchmark_bundle", "role": "feature_concept"},
            {"label": cid, "role": "feature_concept"},
            {"label": f"benchmark.{task}", "role": "target_benchmark"},
            {"label": f"source_benchmark.{source}", "role": "source_benchmark"},
            {"label": f"prompt_rule_style.{style}", "role": "rule_style"},
            {"label": "feature_mode.rule_bundle", "role": "experiment_design"},
            {"label": "implementation_surface.task_or_user_instruction", "role": "implementation_surface"},
            {"label": "heuristic_scope.failure_slice_transfer_candidate", "role": "transfer_scope"},
            {"label": "transfer_scope.cross_benchmark_when_surface_matches", "role": "transfer_scope"},
        ] + [
            {"label": f"gepa_rule.{rule_id}", "role": "included_rule"} for rule_id in rule_ids
        ],
        "scope": {
            "dataset": task,
            "target_benchmark": task,
            "source_benchmark": source,
            "rule_style": style,
            "source_batches": source_batches,
            "source_rule_ids": rule_ids,
            "feature_mode": "rule_bundle",
            "heuristic_scope": "failure_slice_transfer_candidate",
            "transfer_scope": "cross_benchmark_when_surface_matches",
            "scope_boundary": "source_benchmark_not_target_task_protocol",
            "implementation_surface": "task_or_user_instruction",
            "cost_dependency": "prompt_only",
            "prompt_surface": "example_enriched_rule_bundle",
            "source_artifacts": [
                str(GUIDANCE_PATH.relative_to(ROOT)),
                str(CONSTRAINT_PATH.relative_to(ROOT)),
            ],
            "prompt_surface_note": (
                "Concrete examples such as 18-12, 14-14, Quebec/Ontario, and =-B6 are rendered inside "
                "the prompt text, not only in metadata, because they are intended as in-context anchors."
            ),
        },
        "requires": requires,
        "conflicts_with": [
            "dataset_domain_heuristics.wikitable_surface",
            "dataset_domain_heuristics.wikitable_sports_competition",
            "dataset_domain_heuristics.wikitable_election_result",
            "dataset_domain_heuristics.wikitable_media_episode",
            "dataset_domain_heuristics.wikitable_music_chart",
            "dataset_domain_heuristics.wikitable_geo_admin",
            "dataset_domain_heuristics.wikitable_identifier_rank_code",
            "domain_heuristics.wtq",
            "domain_heuristics.sqa",
            "domain_heuristics.tabfact",
            "domain_heuristics.tablebench",
            "domain_heuristics.hitab",
            "domain_heuristics.hover_context",
            "domain_wikitable_source_semantics_pack",
            "domain_table_source_semantics_pack",
        ],
        "rationale": (
            "Source-benchmark GEPA rule bundle. The source axis captures where the failure slice was mined; "
            "the target task registry owns a task-local copy so runtime feature materialization stays simple."
        ),
    }


def table_feature_spec(task: str, source: str, style: str, rules: list[dict[str, str]]) -> dict[str, Any]:
    spec = base_feature_spec(task, source, style, rules, TABLE_REQUIRES)
    spec["primitive_edits"] = [
        {
            "func_type": "insert_node",
            "params": {
                "node_type": "rule",
                "parent_id": TABLE_HANDLING_SECTION_ID,
                "payload": {"content": prompt_content(source, style, rules), "ordinal": 66},
            },
        },
        {
            "func_type": "insert_node",
            "params": {
                "node_type": "rule",
                "parent_id": TABLE_REASONING_SECTION_ID,
                "payload": {"content": guard_content(source, style), "ordinal": 76},
            },
        },
    ]
    return spec


def hover_feature_spec(task: str, source: str, style: str, rules: list[dict[str, str]]) -> dict[str, Any]:
    spec = base_feature_spec(task, source, style, rules, HOVER_REQUIRES)
    retrieval_content = (
        "Use this source-benchmark rule bundle only when a claim or retrieved passage describes table-like "
        "evidence: a result table, ranking, score table, episode list, statistical table, or comparison table.\n"
        + prompt_content(source, style, rules)
        + "\nFor ordinary prose passages where these surfaces do not appear, ignore the bundle."
    )
    summary_content = (
        "When summarizing table-like Wikipedia evidence, preserve the roles named in this source-benchmark bundle.\n"
        + prompt_content(source, style, rules)
        + "\nIf the evidence is ordinary prose and the bundle does not match, ignore it."
    )
    edits: list[dict[str, Any]] = []
    for module in ["create_query_hop2", "create_query_hop3"]:
        edits.append({
            "target_module": module,
            "func_type": "insert_node",
            "params": {
                "node_type": "rule",
                "parent_id": HOVER_RETRIEVAL_SECTION_ID,
                "payload": {"content": retrieval_content, "ordinal": 116},
            },
        })
    for module in ["summarize1", "summarize2"]:
        edits.append({
            "target_module": module,
            "func_type": "insert_node",
            "params": {
                "node_type": "rule",
                "parent_id": HOVER_EVIDENCE_SECTION_ID,
                "payload": {"content": summary_content, "ordinal": 116},
            },
        })
    spec["primitive_edits"] = edits
    return spec


def write_feature_specs(grouped: dict[tuple[str, str], list[dict[str, str]]]) -> list[Path]:
    paths: list[Path] = []
    for task in TASKS:
        feature_dir = ROOT / "features" / task / "domain_heuristic"
        for source in SOURCE_BENCHMARKS:
            for style in STYLES:
                rules = grouped[(source, style)]
                spec = hover_feature_spec(task, source, style, rules) if task == "hover_context" else table_feature_spec(task, source, style, rules)
                path = feature_dir / f"{file_stem(source, style)}.json"
                write_json(path, spec)
                paths.append(path)
    return paths


def bundle_ids() -> list[str]:
    return [canonical_id(source, style) for source in SOURCE_BENCHMARKS for style in STYLES]


def config_for(task: str, task_cfg: dict[str, Any]) -> OrderedDict[str, Any]:
    fixed = list(task_cfg["fixed_coalition"])
    coalitions: OrderedDict[str, list[str]] = OrderedDict()
    coalitions["base"] = fixed
    metadata: OrderedDict[str, dict[str, str]] = OrderedDict()
    metadata["base"] = {"source": "anchor", "interpretation": "fixed surface without GEPA source-benchmark domain heuristic bundle"}
    for source in SOURCE_BENCHMARKS:
        for style in STYLES:
            label = axis_label(source, style)
            cid = canonical_id(source, style)
            coalitions[label] = fixed + [cid]
            metadata[label] = {
                "source_benchmark": source,
                "rule_style": style,
                "canonical_id": cid,
                "interpretation": "one source-benchmark GEPA rule bundle over fixed target surface",
            }

    cfg: OrderedDict[str, Any] = OrderedDict()
    cfg["experiment_type"] = "explicit_coalitions"
    cfg["task"] = task
    cfg["split"] = task_cfg["split"]
    cfg["max_queries"] = task_cfg["max_queries"]
    cfg["sample_seed"] = task_cfg["sample_seed"]
    cfg["seed"] = 42
    cfg["max_tokens"] = task_cfg["max_tokens"]
    cfg["request_timeout"] = task_cfg["request_timeout"]
    cfg["temperature"] = 0
    cfg["top_p"] = 1.0
    cfg["sampling_top_k"] = -1
    cfg["model"] = "Qwen/Qwen2.5-14B-Instruct"
    cfg["ports"] = [8000, 8001, 8002, 8003]
    cfg["num_workers"] = 128
    cfg["db_path"] = CUBE_DB
    cfg["vllm_db"] = VLLM_DB
    cfg["batch_configs"] = True
    cfg["include_base"] = False
    cfg["summary_baseline_label"] = "base"
    cfg["phase"] = task_cfg["phase"]
    cfg["base_features"] = task_cfg["base_features"]
    cfg["experiment_features"] = fixed + bundle_ids()
    cfg["coalitions"] = coalitions
    cfg["coalition_metadata"] = metadata
    cfg["coalition_interpretation"] = (
        "GEPA source-benchmark domain heuristic transfer screen. Each target benchmark runs base plus one "
        "source_benchmark/style bundle. The source axis is where the rules were discovered; the target axis "
        "tests whether those examples and constraints transfer when the surface matches."
    )
    for key in [
        "sqa_data_dir",
        "data_path",
        "wiki_abstracts_path",
        "bm25_index_dir",
        "retrieval_k",
        "retry_errors",
    ]:
        if key in task_cfg:
            cfg[key] = task_cfg[key]
    return cfg


def preflight_for(config_path: Path, task: str, cfg: OrderedDict[str, Any]) -> str:
    rel = config_path.relative_to(ROOT)
    bundles = len(SOURCE_BENCHMARKS) * len(STYLES)
    rows = "\n".join(
        f"- `{axis_label(source, style)}` -> `{canonical_id(source, style)}`"
        for source in SOURCE_BENCHMARKS for style in STYLES
    )
    return f"""# GEPA Source-Benchmark Domain Heuristics Preflight: {task}

Config: `{rel}`

## Purpose

Run the source-benchmark GEPA domain-heuristic transfer screen for `{task}`.

- Source axis: {len(SOURCE_BENCHMARKS)} source benchmarks: {', '.join(SOURCE_BENCHMARKS)}.
- Style axis: {', '.join(STYLES)}.
- Coalitions: `base` plus {bundles} one-bundle treatments.
- Cube: `{cfg['db_path']}`
- VLLM cache: `{cfg['vllm_db']}`

## Bundle Coalitions

{rows}

## Launch Command

```bash
WAIT_FOR_COMPLETION=1 PATH=/nethome/jsu323/miniconda3/envs/tran/bin:/nethome/jsu323/miniconda3/bin:/usr/local/bin:/usr/bin:/bin \\
  bash launch_experiment.sh \\
  {rel.with_suffix('.preflight.md')} \\
  {rel}
```
"""


def write_configs() -> list[Path]:
    paths: list[Path] = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for task, task_cfg in TASKS.items():
        cfg = config_for(task, task_cfg)
        path = OUT_DIR / f"{task_cfg['config_stem']}.json"
        write_json(path, cfg)
        preflight_path = path.with_suffix(".preflight.md")
        preflight_path.write_text(preflight_for(path, task, cfg), encoding="utf-8")
        paths.extend([path, preflight_path])
    return paths


def write_manifest(grouped: dict[tuple[str, str], list[dict[str, str]]]) -> Path:
    data = OrderedDict()
    data["schema_version"] = "wikitable_gepa_source_benchmark_domain_v1"
    data["purpose"] = "Target benchmark x source benchmark x guidance/constraint GEPA domain heuristic transfer screen."
    data["db_path"] = CUBE_DB
    data["vllm_db"] = VLLM_DB
    data["target_benchmark_axis"] = list(TASKS)
    data["source_benchmark_axis"] = SOURCE_BENCHMARKS
    data["style_axis"] = STYLES
    data["bundle_axis"] = [
        {
            "axis_label": axis_label(source, style),
            "canonical_id": canonical_id(source, style),
            "source_benchmark": source,
            "rule_style": style,
            "rule_ids": [rule["rule_id"] for rule in grouped[(source, style)]],
            "n_rules": len(grouped[(source, style)]),
        }
        for source in SOURCE_BENCHMARKS for style in STYLES
    ]
    data["config_dir"] = str(OUT_DIR.relative_to(ROOT))
    data["preview_path"] = str(PREVIEW_PATH.relative_to(ROOT))
    data["source_artifacts"] = [
        str(GUIDANCE_PATH.relative_to(ROOT)),
        str(CONSTRAINT_PATH.relative_to(ROOT)),
    ]
    data["notes"] = [
        "Rules are grouped by the benchmark whose failure slice generated them, not by target benchmark.",
        "Guidance and constraint styles are separate bundles to test whether stricter rule language matters.",
        "Concrete examples are rendered in prompt text because they are part of the in-context intervention.",
        "TabFact is target-only in this version because no TabFact source failure batch exists in the current GEPA artifacts.",
    ]
    write_json(MANIFEST_PATH, data)
    return MANIFEST_PATH


def write_readme(config_paths: list[Path]) -> Path:
    readme = OUT_DIR / "README.md"
    config_jsons = [p for p in config_paths if p.suffix == ".json"]
    lines = [
        "# GEPA Source-Benchmark Domain Heuristics",
        "",
        "Generated by `scripts/generate_gepa_source_benchmark_domain_heuristics.py`.",
        "",
        "Each target benchmark config runs `base` plus one bundle for every `source_benchmark * rule_style` cell.",
        "",
        f"Cube: `{CUBE_DB}`",
        f"VLLM cache: `{VLLM_DB}`",
        "",
        "## Source Bundle Axis",
        "",
    ]
    for source in SOURCE_BENCHMARKS:
        for style in STYLES:
            lines.append(f"- `{axis_label(source, style)}` -> `{canonical_id(source, style)}`")
    lines.extend(["", "## Sequential Commands", "", "```bash"])
    for path in config_jsons:
        rel = path.relative_to(ROOT)
        lines.extend([
            "WAIT_FOR_COMPLETION=1 PATH=/nethome/jsu323/miniconda3/envs/tran/bin:/nethome/jsu323/miniconda3/bin:/usr/local/bin:/usr/bin:/bin \\",
            "  bash launch_experiment.sh \\",
            f"  {rel.with_suffix('.preflight.md')} \\",
            f"  {rel}",
            "",
        ])
    lines.extend(["```", ""])
    readme.write_text("\n".join(lines), encoding="utf-8")
    return readme


def esc(text: Any) -> str:
    import html
    return html.escape(str(text))


def write_preview(grouped: dict[tuple[str, str], list[dict[str, str]]], config_paths: list[Path]) -> Path:
    cards = []
    for source in SOURCE_BENCHMARKS:
        for style in STYLES:
            rules = grouped[(source, style)]
            body = "".join(f"<li><code>{esc(rule['rule_id'])}</code>: {esc(rule['text'])}</li>" for rule in rules)
            cards.append(
                f"<section class='card'><h2>{esc(axis_label(source, style))}</h2>"
                f"<p><code>{esc(canonical_id(source, style))}</code></p>"
                f"<ul>{body}</ul></section>"
            )
    config_rows = []
    for path in [p for p in config_paths if p.suffix == ".json"]:
        task = path.name.split(".", 1)[0]
        config_rows.append(f"<tr><td>{esc(task)}</td><td><code>{esc(path.relative_to(ROOT))}</code></td></tr>")
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GEPA Source-Benchmark Domain Heuristics</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2933; background: #f7f8fa; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ font-size: 17px; margin: 0 0 6px; }}
    .subtle {{ color: #667085; margin-bottom: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 12px; }}
    .card {{ background: white; border: 1px solid #d7dde8; border-radius: 8px; padding: 14px; }}
    ul {{ padding-left: 20px; }}
    li {{ margin: 7px 0; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 18px 0; }}
    th, td {{ border: 1px solid #d7dde8; padding: 8px; text-align: left; }}
    th {{ background: #eef3f8; }}
  </style>
</head>
<body>
  <h1>GEPA Source-Benchmark Domain Heuristics</h1>
  <div class="subtle">Bundles are grouped by source benchmark and rule style. Examples are intentionally inside the prompt text.</div>
  <table><thead><tr><th>Target benchmark</th><th>Config</th></tr></thead><tbody>{''.join(config_rows)}</tbody></table>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW_PATH.write_text(html_doc, encoding="utf-8")
    return PREVIEW_PATH


def main() -> None:
    grouped = group_rules()
    feature_paths = write_feature_specs(grouped)
    config_paths = write_configs()
    manifest_path = write_manifest(grouped)
    readme_path = write_readme(config_paths)
    preview_path = write_preview(grouped, config_paths)
    print(f"Wrote {len(feature_paths)} feature specs")
    print(f"Wrote {len(config_paths)} config/preflight files")
    print(f"Wrote {manifest_path.relative_to(ROOT)}")
    print(f"Wrote {readme_path.relative_to(ROOT)}")
    print(f"Wrote {preview_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
