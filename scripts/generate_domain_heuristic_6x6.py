#!/usr/bin/env python3
"""Generate the WikiTable domain-subpack 6x6 experiment configs.

The design is benchmark x source-domain heuristic:
  6 target benchmarks x 6 WikiTable source-domain subpacks.

This intentionally creates same-canonical feature specs inside each target
task registry, because the current FeatureRegistry is task-local and does not
allow cross-task feature references at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = (
    ROOT
    / "Obsidian/facet_exp/wikitable_transfer/configs/systematic/qwen25_14b"
    / "domain_heuristic_6x6"
)
MANIFEST_PATH = ROOT / "study_layer/coalition_manifests/wikitable_domain_subpack_6x6_v1.json"

CUBE_DB = "/data/users/jsu323/facet/wikitable_domain_subpack_6x6_v1.db"
VLLM_DB = "/data/users/jsu323/facet/wikitable_domain_subpack_6x6_v1_vllm.db"

GROUNDING_ARTIFACTS = [
    "Obsidian/Transferability/Project/coding_agent_logs/codex/code/systematic_design/benchmark_design/domain_heuristic_wiki/finer_domain_heuristics_v1.json",
    "Obsidian/Transferability/Project/coding_agent_logs/codex/code/systematic_design/benchmark_design/domain_heuristic_wiki/benchmark_domain_hint_rules_v1.json",
    "Obsidian/Transferability/Project/coding_agent_logs/codex/code/systematic_design/benchmark_design/domain_heuristic_wiki/source_semantic_annotation_matrix_v1.json",
]

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


DOMAIN_SUBPACKS: list[dict[str, Any]] = [
    {
        "axis_label": "domain.wikitable.sports_competition",
        "canonical_id": "dataset_domain_heuristics.wikitable_sports_competition",
        "file_stem": "dataset_domain_heuristics_wikitable_sports_competition",
        "rule_label": "domain_hint_rule.wikitable.sports_competition",
        "source_family": "wikipedia_table",
        "feature_content": (
            "For Wikipedia sports and competition tables, first decide whether the table is about "
            "matches, seasons, medals, races, or rankings. Treat score strings like 18-12 as "
            "complete results unless the question asks for a margin or difference. Attach duplicate "
            "Score/Result columns to their neighboring team/player columns. Interpret Rank, Place, "
            "Position, Seed, and Pick as ordinal fields where lower numbers are earlier or better; "
            "interpret Points, score totals, attendance, and medals as magnitude fields where larger "
            "numbers are bigger. Preserve full athlete, team, and country labels when they are the answer."
        ),
        "source_rule_ids": ["wtq.score_is_result", "wtq.rank_vs_value", "wtq.adjacent_metric_group"],
    },
    {
        "axis_label": "domain.wikitable.election_result",
        "canonical_id": "dataset_domain_heuristics.wikitable_election_result",
        "file_stem": "dataset_domain_heuristics_wikitable_election_result",
        "rule_label": "domain_hint_rule.wikitable.election_result",
        "source_family": "wikipedia_table",
        "feature_content": (
            "For Wikipedia election tables, parse candidate-result groups before selecting an answer. "
            "Candidate or person, vote count, percentage share, party, district or riding, incumbent, "
            "and result columns are distinct roles. If the question asks for votes, use vote counts; "
            "if it asks for percentage or share, use percentages. For beat, lost by, or majority "
            "questions, compute the margin in the requested unit. Preserve the full candidate or "
            "result cell when the question asks for the candidate entry or election result."
        ),
        "source_rule_ids": ["wtq.adjacent_metric_group", "wtq.rank_vs_value"],
    },
    {
        "axis_label": "domain.wikitable.media_episode",
        "canonical_id": "dataset_domain_heuristics.wikitable_media_episode",
        "file_stem": "dataset_domain_heuristics_wikitable_media_episode",
        "rule_label": "domain_hint_rule.wikitable.media_episode",
        "source_family": "wikipedia_table",
        "feature_content": (
            "For Wikipedia television, episode, and media tables, keep episode title, episode number, "
            "season number, series number, production code, air date, director, writer, viewers, "
            "rating, and share as separate roles. If the question asks which episode, return the "
            "episode title unless it asks for a number or code. If the row summarizes a season, use "
            "the Episodes cell for episode counts. For director or writer counts, count rows only "
            "within the requested season or active set."
        ),
        "source_rule_ids": ["wtq.identifier_vs_named_entity", "wtq.row_order_navigation"],
    },
    {
        "axis_label": "domain.wikitable.music_chart",
        "canonical_id": "dataset_domain_heuristics.wikitable_music_chart",
        "file_stem": "dataset_domain_heuristics_wikitable_music_chart",
        "rule_label": "domain_hint_rule.wikitable.music_chart",
        "source_family": "wikipedia_table",
        "feature_content": (
            "For Wikipedia music and chart tables, do not collapse music roles. Song or single, "
            "album, artist, producer, featured guest, sales, and chart position are distinct. Use "
            "the exact chart column named by the question, such as GER peak chart position. Interpret "
            "chart positions as ordinal ranks where smaller numbers are better; blanks and dashes "
            "usually mean did not chart. Keep sales quantities with commas as one numeric value."
        ),
        "source_rule_ids": ["wtq.chart_country_header", "wtq.rank_vs_value", "wtq.country_code_alias"],
    },
    {
        "axis_label": "domain.wikitable.geo_admin",
        "canonical_id": "dataset_domain_heuristics.wikitable_geo_admin",
        "file_stem": "dataset_domain_heuristics_wikitable_geo_admin",
        "rule_label": "domain_hint_rule.wikitable.geo_admin",
        "source_family": "wikipedia_table",
        "feature_content": (
            "For Wikipedia geography and administrative tables, keep place roles separate. Country, "
            "county, city, state or province, location, venue, nationality, and represent are not "
            "interchangeable. If a cell gives a compound place such as City, Country, preserve the "
            "complete listed place unless the question explicitly asks for only one part. For "
            "same-place questions, exclude the anchor entity unless the question asks to include it."
        ),
        "source_rule_ids": ["wtq.country_code_alias"],
    },
    {
        "axis_label": "domain.wikitable.identifier_rank_code",
        "canonical_id": "dataset_domain_heuristics.wikitable_identifier_rank_code",
        "file_stem": "dataset_domain_heuristics_wikitable_identifier_rank_code",
        "rule_label": "domain_hint_rule.wikitable.identifier_rank_code",
        "source_family": "wikipedia_table",
        "feature_content": (
            "For Wikipedia identifier, rank, and code tables, decide whether the question asks for "
            "an identifier, the entity named by that identifier, or the ranked entity. Route codes, "
            "episode numbers, Unicode glyphs, production codes, ranks, and seeds are identifiers or "
            "ordinal fields, not names. Do not substitute the identifier for the name, title, or "
            "value column unless the question explicitly asks for the code or number. Preserve "
            "identifier prefixes such as Ep. when the identifier surface itself is the answer."
        ),
        "source_rule_ids": ["wtq.identifier_vs_named_entity", "wtq.rank_vs_value"],
    },
]


TASKS: dict[str, dict[str, Any]] = {
    "wtq": {
        "config_stem": "wtq.wikitable_domain_subpack_6x6_v1.full",
        "phase": "wtq_wikitable_domain_subpack_6x6_v1_qwen25_14b_full",
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
        "config_stem": "sqa.wikitable_domain_subpack_6x6_v1.full",
        "phase": "sqa_wikitable_domain_subpack_6x6_v1_qwen25_14b_full",
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
        "config_stem": "tabfact.wikitable_domain_subpack_6x6_v1.n1024",
        "phase": "tabfact_wikitable_domain_subpack_6x6_v1_qwen25_14b_1024",
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
        "config_stem": "tablebench.wikitable_domain_subpack_6x6_v1.n1024",
        "phase": "tablebench_wikitable_domain_subpack_6x6_v1_qwen25_14b_1024",
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
        "config_stem": "hitab.wikitable_domain_subpack_6x6_v1.n1024",
        "phase": "hitab_wikitable_domain_subpack_6x6_v1_qwen25_14b_1024",
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
        "config_stem": "hover.wikitable_domain_subpack_6x6_v1.n300",
        "phase": "hover_wikitable_domain_subpack_6x6_v1_qwen25_14b_300",
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


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n")


def table_feature_spec(task: str, subpack: dict[str, Any]) -> dict[str, Any]:
    content = (
        "Use this WikiTable source-domain block when the source surface matches:\n"
        f"- {subpack['feature_content']}"
    )
    guard = (
        "Apply this source-domain block only when the table, header, cell values, or question "
        "surface match it. If it conflicts with explicit wording or table evidence, follow the "
        "explicit wording and evidence. Do not change the required output format."
    )
    return base_feature_spec(task, subpack, TABLE_REQUIRES) | {
        "primitive_edits": [
            {
                "func_type": "insert_node",
                "params": {
                    "node_type": "rule",
                    "parent_id": TABLE_HANDLING_SECTION_ID,
                    "payload": {"content": content, "ordinal": 64},
                },
            },
            {
                "func_type": "insert_node",
                "params": {
                    "node_type": "rule",
                    "parent_id": TABLE_REASONING_SECTION_ID,
                    "payload": {"content": guard, "ordinal": 74},
                },
            },
        ]
    }


def hover_feature_spec(task: str, subpack: dict[str, Any]) -> dict[str, Any]:
    retrieval_content = (
        "Use this WikiTable source-domain block only when a claim or retrieved passage describes "
        "a table-like list, ranking, chart, result table, episode list, election table, or "
        "administrative table:\n"
        f"- {subpack['feature_content']}\n"
        "Preserve title-like entities, years, and relation anchors needed for retrieval. Do not "
        "force table assumptions onto ordinary prose passages."
    )
    summary_content = (
        "When the retrieved Wikipedia passage describes table-like evidence, preserve the relevant "
        "source-domain roles in the evidence summary:\n"
        f"- {subpack['feature_content']}\n"
        "If the passage is ordinary prose and the block does not match, ignore the block."
    )
    edits = []
    for module in ["create_query_hop2", "create_query_hop3"]:
        edits.append(
            {
                "target_module": module,
                "func_type": "insert_node",
                "params": {
                    "node_type": "rule",
                    "parent_id": HOVER_RETRIEVAL_SECTION_ID,
                    "payload": {"content": retrieval_content, "ordinal": 114},
                },
            }
        )
    for module in ["summarize1", "summarize2"]:
        edits.append(
            {
                "target_module": module,
                "func_type": "insert_node",
                "params": {
                    "node_type": "rule",
                    "parent_id": HOVER_EVIDENCE_SECTION_ID,
                    "payload": {"content": summary_content, "ordinal": 114},
                },
            }
        )
    return base_feature_spec(task, subpack, HOVER_REQUIRES) | {"primitive_edits": edits}


def base_feature_spec(task: str, subpack: dict[str, Any], requires: list[str]) -> dict[str, Any]:
    canonical_id = subpack["canonical_id"]
    benchmark_label = f"benchmark.{task}"
    return {
        "canonical_id": canonical_id,
        "task": task,
        "semantic_labels": [
            {"label": "dataset_domain_heuristic.subpack", "role": "feature_concept"},
            {"label": canonical_id, "role": "feature_concept"},
            {"label": benchmark_label, "role": "dataset"},
            {"label": "source_family.wikipedia_table", "role": "source_family"},
            {"label": "heuristic_scope.source_surface_transfer_candidate", "role": "transfer_scope"},
            {"label": "transfer_scope.cross_benchmark_when_source_surface_matches", "role": "transfer_scope"},
            {"label": "feature_mode.rule_bundle", "role": "experiment_design"},
            {"label": "implementation_surface.task_or_user_instruction", "role": "implementation_surface"},
            {"label": subpack["rule_label"], "role": "included_rule"},
        ],
        "scope": {
            "dataset": task,
            "source_family": subpack["source_family"],
            "feature_mode": "rule_bundle",
            "heuristic_scope": "source_surface_transfer_candidate",
            "transfer_scope": "cross_benchmark_when_source_surface_matches",
            "scope_boundary": "dataset_source_not_task_protocol",
            "implementation_surface": "task_or_user_instruction",
            "cost_dependency": "prompt_only",
            "domain_block_id": subpack["axis_label"],
            "source_artifact": GROUNDING_ARTIFACTS[0],
            "grounding_artifacts": GROUNDING_ARTIFACTS,
            "source_rule_ids": subpack["source_rule_ids"],
            "prompt_surface": "clean_rule_subpack",
            "prompt_surface_note": (
                "Prompt text renders source/data rules only; task protocol, output contract, "
                "and benchmark parser fixes live in separate feature families."
            ),
        },
        "requires": requires,
        "conflicts_with": [
            "dataset_domain_heuristics.wikitable_surface",
            "domain_heuristics.wtq",
            "domain_heuristics.sqa",
            "domain_heuristics.tabfact",
            "domain_heuristics.tablebench",
            "domain_heuristics.hitab",
            "domain_heuristics.hover_context",
        ],
        "rationale": (
            "Fine-grained WikiTable source-domain subpack from the domain heuristic artifact. "
            "Generated for each target task so the task-local registry can run the 6x6 transfer screen."
        ),
    }


def write_feature_specs() -> list[Path]:
    paths: list[Path] = []
    for task in TASKS:
        feature_dir = ROOT / "features" / task / "dataset_domain_heuristic"
        for subpack in DOMAIN_SUBPACKS:
            spec = hover_feature_spec(task, subpack) if task == "hover_context" else table_feature_spec(task, subpack)
            path = feature_dir / f"{subpack['file_stem']}.json"
            write_json(path, spec)
            paths.append(path)
    return paths


def experiment_features_for(task_cfg: dict[str, Any]) -> list[str]:
    ids = list(task_cfg["fixed_coalition"])
    ids.extend(subpack["canonical_id"] for subpack in DOMAIN_SUBPACKS)
    return ids


def config_for(task: str, task_cfg: dict[str, Any]) -> dict[str, Any]:
    fixed = list(task_cfg["fixed_coalition"])
    coalitions = {"base": fixed}
    for subpack in DOMAIN_SUBPACKS:
        coalitions[subpack["axis_label"]] = fixed + [subpack["canonical_id"]]

    cfg: dict[str, Any] = {
        "experiment_type": "explicit_coalitions",
        "task": task,
        "split": task_cfg["split"],
        "max_queries": task_cfg["max_queries"],
        "sample_seed": task_cfg["sample_seed"],
        "seed": 42,
        "max_tokens": task_cfg["max_tokens"],
        "request_timeout": task_cfg["request_timeout"],
        "temperature": 0,
        "top_p": 1.0,
        "sampling_top_k": -1,
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "ports": [8000, 8001, 8002, 8003],
        "num_workers": 128,
        "db_path": CUBE_DB,
        "vllm_db": VLLM_DB,
        "batch_configs": True,
        "include_base": False,
        "summary_baseline_label": "base",
        "phase": task_cfg["phase"],
        "base_features": task_cfg["base_features"],
        "experiment_features": experiment_features_for(task_cfg),
        "coalitions": coalitions,
        "coalition_interpretation": (
            "WikiTable source-domain subpack 6x6 screen. Each target benchmark runs the same "
            "six fine-grained domain heuristic concepts over a fixed base prompt/surface. "
            "This is a transfer and negative-control matrix, not an optimized benchmark prompt."
        ),
    }
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


def preflight_for(config_path: Path, task: str, cfg: dict[str, Any]) -> str:
    rel = config_path.relative_to(ROOT)
    cells = len(DOMAIN_SUBPACKS)
    return f"""# Domain Heuristic 6x6 Preflight: {task}

Config: `{rel}`

## Purpose

Run the WikiTable source-domain subpack transfer screen for `{task}`.

- Domain axis: {cells} fine-grained WikiTable source-domain subpacks.
- Benchmark axis: `{task}` target benchmark.
- Base/surface: fixed per-benchmark scaffold, serialization, and output contract.
- Cube: `{cfg['db_path']}`
- VLLM cache: `{cfg['vllm_db']}`

## Coalitions

`base` plus:

{chr(10).join(f"- `{subpack['axis_label']}` -> `{subpack['canonical_id']}`" for subpack in DOMAIN_SUBPACKS)}

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
        preflight_path.write_text(preflight_for(path, task, cfg))
        paths.extend([path, preflight_path])
    return paths


def write_manifest() -> Path:
    data = {
        "schema_version": "wikitable_domain_subpack_6x6_v1",
        "purpose": "6 target benchmarks x 6 fine WikiTable source-domain heuristic subpacks.",
        "db_path": CUBE_DB,
        "vllm_db": VLLM_DB,
        "benchmark_axis": list(TASKS),
        "domain_axis": [
            {
                "axis_label": subpack["axis_label"],
                "canonical_id": subpack["canonical_id"],
                "source_rule_ids": subpack["source_rule_ids"],
            }
            for subpack in DOMAIN_SUBPACKS
        ],
        "config_dir": str(OUT_DIR.relative_to(ROOT)),
        "grounding_artifacts": GROUNDING_ARTIFACTS,
        "notes": [
            "SQA follow-up/dialog rules are intentionally excluded; they are task heuristics, not dataset-domain heuristics.",
            "HoVer is included as a passage-source negative/control target for table-domain subpacks.",
            "Each config contains base plus six one-subpack coalitions over a fixed per-benchmark surface.",
        ],
    }
    write_json(MANIFEST_PATH, data)
    return MANIFEST_PATH


def write_readme(config_paths: list[Path]) -> Path:
    readme = OUT_DIR / "README.md"
    config_jsons = [p for p in config_paths if p.suffix == ".json"]
    lines = [
        "# WikiTable Domain Subpack 6x6",
        "",
        "Generated by `scripts/generate_domain_heuristic_6x6.py`.",
        "",
        "This directory contains one config per benchmark. Each config runs `base` plus the same six fine WikiTable source-domain heuristic subpacks.",
        "",
        f"Cube: `{CUBE_DB}`",
        f"VLLM cache: `{VLLM_DB}`",
        "",
        "## Sequential Commands",
        "",
        "```bash",
    ]
    for path in config_jsons:
        rel = path.relative_to(ROOT)
        lines.extend(
            [
                "WAIT_FOR_COMPLETION=1 PATH=/nethome/jsu323/miniconda3/envs/tran/bin:/nethome/jsu323/miniconda3/bin:/usr/local/bin:/usr/bin:/bin \\",
                "  bash launch_experiment.sh \\",
                f"  {rel.with_suffix('.preflight.md')} \\",
                f"  {rel}",
                "",
            ]
        )
    lines.extend(["```", ""])
    readme.write_text("\n".join(lines))
    return readme


def main() -> None:
    feature_paths = write_feature_specs()
    config_paths = write_configs()
    manifest_path = write_manifest()
    readme_path = write_readme(config_paths)
    print(f"Wrote {len(feature_paths)} feature specs")
    print(f"Wrote {len(config_paths)} config/preflight files")
    print(f"Wrote {manifest_path.relative_to(ROOT)}")
    print(f"Wrote {readme_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
