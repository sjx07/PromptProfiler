#!/usr/bin/env python3
"""Render dry-run prompts for configs or ad hoc feature coalitions.

This is intentionally offline: it seeds queries/features into a temporary cube,
materializes the configured feature sets, renders prompts, and writes markdown
artifacts. It does not call an LLM and does not write to cfg["db_path"].

Config example:
    python3 tools/render_prompts_from_config.py \
      Obsidian/facet_exp/tablebench/configs/table_bench_pot_loo.json

Ad hoc coalition examples:
    python3 tools/render_prompts_from_config.py --task wtq \
      --features reasoning_scaffold_visible_cot,input_context_column_statistics \
      --split test --max-queries 2 --group-by ''

    python3 tools/render_prompts_from_config.py --task sqa \
      --features reasoning_scaffold_visible_cot,input_context_column_statistics \
      --split test --max-queries 2 --group-by ''

    python3 tools/render_prompts_from_config.py --task tablebench \
      --features reasoning_scaffold_visible_cot,input_context_column_statistics \
      --split test --max-queries 2 --group-by qtype,qsubtype

    python3 tools/render_prompts_from_config.py --task tabfact \
      --features reasoning_scaffold_visible_cot,input_context_column_statistics \
      --split validation --max-queries 2 --group-by ''

    python3 tools/render_prompts_from_config.py --task hitab \
      --features reasoning_scaffold_visible_cot,input_context_column_statistics \
      --split test --max-queries 2 --group-by ''
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import seed_funcs
from core.feature_registry import FeatureRegistry
from core.func_registry import apply_config, apply_config_modules
from core.store import CubeStore
from experiment.config_generators import REGISTRY as GENERATORS
from experiment.config_generators import generate
from task_registry import get_registry


GENERATOR_OPTION_KEYS = (
    "min_features",
    "max_features",
    "min_rules",
    "max_rules",
    "coalitions",
    "config_kind",
)

COMMON_RENDER_BASE = [
    "_section_role",
    "_section_task",
    "_section_rules",
    "_section_table_handling",
    "_section_reasoning",
    "_section_format_fix",
    "facet_dp_scaffold",
    "prompt_format_plain",
    "table_serialization_json_columns_data",
]

TASK_RENDER_BASE_EXTRAS = {
    "sqa": ["sqa_dialog_binding_base"],
}

TASK_DEFAULT_CONTRACT = {
    "wtq": "output_contract_json_answer_list",
    "sqa": "output_contract_json_answer_list",
    "hitab": "output_contract_json_answer_list",
    "tabfact": "output_contract_json_verdict",
    "tablebench": "output_contract_tablebench_answer_string",
}

TASK_DEFAULT_SPLIT = {
    "wtq": "test",
    "sqa": "test",
    "hitab": "test",
    "tabfact": "validation",
    "tablebench": "test",
}

EXAMPLE_TEXT = """examples:
  render an existing experiment config:
    python3 tools/render_prompts_from_config.py Obsidian/facet_exp/wtq/configs/systematic/gpt_oss_20b/existing_clean_coalitions/wtq.fmt_ser_ctx_reasoning.full.json --max-configs 3 --max-queries 2

  render an ad hoc WTQ coalition:
    python3 tools/render_prompts_from_config.py --task wtq --features reasoning_scaffold_visible_cot,input_context_column_statistics --split test --max-queries 2 --group-by ''

  render an ad hoc SQA coalition:
    python3 tools/render_prompts_from_config.py --task sqa --features reasoning_scaffold_visible_cot,input_context_column_statistics --split test --max-queries 2 --group-by ''

  render an ad hoc TableBench coalition:
    python3 tools/render_prompts_from_config.py --task tablebench --features reasoning_scaffold_visible_cot,input_context_column_statistics --split test --max-queries 2 --group-by qtype,qsubtype

  render an ad hoc TabFact coalition:
    python3 tools/render_prompts_from_config.py --task tabfact --features reasoning_scaffold_visible_cot,input_context_column_statistics --split validation --max-queries 2 --group-by ''

  render an ad hoc HiTab coalition:
    python3 tools/render_prompts_from_config.py --task hitab --features reasoning_scaffold_visible_cot,input_context_column_statistics --split test --max-queries 2 --group-by ''
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render prompt dry-run artifacts from an experiment config JSON or ad hoc feature coalition.",
        epilog=EXAMPLE_TEXT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        nargs="?",
        help="Experiment config JSON path. Omit when using --task/--features quick coalition mode.",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task name for quick coalition mode, e.g. wtq, sqa, tablebench, tabfact, hitab.",
    )
    parser.add_argument(
        "--features",
        action="append",
        default=[],
        help="Comma-separated feature canonical IDs for quick coalition mode. Can be repeated.",
    )
    parser.add_argument(
        "--base-features",
        action="append",
        default=[],
        help="Comma-separated base feature canonical IDs for quick coalition mode. Defaults to a task-specific rendering base.",
    )
    parser.add_argument(
        "--coalition-label",
        default=None,
        help="Label for the quick coalition config. Defaults to coalition__<features>.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory. Defaults to <config parent>/../dry_runs/<config stem> "
            "when config is under a configs/ directory, else dry_runs/<config stem>."
        ),
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Override config split for query seeding/selection.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Override config max_queries before seeding. 0 means all, matching run_experiment.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Override config sample_seed before seeding.",
    )
    parser.add_argument(
        "--group-by",
        default="qtype,qsubtype",
        help="Comma-separated query fields used to choose representative prompts. Use '' for one all-query group.",
    )
    parser.add_argument(
        "--samples-per-group",
        type=int,
        default=1,
        help="Number of query examples to render per group per config.",
    )
    parser.add_argument(
        "--only-config",
        action="append",
        default=[],
        help="Render only labels matching this exact label or regex. Can be repeated.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=0,
        help="Render at most this many configs after filtering. 0 means no cap.",
    )
    parser.add_argument(
        "--tmp-db",
        default=None,
        help="Optional path for the temporary dry-run cube. Defaults to a tempfile under /tmp.",
    )
    args = parser.parse_args()

    if args.config:
        cfg_path = Path(args.config)
        with cfg_path.open() as f:
            cfg = json.load(f)
    else:
        cfg_path = Path(f"render_{args.task or 'coalition'}.json")
        cfg = _quick_coalition_config(args)
    if args.split is not None:
        cfg["split"] = args.split
    if args.max_queries is not None:
        cfg["max_queries"] = args.max_queries
    if args.sample_seed is not None:
        cfg["sample_seed"] = args.sample_seed

    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(cfg_path)
    prompts_dir = out_dir / "rendered_prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    if args.tmp_db:
        store = CubeStore(args.tmp_db)
        tmp_ctx = None
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="prompt_profiler_dry_run_")
        store = CubeStore(Path(tmp_ctx.name) / "dry_run.db")

    try:
        manifest = render_from_config(
            cfg,
            cfg_path=cfg_path,
            out_dir=out_dir,
            prompts_dir=prompts_dir,
            store=store,
            group_by=[x.strip() for x in args.group_by.split(",") if x.strip()],
            samples_per_group=max(1, args.samples_per_group),
            only_config=args.only_config,
            max_configs=max(0, args.max_configs),
        )
    finally:
        store.close()
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {manifest['n_prompt_files']} prompt files")
    print(f"Output: {out_dir}")
    print(f"Manifest: {manifest_path}")


def render_from_config(
    cfg: Dict[str, Any],
    *,
    cfg_path: Path,
    out_dir: Path,
    prompts_dir: Path,
    store: CubeStore,
    group_by: List[str],
    samples_per_group: int,
    only_config: List[str],
    max_configs: int,
) -> Dict[str, Any]:
    task_name = cfg.get("task")
    if not task_name:
        raise ValueError("config must define 'task'")

    registry = get_registry()
    if task_name not in registry:
        raise ValueError(f"unknown task {task_name!r}; available={sorted(registry)}")
    task_entry = registry[task_name]
    task_cls = task_entry.task_cls
    if hasattr(task_cls, "configure_from_cfg"):
        task_cls.configure_from_cfg(cfg)

    split = cfg.get("split", "dev")
    experiment_type = cfg.get("experiment_type", "add_one_feature")
    if experiment_type not in GENERATORS:
        raise ValueError(
            f"unsupported experiment_type {experiment_type!r}; available={sorted(GENERATORS)}"
        )

    base_features = list(cfg.get("base_features") or [])
    experiment_features = list(cfg.get("experiment_features") or [])
    if not base_features:
        raise ValueError("config must define non-empty 'base_features'")

    feature_registry = FeatureRegistry.load(task=task_name)
    full_specs, base_ids, bundles, base_feature_hashes, conflicts = _build_feature_bundles(
        feature_registry,
        base_features,
        experiment_features,
    )
    seed_funcs(store, full_specs)

    has_examples = any(spec["func_type"] == "add_example" for spec in full_specs)
    if has_examples:
        example_split = cfg.get("example_split")
        if not example_split:
            raise ValueError("'example_split' is required because at least one feature uses add_example")
        if example_split == split:
            raise ValueError(f"example_split must differ from split; both are {split!r}")
        task_entry.seeder_fn(store, _example_seed_cfg(cfg), example_split)
        example_pool = _load_queries(store, task_entry.dataset_key_fn(cfg), example_split)
    else:
        example_pool = None

    task_entry.seeder_fn(store, cfg, split)
    dataset_key = task_entry.dataset_key_fn(cfg)
    queries = _load_queries(store, dataset_key, split)
    if not queries:
        raise ValueError(f"no queries seeded for dataset={dataset_key!r} split={split!r}")

    base_config_id = store.get_or_create_config(
        base_ids,
        meta={
            "kind": "base",
            "canonical_ids": list(base_features),
            "feature_ids": [base_feature_hashes[c] for c in base_features],
        },
    )
    generated = generate(
        experiment_type,
        store,
        base_ids=base_ids,
        bundles=bundles,
        conflicts=conflicts,
        base_canonical_ids=list(base_features),
        base_feature_ids=[base_feature_hashes[c] for c in base_features],
        **_generator_kwargs(cfg),
    )

    configs = [
        {
            "config_id": base_config_id,
            "label": "base",
            "kind": "base",
            "func_ids": base_ids,
            "meta": {
                "canonical_ids": list(base_features),
                "feature_ids": [base_feature_hashes[c] for c in base_features],
            },
        }
    ]
    for config_id, func_ids, meta in generated:
        configs.append({
            "config_id": config_id,
            "label": _config_label(meta, config_id),
            "kind": str(meta.get("experiment") or meta.get("kind") or experiment_type),
            "func_ids": func_ids,
            "meta": meta,
        })

    configs = _filter_configs(configs, only_config)
    if max_configs:
        configs = configs[:max_configs]

    grouped_queries = _select_grouped_queries(
        queries,
        group_by=group_by,
        samples_per_group=samples_per_group,
    )

    artifacts = []
    for config in configs:
        config_dir = prompts_dir / _safe_name(config["label"])
        config_dir.mkdir(parents=True, exist_ok=True)
        task = task_cls()
        if hasattr(task, "bind_modules"):
            states = apply_config_modules(
                config["func_ids"],
                store,
                module_names=task.module_names(),
            )
            task.bind_modules(states, example_pool=example_pool)
        else:
            state = apply_config(config["func_ids"], store)
            task.bind(state, example_pool=example_pool)

        for group_key, group_queries in grouped_queries:
            for index, query in enumerate(group_queries, start=1):
                system_prompt, user_content = task.build_prompt(query)
                filename = f"{_safe_name(group_key)}"
                if len(group_queries) > 1:
                    filename += f"__sample_{index}"
                filename += ".md"
                path = config_dir / filename
                query_meta = _meta(query)
                rendered = _render_markdown(
                    config=config,
                    group_key=group_key,
                    query=query,
                    query_meta=query_meta,
                    system_prompt=system_prompt,
                    user_content=user_content,
                )
                path.write_text(rendered)
                artifacts.append({
                    "path": str(path.relative_to(out_dir)),
                    "config_id": config["config_id"],
                    "config_label": config["label"],
                    "group": group_key,
                    "query_id": query.get("query_id"),
                    "meta": _summary_meta(query_meta),
                })

    return {
        "config_path": str(cfg_path),
        "task": task_name,
        "dataset": dataset_key,
        "split": split,
        "experiment_type": experiment_type,
        "base_features": base_features,
        "experiment_features": experiment_features,
        "group_by": group_by,
        "samples_per_group": samples_per_group,
        "n_seeded_queries": len(queries),
        "n_configs": len(configs),
        "n_prompt_files": len(artifacts),
        "configs": [
            {
                "config_id": c["config_id"],
                "label": c["label"],
                "kind": c["kind"],
                "canonical_ids": c["meta"].get("canonical_ids", []),
            }
            for c in configs
        ],
        "artifacts": artifacts,
    }


def _quick_coalition_config(args: argparse.Namespace) -> Dict[str, Any]:
    task = str(args.task or "").strip()
    if not task:
        raise ValueError("quick coalition mode requires --task when no config path is provided")
    features = _parse_feature_args(args.features)
    base_features = _parse_feature_args(args.base_features)
    feature_registry = FeatureRegistry.load(task=task)
    if not base_features:
        base_features = _default_render_base_features(task, features, feature_registry)
    feature_registry.validate_feature_set(base_features + features)
    label = args.coalition_label or _coalition_label(features)
    split = args.split or TASK_DEFAULT_SPLIT.get(task, "test")
    max_queries = 2 if args.max_queries is None else args.max_queries
    cfg: Dict[str, Any] = {
        "task": task,
        "split": split,
        "max_queries": max_queries,
        "sample_seed": 0 if args.sample_seed is None else args.sample_seed,
        "experiment_type": "explicit_coalitions" if features else "base_only",
        "base_features": base_features,
        "experiment_features": features,
        "n_samples": 1,
        "seed": 0,
        "config_kind": "rendered_coalition",
    }
    if features:
        cfg["coalitions"] = {label: features}
    return cfg


def _parse_feature_args(values: List[str]) -> List[str]:
    out: List[str] = []
    for raw in values:
        for part in str(raw).split(","):
            item = part.strip()
            if item and item not in out:
                out.append(item)
    return out


def _default_render_base_features(
    task: str,
    selected_features: List[str],
    feature_registry: FeatureRegistry,
) -> List[str]:
    base_candidates = list(COMMON_RENDER_BASE)
    base_candidates.extend(TASK_RENDER_BASE_EXTRAS.get(task, []))
    contract = TASK_DEFAULT_CONTRACT.get(task)
    if contract:
        base_candidates.append(contract)
    return [
        feature
        for feature in base_candidates
        if not _feature_conflicts_with_any(feature, selected_features, feature_registry)
    ]


def _feature_conflicts_with_any(
    feature: str,
    selected_features: List[str],
    feature_registry: FeatureRegistry,
) -> bool:
    feature_spec = feature_registry._by_canonical.get(feature)
    feature_conflicts = set((feature_spec or {}).get("conflicts_with", []))
    for selected in selected_features:
        selected_spec = feature_registry._by_canonical.get(selected)
        if selected_spec is None:
            # Let materialize produce the normal unknown-feature error later.
            continue
        selected_conflicts = set(selected_spec.get("conflicts_with", []))
        if feature in selected_conflicts or selected in feature_conflicts:
            return True
    return False


def _coalition_label(features: List[str]) -> str:
    if not features:
        return "base"
    return "coalition__" + "__".join(features)


def _build_feature_bundles(
    feature_registry: FeatureRegistry,
    base_features: List[str],
    experiment_features: List[str],
) -> Tuple[List[dict], List[str], Dict[str, Tuple[str, List[str]]], Dict[str, str], Dict[str, frozenset]]:
    base_specs, _ = feature_registry.materialize(base_features)
    base_ids = [spec["func_id"] for spec in base_specs]
    base_set = set(base_ids)

    all_specs_by_func_id = {spec["func_id"]: spec for spec in base_specs}
    bundles: Dict[str, Tuple[str, List[str]]] = {}
    conflicts: Dict[str, frozenset] = {}
    for canonical_id in experiment_features:
        specs, feature_to_funcs = feature_registry.materialize(base_features + [canonical_id])
        for spec in specs:
            all_specs_by_func_id[spec["func_id"]] = spec
        feature_hash = feature_registry.feature_id_for(canonical_id)
        feature_func_ids = feature_to_funcs[feature_hash]
        bundles[canonical_id] = (
            feature_hash,
            [func_id for func_id in feature_func_ids if func_id not in base_set],
        )
        feature_spec = feature_registry._by_canonical[canonical_id]
        conflicts[canonical_id] = frozenset(feature_spec.get("conflicts_with", []))

    base_feature_hashes = {
        canonical_id: feature_registry.feature_id_for(canonical_id)
        for canonical_id in base_features
    }
    return list(all_specs_by_func_id.values()), base_ids, bundles, base_feature_hashes, conflicts


def _generator_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = {
        "n_samples": int(cfg.get("n_samples", 200) or 200),
        "seed": int(cfg.get("seed", 42) or 42),
    }
    for key in GENERATOR_OPTION_KEYS:
        if cfg.get(key) is not None:
            kwargs[key] = cfg[key]
    return kwargs


def _example_seed_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    out["max_queries"] = int(
        cfg.get(
            "max_example_queries",
            cfg.get(
                "max_train_queries",
                cfg.get("example_max_queries", cfg.get("max_queries", 0)),
            ),
        )
        or 0
    )
    out["sample_seed"] = int(
        cfg.get(
            "example_sample_seed",
            cfg.get("train_sample_seed", cfg.get("sample_seed", 0)),
        )
        or 0
    )
    return out


def _load_queries(store: CubeStore, dataset: str, split: str) -> List[dict]:
    rows = store._get_conn().execute(
        "SELECT * FROM query WHERE dataset = ? AND json_extract(meta, '$.split') = ? ORDER BY rowid",
        (dataset, split),
    ).fetchall()
    return [dict(row) for row in rows]


def _select_grouped_queries(
    queries: List[dict],
    *,
    group_by: List[str],
    samples_per_group: int,
) -> List[Tuple[str, List[dict]]]:
    if not group_by:
        return [("all", queries[:samples_per_group])]

    buckets: Dict[str, List[dict]] = defaultdict(list)
    for query in queries:
        parts = []
        for field in group_by:
            value = _query_value(query, field)
            parts.append(f"{field}={value if value not in (None, '') else 'unknown'}")
        buckets["__".join(parts)].append(query)

    return [
        (group_key, bucket[:samples_per_group])
        for group_key, bucket in sorted(buckets.items())
    ]


def _query_value(query: dict, field: str) -> Any:
    meta = _meta(query)
    raw = meta.get("_raw", {}) if isinstance(meta.get("_raw"), dict) else {}
    if "." in field:
        return _dotted_value({"query": query, "meta": meta, "_raw": raw}, field)
    if field in query:
        return query[field]
    if field in meta:
        return meta[field]
    if field in raw:
        return raw[field]
    return None


def _dotted_value(scope: Dict[str, Any], field: str) -> Any:
    parts = field.split(".")
    cur: Any
    if parts[0] in scope:
        cur = scope[parts[0]]
        parts = parts[1:]
    else:
        cur = scope.get("query", {})
    for part in parts:
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _meta(query: dict) -> Dict[str, Any]:
    meta = query.get("meta", {})
    if isinstance(meta, str):
        try:
            return json.loads(meta)
        except json.JSONDecodeError:
            return {}
    return meta if isinstance(meta, dict) else {}


def _summary_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    raw = meta.get("_raw", {}) if isinstance(meta.get("_raw"), dict) else {}
    out = {}
    for key in ("split", "qtype", "qsubtype", "statement", "question"):
        if key in meta:
            out[key] = meta[key]
        elif key in raw:
            out[key] = raw[key]
    return out


def _config_label(meta: Dict[str, Any], config_id: int) -> str:
    for key in (
        "label",
        "canonical_id",
        "removed_canonical_id",
        "added_rule",
        "removed_rule",
    ):
        if meta.get(key):
            return str(meta[key])
    if meta.get("subset_canonical_ids"):
        return "coalition__" + "__".join(str(x) for x in meta["subset_canonical_ids"])
    return f"config_{config_id}"


def _filter_configs(configs: List[dict], patterns: List[str]) -> List[dict]:
    if not patterns:
        return configs
    out = []
    for config in configs:
        label = config["label"]
        if any(label == pattern or re.search(pattern, label) for pattern in patterns):
            out.append(config)
    return out


def _render_markdown(
    *,
    config: Dict[str, Any],
    group_key: str,
    query: dict,
    query_meta: Dict[str, Any],
    system_prompt: str,
    user_content: str,
) -> str:
    summary = _summary_meta(query_meta)
    lines = [
        f"# {config['label']} x {group_key}",
        "",
        "## Metadata",
        f"- config_id: {config['config_id']}",
        f"- kind: {config['kind']}",
        f"- query_id: {query.get('query_id', '')}",
    ]
    for key, value in summary.items():
        value_text = str(value).replace("\n", " ")
        if len(value_text) > 240:
            value_text = value_text[:237] + "..."
        lines.append(f"- {key}: {value_text}")
    canonical_ids = config["meta"].get("canonical_ids") or []
    if canonical_ids:
        lines.append("- canonical_ids:")
        lines.extend(f"  - {canonical_id}" for canonical_id in canonical_ids)
    lines.extend([
        "",
        "## System Prompt",
        "",
        "```text",
        system_prompt.rstrip(),
        "```",
        "",
        "## User Content",
        "",
        "```text",
        user_content.rstrip(),
        "```",
        "",
    ])
    return "\n".join(lines)


def _default_out_dir(config_path: Path) -> Path:
    for parent in [config_path.parent, *config_path.parents]:
        if parent.name == "configs":
            return parent.parent / "dry_runs" / config_path.stem
    return Path("dry_runs") / config_path.stem


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value).strip())
    safe = safe.strip("._")
    return safe[:180] or "unnamed"


if __name__ == "__main__":
    main()
