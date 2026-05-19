#!/usr/bin/env python3
"""Generate family-level prompt coalition configs from a manifest.

This script is intentionally a thin compiler: it validates block-level
coalition design, expands blocks into existing feature canonical_ids, and emits
ordinary ``explicit_coalitions`` configs for ``run_experiment.py``.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.feature_registry import FeatureRegistry  # noqa: E402


DEFAULT_MANIFEST = ROOT / "study_layer" / "coalition_manifests" / "wikitable_family_v1.json"
LABEL_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*(?:__[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*)*$|^base$")


class ManifestError(ValueError):
    """Manifest validation error."""


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def ordered_unique(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def block_family(manifest: dict[str, Any], block_id: str) -> str:
    try:
        return manifest["blocks"][block_id]["family"]
    except KeyError as exc:
        raise ManifestError(f"Unknown block: {block_id}") from exc


def family_mode(manifest: dict[str, Any], family: str) -> str:
    try:
        return manifest["families"][family]["mode"]
    except KeyError as exc:
        raise ManifestError(f"Unknown family: {family}") from exc


def block_label(blocks: list[str]) -> str:
    return "__".join(blocks) if blocks else "base"


def add_generated(
    rows: list[dict[str, Any]],
    seen: set[tuple[str, ...]],
    blocks: list[str],
    source: str,
) -> None:
    key = tuple(blocks)
    if key in seen:
        return
    seen.add(key)
    label = block_label(blocks)
    rows.append({
        "label": label,
        "selected_blocks": blocks,
        "source": source,
    })


def expand_generation(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    generation = manifest.get("generation", {})
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()

    add_generated(rows, seen, [], "anchor")

    for block_id in generation.get("main_blocks", []):
        add_generated(rows, seen, [block_id], "main_effect")

    for ladder in generation.get("ladders", []):
        name = ladder.get("name", "ladder")
        for step in ladder.get("steps", []):
            add_generated(rows, seen, list(step), name)

    for spec in generation.get("pairwise", []):
        name = spec.get("name", "pairwise")
        for left in spec.get("left", []):
            for right in spec.get("right", []):
                add_generated(rows, seen, [left, right], name)

    for spec in generation.get("selected", []):
        add_generated(rows, seen, list(spec.get("blocks", [])), spec.get("source", "selected"))

    return rows


def effective_blocks(
    manifest: dict[str, Any],
    fixed_blocks: list[str],
    selected_blocks: list[str],
) -> list[str]:
    drop_families: set[str] = set()
    selected_families: set[str] = set()
    blocks = manifest["blocks"]

    for block_id in selected_blocks:
        block = blocks[block_id]
        family = block["family"]
        selected_families.add(family)
        if family_mode(manifest, family).startswith("exclusive"):
            drop_families.add(family)
        drop_families.update(block.get("excludes_families", []))

    effective: list[str] = []
    for block_id in fixed_blocks:
        family = block_family(manifest, block_id)
        if family in drop_families or family in selected_families and family_mode(manifest, family).startswith("exclusive"):
            continue
        effective.append(block_id)

    effective.extend(selected_blocks)
    return ordered_unique(effective)


def atoms_for_blocks(manifest: dict[str, Any], block_ids: list[str]) -> list[str]:
    atoms: list[str] = []
    for block_id in block_ids:
        atoms.extend(manifest["blocks"][block_id].get("atoms", []))
    return ordered_unique(atoms)


def validate_manifest(manifest: dict[str, Any]) -> None:
    required_top = ["study_id", "common_config", "targets", "families", "blocks", "generation"]
    for key in required_top:
        if key not in manifest:
            raise ManifestError(f"Manifest missing required key: {key}")

    for family_id, spec in manifest["families"].items():
        if not re.match(r"^[a-z][a-z0-9_]*$", family_id):
            raise ManifestError(f"Invalid family id: {family_id}")
        if spec.get("mode") not in {"exclusive", "exclusive_or_fixed", "additive"}:
            raise ManifestError(f"Invalid mode for family {family_id}: {spec.get('mode')}")

    for block_id, spec in manifest["blocks"].items():
        if "." not in block_id:
            raise ManifestError(f"Block id must be <family>.<name>: {block_id}")
        family = spec.get("family")
        if family not in manifest["families"]:
            raise ManifestError(f"Block {block_id} references unknown family: {family}")
        if not isinstance(spec.get("atoms"), list) or not spec["atoms"]:
            raise ManifestError(f"Block {block_id} must define non-empty atoms")
        if not LABEL_RE.match(block_id):
            raise ManifestError(f"Block id is not label-safe: {block_id}")

    for target_name, target in manifest["targets"].items():
        for block_id in target.get("fixed_blocks", []):
            if block_id not in manifest["blocks"]:
                raise ManifestError(f"Target {target_name} fixed block is unknown: {block_id}")

    for row in expand_generation(manifest):
        label = row["label"]
        if not LABEL_RE.match(label):
            raise ManifestError(f"Generated label is not label-safe: {label}")
        for block_id in row["selected_blocks"]:
            if block_id not in manifest["blocks"]:
                raise ManifestError(f"Generated row references unknown block: {block_id}")


def validate_effective_blocks(manifest: dict[str, Any], effective: list[str]) -> list[str]:
    errors: list[str] = []
    by_family: dict[str, list[str]] = {}
    for block_id in effective:
        family = block_family(manifest, block_id)
        by_family.setdefault(family, []).append(block_id)

    for family, block_ids in by_family.items():
        if family_mode(manifest, family).startswith("exclusive") and len(block_ids) > 1:
            errors.append(f"exclusive family {family} has multiple blocks: {', '.join(block_ids)}")

    families = set(by_family)
    block_set = set(effective)
    for block_id in effective:
        block = manifest["blocks"][block_id]
        for required_family in block.get("requires_families", []):
            if required_family not in families:
                errors.append(f"{block_id} requires family {required_family}")
        for required_block in block.get("requires_blocks", []):
            if required_block not in block_set:
                errors.append(f"{block_id} requires block {required_block}")

    return errors


def validate_task_features(
    task: str,
    base_features: list[str],
    atoms: list[str],
) -> list[str]:
    errors: list[str] = []
    try:
        registry = FeatureRegistry.load(task=task)
    except Exception as exc:  # pragma: no cover - defensive CLI path
        return [f"cannot load FeatureRegistry for task={task}: {exc}"]

    for cid in base_features + atoms:
        try:
            registry.feature_id_for(cid)
        except KeyError:
            errors.append(f"unknown feature atom for {task}: {cid}")

    if not errors:
        try:
            registry.validate_feature_set(base_features + atoms)
        except ValueError as exc:
            errors.append(str(exc))

    return errors


def build_target(
    manifest: dict[str, Any],
    target_name: str,
    target: dict[str, Any],
    generated_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task = target["task"]
    base_features = list(target["base_features"])
    fixed_blocks = list(target.get("fixed_blocks", []))
    config: OrderedDict[str, Any] = OrderedDict()
    config.update(manifest["common_config"])
    config.update({k: v for k, v in target.items() if k not in {"config_path", "fixed_blocks"}})
    config["base_features"] = base_features

    coalitions: OrderedDict[str, list[str]] = OrderedDict()
    metadata: OrderedDict[str, dict[str, Any]] = OrderedDict()
    preview_rows: list[dict[str, Any]] = []

    for row in generated_rows:
        selected = list(row["selected_blocks"])
        effective = effective_blocks(manifest, fixed_blocks, selected)
        atoms = atoms_for_blocks(manifest, effective)
        errors = validate_effective_blocks(manifest, effective)
        errors.extend(validate_task_features(task, base_features, atoms))

        label = row["label"]
        status = "valid" if not errors else "invalid"
        preview_rows.append({
            "target": target_name,
            "task": task,
            "label": label,
            "source": row["source"],
            "selected_blocks": selected,
            "effective_blocks": effective,
            "atoms": atoms,
            "status": status,
            "errors": errors,
            "config_path": target["config_path"],
        })

        if not errors:
            coalitions[label] = atoms
            metadata[label] = {
                "selected_blocks": selected,
                "effective_blocks": effective,
                "families": sorted({block_family(manifest, b) for b in effective}),
                "atoms": atoms,
                "generation_rule": row["source"],
            }

    config["experiment_features"] = ordered_unique([atom for atoms in coalitions.values() for atom in atoms])
    config["coalitions"] = coalitions
    config["coalition_metadata"] = metadata
    config["family_manifest"] = {
        "study_id": manifest["study_id"],
        "target": target_name,
        "fixed_blocks": fixed_blocks,
        "config_path": target["config_path"],
    }
    return config, preview_rows


def write_preview(
    path: Path,
    manifest: dict[str, Any],
    configs: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    valid = sum(1 for row in rows if row["status"] == "valid")
    invalid = len(rows) - valid
    target_counts = {}
    for row in rows:
        target_counts.setdefault(row["target"], {"valid": 0, "invalid": 0})
        target_counts[row["target"]][row["status"]] += 1

    def esc(value: Any) -> str:
        return html.escape(str(value))

    def badges(items: list[str], cls: str = "badge") -> str:
        return " ".join(f'<span class="{cls}">{esc(item)}</span>' for item in items)

    table_rows = []
    for row in rows:
        errors = "<br>".join(esc(err) for err in row["errors"]) if row["errors"] else ""
        table_rows.append(
            "<tr>"
            f"<td>{esc(row['target'])}</td>"
            f"<td><code>{esc(row['label'])}</code><div class=\"subtle\">{esc(row['source'])}</div></td>"
            f"<td>{badges(row['selected_blocks']) if row['selected_blocks'] else '<span class=\"muted\">anchor</span>'}</td>"
            f"<td>{badges(row['effective_blocks'])}</td>"
            f"<td>{badges(row['atoms'], 'atom')}</td>"
            f"<td><span class=\"status {esc(row['status'])}\">{esc(row['status'])}</span>{('<div class=\"error\">' + errors + '</div>') if errors else ''}</td>"
            f"<td><code>{esc(row['config_path'])}</code></td>"
            "</tr>"
        )

    config_cards = []
    for target, cfg in configs.items():
        config_cards.append(
            "<section class=\"card\">"
            f"<h2>{esc(target)}</h2>"
            f"<p><b>Config:</b> <code>{esc(manifest['targets'][target]['config_path'])}</code></p>"
            f"<p><b>Base features:</b> {badges(cfg['base_features'], 'atom')}</p>"
            f"<p><b>Experiment feature atoms:</b> {len(cfg['experiment_features'])}</p>"
            f"<p><b>Coalitions:</b> {len(cfg['coalitions'])}</p>"
            "</section>"
        )

    counts = " ".join(
        f"<span class=\"pill\">{esc(target)}: {counts['valid']} valid / {counts['invalid']} invalid</span>"
        for target, counts in sorted(target_counts.items())
    )
    body = "\n".join(table_rows)
    cards = "\n".join(config_cards)
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(manifest['study_id'])} preview</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2933; background: #f7f8fa; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 0 0 10px; font-size: 18px; }}
    .subtle, .muted {{ color: #687382; font-size: 12px; }}
    .summary {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 16px 0; }}
    .pill {{ background: #e8eef7; border: 1px solid #ccd8ea; border-radius: 999px; padding: 5px 10px; font-size: 13px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; margin: 16px 0 22px; }}
    .card {{ background: white; border: 1px solid #d9dee7; border-radius: 8px; padding: 14px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d9dee7; }}
    th, td {{ border-bottom: 1px solid #e4e8ef; padding: 9px 10px; vertical-align: top; text-align: left; }}
    th {{ position: sticky; top: 0; background: #eef3f8; z-index: 1; font-size: 12px; text-transform: uppercase; letter-spacing: 0.03em; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
    .badge, .atom {{ display: inline-block; border-radius: 6px; padding: 2px 6px; margin: 1px 2px 2px 0; font-size: 12px; }}
    .badge {{ background: #edf7ee; border: 1px solid #c7e4cc; }}
    .atom {{ background: #f4f0ff; border: 1px solid #ddd0ff; }}
    .status {{ display: inline-block; border-radius: 6px; padding: 2px 7px; font-size: 12px; font-weight: 600; }}
    .valid {{ background: #e6f6ec; color: #146c2e; }}
    .invalid {{ background: #fdebec; color: #9c1c24; }}
    .error {{ margin-top: 4px; color: #9c1c24; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>{esc(manifest['study_id'])}</h1>
  <div class="subtle">{esc(manifest.get('description', ''))}</div>
  <div class="summary">
    <span class="pill">rows: {len(rows)}</span>
    <span class="pill">valid: {valid}</span>
    <span class="pill">invalid: {invalid}</span>
    {counts}
  </div>
  <div class="cards">{cards}</div>
  <table>
    <thead>
      <tr>
        <th>Target</th>
        <th>Coalition label</th>
        <th>Selected blocks</th>
        <th>Effective blocks</th>
        <th>Feature atoms</th>
        <th>Status</th>
        <th>Config path</th>
      </tr>
    </thead>
    <tbody>
      {body}
    </tbody>
  </table>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_doc, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--preview-html", type=Path, default=None)
    parser.add_argument("--write-configs", action="store_true", help="Write runnable config JSON files.")
    parser.add_argument("--allow-invalid-preview", action="store_true", help="Do not fail if preview rows are invalid.")
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    validate_manifest(manifest)
    generated_rows = expand_generation(manifest)

    configs: dict[str, dict[str, Any]] = {}
    preview_rows: list[dict[str, Any]] = []
    for target_name, target in manifest["targets"].items():
        config, rows = build_target(manifest, target_name, target, generated_rows)
        configs[target_name] = config
        preview_rows.extend(rows)

    invalid_rows = [row for row in preview_rows if row["status"] != "valid"]
    if invalid_rows and not args.allow_invalid_preview:
        for row in invalid_rows:
            print(f"INVALID {row['target']} {row['label']}: {'; '.join(row['errors'])}", file=sys.stderr)
        return 2

    preview_path = args.preview_html or Path(manifest["preview_path"])
    if not preview_path.is_absolute():
        preview_path = ROOT / preview_path
    write_preview(preview_path, manifest, configs, preview_rows)

    if args.write_configs:
        if invalid_rows:
            raise ManifestError("Refusing to write configs while invalid preview rows exist")
        for target_name, config in configs.items():
            path = ROOT / manifest["targets"][target_name]["config_path"]
            write_json(path, config)

    print(f"manifest: {args.manifest}")
    print(f"preview: {preview_path}")
    print(f"targets: {', '.join(configs)}")
    print(f"coalitions_per_target: {len(generated_rows)}")
    print(f"invalid_rows: {len(invalid_rows)}")
    if args.write_configs:
        for target_name in configs:
            print(f"wrote_config: {ROOT / manifest['targets'][target_name]['config_path']}")
    else:
        print("configs: preview only; pass --write-configs to emit runnable JSON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
