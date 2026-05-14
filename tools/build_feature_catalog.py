#!/usr/bin/env python3
"""Build the study-layer feature catalog from existing feature JSON specs."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.feature_registry import compute_feature_id  # noqa: E402


VALID_FAMILIES = {
    "response_mode",
    "reasoning_behavior",
    "rule_or_constraint",
    "input_context_builder",
    "demonstration_policy",
}
VALID_SURFACES = {
    "system_instruction",
    "task_or_user_instruction",
    "input_transform",
    "context_field",
    "output_contract",
    "auxiliary_output_field",
    "parser_or_scorer_contract",
}
VALID_COSTS = {
    "prompt_only",
    "parser_change",
    "scorer_change",
    "input_builder_change",
    "runtime_execution",
    "external_model_or_tool",
}

RESPONSE_LABEL_MAP = {
    "profile.dp": "response.direct_final_only",
    "reasoning.direct": "response.direct_final_only",
    "profile.tcot": "response.visible_cot",
    "reasoning.text_chain_of_thought": "response.visible_cot",
    "reasoning.chain_of_thought": "response.visible_cot",
    "profile.scot": "response.visible_structured_trace",
    "reasoning.structured_chain_of_thought": "response.visible_structured_trace",
    "profile.pot": "response.program_of_thought.python",
    "reasoning.program_of_thought": "response.program_of_thought.python",
}

PREFIX_CONCEPT_MAP = {
    "ag_rules": ("reasoning.aggregation_guidance", "reasoning_behavior"),
    "cp_rules": ("rule.construction_constraint", "rule_or_constraint"),
    "dh_": ("rule.decision_heuristic", "rule_or_constraint"),
    "ech_": ("rule.edge_case_handling", "rule_or_constraint"),
    "es_": ("input_context.evidence_summary", "input_context_builder"),
    "fmt_": ("rule.output_contract_control", "rule_or_constraint"),
    "is_rules": ("rule.idiom_shortcut", "rule_or_constraint"),
    "mdk_": ("input_context.domain_knowledge", "input_context_builder"),
    "of_rules": ("rule.output_contract_control", "rule_or_constraint"),
    "pp_rules": ("rule.prompt_policy", "rule_or_constraint"),
    "qd_": ("reasoning.decomposition", "reasoning_behavior"),
    "rsp_": ("rule.output_contract_control", "rule_or_constraint"),
    "rt_rules": ("reasoning.trace_guidance", "reasoning_behavior"),
    "sr_": ("reasoning.staged_reasoning", "reasoning_behavior"),
    "vbo_": ("reasoning.verify_before_output", "reasoning_behavior"),
    "vi_rules": ("reasoning.verification_guidance", "reasoning_behavior"),
}


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def iter_feature_specs(roots: list[Path]) -> list[tuple[Path, dict[str, Any], str]]:
    specs: list[tuple[Path, dict[str, Any], str]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*/*.json")):
            try:
                spec = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # pragma: no cover - defensive report path
                spec = {"_load_error": str(exc)}
            specs.append((path, spec, root.name))
    return specs


def normalized_labels(raw_labels: Any) -> list[str]:
    if raw_labels is None:
        return []
    if not isinstance(raw_labels, list):
        return []
    labels: list[str] = []
    for item in raw_labels:
        label = None
        if isinstance(item, str):
            label = item
        elif isinstance(item, dict):
            label = item.get("label") or item.get("label_id") or item.get("id")
        if isinstance(label, str) and label.strip():
            labels.append(label.strip())
    return labels


def compute_id(spec: dict[str, Any]) -> str:
    edits = spec.get("primitive_edits")
    if isinstance(edits, list) and edits:
        primitive_edits = deepcopy(edits)
        target = spec.get("target_module")
        if target:
            for edit in primitive_edits:
                if isinstance(edit, dict):
                    edit.setdefault("target_module", target)
        return compute_feature_id(primitive_edits)
    existing = spec.get("feature_id")
    return str(existing) if existing else "unknown"


def primitive_text(spec: dict[str, Any]) -> str:
    parts: list[str] = []
    for edit in spec.get("primitive_edits") or []:
        if not isinstance(edit, dict):
            continue
        parts.append(str(edit.get("func_type", "")))
        params = edit.get("params") or {}
        if isinstance(params, dict):
            parts.append(str(params.get("parent_id", "")))
            parts.append(str(params.get("node_type", "")))
            parts.append(str(params.get("content", "")))
            parts.append(str(params.get("field", "")))
            parts.append(str(params.get("format", "")))
    return " ".join(parts).lower()


def concept_family_from_concept(concept_id: str) -> str:
    if concept_id.startswith("response."):
        return "response_mode"
    if concept_id.startswith("reasoning."):
        return "reasoning_behavior"
    if concept_id.startswith("input_context."):
        return "input_context_builder"
    if concept_id.startswith("demonstration."):
        return "demonstration_policy"
    if concept_id.startswith("rule.") or concept_id.startswith("structure."):
        return "rule_or_constraint"
    return "unknown"


def infer_concept(
    canonical_id: str,
    spec: dict[str, Any],
    labels: list[str],
) -> tuple[str, str, str]:
    """Return (concept_id, semantic_family, confidence)."""
    label_set = set(labels)
    lower_id = canonical_id.lower()
    text = " ".join(
        [
            lower_id,
            str(spec.get("theme", "")).lower(),
            str(spec.get("tier", "")).lower(),
            str(spec.get("rationale", "")).lower(),
            " ".join(labels).lower(),
            primitive_text(spec),
        ]
    )

    for label, concept in RESPONSE_LABEL_MAP.items():
        if label in label_set:
            return concept, "response_mode", "high"

    for label in labels:
        if label.startswith("response."):
            return label, "response_mode", "high"
        if label.startswith("reasoning.") and label not in RESPONSE_LABEL_MAP:
            return label, "reasoning_behavior", "high"
        if label.startswith("input_context."):
            return label, "input_context_builder", "high"
        if label.startswith("context."):
            concept = label.replace("context.", "input_context.", 1)
            return concept, "input_context_builder", "high"
        if label.startswith("demonstration."):
            return label, "demonstration_policy", "high"
        if label.startswith("rule.") or label.startswith("constraint."):
            concept = label.replace("constraint.", "rule.", 1)
            return concept, "rule_or_constraint", "high"

    if lower_id.startswith("_section_"):
        return f"structure.section.{slug(lower_id.removeprefix('_section_'))}", "rule_or_constraint", "low"

    for prefix, (concept, family) in PREFIX_CONCEPT_MAP.items():
        if lower_id.startswith(prefix):
            return concept, family, "medium"

    if any(token in text for token in ["facet_dp", "_dp_", "direct answer", "answer only", "final only"]):
        return "response.direct_final_only", "response_mode", "medium"
    if any(token in text for token in ["facet_tcot", "_tcot_", "enable_cot", "chain of thought"]):
        return "response.visible_cot", "response_mode", "medium"
    if any(token in text for token in ["facet_scot", "_scot_", "structured cot", "structured chain"]):
        return "response.visible_structured_trace", "response_mode", "medium"
    if any(token in text for token in ["facet_pot", "_pot_", "python", "pandas", "program of thought"]):
        return "response.program_of_thought.python", "response_mode", "medium"
    if any(token in text for token in ["sql repair", "repair sql", "repair_target"]):
        return "response.sql_repair", "response_mode", "medium"
    if any(token in lower_id for token in ["sql_program", "sql_response_mode", "enable_sql"]):
        return "response.sql_program", "response_mode", "medium"

    reasoning_patterns = [
        ("filter_then_extract", "reasoning.evidence_localization"),
        ("filter-then-extract", "reasoning.evidence_localization"),
        ("evidence_localization", "reasoning.evidence_localization"),
        ("extract_then_compute", "reasoning.extract_then_compute"),
        ("extract-then-compute", "reasoning.extract_then_compute"),
        ("decompose", "reasoning.decomposition"),
        ("decomposition", "reasoning.decomposition"),
        ("subquestion", "reasoning.decomposition"),
        ("candidate_enumeration", "reasoning.candidate_enumeration"),
        ("enumerate", "reasoning.candidate_enumeration"),
        ("verify", "reasoning.verify_before_output"),
        ("verification", "reasoning.verify_before_output"),
        ("rewrite", "reasoning.query_rewriting"),
    ]
    for token, concept in reasoning_patterns:
        if token in text:
            return concept, "reasoning_behavior", "medium"

    context_patterns = [
        ("context_builder", "input_context.builder"),
        ("input_builder", "input_context.builder"),
        ("type_annotation", "input_context.type_annotation"),
        ("table_context", "input_context.table_or_schema_representation"),
        ("schema_context", "input_context.schema_representation"),
        ("schema_use", "input_context.schema_use_guidance"),
        ("schema", "input_context.schema_representation"),
        ("column", "input_context.table_or_schema_representation"),
        ("table", "input_context.table_or_schema_representation"),
        ("serialization", "input_context.serialization"),
        ("sample", "input_context.value_enrichment"),
        ("evidence", "input_context.evidence"),
        ("prune", "input_context.pruning"),
        ("clean", "input_context.cleaning"),
        ("missing value", "input_context.cleaning"),
    ]
    for token, concept in context_patterns:
        if token in lower_id:
            return concept, "input_context_builder", "medium"

    if any(token in text for token in ["few shot", "few-shot", "few_shot", "worked example", "demo"]):
        return "demonstration.few_shot", "demonstration_policy", "medium"

    if any(token in text for token in ["format", "output", "answer", "contract", "json", "label", "strict"]):
        return "rule.output_contract_control", "rule_or_constraint", "medium"
    if any(token in text for token in ["must", "do not", "never", "always", "constraint", "rule"]):
        return "rule.general_instruction", "rule_or_constraint", "low"

    return f"unknown.{slug(canonical_id) or 'feature'}", "unknown", "low"


def infer_surface(spec: dict[str, Any], concept_id: str) -> str:
    scope = spec.get("scope") if isinstance(spec.get("scope"), dict) else {}
    surface = scope.get("implementation_surface")
    if surface in VALID_SURFACES:
        return surface

    edits = spec.get("primitive_edits") or []
    func_types = {edit.get("func_type") for edit in edits if isinstance(edit, dict)}
    text = primitive_text(spec)
    if "input_transform" in func_types:
        return "input_transform"
    if "set_table_format" in func_types:
        return "context_field"
    if "set_format" in func_types:
        return "parser_or_scorer_contract" if "python" in text or "sql" in text else "output_contract"
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        params = edit.get("params") or {}
        if not isinstance(params, dict):
            continue
        node_type = str(params.get("node_type", "")).lower()
        parent_id = str(params.get("parent_id", "")).lower()
        content = str(params.get("content", "")).lower()
        if node_type == "output_field":
            if "python" in content or "sql" in content or "code" in content:
                return "parser_or_scorer_contract"
            return "auxiliary_output_field"
        if "format" in parent_id:
            return "output_contract"
        if node_type == "section":
            return "system_instruction"
    return "task_or_user_instruction"


def infer_cost(spec: dict[str, Any], concept_id: str, surface: str) -> str:
    scope = spec.get("scope") if isinstance(spec.get("scope"), dict) else {}
    cost = scope.get("cost_dependency")
    if cost in VALID_COSTS:
        return cost
    text = " ".join(
        [
            concept_id,
            " ".join(normalized_labels(spec.get("semantic_labels"))),
            primitive_text(spec),
        ]
    ).lower()
    if concept_id in {"response.program_of_thought.python", "response.sql_program", "response.sql_repair"}:
        return "runtime_execution"
    if any(token in text for token in ["python", "pandas", "execute", "runtime"]):
        return "runtime_execution"
    if surface == "parser_or_scorer_contract":
        return "parser_change"
    if surface in {"input_transform", "context_field"}:
        return "input_builder_change"
    if surface in {"output_contract", "auxiliary_output_field"}:
        return "parser_change"
    return "prompt_only"


def build_catalog(roots: list[Path]) -> tuple[list[dict[str, Any]], Counter[str]]:
    specs = iter_feature_specs(roots)
    key_counts: Counter[str] = Counter()
    for path, spec, _source_root in specs:
        task = str(spec.get("task") or path.parent.name)
        canonical_id = str(spec.get("canonical_id") or spec.get("feature_id") or path.stem)
        key_counts[f"{task}::{canonical_id}"] += 1

    rows: list[dict[str, Any]] = []
    for path, spec, source_root in specs:
        task = str(spec.get("task") or path.parent.name)
        canonical_id = str(spec.get("canonical_id") or spec.get("feature_id") or path.stem)
        base_component_id = f"{task}::{canonical_id}"
        component_id = base_component_id
        notes: list[str] = []
        if key_counts[base_component_id] > 1:
            component_id = f"{base_component_id}@{source_root}"
            notes.append("duplicate task::canonical_id; namespace-qualified with source root")

        labels = normalized_labels(spec.get("semantic_labels"))
        concept_id, family, confidence = infer_concept(canonical_id, spec, labels)
        if family == "unknown":
            family = concept_family_from_concept(concept_id)
        surface = infer_surface(spec, concept_id)
        cost = infer_cost(spec, concept_id, surface)

        if concept_id.startswith("unknown."):
            notes.append("concept inferred as unknown")
        if confidence == "low":
            notes.append("low-confidence taxonomy inference")
        if not labels:
            notes.append("no semantic_labels metadata")
        if canonical_id.startswith("_section_"):
            notes.append("structural section component")

        rows.append(
            {
                "component_id": component_id,
                "feature_id": compute_id(spec),
                "canonical_id": canonical_id,
                "task": task,
                "source_path": rel(path),
                "concept_id": concept_id,
                "concept_confidence": confidence,
                "semantic_family": family,
                "implementation_surface": surface,
                "cost_dependency": cost,
                "requires": spec.get("requires") or [],
                "conflicts_with": spec.get("conflicts_with") or [],
                "semantic_labels_raw": spec.get("semantic_labels") or [],
                "notes": "; ".join(notes),
            }
        )

    rows.sort(key=lambda row: (row["task"], row["concept_id"], row["canonical_id"], row["source_path"]))
    return rows, key_counts


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def table(counter: Counter[str], limit: int | None = None) -> str:
    items = counter.most_common(limit)
    if not items:
        return "- none\n"
    width = max(len(k) for k, _v in items)
    return "\n".join(f"- `{k}`{' ' * (width - len(k))} : {v}" for k, v in items) + "\n"


def write_report(rows: list[dict[str, Any]], key_counts: Counter[str], path: Path) -> None:
    by_task = Counter(row["task"] for row in rows)
    by_concept = Counter(row["concept_id"] for row in rows)
    by_family = Counter(row["semantic_family"] for row in rows)
    by_surface = Counter(row["implementation_surface"] for row in rows)
    by_cost = Counter(row["cost_dependency"] for row in rows)
    by_confidence = Counter(row["concept_confidence"] for row in rows)
    unknown_rows = [row for row in rows if row["concept_id"].startswith("unknown.") or row["semantic_family"] == "unknown"]
    low_conf_rows = [row for row in rows if row["concept_confidence"] == "low"]
    duplicate_keys = [key for key, count in key_counts.items() if count > 1]

    concept_tasks: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        concept_tasks[row["concept_id"]].add(row["task"])
    cross_task = {
        concept: len(tasks)
        for concept, tasks in concept_tasks.items()
        if len(tasks) > 1 and not concept.startswith("unknown.")
    }

    lines = [
        "# Feature Catalog v0 Report",
        "",
        "Generated from existing feature JSON specs only. This round does not add new features or modify the experiment runner.",
        "",
        "## Inventory",
        "",
        f"- Rows: {len(rows)}",
        f"- Tasks: {len(by_task)}",
        f"- Concepts: {len(by_concept)}",
        f"- Cross-task concepts: {len(cross_task)}",
        f"- Duplicate `task::canonical_id` keys namespace-qualified: {len(duplicate_keys)}",
        f"- Unknown concept rows: {len(unknown_rows)}",
        f"- Low-confidence rows: {len(low_conf_rows)}",
        "",
        "## Tasks",
        "",
        table(by_task),
        "## Semantic Families",
        "",
        table(by_family),
        "## Implementation Surfaces",
        "",
        table(by_surface),
        "## Cost Dependencies",
        "",
        table(by_cost),
        "## Concept Confidence",
        "",
        table(by_confidence),
        "## Top Concepts",
        "",
        table(by_concept, limit=40),
        "## Cross-Task Transfer Candidates",
        "",
    ]
    if cross_task:
        for concept, task_count in sorted(cross_task.items(), key=lambda item: (-item[1], item[0]))[:60]:
            tasks = ", ".join(sorted(concept_tasks[concept]))
            lines.append(f"- `{concept}`: {task_count} tasks ({tasks})")
    else:
        lines.append("- none")
    lines += [
        "",
        "## Rows Needing Review",
        "",
        "These rows are usable for inventory, but should not be treated as clean study treatments until reviewed.",
        "",
    ]
    review_rows = sorted(
        low_conf_rows + [row for row in unknown_rows if row not in low_conf_rows],
        key=lambda row: (row["task"], row["concept_id"], row["canonical_id"]),
    )
    if review_rows:
        for row in review_rows[:120]:
            lines.append(
                f"- `{row['component_id']}` -> `{row['concept_id']}` "
                f"({row['semantic_family']}, {row['implementation_surface']}, {row['cost_dependency']}): {row['notes']}"
            )
        if len(review_rows) > 120:
            lines.append(f"- ... {len(review_rows) - 120} additional review rows omitted from report")
    else:
        lines.append("- none")

    lines += [
        "",
        "## Interpretation",
        "",
        "- `feature_id` remains the content/provenance key.",
        "- `component_id` is the runnable row key. Duplicate `task::canonical_id` rows from active and legacy roots are suffixed with `@features` or `@features_legacy`.",
        "- `concept_id` is the study-facing transfer label. It is intentionally coarser than concrete implementation.",
        "- Response modes are represented as features, but they can still serve as baselines in later EffectSpec rows.",
        "- Rows with low confidence or unknown concepts are the right next review target before controlled experiments.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_html(rows: list[dict[str, Any]], path: Path) -> None:
    data = json.dumps(rows, sort_keys=True).replace("</", "<\\/")
    total = len(rows)
    concepts = len({row["concept_id"] for row in rows})
    tasks = len({row["task"] for row in rows})
    unknown = sum(1 for row in rows if row["concept_id"].startswith("unknown."))
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Feature Catalog v0</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    body {{ margin: 0; background: #f6f7f9; color: #1e2732; }}
    header {{ padding: 24px 28px 18px; background: #fff; border-bottom: 1px solid #dbe1ea; }}
    h1 {{ margin: 0 0 10px; font-size: 24px; font-weight: 700; letter-spacing: 0; }}
    .summary {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .metric {{ border: 1px solid #dbe1ea; background: #f9fafc; border-radius: 6px; padding: 8px 10px; min-width: 116px; }}
    .metric b {{ display: block; font-size: 18px; }}
    main {{ padding: 18px 28px 32px; }}
    .filters {{ display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 10px; margin-bottom: 14px; }}
    input, select {{ width: 100%; box-sizing: border-box; border: 1px solid #cbd4df; border-radius: 6px; padding: 8px 9px; background: #fff; color: #1e2732; }}
    .count {{ margin: 8px 0 12px; color: #526172; font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #dbe1ea; }}
    th, td {{ border-bottom: 1px solid #e5eaf0; padding: 8px 9px; text-align: left; vertical-align: top; font-size: 12px; }}
    th {{ position: sticky; top: 0; background: #eef2f6; z-index: 1; font-size: 11px; text-transform: uppercase; color: #465568; }}
    tr:hover td {{ background: #f8fbff; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 11px; }}
    details {{ max-width: 360px; }}
    summary {{ cursor: pointer; color: #165a9f; }}
    .pill {{ display: inline-block; border-radius: 999px; padding: 2px 7px; background: #edf2f7; border: 1px solid #d8e0e8; white-space: nowrap; }}
    .low {{ background: #fff7ed; border-color: #fed7aa; }}
    .high {{ background: #ecfdf5; border-color: #bbf7d0; }}
    .medium {{ background: #eff6ff; border-color: #bfdbfe; }}
  </style>
</head>
<body>
  <header>
    <h1>Feature Catalog v0</h1>
    <div class="summary">
      <div class="metric"><b>{total}</b>rows</div>
      <div class="metric"><b>{tasks}</b>tasks</div>
      <div class="metric"><b>{concepts}</b>concepts</div>
      <div class="metric"><b>{unknown}</b>unknown concepts</div>
    </div>
  </header>
  <main>
    <div class="filters">
      <input id="q" placeholder="Search component, concept, notes">
      <select id="task"></select>
      <select id="concept"></select>
      <select id="family"></select>
      <select id="surface"></select>
      <select id="cost"></select>
    </div>
    <div class="count" id="count"></div>
    <table>
      <thead>
        <tr>
          <th>Component</th>
          <th>Concept</th>
          <th>Family</th>
          <th>Surface</th>
          <th>Cost</th>
          <th>Confidence</th>
          <th>Evidence</th>
        </tr>
      </thead>
      <tbody id="rows"></tbody>
    </table>
  </main>
  <script>
    const rows = {data};
    const fields = ["task", "concept", "family", "surface", "cost"];
    const ids = {{ task: "task", concept: "concept_id", family: "semantic_family", surface: "implementation_surface", cost: "cost_dependency" }};
    function options(id, values) {{
      const el = document.getElementById(id);
      el.innerHTML = `<option value="">All ${{id}}</option>` + values.map(v => `<option>${{escapeHtml(v)}}</option>`).join("");
    }}
    for (const id of fields) {{
      const key = ids[id];
      options(id, [...new Set(rows.map(r => r[key]))].sort());
    }}
    function escapeHtml(s) {{
      return String(s).replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[c]));
    }}
    function render() {{
      const q = document.getElementById("q").value.toLowerCase();
      const filters = Object.fromEntries(fields.map(id => [ids[id], document.getElementById(id).value]));
      const filtered = rows.filter(row => {{
        for (const [key, value] of Object.entries(filters)) {{
          if (value && row[key] !== value) return false;
        }}
        if (!q) return true;
        return JSON.stringify(row).toLowerCase().includes(q);
      }});
      document.getElementById("count").textContent = `${{filtered.length}} / ${{rows.length}} rows`;
      document.getElementById("rows").innerHTML = filtered.map(row => `
        <tr>
          <td><code>${{escapeHtml(row.component_id)}}</code><br><code>${{escapeHtml(row.feature_id)}}</code><br>${{escapeHtml(row.source_path)}}</td>
          <td><code>${{escapeHtml(row.concept_id)}}</code></td>
          <td><span class="pill">${{escapeHtml(row.semantic_family)}}</span></td>
          <td><span class="pill">${{escapeHtml(row.implementation_surface)}}</span></td>
          <td><span class="pill">${{escapeHtml(row.cost_dependency)}}</span></td>
          <td><span class="pill ${{escapeHtml(row.concept_confidence)}}">${{escapeHtml(row.concept_confidence)}}</span></td>
          <td>
            <details>
              <summary>details</summary>
              <div><b>canonical:</b> <code>${{escapeHtml(row.canonical_id)}}</code></div>
              <div><b>requires:</b> ${{escapeHtml(JSON.stringify(row.requires))}}</div>
              <div><b>conflicts:</b> ${{escapeHtml(JSON.stringify(row.conflicts_with))}}</div>
              <div><b>labels:</b> ${{escapeHtml(JSON.stringify(row.semantic_labels_raw))}}</div>
              <div><b>notes:</b> ${{escapeHtml(row.notes || "")}}</div>
            </details>
          </td>
        </tr>
      `).join("");
    }}
    for (const id of ["q", ...fields]) document.getElementById(id).addEventListener("input", render);
    render();
  </script>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="*", default=["features", "features_legacy"])
    parser.add_argument("--out-dir", default="study_layer")
    args = parser.parse_args()

    roots = [ROOT / root for root in args.roots]
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, key_counts = build_catalog(roots)
    jsonl_path = out_dir / "feature_catalog_v0.jsonl"
    report_path = out_dir / "feature_catalog_report_v0.md"
    html_path = out_dir / "feature_catalog_v0.html"
    write_jsonl(rows, jsonl_path)
    write_report(rows, key_counts, report_path)
    write_html(rows, html_path)

    print(f"wrote {rel(jsonl_path)} ({len(rows)} rows)")
    print(f"wrote {rel(report_path)}")
    print(f"wrote {rel(html_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
