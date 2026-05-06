"""TableBench task — single-prompt textual answer over (FactChecking | NumericalReasoning | DataAnalysis).

Out of scope: Visualization (chart-code), filtered at loader.
"""
from __future__ import annotations

import json
from dataclasses import replace
from typing import Dict, Iterable

from prompt.prompt_state import PromptState
from prompt.rules import RuleItem, RuleSection, RuleTree
from task import BaseTask
from tasks.tablebench.official_parser import (
    parse_general_code_then_exec,
)
from tasks.tablebench.official_scorer import normalize_answer, score_one


_OUTPUT_TO_TABLE_FORMAT = {
    "json":       "json_records",
    "markdown":   "markdown",
    "plain":      "markdown",
    "yaml":       "markdown",
    "code_block": "json_records",
}

_DATAANALYSIS_ANSWER_CONTRACTS = {
    "ImpactAnalysis": (
        'Ensure the answer is in the "AnswerName1, AnswerName2..." form, no other form.',
        "Ensure the answer is a entity name or a impact description(No clear impact, Negtive impact or Positive impact), as short as possible, without any explanation.",
    ),
    "AnomalyDetection": (
        "Answer should point out the abnormal data with total number then explain why for each anomaly as short as possible.",
        'If no anomaly is detected, the answer should be "No anomalies are detected in the table."',
        "Examples:",
        "The three anomalies are row 5 with Tom having an unusually high score 101 in the Math column, row 7 with an unusually low score 3 in the English column, and row 9 with an unusually high score 200 in the Science column.",
        "No anomalies are detected in the table.",
    ),
    "CorrelationAnalysis": (
        'Ensure the answer is in the "CorrelationRelation, CorrelationCoefficient." form, no other form.',
        'Ensure that: the correlation coefficient should be a float number with two decimal places; the correlation relation can only be "No correlation" with the correlation coefficient between -0.3 to +0.3, "Weak positive correlation" with the correlation coefficient between +0.3 to +0.7, "Weak negative correlation" with the correlation coefficient between -0.3 to -0.7, "Strong positive correlation" with the correlation coefficient between +0.7 to +1, or "Strong negative correlation" with the correlation coefficient between -0.7 to -1.',
        "Examples:",
        "No correlation, 0.15",
        "Strong positive correlation, 0.82",
    ),
    "TrendForecasting": (
        'Ensure the answer is in the "AnswerName1, AnswerName2..." form, no other form.',
        'Ensure the "AnswerName" is a number, entity name or a trend description(No clear trend, Increasing trend or Decreasing trend), as short as possible, without any explanation.',
        "Examples:",
        "1, 2, 3",
        "Increasing trend",
        "Increasing trend, 13.5",
    ),
    "CausalAnalysis": (
        'Ensure the answer is in the "Answer" form, no other form.',
        "Ensure answer should give the conclusion then provide a brief explanation of the causal analysis results as concise as possible.",
        "Examples:",
        "Yes, Higher interest positively influences deposit balances change (correlation coefficient of 0.89).",
        "No, Analysis reveals a negligible inverse correlation (0.21), suggesting the gdp does not causally influence the population.",
        "The water level of a river exhibits a stronger causal relationship with rainfall (0.82) compared to snowfall (0.32).",
    ),
    "DescriptiveAnalysis": (
        "The table presents the shooting accuracy of 8 different bullet types (including .308 Winchester and .300 Winchester Magnum) at 100 meters and 300 meters, measured in millimeters and Minutes of Angle (MOA) for dispersion. The data indicates that .300 Winchester Magnum bullets exhibit higher precision at 300 meters, with smaller dispersion ranges.",
    ),
    "StatisticalAnalysis": (
        'Ensure the answer is in the "AnswerName1, AnswerName2..." form, no other form.',
        'Ensure answer format is in the "AnswerName1, AnswerName2..." form, no other form. Ensure the "AnswerName" is a number or entity name, as short as possible, without any explanation.',
    ),
}

_DEFAULT_ANSWER_RULE = (
    "Use `AnswerName1, AnswerName2...`; each AnswerName must be a number "
    "or entity name, as short as possible, with no explanation."
)
_COUNTING_REASONING_CONTRACT = (
    "When the question asks how many items match a condition, enumerate each "
    "table row or entity that satisfies the condition, keep a running count, "
    "and put only the final count in `answer`."
)


class TableBench(BaseTask):
    name = "tablebench"
    scorer = "tb_official_acc"
    _parser_module_path = "tasks.wtq.parsers"   # reuse the "answer" parser
    default_input_fields: Dict[str, str] = {
        "table": "The table data to answer the question about",
        "question": "The question to answer using the table",
    }
    default_output_fields: Dict[str, str] = {
        "answer": "The answer extracted from the table",
    }

    def _gold_output(self, meta: dict, raw: dict) -> dict:
        return {"answer": str(raw.get("answer", meta.get("gold_answer", "")))}

    def build_record(self, query: dict, meta: dict, raw: dict) -> dict:
        from tasks.wtq.table_formats import get_table_formatter

        table_data = raw.get("table", {})
        header = list(table_data.get("header", []))
        rows = [list(r) for r in table_data.get("rows", [])]
        name = table_data.get("name", "")
        question = raw.get("question", query.get("content", ""))

        header, rows = self._apply_transforms(header, rows, question)

        fmt = "markdown"
        if self._prompt_state is not None:
            explicit = self._prompt_state.metadata.get("table_format")
            if explicit:
                fmt = explicit
            else:
                style = self._prompt_state.format_style_name
                fmt = _OUTPUT_TO_TABLE_FORMAT.get(style, "markdown")

        formatter = get_table_formatter(fmt)
        table_str = formatter(header, rows, name)

        if self._pending_stats:
            table_str = self._pending_stats + "\n\n" + table_str

        return {
            "table": table_str,
            "question": question,
        }

    def configure_prompt_state_for_record(
        self,
        prompt_state: PromptState,
        record: dict,
        meta: dict,
        raw: dict,
    ) -> PromptState:
        """Add record-specific answer contracts without mutating bound state."""
        semantic = prompt_state.semantic
        output_fields = dict(semantic.output_fields)
        rule_sections = semantic.rule_sections

        extra_rules: list[str] = []
        qtype = str(raw.get("qtype", meta.get("qtype", "")))
        qsubtype = str(raw.get("qsubtype", meta.get("qsubtype", "")))

        if qtype == "DataAnalysis":
            contract = _DATAANALYSIS_ANSWER_CONTRACTS.get(qsubtype)
            if contract and _uses_official_tablebench_output_contract(output_fields):
                field_desc, rule_texts = _render_answer_contract(contract)
                if "answer" in output_fields:
                    output_fields["answer"] = field_desc
                    extra_rules.append(_answer_field_rule())
                    extra_rules.extend(rule_texts)
                elif "code" in output_fields:
                    output_fields["code"] = _code_field_description(field_desc)
                    extra_rules.append(_code_answer_rule())
                    if field_desc and field_desc not in rule_texts:
                        extra_rules.append(field_desc)
                    extra_rules.extend(rule_texts)
        else:
            if "answer" in output_fields:
                extra_rules.append(_answer_field_rule())
                extra_rules.append(_DEFAULT_ANSWER_RULE)
            elif "code" in output_fields:
                output_fields["code"] = _code_field_description()
                extra_rules.append(_code_answer_rule())
                extra_rules.append(_DEFAULT_ANSWER_RULE)
        if _should_add_counting_contract(qtype, qsubtype, record) and (
            "reasoning" in output_fields or "symbolic_trace" in output_fields
        ):
            extra_rules.append(_COUNTING_REASONING_CONTRACT)

        if not extra_rules and output_fields == semantic.output_fields:
            return prompt_state

        if extra_rules:
            rule_sections = _append_rules_to_format_fix_section(rule_sections, extra_rules)

        rule_tree = _rule_tree_for_sections(rule_sections, semantic.tree)

        return prompt_state.clone(
            semantic=replace(
                semantic,
                output_fields=output_fields,
                tree=rule_tree,
                rule_sections=rule_sections,
            ),
        )

    def parse_response(self, raw_response: str) -> str:
        """Dispatch through the configured FACET output field parser."""
        return super().parse_response(raw_response)

    def score(self, prediction: str, query_meta: dict) -> tuple[float, dict]:
        """Eval-thread scorer. Dispatches purely from the persisted prediction
        string (`__CODE__` prefix → execute) and `query_meta` — no prompt_state
        reads. This survives the eval-thread boundary by construction."""
        if isinstance(query_meta, str):
            query_meta = json.loads(query_meta)

        raw = query_meta.get("_raw", {})
        gold = str(raw.get("answer", query_meta.get("gold_answer", "")))
        qtype = raw.get("qtype", query_meta.get("qtype", "unknown"))
        qsubtype = raw.get("qsubtype", query_meta.get("qsubtype", "unknown"))

        raw_prediction = str(prediction or "").strip()
        pred = raw_prediction
        ecr_1 = None
        executed_python = raw_prediction.startswith("__CODE__")
        if executed_python:
            code = raw_prediction[len("__CODE__"):].strip()
            code_or_response = f"```python\n{code}\n```"
            pred, ecr_1 = parse_general_code_then_exec(
                code_or_response,
                raw.get("table", {}),
            )

        pred = str(pred or "").strip()
        normalized_gold = normalize_answer(gold)
        normalized_pred = normalize_answer(pred)
        score_val, method = score_one(qtype, qsubtype, gold, pred)
        if ecr_1 is not None:
            method = f"python_exec_{method}"

        return score_val, {
            "status": "ok",
            "prediction": pred,
            "normalized_prediction": normalized_pred,
            "raw_prediction": raw_prediction[:500],
            "gold": gold,
            "normalized_gold": normalized_gold,
            "qtype": qtype,
            "qsubtype": qsubtype,
            "method": method,
            "output_mode": "python_exec" if executed_python else "direct",
            "ECR@1": ecr_1,
        }


def _append_rules_to_format_fix_section(
    sections: Iterable[RuleSection],
    extra_rules: list[str],
) -> list[RuleSection]:
    """Add record-specific rules to the existing format_fix section.

    The static official features already define a top-level ``format_fix``
    section. Appending another section with the same title renders as duplicate
    headers in plain/markdown formats and overwrites by key in JSON formats.
    """
    merged: list[RuleSection] = []
    inserted = False
    for section in sections:
        if section.title == "format_fix" and not inserted:
            parent_id = section.node_id
            runtime_items = [
                RuleItem(
                    text=rule,
                    rule_kind="bullet",
                    node_id=f"__tablebench_record_format_fix_{i}",
                    parent_id=parent_id,
                )
                for i, rule in enumerate(extra_rules, start=1)
            ]
            merged.append(replace(section, children=list(section.children) + runtime_items))
            inserted = True
        else:
            merged.append(section)

    if not inserted:
        merged.append(RuleSection(title="format_fix", content="\n".join(extra_rules)))
    return merged


def _rule_tree_for_sections(sections: list[RuleSection], old_tree: RuleTree) -> RuleTree:
    tree = RuleTree(roots=sections, mask=old_tree.mask)
    tree._rebuild_index()
    return tree


def _uses_official_tablebench_output_contract(output_fields: dict[str, str]) -> bool:
    text = "\n".join(str(desc) for desc in output_fields.values())
    return (
        "AnswerName1" in text
        or "prints only the answer value" in text
        or "code" in output_fields
    )


def _render_answer_contract(contract: Iterable[str]) -> tuple[str, list[str]]:
    raw_parts = (contract,) if isinstance(contract, str) else contract
    parts = [str(part).strip() for part in raw_parts if str(part).strip()]
    if not parts:
        return "", []
    return parts[0], parts[1:] or [parts[0]]


def _answer_field_rule() -> str:
    return (
        "Put the final answer directly in the `answer` field with no label, "
        "prefix, or explanation."
    )


def _code_field_description(answer_description: str | None = None) -> str:
    if answer_description:
        return (
            "Executable Python code that analyzes `table.csv`, assigns the "
            f"requested answer value as: {answer_description}, and prints only "
            "that answer value with no label, prefix, or explanation."
        )
    return (
        "Executable Python code that analyzes `table.csv`, assigns the final "
        "answer value to `answer`, and prints only that value with "
        "`print(answer)`."
    )


def _code_answer_rule() -> str:
    return (
        "Set `answer` to the final answer value and print only that value with "
        "`print(answer)`, with no label, prefix, or explanatory text."
    )


def _should_add_counting_contract(qtype: str, qsubtype: str, record: dict) -> bool:
    if qtype != "NumericalReasoning":
        return False
    if qsubtype == "Counting":
        return True

    question = str(record.get("question", "")).strip().lower()
    return (
        question.startswith("how many ")
        or question.startswith("number of ")
        or " count " in f" {question} "
    )
