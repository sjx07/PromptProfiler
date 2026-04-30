"""Sequential Question Answering task — parse answers, score via denotation accuracy."""
from __future__ import annotations

import json
import re
from typing import Dict, List

from task import BaseTask


class SequentialQA(BaseTask):
    name = "sequential_qa"
    scorer = "denotation_acc"
    default_input_fields: Dict[str, str] = {
        "table": "The table data to answer the question about",
        "conversation_history": "Previous questions and answers in this conversation",
        "question": "The current question to answer using the table",
    }
    default_output_fields: Dict[str, str] = {
        "answer": "The answer extracted from the table",
    }

    def _gold_output(self, meta: dict, raw: dict) -> dict:
        answer_text = raw.get("answer_text", meta.get("gold_answer", []))
        if isinstance(answer_text, str):
            answer_text = [answer_text]
        return {"answer": ", ".join(str(v) for v in answer_text)}

    def build_record(self, query: dict, meta: dict, raw: dict) -> dict:
        from tasks.sqa.loaders import table_to_markdown

        table = raw.get("table", {})
        question = raw.get("question", query.get("content", ""))
        history = raw.get("history", [])

        table_md = table_to_markdown(table)

        history_str = ""
        if history:
            parts = []
            for turn in history:
                q = turn.get("question", "")
                a = turn.get("answer", [])
                a_str = ", ".join(str(v) for v in a) if isinstance(a, list) else str(a)
                parts.append(f"Q: {q}\nA: {a_str}")
            history_str = "\n".join(parts)

        return {
            "table": table_md,
            "conversation_history": history_str,
            "question": question,
        }

    def parse_response(self, raw_response: str) -> str:
        if self._prompt_state is not None:
            parsed = self._prompt_state.parse_output(raw_response)
            if parsed:
                val = parsed.get("answer", "")
                if isinstance(val, list):
                    answer = ", ".join(str(v) for v in val).strip()
                else:
                    answer = str(val).strip()
                if answer:
                    return answer
        return _extract_answer(raw_response)

    def score(self, prediction: str, query_meta: dict) -> tuple[float, dict]:
        if isinstance(query_meta, str):
            query_meta = json.loads(query_meta)

        raw = query_meta.get("_raw", {})
        gold_answer = raw.get("answer_text", query_meta.get("gold_answer", []))
        if isinstance(gold_answer, str):
            gold_answer = [gold_answer]

        gold_norm = {_normalize(str(v)) for v in gold_answer}
        pred_values = _parse_prediction(prediction)
        pred_norm = {_normalize(v) for v in pred_values}

        match = pred_norm == gold_norm
        score_val = 1.0 if match else 0.0

        return score_val, {
            "status": "ok",
            "prediction": prediction,
            "pred_normalized": sorted(pred_norm),
            "gold": gold_answer,
            "gold_normalized": sorted(gold_norm),
        }


# ── helpers ──────────────────────────────────────────────────────────


def _extract_answer(text: str) -> str:
    """Extract answer from LLM response.

    Tries JSON, then a labeled `answer:` pattern, then a last-line
    heuristic with common prefixes.
    """
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            for key in ("answer", "result", "value", "response"):
                if key in parsed:
                    val = parsed[key]
                    if isinstance(val, list):
                        return ", ".join(str(v) for v in val)
                    return str(val).strip()
        if isinstance(parsed, list):
            return ", ".join(str(v) for v in parsed)
    except (json.JSONDecodeError, TypeError):
        pass

    m = re.search(
        r"(?:answer|result|value)\s*[:\-]\s*(.+?)(?:\n|$)",
        text,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).strip().rstrip(".")

    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if lines:
        last = lines[-1]
        for prefix in ["The answer is", "Answer:", "Therefore,", "So,", "Thus,"]:
            if last.lower().startswith(prefix.lower()):
                return last[len(prefix):].strip().rstrip(".")
        if len(last) < 200:
            return last

    return text


def _normalize(v: str) -> str:
    """Normalize an answer value for set-comparison."""
    v = v.strip().lower()
    v = v.strip(".,;:!?\"'")
    v = re.sub(r"\s+", " ", v)
    v = v.replace("\xa0", " ")
    v = v.replace(",", "")
    try:
        num = float(v)
        if num == int(num):
            v = str(int(num))
        else:
            v = f"{num:.4g}"
    except ValueError:
        pass
    return v


def _parse_prediction(prediction: str) -> List[str]:
    """Parse a prediction string into a list of values."""
    prediction = prediction.strip()

    try:
        parsed = json.loads(prediction)
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed]
    except (json.JSONDecodeError, TypeError):
        pass

    if "|" in prediction:
        parts = prediction.split("|")
    elif "," in prediction and "\n" not in prediction:
        parts = prediction.split(",")
    elif "\n" in prediction:
        parts = prediction.split("\n")
    else:
        parts = [prediction]

    return [p.strip() for p in parts if p.strip()]
