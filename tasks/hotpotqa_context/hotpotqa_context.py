"""HotpotQA compound QA task using each example's provided context.

This is intentionally not the paper's full Wikipedia/BM25 retrieval setup.
It keeps retrieval local to the HotpotQA example context so prompt behavior is
the variable under test.
"""
from __future__ import annotations

import json
import re
import string
from collections import Counter
from typing import Any, Mapping

from task import CompoundTask, ModuleRuntime, ModuleSpec


class HotpotQAContextTask(CompoundTask):
    """Four-stage multi-hop QA over provided HotpotQA context passages."""

    name = "hotpotqa_context"
    scorer = "hotpotqa_context_f1"
    retrieval_k = 7
    module_specs = {
        "summarize1": ModuleSpec(
            input_fields={
                "question": "The original HotpotQA question.",
                "passages": "First-hop context passages retrieved from the example context.",
            },
            output_fields={
                "reasoning": "Brief reasoning about which retrieved facts are relevant.",
                "summary": "Concise first-hop facts relevant to answering the question.",
            },
        ),
        "create_query_hop2": ModuleSpec(
            input_fields={
                "question": "The original HotpotQA question.",
                "summary_1": "The first-hop summary.",
            },
            output_fields={
                "reasoning": "Brief reasoning about the missing evidence needed for the second hop.",
                "query": "A focused retrieval query for missing second-hop evidence.",
            },
        ),
        "summarize2": ModuleSpec(
            input_fields={
                "question": "The original HotpotQA question.",
                "summary_1": "The first-hop summary.",
                "passages": "Second-hop context passages retrieved from the example context.",
            },
            output_fields={
                "reasoning": "Brief reasoning about how the second-hop passages connect to the question.",
                "summary": "Concise second-hop facts that support the final answer.",
            },
        ),
        "final_answer": ModuleSpec(
            input_fields={
                "question": "The original HotpotQA question.",
                "summary_1": "The first-hop summary.",
                "summary_2": "The second-hop summary.",
            },
            output_fields={
                "reasoning": "Brief reasoning that uses the two summaries to identify the answer.",
                "answer": "Shortest supported answer span.",
            },
        ),
    }

    def run(self, query: dict, runtime: ModuleRuntime) -> str:
        question = query.get("content", "")
        meta = _as_meta(query.get("meta", {}))
        context = _context_from_meta(meta)

        passages1 = retrieve_context_passages(question, context, k=self.retrieval_k)
        system_prompt, user_content = self.build_module_prompt(
            "summarize1",
            {
                "question": question,
                "passages": format_passages(passages1),
            },
        )
        summary1 = runtime.call(
            "summarize1",
            system_prompt,
            user_content,
            parse=parse_summary,
            meta={
                "retrieval_query": question,
                "retrieved_titles": [p["title"] for p in passages1],
            },
        ).parsed_output

        system_prompt, user_content = self.build_module_prompt(
            "create_query_hop2",
            {
                "question": question,
                "summary_1": summary1,
            },
        )
        hop2_query = runtime.call(
            "create_query_hop2",
            system_prompt,
            user_content,
            parse=parse_hop_query,
        ).parsed_output

        retrieval_query = hop2_query or question
        passages2 = retrieve_context_passages(retrieval_query, context, k=self.retrieval_k)
        system_prompt, user_content = self.build_module_prompt(
            "summarize2",
            {
                "question": question,
                "summary_1": summary1,
                "passages": format_passages(passages2),
            },
        )
        summary2 = runtime.call(
            "summarize2",
            system_prompt,
            user_content,
            parse=parse_summary,
            meta={
                "retrieval_query": retrieval_query,
                "retrieved_titles": [p["title"] for p in passages2],
            },
        ).parsed_output

        system_prompt, user_content = self.build_module_prompt(
            "final_answer",
            {
                "question": question,
                "summary_1": summary1,
                "summary_2": summary2,
            },
        )
        return runtime.call(
            "final_answer",
            system_prompt,
            user_content,
            parse=parse_answer,
        ).parsed_output

    def score(self, prediction: str, query_meta: dict) -> tuple[float, dict]:
        meta = _as_meta(query_meta)
        raw = meta.get("_raw", {}) if isinstance(meta.get("_raw", {}), dict) else {}
        answer = _first_text(meta, raw, ["answer", "gold", "target", "reference"])
        if not answer:
            return 0.0, {
                "status": "missing_answer",
                "answer_present": False,
                "exact_match": 0.0,
                "f1": 0.0,
            }

        exact = 1.0 if normalize_answer(prediction) == normalize_answer(answer) else 0.0
        f1 = answer_f1(prediction, answer)
        return f1, {
            "status": "ok",
            "answer_present": True,
            "exact_match": exact,
            "f1": f1,
        }


def retrieve_context_passages(query: str, context: Any, *, k: int = 7) -> list[dict[str, Any]]:
    """Return top-k context passages by deterministic lexical overlap."""
    if k <= 0:
        return []
    passages = _flatten_context(context)
    if not passages:
        return []

    query_tokens = _tokens(query)
    scored = [
        (_passage_score(query_tokens, passage), idx, passage)
        for idx, passage in enumerate(passages)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [passage for _, _, passage in scored[:k]]


def format_passages(passages: list[dict[str, Any]]) -> str:
    if not passages:
        return "No context passages available."
    lines = []
    for idx, passage in enumerate(passages, start=1):
        title = passage.get("title", "")
        text = passage.get("text", "")
        lines.append(f"Passage {idx} ({title}): {text}")
    return "\n\n".join(lines)


def parse_summary(raw_response: str) -> str:
    return _parse_field(raw_response, ["summary", "summary_1", "facts"])


def parse_hop_query(raw_response: str) -> str:
    return _parse_field(raw_response, ["query", "search_query", "hop2_query"])


def parse_answer(raw_response: str) -> str:
    return _parse_field(raw_response, ["answer", "final_answer", "response"])


def normalize_answer(text: str) -> str:
    lowered = str(text).lower()
    no_punc = lowered.translate(str.maketrans("", "", string.punctuation))
    no_articles = re.sub(r"\b(a|an|the)\b", " ", no_punc)
    return " ".join(no_articles.split())


def answer_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _context_from_meta(meta: dict) -> Any:
    raw = meta.get("_raw", {}) if isinstance(meta.get("_raw", {}), dict) else {}
    return (
        meta.get("context")
        or raw.get("context")
        or raw.get("contexts")
        or raw.get("paragraphs")
        or []
    )


def _as_meta(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _flatten_context(context: Any) -> list[dict[str, Any]]:
    if context is None:
        return []
    if isinstance(context, str):
        text = context.strip()
        if not text:
            return []
        if text.startswith("{") or text.startswith("["):
            try:
                return _flatten_context(json.loads(text))
            except json.JSONDecodeError:
                pass
        return [_make_passage("context", text)]
    if isinstance(context, Mapping):
        if "title" in context and "sentences" in context:
            titles = context.get("title")
            sentences = context.get("sentences")
            if isinstance(titles, list) and isinstance(sentences, list):
                return [
                    _make_passage(title, sent)
                    for title, sent in zip(titles, sentences)
                ]
            return [_make_passage(titles, sentences)]
        if "context" in context:
            return _flatten_context(context["context"])
        return [
            _make_passage(title, sentences)
            for title, sentences in context.items()
        ]
    if isinstance(context, (list, tuple)):
        passages = []
        for idx, item in enumerate(context):
            if isinstance(item, Mapping):
                title = item.get("title") or item.get("name") or f"passage_{idx}"
                content = (
                    item.get("sentences")
                    or item.get("text")
                    or item.get("paragraph")
                    or item.get("content")
                    or ""
                )
                passages.append(_make_passage(title, content))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                passages.append(_make_passage(item[0], item[1]))
            else:
                passages.append(_make_passage(f"passage_{idx}", item))
        return [p for p in passages if p["text"]]
    return [_make_passage("context", context)]


def _make_passage(title: Any, content: Any) -> dict[str, Any]:
    sentences = _sentence_list(content)
    text = " ".join(sentence.strip() for sentence in sentences if sentence.strip())
    return {
        "title": str(title or "").strip(),
        "sentences": sentences,
        "text": text,
    }


def _sentence_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        sentences = []
        for item in value:
            sentences.extend(_sentence_list(item))
        return sentences
    return [str(value)]


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "what",
    "which",
    "who",
}


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if token not in _STOPWORDS
    ]


def _passage_score(query_tokens: list[str], passage: dict[str, Any]) -> int:
    if not query_tokens:
        return 0
    query_set = set(query_tokens)
    title_tokens = set(_tokens(passage.get("title", "")))
    passage_tokens = _tokens(f"{passage.get('title', '')} {passage.get('text', '')}")
    counts = Counter(passage_tokens)
    overlap = sum(counts[token] for token in query_set)
    title_bonus = sum(2 for token in query_set if token in title_tokens)
    phrase = " ".join(query_tokens)
    phrase_bonus = 3 if phrase and phrase in " ".join(passage_tokens) else 0
    return overlap + title_bonus + phrase_bonus


def _parse_field(raw_response: str, field_names: list[str]) -> str:
    raw_response = _strip_thinking(raw_response)
    parsed = _parse_json_object(raw_response)
    if isinstance(parsed, dict):
        for field_name in field_names:
            if field_name in parsed and parsed[field_name] not in (None, ""):
                return str(parsed[field_name]).strip()

    labeled = _extract_labeled_block(raw_response, field_names)
    if labeled:
        return labeled
    return raw_response.strip()


def _parse_json_object(text: str) -> Any:
    stripped = _strip_fence(str(text).strip())
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def _strip_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return re.sub(r"<think>.*\Z", "", text, flags=re.DOTALL).strip()


def _extract_labeled_block(text: str, labels: list[str]) -> str:
    label_pattern = "|".join(re.escape(label) for label in labels)
    match = re.search(
        rf"(?ims)^\s*(?:{label_pattern})\s*:\s*(.*?)(?=^\s*[a-zA-Z_][\w ]{{0,40}}\s*:|\Z)",
        text,
    )
    return match.group(1).strip() if match else ""


def _first_text(meta: dict, raw: dict, keys: list[str]) -> str:
    for source in (meta, raw):
        for key in keys:
            value = source.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return ""
