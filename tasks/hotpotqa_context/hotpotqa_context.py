"""HotpotQA compound QA task with wiki BM25 retrieval.

4-module pipeline mirroring the GEPA paper's modified HoVerMultiHop:
  summarize1 → create_query_hop2 → summarize2 → final_answer

Retriever: `bm25s` over `wiki.abstracts.2017.jsonl` (DSPy's canonical
multi-hop config: k1=0.9, b=0.4, English stemmer + stopwords).

Scoring: strict EM + HotpotQA F1 via `dspy.evaluate.metrics`
(which copy the official `hotpot_evaluate_v1.py` formulas).
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Mapping

from task import CompoundTask, ModuleRuntime, ModuleSpec

try:
    from dspy.evaluate.metrics import hotpot_f1_score, normalize_text  # type: ignore[import-not-found]
except ImportError:
    def normalize_text(text: str) -> str:
        """DSPy-compatible HotpotQA answer normalization fallback."""
        import string

        def remove_articles(s: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", s)

        def white_space_fix(s: str) -> str:
            return " ".join(s.split())

        def remove_punc(s: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in s if ch not in exclude)

        return white_space_fix(remove_articles(remove_punc(text.lower())))

    def hotpot_f1_score(prediction: str, ground_truth: str) -> float:
        """Token F1 fallback matching the official HotpotQA metric shape."""
        pred_tokens = normalize_text(prediction).split()
        gold_tokens = normalize_text(ground_truth).split()
        common = set(pred_tokens) & set(gold_tokens)
        num_same = sum(min(pred_tokens.count(tok), gold_tokens.count(tok)) for tok in common)
        if len(pred_tokens) == 0 or len(gold_tokens) == 0:
            return float(pred_tokens == gold_tokens)
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)


# ── Task ─────────────────────────────────────────────────────────────

class HotpotQAContextTask(CompoundTask):
    """Four-stage multi-hop QA with wiki BM25 retrieval."""

    name = "hotpotqa_context"
    scorer = "hotpotqa_context_exact_match"
    retrieval_k = 7
    wiki_abstracts_path: str | None = None
    bm25_index_dir: str | None = None
    module_specs = {
        "summarize1": ModuleSpec(
            input_fields={
                "question": "The original HotpotQA question.",
                "passages": "First-hop context passages retrieved from the retriever.",
            },
            output_fields={
                "reasoning": "Brief chain-of-thought over the passages before committing to a summary.",
                "summary": "Concise first-hop facts relevant to answering the question.",
            },
        ),
        "create_query_hop2": ModuleSpec(
            input_fields={
                "question": "The original HotpotQA question.",
                "summary_1": "The first-hop summary.",
            },
            output_fields={
                "reasoning": "Brief reasoning about what evidence is still missing after hop 1.",
                "query": "A focused retrieval query for missing second-hop evidence.",
            },
        ),
        "summarize2": ModuleSpec(
            input_fields={
                "question": "The original HotpotQA question.",
                "context": "The first-hop summary used as intermediate context.",
                "passages": "Second-hop context passages retrieved from the retriever.",
            },
            output_fields={
                "reasoning": "Brief reasoning connecting hop 2 passages to hop 1 summary and the question.",
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
                "reasoning": "Brief reasoning connecting the summaries to the answer span.",
                "answer": "Shortest supported answer span.",
            },
        ),
    }

    @classmethod
    def configure_from_cfg(cls, cfg: Mapping[str, Any]) -> None:
        cls.wiki_abstracts_path = (
            cfg.get("wiki_abstracts_path")
            or os.environ.get("HOTPOTQA_WIKI_ABSTRACTS")
        )
        cls.bm25_index_dir = (
            cfg.get("bm25_index_dir")
            or os.environ.get("HOTPOTQA_BM25_INDEX_DIR")
        )
        if cfg.get("retrieval_k") is not None:
            cls.retrieval_k = int(cfg["retrieval_k"])

    def __init__(
        self,
        *,
        wiki_abstracts_path: str | None = None,
        bm25_index_dir: str | None = None,
        retriever: Any | None = None,
    ) -> None:
        super().__init__()
        if retriever is not None:
            self._retriever = retriever
        else:
            self._retriever = get_wiki_bm25_retriever(
                wiki_abstracts_path or self.__class__.wiki_abstracts_path,
                bm25_index_dir or self.__class__.bm25_index_dir,
            )

    def run(self, query: dict, runtime: ModuleRuntime) -> str:
        question = query.get("content", "")

        # hop 1
        passages1 = self._retriever.search(question, k=self.retrieval_k)
        sys_p, user_p = self.build_module_prompt(
            "summarize1",
            {"question": question, "passages": format_passages(passages1)},
        )
        summary1 = runtime.call(
            "summarize1", sys_p, user_p, parse=parse_summary,
            meta={"retrieval_query": question,
                  "retrieved_titles": [p["title"] for p in passages1]},
        ).parsed_output

        # hop 2 query
        sys_p, user_p = self.build_module_prompt(
            "create_query_hop2",
            {"question": question, "summary_1": summary1},
        )
        hop2_query = runtime.call(
            "create_query_hop2", sys_p, user_p, parse=parse_hop_query,
        ).parsed_output

        # hop 2 retrieve + summarize
        retrieval_query = hop2_query or question
        passages2 = self._retriever.search(retrieval_query, k=self.retrieval_k)
        sys_p, user_p = self.build_module_prompt(
            "summarize2",
            {"question": question, "context": summary1,
             "passages": format_passages(passages2)},
        )
        summary2 = runtime.call(
            "summarize2", sys_p, user_p, parse=parse_summary,
            meta={"retrieval_query": retrieval_query,
                  "retrieved_titles": [p["title"] for p in passages2]},
        ).parsed_output

        # final answer
        sys_p, user_p = self.build_module_prompt(
            "final_answer",
            {"question": question, "summary_1": summary1, "summary_2": summary2},
        )
        return runtime.call(
            "final_answer", sys_p, user_p, parse=parse_answer,
        ).parsed_output

    def score(self, prediction: str, query_meta: dict) -> tuple[float, dict]:
        raw = query_meta.get("_raw") if isinstance(query_meta.get("_raw"), dict) else {}
        answer = query_meta.get("answer") or (raw.get("answer") if raw else "") or ""
        if not answer:
            return 0.0, {"status": "missing_answer",
                         "exact_match": 0.0, "f1": 0.0}
        em = 1.0 if normalize_text(prediction) == normalize_text(answer) else 0.0
        f1 = hotpot_f1_score(prediction, answer)
        return em, {"status": "ok", "exact_match": em, "f1": f1}


# ── Retriever ────────────────────────────────────────────────────────

class WikiBM25Retriever:
    """BM25 over a local `wiki.abstracts.2017.jsonl` corpus (DSPy-canonical config)."""

    def __init__(self, corpus_path: str | None, index_dir: str | None = None) -> None:
        self.corpus_path = Path(corpus_path).expanduser() if corpus_path else None
        self.index_dir = (
            Path(index_dir).expanduser() if index_dir else self._default_index_dir()
        )
        self._lock = threading.Lock()
        self._ready = False
        self._retriever: Any = None  # bm25s.BM25 after _ensure_ready
        self._stemmer: Any = None
        self._corpus: list[str] = []

    def search(self, query: str, *, k: int = 7, context: Any = None) -> list[dict[str, Any]]:
        _ = context  # accepted for retriever-interface parity; unused
        if k <= 0:
            return []
        self._ensure_ready()
        import bm25s  # type: ignore[import-not-found]

        tokens = bm25s.tokenize(
            str(query), stopwords="en", stemmer=self._stemmer, show_progress=False,
        )
        results, _ = self._retriever.retrieve(
            tokens, k=k, n_threads=1, show_progress=False,
        )
        return [_passage_from_corpus_text(self._corpus[int(d)]) for d in results[0][:k]]

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            try:
                import bm25s  # type: ignore[import-not-found]
                import Stemmer  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError(
                    "HotpotQA wiki BM25 needs packages 'bm25s' and 'PyStemmer'."
                ) from exc

            if not self.corpus_path or not self.corpus_path.exists():
                raise FileNotFoundError(
                    "HotpotQA wiki BM25 needs a local wiki.abstracts.2017.jsonl — "
                    "set cfg.wiki_abstracts_path or HOTPOTQA_WIKI_ABSTRACTS."
                )

            self.index_dir.mkdir(parents=True, exist_ok=True)
            self._corpus = list(_iter_wiki_abstract_corpus(self.corpus_path))
            self._stemmer = Stemmer.Stemmer("english")

            if any(self.index_dir.iterdir()):
                try:
                    self._retriever = bm25s.BM25.load(str(self.index_dir))
                except Exception:
                    self._retriever = None

            if self._retriever is None:
                tokens = bm25s.tokenize(
                    self._corpus, stopwords="en", stemmer=self._stemmer, show_progress=True,
                )
                self._retriever = bm25s.BM25(k1=0.9, b=0.4)
                self._retriever.index(tokens)
                self._retriever.save(str(self.index_dir))
            self._ready = True

    @staticmethod
    def _default_index_dir() -> Path:
        return Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / (
            "prompt_profiler/hotpotqa_wiki_bm25"
        )


_WIKI_RETRIEVER_CACHE: dict[tuple[str | None, str | None], WikiBM25Retriever] = {}
_WIKI_RETRIEVER_CACHE_LOCK = threading.Lock()


def get_wiki_bm25_retriever(
    corpus_path: str | None, index_dir: str | None = None,
) -> WikiBM25Retriever:
    key = (str(corpus_path) if corpus_path else None,
           str(index_dir) if index_dir else None)
    with _WIKI_RETRIEVER_CACHE_LOCK:
        if key not in _WIKI_RETRIEVER_CACHE:
            _WIKI_RETRIEVER_CACHE[key] = WikiBM25Retriever(corpus_path, index_dir)
        return _WIKI_RETRIEVER_CACHE[key]


def _iter_wiki_abstract_corpus(path: Path):
    try:
        import ujson as json_lib  # type: ignore[import-not-found]
    except ImportError:
        json_lib = json
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json_lib.loads(line)
            title = str(row.get("title", "")).strip()
            text_val = row.get("text", "")
            text = " ".join(map(str, text_val)) if isinstance(text_val, list) else str(text_val)
            if title or text:
                yield f"{title} | {text}"


def _passage_from_corpus_text(text: str) -> dict[str, Any]:
    title, sep, body = str(text).partition(" | ")
    if not sep:
        title, body = "context", text
    return {"title": title.strip(), "text": body.strip()}


# ── Formatting + parsers ─────────────────────────────────────────────

def format_passages(passages: list[dict[str, Any]]) -> str:
    if not passages:
        return "No context passages available."
    return "\n\n".join(
        f"Passage {i} ({p.get('title', '')}): {p.get('text', '')}"
        for i, p in enumerate(passages, start=1)
    )


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_TRAILING_THINK_RE = re.compile(r"<think>.*\Z", re.DOTALL)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_field(raw: str, *fields: str) -> str:
    """Extract the first matching JSON field from `raw`; fall back to stripped raw text.

    Defensive against:
      - `<think>...</think>` prefixes (Qwen / DeepSeek thinking-mode).
      - ```fenced``` JSON blocks.
      - Empty-string values — return the empty string rather than falling through
        (the model genuinely said "I don't know").
      - Truncated JSON (missing closing brace due to max_tokens cutoff):
        after a failed `json.loads`, regex-extract each field individually.
    """
    text = _TRAILING_THINK_RE.sub("", _THINK_RE.sub("", raw)).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    m = _JSON_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            for f in fields:
                if f in obj and obj[f] is not None:
                    val = obj[f]
                    # Nested structure (e.g. GEPA+Merge's `summarize1` returns a dict
                    # of "Entity/Person Mention", "Direct Answer", "Clues for Next
                    # Steps" because its prompt asks for it). Emit JSON so the next
                    # module reads structured evidence, not Python `repr(dict)`.
                    if isinstance(val, (dict, list)):
                        return json.dumps(val, indent=2, ensure_ascii=False).strip()
                    return str(val).strip()
    # Truncated / malformed JSON: regex per field, tolerant of missing brace.
    for f in fields:
        pat = re.compile(r'"' + re.escape(f) + r'"\s*:\s*"((?:[^"\\]|\\.)*)"', re.DOTALL)
        fm = pat.search(text)
        if fm:
            raw_val = fm.group(1)
            try:
                return bytes(raw_val, "utf-8").decode("unicode_escape").strip()
            except UnicodeDecodeError:
                return raw_val.strip()
    return text.strip()


def parse_summary(raw_response: str) -> str:
    return _parse_json_field(raw_response, "summary", "summary_1", "facts")


def parse_hop_query(raw_response: str) -> str:
    return _parse_json_field(raw_response, "query", "search_query", "hop2_query")


def parse_answer(raw_response: str) -> str:
    return _parse_json_field(raw_response, "answer", "final_answer", "response")
