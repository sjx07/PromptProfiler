"""HoVer compound fact-verification task with 3-hop wiki BM25 retrieval.

4-module pipeline mirroring langProBe's `HoverMultiHop` (the same program GEPA's
paper adapts for HotpotQA). Scoring is `discrete_retrieval_eval`: all gold
supporting-fact titles must be found among the retrieved doc titles across
hops. No answer generation — purely a query-writing / retrieval task.

Pipeline:
  hop1  retrieve(claim)                           → summarize1(claim, passages)       → summary_1
  hop2  create_query_hop2(claim, summary_1)       → retrieve(query) → summarize2(...) → summary_2
  hop3  create_query_hop3(claim, summary_1, summary_2) → retrieve(query)
  prediction = "\\n".join(unique retrieved titles across 3 hops)

Retriever: `bm25s` over `wiki.abstracts.2017.jsonl` (same as `HotpotQAContextTask`).
"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping

from task import CompoundTask, ModuleRuntime, ModuleSpec
from tasks.hotpotqa_context.hotpotqa_context import (
    WikiBM25Retriever,
    format_passages,
    get_wiki_bm25_retriever,
    _parse_json_field,
    normalize_text,
)


# ── Task ─────────────────────────────────────────────────────────────

class HoverContextTask(CompoundTask):
    """Three-hop fact-verification retrieval, 2 query writers + 2 summary modules."""

    name = "hover_context"
    scorer = "hover_discrete_retrieval"
    retrieval_k = 7
    wiki_abstracts_path: str | None = None
    bm25_index_dir: str | None = None
    module_specs = {
        "summarize1": ModuleSpec(
            input_fields={
                "claim": "The HoVer claim to verify.",
                "passages": "First-hop context passages retrieved from the retriever.",
            },
            output_fields={
                "summary": "Concise first-hop analysis of whether passages support or refute the claim.",
            },
        ),
        "create_query_hop2": ModuleSpec(
            input_fields={
                "claim": "The HoVer claim to verify.",
                "summary_1": "The first-hop summary.",
            },
            output_fields={
                "query": "A focused retrieval query for missing second-hop evidence.",
            },
        ),
        "summarize2": ModuleSpec(
            input_fields={
                "claim": "The HoVer claim to verify.",
                "context": "The first-hop summary used as intermediate context.",
                "passages": "Second-hop context passages retrieved from the retriever.",
            },
            output_fields={
                "summary": "Concise second-hop analysis connecting hop-2 passages to hop-1 context.",
            },
        ),
        "create_query_hop3": ModuleSpec(
            input_fields={
                "claim": "The HoVer claim to verify.",
                "summary_1": "The first-hop summary.",
                "summary_2": "The second-hop summary.",
            },
            output_fields={
                "query": "A focused retrieval query for missing third-hop evidence.",
            },
        ),
    }

    @classmethod
    def configure_from_cfg(cls, cfg: Mapping[str, Any]) -> None:
        import os
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
        claim = query.get("content", "")

        # hop 1
        passages1 = self._retriever.search(claim, k=self.retrieval_k)
        sys_p, user_p = self.build_module_prompt(
            "summarize1",
            {"claim": claim, "passages": format_passages(passages1)},
        )
        summary1 = runtime.call(
            "summarize1", sys_p, user_p, parse=parse_summary,
            meta={"retrieval_query": claim,
                  "retrieved_titles": [p["title"] for p in passages1]},
        ).parsed_output

        # hop 2 query
        sys_p, user_p = self.build_module_prompt(
            "create_query_hop2",
            {"claim": claim, "summary_1": summary1},
        )
        hop2_query = runtime.call(
            "create_query_hop2", sys_p, user_p, parse=parse_hop_query,
        ).parsed_output

        # hop 2 retrieve + summarize
        q2 = hop2_query or claim
        passages2 = self._retriever.search(q2, k=self.retrieval_k)
        sys_p, user_p = self.build_module_prompt(
            "summarize2",
            {"claim": claim, "context": summary1,
             "passages": format_passages(passages2)},
        )
        summary2 = runtime.call(
            "summarize2", sys_p, user_p, parse=parse_summary,
            meta={"retrieval_query": q2,
                  "retrieved_titles": [p["title"] for p in passages2]},
        ).parsed_output

        # hop 3 query
        sys_p, user_p = self.build_module_prompt(
            "create_query_hop3",
            {"claim": claim, "summary_1": summary1, "summary_2": summary2},
        )
        hop3_query = runtime.call(
            "create_query_hop3", sys_p, user_p, parse=parse_hop_query,
            meta={"hop3_input_summary_1": summary1[:200],
                  "hop3_input_summary_2": summary2[:200]},
        ).parsed_output

        # hop 3 retrieve (no summarize3 — last hop just retrieves)
        q3 = hop3_query or claim
        passages3 = self._retriever.search(q3, k=self.retrieval_k)

        # Prediction = union of unique titles across all 3 hops, sorted for determinism.
        # Downstream scorer parses this by splitting on newlines.
        titles = sorted({p["title"] for p in passages1 + passages2 + passages3 if p.get("title")})

        # Stash hop3 retrieval meta on the last module trace via runtime.call would require
        # an extra module; instead, rely on titles flowing through the prediction.
        _ = (passages3,)  # keep to make intent explicit
        return "\n".join(titles)

    def score(self, prediction: str, query_meta: dict) -> tuple[float, dict]:
        raw = query_meta.get("_raw") if isinstance(query_meta.get("_raw"), dict) else {}
        supporting = (
            query_meta.get("supporting_facts")
            or (raw.get("supporting_facts") if raw else None)
            or []
        )
        gold_titles = set()
        for f in supporting:
            if isinstance(f, (list, tuple)) and f:
                gold_titles.add(normalize_text(str(f[0])))
            elif isinstance(f, dict):
                key = f.get("key") or f.get("title") or ""
                if key:
                    gold_titles.add(normalize_text(str(key)))
        if not gold_titles:
            return 0.0, {"status": "missing_gold_titles",
                         "supported": 0.0, "recall": 0.0,
                         "n_gold": 0, "n_retrieved": 0}

        retrieved = {normalize_text(t) for t in prediction.splitlines() if t.strip()}
        hits = gold_titles & retrieved
        recall = len(hits) / len(gold_titles)
        supported = 1.0 if gold_titles.issubset(retrieved) else 0.0
        return supported, {
            "status": "ok",
            "supported": supported,
            "recall": recall,
            "n_gold": len(gold_titles),
            "n_retrieved": len(retrieved),
            "n_hits": len(hits),
        }


# ── Parsers (reuse hotpotqa's _parse_json_field) ─────────────────────

def parse_summary(raw_response: str) -> str:
    return _parse_json_field(raw_response, "summary", "analysis", "reasoning")


def parse_hop_query(raw_response: str) -> str:
    return _parse_json_field(raw_response, "query", "search_query", "hop_query")
