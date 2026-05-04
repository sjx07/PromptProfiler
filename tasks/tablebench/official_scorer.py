"""Port of TableBench's official scoring logic.

Source: github.com/TableBench/TableBench (metrics/qa_metrics.py + custom_em_metric.py).

Scoring rules:
  - FactChecking, NumericalReasoning: strict EM (after normalize_answer)
  - DataAnalysis subtypes:
      * CorrelationAnalysis, TrendForecasting, StatisticalAnalysis → EM_with_error_10
      * ImpactAnalysis → strict EM
      * Others → ROUGE-L
"""
from __future__ import annotations

import re
import string
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Tuple


# ── normalization ─────────────────────────────────────────────────────

def _remove_articles(t: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", t)


def _remove_punc(t: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in t if ch not in exclude)


def normalize_answer(s: str) -> str:
    return " ".join(_remove_articles(_remove_punc(s.lower())).split())


# ── EM with numeric precision matching ────────────────────────────────

_NUM_RE = re.compile(r"^-?\d+(\.\d+)?%?$")


def _is_number(v: str) -> bool:
    return bool(_NUM_RE.match(v.strip()))


def _normalize_number(v: str) -> Decimal:
    v = v.strip()
    if v.endswith("%"):
        return (Decimal(v[:-1]) / Decimal("100")).quantize(
            Decimal("1.0000"), rounding=ROUND_HALF_UP
        )
    return Decimal(v)


def _decimal_precision(values: List[str]) -> int:
    precs = []
    for val in values:
        if val.endswith("%"):
            continue
        if "." in val:
            precs.append(len(val.split(".")[-1]))
        else:
            precs.append(0)
    return min(precs) if precs else 0


def _round_to(v: Decimal, p: int) -> str:
    fmt = f"1.{'0' * p}"
    return str(v.quantize(Decimal(fmt), rounding=ROUND_HALF_UP))


def compute_em(reference: str, prediction: str) -> float:
    """Single-pair EM (after normalize_answer). Mirrors compute_em loop body."""
    ref_answers = [x.strip() for x in reference.split(",")]
    pred_answers = [x.strip() for x in prediction.split(",")]
    score = 0.0
    weight = 1.0 / max(len(ref_answers), 1)

    for i, r in enumerate(ref_answers):
        if i >= len(pred_answers):
            continue
        p = pred_answers[i]
        if _is_number(r):
            try:
                if r.endswith("%"):
                    if _normalize_number(r) == _normalize_number(p):
                        score += weight
                else:
                    ref_vals = [x for x in ref_answers
                                if _is_number(x) and not x.endswith("%")]
                    prec = _decimal_precision(ref_vals)
                    if _round_to(_normalize_number(r), prec) == \
                       _round_to(_normalize_number(p), prec):
                        score += weight
            except Exception:
                continue
        else:
            if r == p:
                score += weight
    return score


def compute_em_with_tolerance(reference: str, prediction: str,
                               error_pct: float = 10.0) -> float:
    """EM where numeric values match if abs(p-r)/abs(r) ≤ error_pct/100."""
    ref_answers = [x.strip() for x in reference.split(",")]
    pred_answers = [x.strip() for x in prediction.split(",")]
    score = 0.0
    weight = 1.0 / max(len(ref_answers), 1)

    for i, r in enumerate(ref_answers):
        if i >= len(pred_answers):
            continue
        p = pred_answers[i]
        if _is_number(r) and _is_number(p):
            try:
                rn = _normalize_number(r)
                pn = _normalize_number(p)
                if rn == 0:
                    if abs(pn) <= Decimal(error_pct) / Decimal("100"):
                        score += weight
                else:
                    if abs(pn - rn) / abs(rn) <= Decimal(error_pct) / Decimal("100"):
                        score += weight
            except Exception:
                continue
        else:
            if r == p:
                score += weight
    return score


# ── ROUGE-L (inline LCS implementation; matches rouge_score's rougeL F1) ──

def _lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    # space-optimized LCS DP
    prev = [0] * (len(b) + 1)
    curr = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
        for k in range(len(curr)):
            curr[k] = 0
    return prev[len(b)]


def _rouge_l(reference: str, prediction: str) -> float:
    ref_toks = reference.split()
    pred_toks = prediction.split()
    if not ref_toks or not pred_toks:
        return 0.0
    lcs = _lcs_len(ref_toks, pred_toks)
    if lcs == 0:
        return 0.0
    p = lcs / len(pred_toks)
    r = lcs / len(ref_toks)
    return 2 * p * r / (p + r)


# ── per-subtype dispatch ──────────────────────────────────────────────

_DA_EM_ERROR10 = {"CorrelationAnalysis", "TrendForecasting", "StatisticalAnalysis"}
_DA_EM_STRICT = {"ImpactAnalysis"}
# Anything else under DataAnalysis (CausalAnalysis, DescriptiveAnalysis,
# AnomalyDetection, ...) falls back to ROUGE-L.


def score_one(qtype: str, qsubtype: str, reference: str, prediction: str
              ) -> Tuple[float, str]:
    """Score a single (prediction, reference) following the official rules.

    Returns (score in [0,1], method_label).
    """
    ref = normalize_answer(str(reference or ""))
    pred = normalize_answer(str(prediction or ""))

    if qtype == "FactChecking":
        return compute_em(ref, pred), "fc_em"
    if qtype == "NumericalReasoning":
        return compute_em(ref, pred), "nr_em"
    if qtype == "DataAnalysis":
        if qsubtype in _DA_EM_ERROR10:
            return compute_em_with_tolerance(ref, pred, 10.0), "da_em_error10"
        if qsubtype in _DA_EM_STRICT:
            return compute_em(ref, pred), "da_em_strict"
        return _rouge_l(ref, pred), "da_rouge_l"
    return compute_em(ref, pred), "fallback_em"
