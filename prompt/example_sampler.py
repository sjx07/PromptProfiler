"""Task-agnostic example samplers for few-shot selection.

Samplers conform to the demo_selector interface:
    (record: Dict[str, Any], examples: List[Dict]) -> List[Dict]

The caller provides pre-built pool (train split only), text for similarity,
and an optional predicate extractor. The sampler handles selection logic only.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional

DemoSelector = Callable[[Dict[str, Any], List[Dict]], List[Dict]]


def build_sampler(
    strategy: str,
    k: int,
    pool: List[Dict],
    pool_texts: List[str],
    *,
    text_fn: Callable[[Dict], str] = lambda r: r.get("question", r.get("content", "")),
    seed: int = 42,
    predicate_fn: Optional[Callable[[str], List[float]]] = None,
    diversity: float = 0.3,
) -> DemoSelector:
    """Build a demo_selector callable.

    Args:
        strategy: "random", "knn", "predicate", or "hybrid".
        k: Number of examples to select per query.
        pool: Pre-built example dicts ({"inputs": {...}, "outputs": {...}}).
              Must be train-split only — no test leakage.
        pool_texts: Text for each pool example (for similarity computation).
        text_fn: Extracts query text from a record dict at inference time.
        seed: Random seed.
        predicate_fn: Extracts a numeric feature vector from text.
                      Required for "predicate" and "hybrid".
        diversity: MMR lambda for knn/hybrid (0=max diversity, 1=pure relevance).
    """
    if strategy == "random":
        return _build_random(k, pool, seed)
    elif strategy == "knn":
        return _build_knn(k, pool, pool_texts, text_fn, diversity)
    elif strategy == "predicate":
        if predicate_fn is None:
            raise ValueError("predicate_fn is required for predicate strategy")
        return _build_predicate(k, pool, pool_texts, text_fn, predicate_fn)
    elif strategy == "hybrid":
        if predicate_fn is None:
            raise ValueError("predicate_fn is required for hybrid strategy")
        return _build_hybrid(k, pool, pool_texts, text_fn, predicate_fn, diversity)
    else:
        raise ValueError(f"Unknown example strategy: {strategy}")


def _build_random(k, pool, seed):
    def selector(record, examples):
        query_key = str(record.get("query_id", record.get("question", "")))
        rng = random.Random(seed + hash(query_key) % (2**31))
        return rng.sample(pool, min(k, len(pool)))
    return selector


def _mmr_select(relevance_scores, pool_emb, k, lam):
    """Maximal Marginal Relevance selection.

    Iteratively picks the candidate that maximizes:
        lam * relevance(candidate) - (1-lam) * max_sim(candidate, already_selected)
    """
    import numpy as np

    n = len(relevance_scores)
    if n <= k:
        return list(range(n))

    selected = []
    remaining = set(range(n))

    for _ in range(k):
        best_idx = -1
        best_score = -float("inf")
        for i in remaining:
            rel = relevance_scores[i]
            if selected:
                max_sim = max(float(pool_emb[i] @ pool_emb[j]) for j in selected)
            else:
                max_sim = 0.0
            score = lam * rel - (1 - lam) * max_sim
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(best_idx)
        remaining.discard(best_idx)

    return selected


def _build_knn(k, pool, pool_texts, text_fn, diversity):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    pool_emb = model.encode(pool_texts, normalize_embeddings=True,
                            show_progress_bar=True, batch_size=128)

    # Pre-filter: keep only unique texts (dedup exact duplicates)
    seen = {}
    unique_idxs = []
    for i, t in enumerate(pool_texts):
        if t not in seen:
            seen[t] = i
            unique_idxs.append(i)
    unique_pool = [pool[i] for i in unique_idxs]
    unique_emb = pool_emb[unique_idxs]

    def selector(record, examples):
        q_emb = model.encode([text_fn(record)], normalize_embeddings=True)[0]
        scores = unique_emb @ q_emb
        # Pre-filter to top candidates for MMR efficiency
        n_candidates = min(k * 10, len(unique_pool))
        top_candidates = np.argsort(-scores)[:n_candidates]
        cand_scores = scores[top_candidates]
        cand_emb = unique_emb[top_candidates]
        selected = _mmr_select(cand_scores, cand_emb, k, 1 - diversity)
        return [unique_pool[top_candidates[i]] for i in selected]
    return selector


def _build_predicate(k, pool, pool_texts, text_fn, predicate_fn):
    import numpy as np

    pool_preds = np.array([predicate_fn(t) for t in pool_texts])

    def selector(record, examples):
        q_pred = np.array(predicate_fn(text_fn(record)))
        sims = np.array([
            float(np.minimum(q_pred, p).sum() / max(np.maximum(q_pred, p).sum(), 1e-8))
            for p in pool_preds
        ])
        top = np.argsort(-sims)[:k]
        return [pool[i] for i in top]
    return selector


def _build_hybrid(k, pool, pool_texts, text_fn, predicate_fn, diversity, alpha=0.5):
    """Hybrid: alpha * knn_score + (1-alpha) * predicate_score, with MMR diversity."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    pool_emb = model.encode(pool_texts, normalize_embeddings=True,
                            show_progress_bar=True, batch_size=128)
    pool_preds = np.array([predicate_fn(t) for t in pool_texts])

    # Dedup
    seen = {}
    unique_idxs = []
    for i, t in enumerate(pool_texts):
        if t not in seen:
            seen[t] = i
            unique_idxs.append(i)
    unique_pool = [pool[i] for i in unique_idxs]
    unique_emb = pool_emb[unique_idxs]
    unique_preds = pool_preds[unique_idxs]

    def selector(record, examples):
        q_text = text_fn(record)
        q_emb = model.encode([q_text], normalize_embeddings=True)[0]
        q_pred = np.array(predicate_fn(q_text))

        knn_scores = unique_emb @ q_emb
        pred_scores = np.array([
            float(np.minimum(q_pred, p).sum() / max(np.maximum(q_pred, p).sum(), 1e-8))
            for p in unique_preds
        ])

        combined = alpha * knn_scores + (1 - alpha) * pred_scores

        n_candidates = min(k * 10, len(unique_pool))
        top_candidates = np.argsort(-combined)[:n_candidates]
        cand_scores = combined[top_candidates]
        cand_emb = unique_emb[top_candidates]
        selected = _mmr_select(cand_scores, cand_emb, k, 1 - diversity)
        return [unique_pool[top_candidates[i]] for i in selected]
    return selector
