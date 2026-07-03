"""
Hybrid recommendation scorer.

Blends collaborative filtering (CF) and content-based (CB) scores
using a configurable alpha parameter. Falls back to pure CB for
cold-start users (fewer than MIN_INTERACTIONS interactions).
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

MIN_INTERACTIONS = 5
DEFAULT_ALPHA = 0.65


@dataclass
class RecommendationResult:
    sku: str
    score: float
    source: str  # "collaborative" | "content" | "hybrid"


@dataclass
class HybridRecommender:
    cf_model: object
    cb_model: object
    item_index: dict
    user_index: dict
    alpha: float = DEFAULT_ALPHA
    _interaction_counts: dict = field(default_factory=dict)

    def set_interaction_counts(self, counts: dict) -> None:
        self._interaction_counts = counts

    def _is_cold_start(self, user_id: str) -> bool:
        return self._interaction_counts.get(user_id, 0) < MIN_INTERACTIONS

    def _cf_scores(self, user_id: str, candidate_skus: list) -> np.ndarray:
        uid = self.user_index.get(user_id)
        if uid is None:
            return np.zeros(len(candidate_skus))
        scores = []
        for sku in candidate_skus:
            iid = self.item_index.get(sku, -1)
            if iid == -1:
                scores.append(0.0)
            else:
                try:
                    scores.append(float(self.cf_model.predict(uid, iid)))
                except Exception:
                    scores.append(0.0)
        return np.array(scores)

    def _cb_scores(self, user_id: str, candidate_skus: list) -> np.ndarray:
        try:
            return self.cb_model.score_for_user(user_id, candidate_skus)
        except Exception as exc:
            logger.warning("CB scoring failed for %s: %s", user_id, exc)
            return np.zeros(len(candidate_skus))

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        mn, mx = scores.min(), scores.max()
        if mx == mn:
            return np.ones_like(scores) * 0.5
        return (scores - mn) / (mx - mn)

    def recommend(self, user_id: str, candidate_skus: list, top_k: int = 10,
                  exclude_skus: Optional[list] = None) -> list:
        if not candidate_skus:
            return []
        exclude = set(exclude_skus or [])
        candidates = [s for s in candidate_skus if s not in exclude]
        cold = self._is_cold_start(user_id)
        if cold:
            scores = self._normalize(self._cb_scores(user_id, candidates))
            source_map = {s: "content" for s in candidates}
        else:
            cf = self._normalize(self._cf_scores(user_id, candidates))
            cb = self._normalize(self._cb_scores(user_id, candidates))
            scores = self.alpha * cf + (1 - self.alpha) * cb
            source_map = {}
            for i, sku in enumerate(candidates):
                if cf[i] > cb[i] * 1.5:
                    source_map[sku] = "collaborative"
                elif cb[i] > cf[i] * 1.5:
                    source_map[sku] = "content"
                else:
                    source_map[sku] = "hybrid"
        ranked = np.argsort(scores)[::-1][:top_k]
        return [RecommendationResult(sku=candidates[i], score=round(float(scores[i]),4),
                source=source_map[candidates[i]]) for i in ranked]
