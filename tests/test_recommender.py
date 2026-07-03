"""Unit tests for the hybrid recommender."""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from recommender import HybridRecommender, RecommendationResult

class _FakeCF:
    def predict(self, uid, iid): return 1.0 / (iid + 1)

class _FakeCB:
    def score_for_user(self, user_id, skus): return np.linspace(0.3, 0.9, len(skus))

ITEMS = {"SKU-A":0,"SKU-B":1,"SKU-C":2,"SKU-D":3,"SKU-E":4}
USERS = {"warm":0,"cold":1}

@pytest.fixture
def rec():
    r = HybridRecommender(cf_model=_FakeCF(), cb_model=_FakeCB(),
                          item_index=ITEMS, user_index=USERS, alpha=0.65)
    r.set_interaction_counts({"warm":20,"cold":2})
    return r

def test_returns_top_k(rec):
    res = rec.recommend("warm", list(ITEMS), top_k=3)
    assert len(res) == 3 and all(isinstance(r, RecommendationResult) for r in res)

def test_excludes_skus(rec):
    res = rec.recommend("warm", list(ITEMS), top_k=5, exclude_skus=["SKU-A","SKU-B"])
    assert "SKU-A" not in {r.sku for r in res}

def test_cold_start_content_only(rec):
    res = rec.recommend("cold", list(ITEMS), top_k=5)
    assert all(r.source == "content" for r in res)

def test_warm_uses_hybrid_or_collab(rec):
    res = rec.recommend("warm", list(ITEMS), top_k=5)
    assert {r.source for r in res} & {"collaborative","hybrid"}

def test_scores_normalized(rec):
    res = rec.recommend("warm", list(ITEMS), top_k=5)
    assert all(0.0 <= r.score <= 1.0 for r in res)

def test_empty_candidates(rec):
    assert rec.recommend("warm", [], top_k=5) == []

def test_unknown_user_graceful(rec):
    res = rec.recommend("ghost", list(ITEMS), top_k=3)
    assert len(res) == 3
