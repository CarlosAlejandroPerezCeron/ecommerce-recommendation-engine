"""Content-based similarity using TF-IDF on product metadata."""
from __future__ import annotations
import logging, pickle
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
logger = logging.getLogger(__name__)

class ContentBasedModel:
    def __init__(self, max_features=50_000, ngram_range=(1,2), min_df=2):
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range,
            min_df=min_df, sublinear_tf=True, strip_accents="unicode")
        self._tfidf_matrix = None
        self.sku_index: dict = {}
        self._user_seeds: dict = {}

    def _corpus(self, catalog: pd.DataFrame) -> list:
        return [" ".join(str(r.get(c,"")) for c in ["title","brand","category","description"])
                for _, r in catalog.iterrows()]

    def fit(self, catalog: pd.DataFrame) -> "ContentBasedModel":
        self.sku_index = {s: i for i, s in enumerate(catalog["sku"])}
        self._tfidf_matrix = self.vectorizer.fit_transform(self._corpus(catalog))
        logger.info("TF-IDF: %s, vocab: %d", self._tfidf_matrix.shape, len(self.vectorizer.vocabulary_))
        return self

    def set_user_seeds(self, seeds: dict) -> None:
        self._user_seeds = seeds

    def _seed_vector(self, user_id: str) -> Optional[np.ndarray]:
        idxs = [self.sku_index[s] for s in self._user_seeds.get(user_id,[]) if s in self.sku_index]
        if not idxs: return None
        return np.asarray(self._tfidf_matrix[idxs].mean(axis=0))

    def score_for_user(self, user_id: str, candidates: list) -> np.ndarray:
        if self._tfidf_matrix is None: raise RuntimeError("Model not fitted.")
        seed = self._seed_vector(user_id)
        if seed is None: return np.ones(len(candidates)) * 0.5
        idxs = [self.sku_index.get(s, -1) for s in candidates]
        scores = np.zeros(len(candidates))
        valid = [(i, idx) for i, idx in enumerate(idxs) if idx != -1]
        if valid:
            pos, iids = zip(*valid)
            sims = cosine_similarity(seed, self._tfidf_matrix[list(iids)]).flatten()
            for p, s in zip(pos, sims): scores[p] = s
        return scores

    def similar_items(self, sku: str, top_k=20) -> list:
        idx = self.sku_index.get(sku)
        if idx is None: return []
        sims = cosine_similarity(self._tfidf_matrix[idx], self._tfidf_matrix).flatten()
        sims[idx] = -1
        inv = {v: k for k, v in self.sku_index.items()}
        return [(inv[i], float(sims[i])) for i in np.argsort(sims)[::-1][:top_k]]

    def save(self, path):
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        with open(p / "cb_model.pkl", "wb") as f: pickle.dump(self, f)

    @classmethod
    def load(cls, path) -> "ContentBasedModel":
        with open(Path(path) / "cb_model.pkl", "rb") as f: return pickle.load(f)
