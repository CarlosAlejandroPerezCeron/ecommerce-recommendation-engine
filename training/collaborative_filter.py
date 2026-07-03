"""Collaborative filtering — SVD (sklearn) or ALS (implicit)."""
from __future__ import annotations
import logging, pickle
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
logger = logging.getLogger(__name__)

class CollaborativeFilter:
    def __init__(self, method: Literal["svd","als"]="als", n_factors=100,
                 regularization=0.01, iterations=20, random_state=42):
        self.method = method
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state
        self._model = None
        self.user_index: dict = {}
        self.item_index: dict = {}
        self._user_factors = None
        self._item_factors = None

    def _build_matrix(self, df: pd.DataFrame) -> csr_matrix:
        users = df["user_id"].unique()
        items = df["sku"].unique()
        self.user_index = {u: i for i, u in enumerate(users)}
        self.item_index = {s: i for i, s in enumerate(items)}
        rows = df["user_id"].map(self.user_index).values
        cols = df["sku"].map(self.item_index).values
        return csr_matrix((df["weight"].values.astype(np.float32), (rows, cols)),
                          shape=(len(users), len(items)))

    def fit(self, df: pd.DataFrame) -> "CollaborativeFilter":
        matrix = self._build_matrix(df)
        if self.method == "als":
            try:
                import implicit
                m = implicit.als.AlternatingLeastSquares(
                    factors=self.n_factors, regularization=self.regularization,
                    iterations=self.iterations, random_state=self.random_state)
                m.fit(matrix)
                self._user_factors = m.user_factors
                self._item_factors = m.item_factors
                self._model = m
                logger.info("ALS trained: %d users, %d items", len(self.user_index), len(self.item_index))
                return self
            except ImportError:
                self.method = "svd"
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=self.n_factors, random_state=self.random_state)
        self._user_factors = svd.fit_transform(matrix)
        self._item_factors = svd.components_.T
        self._model = svd
        logger.info("SVD trained: %d users, %d items", len(self.user_index), len(self.item_index))
        return self

    def predict(self, uid: int, iid: int) -> float:
        return float(np.dot(self._user_factors[uid], self._item_factors[iid]))

    def recommend_for_user(self, user_id: str, top_k=50) -> list:
        uid = self.user_index.get(user_id)
        if uid is None: return []
        scores = self._item_factors @ self._user_factors[uid]
        top = np.argsort(scores)[::-1][:top_k]
        inv = {v: k for k, v in self.item_index.items()}
        return [(inv[i], float(scores[i])) for i in top]

    def save(self, path):
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        with open(p / "cf_model.pkl", "wb") as f: pickle.dump(self, f)

    @classmethod
    def load(cls, path) -> "CollaborativeFilter":
        with open(Path(path) / "cf_model.pkl", "rb") as f: return pickle.load(f)
