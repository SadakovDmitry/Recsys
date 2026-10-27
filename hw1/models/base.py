"""Shared deterministic full-catalog recommendation interface."""
import numpy as np


class Recommender:
    def score(self, users):
        raise NotImplementedError

    def recommend(self, users, seen, k=10):
        """Stable ties: ascending internal item ID (sorted original StockCode)."""
        users = np.asarray(users, dtype=np.int64)
        scores = np.asarray(self.score(users), dtype=np.float64)
        if scores.shape != (len(users), seen.shape[1]):
            raise ValueError("Invalid score shape")
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite predictions")
        for row, user in enumerate(users):
            known = seen.indices[seen.indptr[user]:seen.indptr[user + 1]]
            scores[row, known] = -np.inf
        order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        if not np.isfinite(np.take_along_axis(scores, order, axis=1)).all():
            raise ValueError("Fewer than k unseen candidates")
        return order
