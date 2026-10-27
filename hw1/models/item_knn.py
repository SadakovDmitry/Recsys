"""Item-based cosine neighbors, implemented with NumPy/SciPy only."""
import numpy as np
from scipy import sparse
from .base import Recommender


class ItemKNN(Recommender):
    def __init__(self, neighbors=80, shrink=10.0):
        self.neighbors, self.shrink = neighbors, shrink

    def fit(self, data):
        self.x = data.x.astype(np.float64)
        gram = (self.x.T @ self.x).toarray()
        norms = np.sqrt(np.diag(gram))
        similarity = gram / (norms[:, None] * norms[None, :] + self.shrink)
        np.fill_diagonal(similarity, 0.0)
        # For each target item j retain its strongest source neighbors i.
        rows, cols, values = [], [], []
        for j in range(similarity.shape[1]):
            idx = np.argsort(-similarity[:, j], kind="stable")[:self.neighbors]
            idx = idx[similarity[idx, j] > 0]
            rows.extend(idx)
            cols.extend([j] * len(idx))
            values.extend(similarity[idx, j])
        self.weights = sparse.csc_matrix((values, (rows, cols)), shape=gram.shape)
        return self

    def score(self, users):
        return (self.x[users] @ self.weights).toarray()
