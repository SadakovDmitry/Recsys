"""Alternating exact weighted least-squares updates; no recommender library."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .base import Recommender


class ImplicitALS(Recommender):
    def __init__(self, factors=32, regularization=1.0, alpha=20.0,
                 iterations=10, seed=42):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    def targets(self, data):
        target = data.counts.copy().astype(np.float64)
        target.data[:] = 1.0
        return target

    def _update(self, confidence, targets, fixed):
        baseline = fixed.T @ fixed + self.regularization * np.eye(self.factors)
        result = np.zeros((confidence.shape[0], self.factors), dtype=np.float64)
        for row in range(confidence.shape[0]):
            start, stop = confidence.indptr[row:row + 2]
            idx = confidence.indices[start:stop]
            c = confidence.data[start:stop]
            p = targets.data[start:stop]
            y = fixed[idx]
            a = baseline + (y.T * (c - 1.0)) @ y
            b = y.T @ (c * p)
            result[row] = cho_solve(cho_factor(a, lower=True, check_finite=False),
                                    b, check_finite=False)
        return result

    def fit(self, data):
        confidence = data.counts.copy().astype(np.float64)
        confidence.data = 1.0 + self.alpha * np.log1p(confidence.data)
        targets = self.targets(data)
        confidence.sort_indices()
        targets.sort_indices()
        assert np.array_equal(confidence.indices, targets.indices)
        assert np.array_equal(confidence.indptr, targets.indptr)
        c_t, p_t = confidence.T.tocsr(), targets.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.items = rng.normal(0, 0.01, (data.x.shape[1], self.factors))
        for _ in range(self.iterations):
            self.users = self._update(confidence, targets, self.items)
            self.items = self._update(c_t, p_t, self.users)
        return self

    def score(self, users):
        return self.users[users] @ self.items.T
