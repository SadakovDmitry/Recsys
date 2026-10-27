"""Closed-form EASE with zero diagonal."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .base import Recommender


class EASE(Recommender):
    def __init__(self, regularization=500.0):
        self.regularization = regularization

    def fit(self, data):
        self.x = data.x.astype(np.float64)
        gram = (self.x.T @ self.x).toarray()
        gram.flat[::gram.shape[0] + 1] += self.regularization
        precision = cho_solve(cho_factor(gram, lower=True, check_finite=False),
                              np.eye(gram.shape[0]), check_finite=False)
        self.weights = -precision / np.diag(precision)[None, :]
        np.fill_diagonal(self.weights, 0.0)
        return self

    def score(self, users):
        return self.x[users] @ self.weights
