"""Small independent numerical checks of formulas and ranking invariants."""
from types import SimpleNamespace
import unittest
import numpy as np
from scipy import sparse
import torch
from models import EASE, ImplicitALS, CustomTargetALS, ItemKNN
from models.fm import FMNetwork
from models.base import Recommender
from evaluation import ranking_metrics


class NumericalChecks(unittest.TestCase):
    def setUp(self):
        x = sparse.csr_matrix(np.array([[1., 1., 0.], [0., 1., 1.], [1., 0., 1.]]))
        counts = x.copy()
        counts.data[:] = np.arange(1, 7)
        age = x.copy()
        age.data[:] = [1., 5., 10., 30., 60., 120.]
        self.data = SimpleNamespace(x=x, counts=counts, age_days=age)

    def test_ease_constrained_normal_equations(self):
        model = EASE(regularization=2.).fit(self.data)
        x = self.data.x.toarray()
        for j in range(x.shape[1]):
            other = np.arange(x.shape[1]) != j
            # Independent ridge regression on all columns except the target.
            expected = np.linalg.solve(x[:, other].T @ x[:, other] + 2*np.eye(2),
                                       x[:, other].T @ x[:, j])
            np.testing.assert_allclose(model.weights[other, j], expected, atol=1e-12)
        np.testing.assert_array_equal(np.diag(model.weights), 0)

    def test_als_update_matches_dense_weighted_problem(self):
        model = ImplicitALS(factors=2, regularization=0.7)
        y = np.array([[0.2, 0.7], [0.5, 0.3], [0.9, -0.1]])
        c = self.data.counts.copy()
        c.data = 1 + 20*np.log1p(c.data)
        p = CustomTargetALS().targets(self.data)
        result = model._update(c, p, y)
        for u in range(3):
            weights = np.ones(3)
            weights[c[u].indices] = c[u].data
            expected = np.linalg.solve(y.T @ np.diag(weights) @ y + 0.7*np.eye(2),
                                       y.T @ (weights*p[u].toarray().ravel()))
            np.testing.assert_allclose(result[u], expected, atol=1e-12)

    def test_custom_target_changes_preferences(self):
        base = ImplicitALS().targets(self.data)
        custom = CustomTargetALS().targets(self.data)
        self.assertFalse(np.allclose(base.data, custom.data))
        self.assertTrue(np.all((custom.data > 0) & (custom.data <= 1)))
        np.testing.assert_array_equal(base.indices, custom.indices)

    def test_item_cosine_direction(self):
        model = ItemKNN(neighbors=2, shrink=1.).fit(self.data)
        expected = np.full((3, 3), 1/3)
        np.fill_diagonal(expected, 0)
        np.testing.assert_allclose(model.weights.toarray(), expected)

    def test_fm_equals_explicit_pair_sum(self):
        torch.manual_seed(42)
        model = FMNetwork(6, 3)
        f = torch.tensor([[1, 2, 3, 0], [2, 4, 5, 0]])
        actual = model(f)
        expected = model.bias + model.linear(f).sum(dim=(1, 2))
        v = model.latent(f)
        for i in range(f.shape[1]):
            for j in range(i + 1, f.shape[1]):
                expected = expected + (v[:, i]*v[:, j]).sum(dim=1)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(model.latent.weight.grad).all())

    def test_metrics_hand_example(self):
        scores = ranking_metrics([3, 7, 9, 1, 2, 4, 5, 6, 8, 10], {3, 9, 100})
        ideal = 1 + 1/np.log2(3) + 1/2
        self.assertAlmostEqual(scores['NDCG@10'], 1.5/ideal)
        self.assertAlmostEqual(scores['Recall@10'], 2/3)
        self.assertAlmostEqual(scores['Precision@10'], 0.2)

    def test_seen_exclusion_and_deterministic_ties(self):
        class EqualScores(Recommender):
            def score(self, users):
                return np.ones((len(users), 15))
        seen = sparse.csr_matrix(([1., 1.], ([0, 0], [0, 2])), shape=(1, 15))
        rec = EqualScores().recommend([0], seen)[0]
        np.testing.assert_array_equal(rec, [1, 3, 4, 5, 6, 7, 8, 9, 10, 11])


if __name__ == "__main__":
    unittest.main(verbosity=2)
