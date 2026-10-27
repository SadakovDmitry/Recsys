"""Second-order FM and pointwise negative-sampling training from scratch."""
import numpy as np
import torch
from torch import nn
from .base import Recommender


class FMNetwork(nn.Module):
    def __init__(self, n_features, factors):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.linear = nn.Embedding(n_features, 1, padding_idx=0)
        self.latent = nn.Embedding(n_features, factors, padding_idx=0)
        nn.init.zeros_(self.linear.weight)
        nn.init.normal_(self.latent.weight, std=0.02)
        with torch.no_grad():
            self.latent.weight[0].zero_()

    def forward(self, features):
        v = self.latent(features)
        pairwise = 0.5 * (v.sum(dim=1).square() - v.square().sum(dim=1)).sum(dim=1)
        return self.bias + self.linear(features).sum(dim=(1, 2)) + pairwise


class FactorizationMachine(Recommender):
    def __init__(self, factors=16, epochs=8, learning_rate=0.01,
                 regularization=1e-5, negatives=3, batch_size=8192,
                 seed=42, metadata=True):
        self.factors, self.epochs = factors, epochs
        self.learning_rate, self.regularization = learning_rate, regularization
        self.negatives, self.batch_size = negatives, batch_size
        self.seed, self.metadata = seed, metadata

    def _features(self, users, items):
        ids = [users[:, None] + 1, items[:, None] + self.n_users + 1]
        if self.metadata:
            ids.extend([self.user_meta[users], self.item_meta[items]])
        return torch.cat(ids, dim=1)

    def fit(self, data):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.n_users, self.n_items = data.x.shape
        self.user_meta = torch.from_numpy(data.user_meta)
        self.item_meta = torch.from_numpy(data.item_meta)
        n_features = data.n_features if self.metadata else 1 + self.n_users + self.n_items
        self.network = FMNetwork(n_features, self.factors)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate,
                                     weight_decay=self.regularization)
        pos_users, pos_items = data.x.nonzero()
        known = data.x.toarray().astype(bool)
        if known.all(axis=1).any():
            raise ValueError("A user has no available negative items")
        self.losses = []
        self.network.train()
        for _ in range(self.epochs):
            neg_users = np.repeat(pos_users, self.negatives)
            neg_items = rng.integers(self.n_items, size=len(neg_users))
            bad = known[neg_users, neg_items]
            while bad.any():
                neg_items[bad] = rng.integers(self.n_items, size=int(bad.sum()))
                bad = known[neg_users, neg_items]
            users = torch.from_numpy(np.r_[pos_users, neg_users].astype(np.int64))
            items = torch.from_numpy(np.r_[pos_items, neg_items].astype(np.int64))
            labels = torch.from_numpy(np.r_[np.ones(len(pos_users)),
                                            np.zeros(len(neg_users))].astype(np.float32))
            order = rng.permutation(len(users))
            total_loss = 0.0
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                logits = self.network(self._features(users[idx], items[idx]))
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(idx)
            self.losses.append(total_loss / len(order))
        self.network.eval()
        return self

    def score(self, users):
        users = np.asarray(users, dtype=np.int64)
        result = np.empty((len(users), self.n_items), dtype=np.float32)
        with torch.no_grad():
            # Limit temporary embedding tensors during full-catalog evaluation.
            flat = result.reshape(-1)
            for start in range(0, len(flat), self.batch_size):
                positions = np.arange(start, min(start + self.batch_size, len(flat)))
                u = torch.from_numpy(users[positions // self.n_items])
                i = torch.from_numpy(positions % self.n_items)
                flat[start:start + len(positions)] = self.network(self._features(u, i)).numpy()
        return result
