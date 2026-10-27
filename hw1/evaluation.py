"""Ranking metrics, exhaustive candidate evaluation, paired uncertainty."""
import numpy as np
import pandas as pd


def ranking_metrics(recommendations, relevant, k=10):
    hits = np.array([item in relevant for item in recommendations[:k]], dtype=float)
    dcg = np.sum(hits / np.log2(np.arange(2, len(hits) + 2)))
    ideal = np.sum(1 / np.log2(np.arange(2, min(k, len(relevant)) + 2)))
    return {"NDCG@10": float(dcg / ideal), "Recall@10": float(hits.sum() / len(relevant)),
            "Precision@10": float(hits.sum() / k)}


def evaluate(model, data, truth, batch_size=64):
    users = np.array(sorted(truth), dtype=np.int64)
    records, catalog = [], set()
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        recommendations = model.recommend(batch, data.x, k=10)
        for u, rec in zip(batch, recommendations):
            assert len(set(rec)) == 10
            assert data.x[u, rec].sum() == 0
            catalog.update(rec)
            records.append({"user": int(u), **ranking_metrics(rec, truth[u])})
    per_user = pd.DataFrame(records).set_index("user")
    result = per_user.mean().to_dict()
    result["Coverage@10"] = len(catalog) / len(data.items)
    result["users"] = len(users)
    return result, per_user


def paired_bootstrap(left, right, metric="NDCG@10", seed=42, repeats=2000):
    assert left.index.equals(right.index)
    delta = (left[metric] - right[metric]).to_numpy()
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(repeats)])
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean_difference": float(delta.mean()), "ci95_low": float(lo), "ci95_high": float(hi)}
