"""Predeclared search and one final held-out evaluation."""
from itertools import product
from pathlib import Path
import gc
import json
import time
import numpy as np
import pandas as pd
from models import ItemKNN, ImplicitALS, EASE, FactorizationMachine, CustomTargetALS
from evaluation import evaluate, paired_bootstrap

SEED = 42
PRIMARY = "NDCG@10"
MODEL_CLASSES = {"ItemKNN": ItemKNN, "iALS": ImplicitALS, "EASE": EASE,
                 "FM": FactorizationMachine, "CustomALS": CustomTargetALS}
ALS_GRID = [dict(factors=f, regularization=r, alpha=20.0, iterations=10, seed=SEED)
            for f, r in product([32, 64], [1.0, 10.0])]
SEARCH = {
    "ItemKNN": [dict(neighbors=n, shrink=s) for n, s in product([40, 100], [10.0, 100.0])],
    "iALS": ALS_GRID,
    "EASE": [dict(regularization=r) for r in [100.0, 500.0, 1000.0]],
    "FM": [dict(factors=f, epochs=e, learning_rate=0.01, regularization=1e-5,
                negatives=3, batch_size=8192, seed=SEED, metadata=True)
           for f, e in product([16, 32], [6, 12])],
    "CustomALS": [dict(x) for x in ALS_GRID],
}


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validation_search(data, truth, output=Path("results")):
    output = Path(output)
    output.mkdir(exist_ok=True)
    write_json(output / "search_space.json", {"seed": SEED, "primary_metric": PRIMARY,
                                              "configurations": SEARCH})
    rows, selected = [], {}
    for name, configurations in SEARCH.items():
        best = -np.inf
        for trial, params in enumerate(configurations, 1):
            start = time.perf_counter()
            model = MODEL_CLASSES[name](**params).fit(data)
            fit_seconds = time.perf_counter() - start
            metrics, _ = evaluate(model, data, truth)
            row = {"model": name, "trial": trial, "params": json.dumps(params, sort_keys=True),
                   "fit_seconds": fit_seconds, **metrics}
            rows.append(row)
            # Strict greater-than gives deterministic first-configuration tie breaking.
            if metrics[PRIMARY] > best:
                selected[name] = dict(params)
                best = metrics[PRIMARY]
            pd.DataFrame(rows).to_csv(output / "validation.csv", index=False)
            print(f"{name} {trial}/{len(configurations)}: NDCG={metrics[PRIMARY]:.6f}, "
                  f"Recall={metrics['Recall@10']:.6f}, fit={fit_seconds:.1f}s", flush=True)
            del model
            gc.collect()
        write_json(output / "selected_params.json", selected)
    return pd.DataFrame(rows), selected


def final_comparison(data, truth, selected, validation, output=Path("results")):
    output = Path(output)
    # Main five models, then two predeclared controlled experiments.
    trials = [(n, MODEL_CLASSES[n], p, "main") for n, p in selected.items()]
    trials += [("FM_ID_only", FactorizationMachine,
                {**selected["FM"], "metadata": False}, "ablation"),
               ("CustomALS_matched", CustomTargetALS, selected["iALS"], "ablation")]
    rows, per_user, examples, losses = [], {}, {}, {}
    example_users = np.array(sorted(truth)[:3], dtype=np.int64)
    for name, cls, params, role in trials:
        start = time.perf_counter()
        model = cls(**params).fit(data)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        metrics, per_user[name] = evaluate(model, data, truth)
        ranking_seconds = time.perf_counter() - start
        val = validation.loc[validation.model == name, PRIMARY]
        rows.append({"model": name, "role": role, "params": json.dumps(params, sort_keys=True),
                     "validation_NDCG@10": float(val.max()) if len(val) else np.nan,
                     "fit_seconds": fit_seconds, "ranking_seconds": ranking_seconds, **metrics})
        pd.DataFrame(rows).to_csv(output / "test_results.csv", index=False)
        per_user[name].to_csv(output / f"per_user_{name}.csv")
        for u, rec in zip(example_users, model.recommend(example_users, data.x)):
            examples[f"{name}:{data.users[u]}"] = [
                {"rank": rank, "stock_code": data.items[i], "title": data.item_descriptions[i],
                 "relevant_in_test": bool(i in truth[u])}
                for rank, i in enumerate(rec, 1)]
        if hasattr(model, "losses"):
            losses[name] = model.losses
        print(f"TEST {name}: NDCG={metrics[PRIMARY]:.6f}, Recall={metrics['Recall@10']:.6f}", flush=True)
        del model
        gc.collect()
    comparisons = {}
    for left, right in [("CustomALS", "iALS"), ("CustomALS_matched", "iALS"), ("FM", "FM_ID_only")]:
        comparisons[f"{left} minus {right}"] = {
            metric: paired_bootstrap(per_user[left], per_user[right], metric=metric)
            for metric in ["NDCG@10", "Recall@10"]}
    write_json(output / "paired_comparisons.json", comparisons)
    write_json(output / "recommendation_examples.json", examples)
    write_json(output / "fm_losses.json", losses)
    return pd.DataFrame(rows), comparisons, examples, losses
