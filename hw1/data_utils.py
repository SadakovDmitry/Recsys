"""Download, cleaning, time splits and features with train-only fitting."""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from scipy import sparse

SOURCE = "https://archive.ics.uci.edu/dataset/352/online+retail"
DOWNLOAD = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
EXPECTED_SHA256 = "f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b"
TRAIN_END = pd.Timestamp("2011-10-01")
VALID_END = pd.Timestamp("2011-11-01")


def load_raw(cache=Path(".cache")):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "online_retail.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".part")
        urllib.request.urlretrieve(DOWNLOAD, temporary)
        temporary.replace(archive)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    if checksum != EXPECTED_SHA256:
        raise ValueError("Source archive differs from the version used in this experiment")
    # CSV cache avoids parsing the XLSX repeatedly; its name includes source hash.
    csv = cache / ("raw_" + checksum[:16] + ".csv.gz")
    if csv.exists():
        frame = pd.read_csv(csv, dtype={"InvoiceNo": str, "StockCode": str},
                            parse_dates=["InvoiceDate"])
    else:
        with zipfile.ZipFile(archive) as z:
            frame = pd.read_excel(z.open("Online Retail.xlsx"), engine="openpyxl",
                                  dtype={"InvoiceNo": str, "StockCode": str})
        frame.to_csv(csv, index=False, compression="gzip")
    return frame, checksum


def clean_data(raw):
    steps = []
    frame = raw.copy()
    def record(name):
        steps.append({"step": name, "rows": len(frame),
                      "users": frame.CustomerID.nunique(),
                      "items": frame.StockCode.nunique()})
    record("Исходные строки")
    frame = frame.drop_duplicates()
    record("Без точных дубликатов")
    frame = frame.dropna(subset=["CustomerID", "StockCode", "InvoiceDate"])
    record("Есть пользователь, товар и время")
    frame = frame.loc[(frame.Quantity > 0) & (frame.UnitPrice > 0)
                      & ~frame.InvoiceNo.astype(str).str.upper().str.startswith("C")].copy()
    record("Положительные покупки без отмен")
    frame["StockCode"] = frame.StockCode.astype(str).str.strip().str.upper()
    frame = frame.loc[frame.StockCode.str.fullmatch(r"\d{5}[A-Z]{0,2}")].copy()
    record("Только товарные артикулы")
    frame["CustomerID"] = frame.CustomerID.astype(np.int64).astype(str)
    frame["Description"] = frame.Description.fillna("UNKNOWN").str.strip().str.upper()
    frame["Country"] = frame.Country.fillna("UNKNOWN").str.strip()
    frame = frame.sort_values(["InvoiceDate", "InvoiceNo", "CustomerID", "StockCode"],
                              kind="stable").reset_index(drop=True)
    assert 100_000 <= len(frame) <= 2_000_000
    assert frame.CustomerID.nunique() >= 1000 and frame.StockCode.nunique() >= 1000
    return frame, pd.DataFrame(steps)


def split_data(clean):
    train = clean.loc[clean.InvoiceDate < TRAIN_END].copy()
    valid = clean.loc[(clean.InvoiceDate >= TRAIN_END) & (clean.InvoiceDate < VALID_END)].copy()
    test = clean.loc[clean.InvoiceDate >= VALID_END].copy()
    assert train.InvoiceDate.max() < valid.InvoiceDate.min()
    assert valid.InvoiceDate.max() < test.InvoiceDate.min()
    return train, valid, test


@dataclass
class TrainingData:
    x: sparse.csr_matrix
    counts: sparse.csr_matrix
    age_days: sparse.csr_matrix
    users: list
    items: list
    user_map: dict
    item_map: dict
    user_meta: np.ndarray
    item_meta: np.ndarray
    n_features: int
    item_descriptions: list
    feature_summary: dict


def build_training(frame, cutoff):
    assert frame.InvoiceDate.max() < cutoff
    users = sorted(frame.CustomerID.unique())
    items = sorted(frame.StockCode.unique())
    um, im = {v: i for i, v in enumerate(users)}, {v: i for i, v in enumerate(items)}
    pairs = frame.groupby(["CustomerID", "StockCode"], sort=True).agg(
        count=("InvoiceNo", "nunique"), last=("InvoiceDate", "max")).reset_index()
    rows = pairs.CustomerID.map(um).to_numpy()
    cols = pairs.StockCode.map(im).to_numpy()
    shape = (len(users), len(items))
    counts = sparse.csr_matrix((pairs["count"].to_numpy(dtype=float), (rows, cols)), shape=shape)
    age = (cutoff - pairs["last"]).dt.total_seconds().to_numpy() / 86400.0
    age_days = sparse.csr_matrix((age, (rows, cols)), shape=shape)
    x = counts.copy()
    x.data[:] = 1.0
    assert np.array_equal(counts.indices, age_days.indices)
    # Every feature below is estimated strictly from the training window.
    last_users = frame.sort_values("InvoiceDate", kind="stable").groupby("CustomerID").tail(1)
    country = last_users.set_index("CustomerID").Country.reindex(users)
    countries = sorted(country.unique())
    offset = 1 + len(users) + len(items)  # 0 is padding in FM.
    country_map = {v: offset + i for i, v in enumerate(countries)}
    user_meta = country.map(country_map).to_numpy(dtype=np.int64)[:, None]
    offset += len(countries)
    prices = frame.groupby("StockCode").UnitPrice.median().reindex(items).to_numpy()
    boundaries = np.unique(np.quantile(np.log1p(prices), [0.2, 0.4, 0.6, 0.8]))
    price_bins = np.searchsorted(boundaries, np.log1p(prices), side="right") + offset
    offset += len(boundaries) + 1
    descriptions = (frame.sort_values("InvoiceDate", kind="stable")
                    .groupby("StockCode").tail(1).set_index("StockCode")
                    .Description.reindex(items).tolist())
    # A fixed-size vocabulary of training product-title words, not a pretrained model.
    tokens = [list(dict.fromkeys(re.findall(r"[A-Z]{3,}", text))) for text in descriptions]
    frequencies = Counter(word for words in tokens for word in words)
    vocabulary = sorted([w for w in frequencies if frequencies[w] >= 3],
                        key=lambda w: (-frequencies[w], w))[:512]
    word_ids = {w: offset + i for i, w in enumerate(vocabulary)}
    item_meta = np.zeros((len(items), 7), dtype=np.int64)
    item_meta[:, 0] = price_bins
    for i, words in enumerate(tokens):
        ids = [word_ids[w] for w in words if w in word_ids][:6]
        item_meta[i, 1:1 + len(ids)] = ids
    return TrainingData(x, counts, age_days, users, items, um, im, user_meta,
                        item_meta, offset + len(vocabulary), descriptions,
                        {"countries": len(countries), "price_bins": len(boundaries) + 1,
                         "title_vocabulary": len(vocabulary), "title_words_max": 6,
                         "price_log_boundaries": boundaries.tolist(),
                         "fitted_before": str(cutoff)})


def evaluation_cohort(data, heldout, k=10):
    """Same cohort and exhaustive unseen candidates for every model in a stage."""
    unique = heldout[["CustomerID", "StockCode"]].drop_duplicates()
    warm_user = unique.CustomerID.isin(data.user_map)
    warm_item = unique.StockCode.isin(data.item_map)
    available = unique.loc[warm_user & warm_item]
    truth = {}
    already_known = 0
    for user, item in available.itertuples(index=False, name=None):
        u, i = data.user_map[user], data.item_map[item]
        if data.x[u, i] != 0:
            already_known += 1
        else:
            truth.setdefault(u, set()).add(i)
    excluded_few_candidates = sum(data.x.shape[1] - data.x[u].nnz < k for u in truth)
    truth = {u: v for u, v in sorted(truth.items()) if data.x.shape[1] - data.x[u].nnz >= k}
    report = {"heldout_rows": len(heldout), "heldout_users": heldout.CustomerID.nunique(),
              "heldout_unique_pairs": len(unique), "cold_user_pairs": int((~warm_user).sum()),
              "cold_item_pairs": int((~warm_item).sum()),
              "cold_user_or_item_pairs": int((~(warm_user & warm_item)).sum()),
              "seen_pairs_excluded": already_known, "evaluated_users": len(truth),
              "evaluated_relevant_pairs": sum(map(len, truth.values())),
              "users_with_fewer_than_k_candidates": int(excluded_few_candidates),
              "catalog_items": len(data.items)}
    return truth, report
