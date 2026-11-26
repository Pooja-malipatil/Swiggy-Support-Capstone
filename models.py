"""Shared model code for Parts 2 and 4 (importing this file does not run a Part)."""
import numpy as np
from common import load_data, sample_sd


def stratified_split(outcome):
    """Per class, the first floor(7*count/10) rows (file order) go to train; the rest to test."""
    train = np.zeros(len(outcome), dtype=bool)
    for cls in (0, 1):
        idx = np.where(outcome == cls)[0]
        k = (7 * len(idx)) // 10
        train[idx[:k]] = True
    return np.where(train)[0], np.where(~train)[0]


def get_split():
    """Return raw and standardised train/test data. Statistics come from the training rows only."""
    X, outcome, value = load_data()
    tr, te = stratified_split(outcome)
    mean = X[tr].mean(axis=0)
    sd = np.array([sample_sd(X[tr][:, j]) for j in range(7)])
    return {
        "X_raw_tr": X[tr], "X_raw_te": X[te],
        "Xtr": (X[tr] - mean) / sd, "Xte": (X[te] - mean) / sd,
        "ytr": outcome[tr], "yte": outcome[te],
    }


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce(X, y, w, b):
    p = sigmoid(X @ w + b)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train_logistic(X, y, eta, updates, record=()):
    """Batch gradient descent from zero. Returns w, b and a dict of losses at the recorded steps."""
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    losses = {0: bce(X, y, w, b)}
    for step in range(1, updates + 1):
        err = sigmoid(X @ w + b) - y
        gw = X.T @ err / n
        gb = err.sum() / n
        w = w - eta * gw
        b = b - eta * gb
        if step in record:
            losses[step] = bce(X, y, w, b)
    return w, b, losses


def predict_proba(X, w, b):
    return sigmoid(X @ w + b)


def confusion(pred, y):
    pred = np.asarray(pred).astype(bool)
    y = np.asarray(y).astype(bool)
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    tn = int(np.sum(~pred & ~y))
    fn = int(np.sum(~pred & y))
    return tp, fp, tn, fn


def safe_div(a, b):
    return a / b if b else 0.0


def metrics(tp, fp, tn, fn):
    acc = safe_div(tp + tn, tp + fp + tn + fn)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    f1 = safe_div(2 * prec * rec, prec + rec)
    return {"accuracy": acc, "precision": prec, "recall": rec, "specificity": spec, "f1": f1,
            "fpr": safe_div(fp, fp + tn)}


def knn_neighbour_share(ref_X, ref_y, query_X, k):
    """Share of positive neighbours among the K nearest (ties broken by reference file order)."""
    shares = np.empty(len(query_X))
    for i, q in enumerate(query_X):
        d2 = np.sum((ref_X - q) ** 2, axis=1)
        order = np.argsort(d2, kind="stable")[:k]
        shares[i] = ref_y[order].sum() / k
    return shares


def knn_predict(ref_X, ref_y, query_X, k):
    share = knn_neighbour_share(ref_X, ref_y, query_X, k)
    # positive when at least (K+1)/2 neighbours are positive
    pred = (share * k) >= (k + 1) / 2 - 1e-9
    return pred, share


def choose_knn_k(Xtr, ytr):
    """5-fold CV on the training rows (fold = position mod 5). Returns CV rows and best K (ties -> smaller K)."""
    fold = np.arange(len(ytr)) % 5
    cv_rows = []
    for k in range(1, 22, 2):
        f1s = []
        for f in range(5):
            ref, held = fold != f, fold == f
            pred, _ = knn_predict(Xtr[ref], ytr[ref], Xtr[held], k)
            f1s.append(metrics(*confusion(pred, ytr[held]))["f1"])
        cv_rows.append((k, float(np.mean(f1s))))
    best_k = max(cv_rows, key=lambda r: (r[1], -r[0]))[0]
    return cv_rows, best_k
