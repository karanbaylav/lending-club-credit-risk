"""Validation metrics: AUC/Gini, KS, calibration by decile, PSI."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def ks(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def calibration_table(y, p, n=10):
    d = pd.DataFrame({"y": y, "p": p})
    d["decile"] = pd.qcut(d.p.rank(method="first"), n, labels=False) + 1
    return d.groupby("decile").agg(loans=("y", "size"), predicted_pd=("p", "mean"), observed_rate=("y", "mean")).reset_index()


def psi(expected, actual, bins=10):
    """Population stability index of `actual` against `expected`, deciles of `expected`."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(expected, actual):
    """PSI over the distinct values (bins) of a discrete variable such as a WoE feature."""
    e = pd.Series(expected).round(6).value_counts(normalize=True)
    a = pd.Series(actual).round(6).value_counts(normalize=True)
    idx = e.index.union(a.index)
    e, a = e.reindex(idx, fill_value=1e-6).clip(lower=1e-6), a.reindex(idx, fill_value=1e-6).clip(lower=1e-6)
    return float(np.sum((a - e) * np.log(a / e)))
