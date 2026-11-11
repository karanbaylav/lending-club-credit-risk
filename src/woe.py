"""Weight-of-evidence binning fitted on training data only.

WoE convention here: WoE = ln(bad share / good share), so a higher WoE means riskier and the
logistic coefficients on WoE features should be positive."""
import numpy as np
import pandas as pd

EPS = 0.5  # added to bad and good counts so empty cells do not give infinite WoE


def _stats(bad, n):
    bad = np.asarray(bad, float)
    good = np.asarray(n, float) - bad
    return bad, good


def _woe(bad, good, tot_bad, tot_good):
    return np.log(((bad + EPS) / tot_bad) / ((good + EPS) / tot_good))


class NumericBinner:
    """Quantile bins, then merge neighbours until the bad rate follows the expected direction
    (or, with no prior, until neighbours differ by at least `min_gap` in log-odds). Missing = own bin."""

    def __init__(self, n_bins=10, direction=0, min_gap=0.05, min_share=0.02):
        self.n_bins, self.direction, self.min_gap, self.min_share = n_bins, direction, min_gap, min_share

    def fit(self, x, y):
        x, y = pd.Series(x).astype(float), pd.Series(y).astype(float)
        ok = x.notna()
        qs = np.unique(np.nanquantile(x[ok], np.linspace(0, 1, self.n_bins + 1)[1:-1]))
        edges = [-np.inf, *qs, np.inf]
        while True:
            cut = pd.cut(x[ok], edges, right=True)
            g = y[ok].groupby(cut, observed=False).agg(["sum", "count"])
            rate = (g["sum"] + EPS) / (g["count"] + 2 * EPS)
            lo = np.log(rate / (1 - rate)).to_numpy()
            share = (g["count"] / ok.sum()).to_numpy()
            if len(lo) <= 1:
                break
            diffs = np.diff(lo)
            if self.direction > 0:
                bad = np.where(diffs < 0)[0]
            elif self.direction < 0:
                bad = np.where(diffs > 0)[0]
            else:
                bad = np.where(np.abs(diffs) < self.min_gap)[0]
            small = np.where(share < self.min_share)[0]
            if len(bad):
                j = bad[np.argmin(np.abs(diffs[bad]))]          # merge the closest violating pair
            elif len(small):
                k = small[0]
                j = k if k < len(lo) - 1 else k - 1
            else:
                break
            edges = edges[: j + 1] + edges[j + 2:]
        self.edges = edges
        tot_bad, tot_good = y.sum(), len(y) - y.sum()
        cut = pd.cut(x, edges, right=True).cat.add_categories("MISSING").fillna("MISSING")
        g = y.groupby(cut, observed=False).agg(["sum", "count"])
        bad_, good_ = _stats(g["sum"], g["count"])
        self.table = pd.DataFrame({"bin": g.index.astype(str), "loans": g["count"].to_numpy(),
                                   "bad_rate": (g["sum"] / g["count"].replace(0, np.nan)).to_numpy(),
                                   "woe": _woe(bad_, good_, tot_bad, tot_good)})
        self.table["iv_part"] = ((bad_ / tot_bad) - (good_ / tot_good)) * self.table.woe
        self.iv = float(self.table.iv_part.sum())
        self.map = dict(zip(self.table.bin, self.table.woe))
        if self.table.loc[self.table.bin == "MISSING", "loans"].iloc[0] == 0:
            self.map["MISSING"] = 0.0
        return self

    def transform(self, x):
        cut = pd.cut(pd.Series(x).astype(float), self.edges, right=True).astype(str).replace("nan", "MISSING")
        return cut.map(self.map).astype(float).fillna(0.0).to_numpy()


class CategoricalBinner:
    """One bin per category; categories under `min_share` of training rows are pooled as OTHER."""

    def __init__(self, min_share=0.005):
        self.min_share = min_share

    def fit(self, x, y):
        x, y = pd.Series(x).astype(str), pd.Series(y).astype(float)
        share = x.value_counts(normalize=True)
        self.keep = set(share[share >= self.min_share].index)
        xx = x.where(x.isin(self.keep), "OTHER")
        tot_bad, tot_good = y.sum(), len(y) - y.sum()
        g = y.groupby(xx).agg(["sum", "count"])
        bad_, good_ = _stats(g["sum"], g["count"])
        self.table = pd.DataFrame({"bin": g.index, "loans": g["count"].to_numpy(),
                                   "bad_rate": (g["sum"] / g["count"]).to_numpy(),
                                   "woe": _woe(bad_, good_, tot_bad, tot_good)})
        self.table["iv_part"] = ((bad_ / tot_bad) - (good_ / tot_good)) * self.table.woe
        self.iv = float(self.table.iv_part.sum())
        self.map = dict(zip(self.table.bin, self.table.woe))
        return self

    def transform(self, x):
        xx = pd.Series(x).astype(str)
        xx = xx.where(xx.isin(self.keep), "OTHER")
        return xx.map(self.map).astype(float).fillna(0.0).to_numpy()
