import numpy as np

from src.validation import gini, ks, psi
from src.woe import CategoricalBinner, NumericBinner


def test_monotone_binning_follows_direction():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 40, 20000)
    y = rng.random(20000) < 0.03 + 0.004 * x          # risk rises with x
    b = NumericBinner(direction=1).fit(x, y)
    w = b.table[b.table.bin != "MISSING"].woe.to_numpy()
    assert np.all(np.diff(w) >= -1e-9)
    assert b.iv > 0


def test_missing_gets_its_own_woe():
    x = np.array([1.0, 2.0, np.nan, np.nan] * 500)
    y = np.array([0, 0, 1, 1] * 500)
    b = NumericBinner(n_bins=2).fit(x, y)
    assert b.transform([np.nan])[0] > b.transform([1.0])[0]


def test_categorical_other_pool():
    x = ["a"] * 990 + ["rare"] * 2 + ["b"] * 1008
    y = [0] * 990 + [1, 1] + [1] * 8 + [0] * 1000
    b = CategoricalBinner(min_share=0.01).fit(x, y)
    assert "OTHER" in set(b.table.bin)
    assert np.isfinite(b.transform(["unseen"])[0])


def test_metrics():
    y = np.array([0, 0, 1, 1])
    assert gini(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert ks(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    z = np.random.default_rng(1).normal(size=5000)
    assert psi(z, z) < 1e-9 and psi(z, z + 1) > 0.25
