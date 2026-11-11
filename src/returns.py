"""Annualised realised return per loan (an approximate IRR).

Cash flows are rebuilt from Lending Club's totals, because the file has no payment history:
- payments other than recoveries, P = total_pymnt - recoveries, are spread as the scheduled
  instalment each month up to the last payment month k, with the remainder in month k
  (or spread evenly over k months if fewer than k instalments were paid);
- recoveries net of the collection fee arrive 6 months after the last payment.
The monthly IRR solves NPV = 0 by bisection; annualised = (1 + r)^12 - 1.
Ignores Lending Club's investor servicing fee and reinvestment; this is a simplification,
not the investor return Lending Club reported."""
import numpy as np

RECOVERY_LAG = 6


def _annuity(v, n):
    """sum_{t=1}^{n} v^t, safe for v -> 1."""
    v = np.asarray(v, dtype=float)
    n = np.asarray(n, dtype=float)
    out = np.where(np.abs(1 - v) < 1e-12, n, v * (1 - v ** n) / np.where(np.abs(1 - v) < 1e-12, 1, 1 - v))
    return out


def loan_irr(funded, installment, total_pymnt, recoveries, recovery_fee, last_age, iters=80):
    funded, inst = np.asarray(funded, float), np.asarray(installment, float)
    pay = np.asarray(total_pymnt, float) - np.asarray(recoveries, float)
    rec = np.clip(np.asarray(recoveries, float) - np.asarray(recovery_fee, float), 0, None)
    k = np.maximum(np.asarray(last_age, float), 1)
    regular = np.minimum(inst, pay / k)                       # level payment actually sustained
    n_reg = k - 1
    final = pay - regular * n_reg

    def npv(r):
        v = 1 / (1 + r)
        return -funded + regular * _annuity(v, n_reg) + final * v ** k + rec * v ** (k + RECOVERY_LAG)

    lo, hi = np.full(funded.shape, -0.999), np.full(funded.shape, 0.5)
    for _ in range(iters):
        mid = (lo + hi) / 2
        pos = npv(mid) > 0                                     # NPV falls as r rises
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    r = (lo + hi) / 2
    return (1 + r) ** 12 - 1
