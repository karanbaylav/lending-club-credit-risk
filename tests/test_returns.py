import numpy as np

from src.returns import loan_irr


def schedule_installment(p, annual_rate, n):
    r = annual_rate / 12
    return p * r / (1 - (1 + r) ** -n)


def test_full_schedule_returns_quoted_rate():
    inst = schedule_installment(10_000, 0.12, 36)
    a = loan_irr([10_000], [inst], [inst * 36], [0], [0], [36])[0]
    assert abs(a - ((1 + 0.01) ** 12 - 1)) < 1e-6          # monthly 1% compounds to 12.68%


def test_total_loss_near_minus_100():
    a = loan_irr([10_000], [300], [0], [0], [0], [1])[0]
    assert a < -0.99


def test_default_is_worse_than_payoff():
    inst = schedule_installment(10_000, 0.15, 36)
    paid = loan_irr([10_000], [inst], [inst * 36], [0], [0], [36])[0]
    dflt = loan_irr([10_000], [inst], [inst * 10 + 800], [800], [80], [10])[0]
    assert dflt < 0 < paid
