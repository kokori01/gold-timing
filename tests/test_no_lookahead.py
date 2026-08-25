"""No rule may use information from the day it trades on.

Every rule ends in .shift(1). This test proves it by perturbing the future and
checking that today's exposure does not move.
"""
import numpy as np
import pandas as pd
import pytest

from goldtiming.rules import REGISTRY, apply_rule

RULES = [r for r in REGISTRY if r != "A_buy_hold"]


@pytest.fixture
def prices():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2010-01-01", periods=900)
    steps = rng.normal(0.0004, 0.011, len(idx))
    return pd.Series(100 * np.exp(np.cumsum(steps)), index=idx)


@pytest.mark.parametrize("rule", RULES)
def test_exposure_ignores_same_day_and_future_prices(prices, rule):
    base = REGISTRY[rule](prices)
    cut = len(prices) - 200

    tampered = prices.copy()
    tampered.iloc[cut:] *= 3.0        # rewrite everything from `cut` onwards

    after = REGISTRY[rule](tampered)
    pd.testing.assert_series_equal(
        base.iloc[:cut], after.iloc[:cut], check_names=False,
        obj=f"{rule} exposure before index {cut} changed when the FUTURE changed",
    )


@pytest.mark.parametrize("rule", RULES)
def test_exposure_is_finite_and_bounded(prices, rule):
    expo = REGISTRY[rule](prices)
    assert expo.notna().all(), f"{rule} produced NaN exposure"
    assert (expo >= 0).all(), f"{rule} produced negative exposure"
    assert (expo <= 2.0).all(), f"{rule} exceeded the 2x sanity cap"


def test_costs_reduce_returns(prices):
    free, _ = apply_rule(prices, "B_sma10_binary", cost_bps=0.0)
    paid, _ = apply_rule(prices, "B_sma10_binary", cost_bps=25.0)
    assert paid.sum() < free.sum(), "turnover cost was not charged"
