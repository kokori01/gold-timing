"""The two Phase 8X bugs, written as tests BEFORE any strategy code ran.

These are the reason this repo exists rather than a restored AKOCT. Neither
bug can silently return.
"""
import numpy as np
import pandas as pd
import pytest

from goldtiming.data import tradable_returns
from goldtiming.gate import evaluate


# --------------------------------------------------------------- Phase 8X bug 1
def test_roll_gap_never_enters_pnl():
    """A spliced continuous series jumps at roll boundaries. Those jumps are
    not tradable. Booking them as P&L manufactured false momentum profit."""
    idx = pd.bdate_range("2020-01-01", periods=20)
    px = pd.Series(np.linspace(100, 102, 20), index=idx)
    roll = idx[10]
    px.iloc[10:] += 15.0                      # artificial roll gap

    raw = px.pct_change().fillna(0.0)
    safe = tradable_returns(px, roll_dates=[roll])

    assert raw.loc[roll] > 0.10, "test setup: the gap should be large"
    assert safe.loc[roll] == 0.0, "roll-boundary return leaked into P&L"
    assert (1 + safe).prod() < (1 + raw).prod(), "roll gap still inflating returns"


def test_tradable_returns_untouched_without_rolls():
    idx = pd.bdate_range("2020-01-01", periods=10)
    px = pd.Series(np.linspace(100, 110, 10), index=idx)
    pd.testing.assert_series_equal(
        tradable_returns(px), px.pct_change().fillna(0.0), check_names=False
    )


# --------------------------------------------------------------- Phase 8X bug 2
def test_gate_rejects_partial_pass():
    """Phase 8F passed 50/50 on MaxDD alone while failing CAGR and Sharpe.
    A candidate that wins one criterion and loses another must be rejected."""
    good_dd_bad_return = {
        "capture_spread": 0.30,        # passes
        "excess_at_equal_dd": -0.02,   # FAILS
        "calmar": 0.40,
    }
    bench = {"calmar": 0.20}
    res = evaluate(good_dd_bad_return, bench)
    assert not res.passed, "gate accepted a candidate failing excess_at_equal_dd"
    assert res.checks["capture_spread"] is True
    assert res.checks["excess_at_equal_dd"] is False


def test_gate_requires_all_criteria():
    bench = {"calmar": 0.20}
    passing = {"capture_spread": 0.30, "excess_at_equal_dd": 0.03, "calmar": 0.40}
    assert evaluate(passing, bench).passed
    for field in ("capture_spread", "excess_at_equal_dd", "calmar"):
        broken = dict(passing)
        broken[field] = -1.0
        assert not evaluate(broken, bench).passed, f"{field} was not binding"


def test_gate_missing_metric_is_failure_not_pass():
    """A metric that was never computed must not count as satisfied."""
    assert not evaluate({}, {"calmar": 0.2}).passed
