"""Known-answer tests for the metrics that decide everything."""
import numpy as np
import pandas as pd
import pytest

from goldtiming import metrics as M
from goldtiming.decompose import attribution, episodes, episode_summary


@pytest.fixture
def bench():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2010-01-01", periods=252 * 6)
    return pd.Series(rng.normal(0.0003, 0.010, len(idx)), index=idx)


def test_cagr_exact_on_constant_growth():
    idx = pd.bdate_range("2010-01-01", periods=252)
    r = pd.Series(0.0, index=idx)
    r.iloc[:] = 1.10 ** (1 / 252) - 1
    assert M.cagr(r) == pytest.approx(0.10, abs=1e-3)


def test_max_drawdown_exact():
    r = pd.Series([0.5, -0.5, 0.0], index=pd.bdate_range("2020-01-01", periods=3))
    # 1.0 -> 1.5 -> 0.75 : peak 1.5, trough 0.75 => -50%
    assert M.max_drawdown(r) == pytest.approx(-0.5)


def test_half_exposure_gives_half_capture(bench):
    """A constant 50% position captures 50% of both directions.
    Capture spread is then zero: no asymmetry, no value added."""
    half = bench * 0.5
    c = M.capture(half, bench)
    assert c["upside_capture"] == pytest.approx(0.5, abs=0.05)
    assert c["downside_capture"] == pytest.approx(0.5, abs=0.05)
    assert c["capture_spread"] == pytest.approx(0.0, abs=0.05)


def test_buy_hold_captures_everything(bench):
    c = M.capture(bench, bench)
    assert c["upside_capture"] == pytest.approx(1.0, abs=1e-6)
    assert c["downside_capture"] == pytest.approx(1.0, abs=1e-6)


def test_equal_drawdown_levers_a_derisked_strategy_back_up(bench):
    """Without this correction any strategy 'wins' on drawdown by holding less."""
    half = bench * 0.5
    assert M.cagr_at_equal_drawdown(half, bench) == pytest.approx(M.cagr(bench), rel=0.15)


def test_sharpe_se_flags_one_year_as_unmeasurable():
    """The Vertus arithmetic: at SR~2 on 252 daily obs, SE is about 1.0,
    so a single-year Sharpe cannot be distinguished from a mediocre one."""
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2025-01-01", periods=252)
    r = pd.Series(rng.normal(2.0 / 252, 1.0 / np.sqrt(252), len(idx)), index=idx)
    assert M.sharpe_se(r) == pytest.approx(1.0, abs=0.25)


def test_deflated_sharpe_falls_as_trials_rise(bench):
    good = bench + 0.0006
    few = M.deflated_sharpe(good, n_trials=5, trial_sr_var=0.25)
    many = M.deflated_sharpe(good, n_trials=5000, trial_sr_var=0.25)
    assert many < few, "DSR must penalise a larger search"


# ----------------------------------------------------------------- decomposition
def test_attribution_splits_protection_from_whipsaw():
    idx = pd.bdate_range("2020-01-01", periods=4)
    r = pd.Series([-0.10, 0.10, -0.10, 0.10], index=idx)
    expo = pd.Series([0.0, 0.0, 1.0, 1.0], index=idx)   # out for the first two days
    a = attribution(expo, r)
    assert a["protection_total"] == pytest.approx(0.10)   # dodged a -10% day
    assert a["whipsaw_total"] == pytest.approx(-0.10)     # missed a +10% day
    assert a["net_total"] == pytest.approx(0.0)


def test_episodes_classify_fake_breakdowns_as_whipsaw():
    idx = pd.bdate_range("2020-01-01", periods=6)
    px = pd.Series([100, 90, 95, 105, 110, 115], index=idx, dtype=float)
    expo = pd.Series([1.0, 0.0, 0.0, 1.0, 1.0, 1.0], index=idx)
    ep = episodes(expo, px)
    assert len(ep) == 1
    assert ep.iloc[0]["kind"] == "whipsaw"      # stepped out, price rose
    s = episode_summary(ep)
    assert s["n_whipsaw"] == 1 and s["n_protective"] == 0


# ------------------------------------------------------------------- insurance
def test_put_spread_pays_out_in_a_crash():
    """A deep decline must produce a positive net contribution, otherwise the
    expiry payoff is being dropped and every insurance result is meaningless."""
    import numpy as np
    from goldtiming.options import put_spread_overlay

    idx = pd.bdate_range("2010-01-01", periods=252 * 2)
    px = pd.Series(np.linspace(100, 55, len(idx)), index=idx)   # -45% grind
    iv = pd.Series(0.18, index=idx)
    ov = put_spread_overlay(px, iv, otm_long=0.10, otm_short=0.25, tenor_months=12)
    assert ov["payoff_received"].sum() > 0, "expiry payoff never collected"
    assert ov["pnl"].sum() > 0, "insurance lost money through a -45% decline"


def test_put_spread_costs_premium_in_a_rally():
    import numpy as np
    from goldtiming.options import put_spread_overlay

    idx = pd.bdate_range("2010-01-01", periods=252 * 2)
    px = pd.Series(np.linspace(100, 160, len(idx)), index=idx)
    iv = pd.Series(0.18, index=idx)
    ov = put_spread_overlay(px, iv, otm_long=0.10, otm_short=0.25, tenor_months=12)
    assert ov["pnl"].sum() < 0, "insurance should cost money in a straight rally"
    assert ov["premium_paid"].sum() > 0


def test_put_spread_flat_market_costs_exactly_the_premium():
    """In a flat market every spread expires worthless, so net P&L must equal
    minus the premium outlay -- not twice it. Charging the premium at entry AND
    letting it flow through the mark-to-market diffs double-counts each roll."""
    import numpy as np
    from goldtiming.options import put_spread_overlay

    idx = pd.bdate_range("2010-01-01", periods=252 * 5)
    px = pd.Series(100.0, index=idx)
    iv = pd.Series(0.18, index=idx)
    ov = put_spread_overlay(px, iv, otm_long=0.10, otm_short=0.25, tenor_months=12)
    assert ov["payoff_received"].sum() == pytest.approx(0.0, abs=1e-6)
    assert ov["pnl"].sum() == pytest.approx(-ov["premium_paid"].sum(), rel=1e-6)


def test_put_spread_net_pnl_is_payoff_minus_premium():
    import numpy as np
    from goldtiming.options import put_spread_overlay

    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2010-01-01", periods=252 * 8)
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.011, len(idx)))), index=idx)
    iv = pd.Series(0.18, index=idx)
    ov = put_spread_overlay(px, iv, otm_long=0.10, otm_short=0.25, tenor_months=12)
    expected = ov["payoff_received"].sum() - ov["premium_paid"].sum()
    assert ov["pnl"].sum() == pytest.approx(expected, abs=1e-6)
