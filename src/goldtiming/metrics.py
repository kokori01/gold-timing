"""Metrics. Calmar and capture ratios are the primary gate, not Sharpe.

The stated objective is asymmetric: participate in rises, avoid giving back
profits in falls. Sharpe does not measure that. Capture ratios do.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329


def to_equity(returns: pd.Series) -> pd.Series:
    return (1.0 + returns.fillna(0.0)).cumprod()


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    eq = to_equity(returns)
    years = len(returns) / periods_per_year
    if years <= 0 or eq.iloc[-1] <= 0:
        return float("nan")
    return float(eq.iloc[-1] ** (1.0 / years) - 1.0)


def ann_vol(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Returned as a negative number."""
    eq = to_equity(returns)
    return float((eq / eq.cummax() - 1.0).min())


def calmar(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0:
        return float("nan")
    return cagr(returns, periods_per_year) / abs(mdd)


def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    ex = returns - rf / periods_per_year
    sd = ex.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(ex.mean() / sd * np.sqrt(periods_per_year))


def sharpe_se(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    """Lo (2002) standard error of an annualised Sharpe estimate.

    One year of daily data gives SE ~1.0 at SR~2. Report this next to every
    Sharpe so a single-year number is never mistaken for a measurement.
    """
    n = len(returns)
    if n < 3:
        return float("nan")
    sr_p = sharpe(returns, periods_per_year=periods_per_year) / np.sqrt(periods_per_year)
    se_p = np.sqrt((1.0 + 0.5 * sr_p ** 2) / n)
    return float(se_p * np.sqrt(periods_per_year))


def _geom(x: pd.Series) -> float:
    return float((1.0 + x).prod() ** (1.0 / len(x)) - 1.0) if len(x) else float("nan")


def capture(strategy: pd.Series, benchmark: pd.Series) -> dict:
    """Upside / downside capture on monthly returns.

    upside  : share of the benchmark's up-move you keep   (higher is better)
    downside: share of the benchmark's down-move you eat  (lower is better)

    The whole question is whether upside > downside by enough to pay for itself.
    """
    s = strategy.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    b = benchmark.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    s, b = s.align(b, join="inner")
    up, dn = b > 0, b < 0
    uc = _geom(s[up]) / _geom(b[up]) if up.sum() else float("nan")
    dc = _geom(s[dn]) / _geom(b[dn]) if dn.sum() else float("nan")
    return {
        "upside_capture": uc,
        "downside_capture": dc,
        "capture_spread": uc - dc,      # > 0 means the asymmetry works in your favour
        "n_up_months": int(up.sum()),
        "n_down_months": int(dn.sum()),
    }


def cagr_at_equal_drawdown(strategy: pd.Series, benchmark: pd.Series,
                           periods_per_year: int = TRADING_DAYS,
                           financing_rate: float = 0.0) -> float:
    """Lever the strategy to the benchmark's MaxDD, then report its CAGR.

    The only honest way to compare a de-risked strategy against buy-and-hold:
    without this, any strategy 'wins' on drawdown simply by holding less.

    `financing_rate` charges the borrowed portion. Leverage is not free -- a
    1.8x position funded at 5% costs ~4%/yr, which is larger than any edge
    measured here. Reporting the unfinanced number alone flatters every
    de-risked rule, so both are computed.
    """
    s_dd, b_dd = abs(max_drawdown(strategy)), abs(max_drawdown(benchmark))
    if s_dd == 0:
        return float("nan")
    lev = b_dd / s_dd
    levered = strategy * lev
    if financing_rate and lev > 1:
        levered = levered - (lev - 1.0) * financing_rate / periods_per_year
    return cagr(levered, periods_per_year)


def deflated_sharpe(returns: pd.Series, n_trials: int, trial_sr_var: float,
                    periods_per_year: int = TRADING_DAYS) -> float:
    """Bailey & Lopez de Prado DSR. Requires the honest trial count.

    n_trials comes from trials.jsonl, not from memory. This is precisely the
    number that six years of undocumented experimentation destroyed.
    """
    n = len(returns)
    if n < 3 or n_trials < 2:
        return float("nan")
    sr = sharpe(returns, periods_per_year=periods_per_year) / np.sqrt(periods_per_year)
    g3 = float(returns.skew())
    g4 = float(returns.kurtosis()) + 3.0            # pandas gives excess kurtosis
    sr0 = np.sqrt(trial_sr_var) * (
        (1 - EULER_GAMMA) * norm.ppf(1 - 1.0 / n_trials)
        + EULER_GAMMA * norm.ppf(1 - 1.0 / (n_trials * np.e))
    )
    denom = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2))
    return float(norm.cdf((sr - sr0) * np.sqrt(n - 1) / denom))


def summary(returns: pd.Series, benchmark: pd.Series | None = None,
            periods_per_year: int = TRADING_DAYS,
            financing_rate: float = 0.05) -> dict:
    out = {
        "cagr": cagr(returns, periods_per_year),
        "ann_vol": ann_vol(returns, periods_per_year),
        "max_dd": max_drawdown(returns),
        "calmar": calmar(returns, periods_per_year),
        "sharpe": sharpe(returns, periods_per_year=periods_per_year),
        "sharpe_se": sharpe_se(returns, periods_per_year),
        "n_obs": len(returns),
    }
    if benchmark is not None:
        out.update(capture(returns, benchmark))
        out["cagr_at_equal_dd"] = cagr_at_equal_drawdown(returns, benchmark, periods_per_year)
        out["cagr_at_equal_dd_financed"] = cagr_at_equal_drawdown(
            returns, benchmark, periods_per_year, financing_rate=financing_rate
        )
        out["implied_leverage"] = abs(max_drawdown(benchmark)) / abs(max_drawdown(returns))
        out["bench_cagr"] = cagr(benchmark, periods_per_year)
        out["excess_at_equal_dd"] = out["cagr_at_equal_dd_financed"] - out["bench_cagr"]
        out["excess_at_equal_dd_unfinanced"] = out["cagr_at_equal_dd"] - out["bench_cagr"]
    return out
