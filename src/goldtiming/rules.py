"""Exposure rules.

Contract for every rule: given a daily price series, return a daily exposure
series where exposure[t] is the position HELD during day t, decided using
information available at t-1. Every rule ends in .shift(1); tests/test_no_lookahead
enforces it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _monthly_hold(signal_me: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Month-end decision, held through the next month, lagged one day."""
    return signal_me.reindex(index, method="ffill").shift(1).fillna(0.0)


def realized_vol(prices: pd.Series, window: int = 60) -> pd.Series:
    return prices.pct_change().rolling(window).std() * np.sqrt(TRADING_DAYS)


def downside_vol(prices: pd.Series, window: int = 60) -> pd.Series:
    """Semi-deviation of negative returns, scaled to be comparable with total vol.

    This is the fix for the objection that total volatility is direction-blind:
    a violent rally raises sigma and cuts the position for no reason. Only
    downside dispersion should cut it.
    """
    r = prices.pct_change()
    neg = r.where(r < 0.0, 0.0)
    semi = np.sqrt((neg ** 2).rolling(window).mean() * TRADING_DAYS)
    return semi * np.sqrt(2.0)          # symmetric-case parity with realized_vol


# --------------------------------------------------------------------------- rules

def buy_hold(prices: pd.Series) -> pd.Series:
    return pd.Series(1.0, index=prices.index)


def sma_binary(prices: pd.Series, months: int = 10) -> pd.Series:
    """Faber: month-end close vs its N-month SMA, binary in/out."""
    me = prices.resample("ME").last()
    sig = (me > me.rolling(months).mean()).astype(float)
    return _monthly_hold(sig, prices.index)


def vol_target(prices: pd.Series, target: float = 0.15, window: int = 60,
               cap: float = 1.5, monthly: bool = True) -> pd.Series:
    """Total-volatility targeting. Included as the control that the
    direction-blindness objection predicts will underperform on gold."""
    expo = (target / realized_vol(prices, window)).clip(upper=cap)
    if monthly:
        return _monthly_hold(expo.resample("ME").last(), prices.index)
    return expo.shift(1).fillna(0.0)


def downside_vol_target(prices: pd.Series, target: float = 0.15, window: int = 60,
                        cap: float = 1.5, monthly: bool = True) -> pd.Series:
    expo = (target / downside_vol(prices, window)).clip(upper=cap)
    if monthly:
        return _monthly_hold(expo.resample("ME").last(), prices.index)
    return expo.shift(1).fillna(0.0)


def asymmetric(prices: pd.Series, dd_limit: float = 0.10, entry_window: int = 63,
               reduced: float = 0.40, dvol_window: int = 60,
               dvol_mult: float = 1.5) -> pd.Series:
    """Cut on downside stress, restore on PRICE, not on volatility.

    This is the direct answer to the recovery-lag objection. A V-shaped rebound
    makes new highs while realized vol is still carrying the crash inside its
    window, so any vol-based re-entry stays small through the best part of the
    move. Re-entering on a price high decouples the two decisions:

        risk-off  <- drawdown from trailing peak, or a downside-vol spike
        risk-on   <- price reclaims its N-day high, regardless of vol
    """
    dvol = downside_vol(prices, dvol_window)
    dvol_ref = dvol.rolling(TRADING_DAYS).median()
    peak = prices.cummax()
    dd = prices / peak - 1.0
    hi = prices.rolling(entry_window).max()

    stress = (dd < -dd_limit) | (dvol > dvol_mult * dvol_ref)
    reclaim = prices >= hi

    state = np.ones(len(prices))
    cur = 1.0
    s_arr, r_arr = stress.to_numpy(), reclaim.to_numpy()
    for i in range(len(prices)):
        if cur == 1.0 and s_arr[i]:
            cur = reduced
        elif cur != 1.0 and r_arr[i]:
            cur = 1.0
        state[i] = cur
    return pd.Series(state, index=prices.index).shift(1).fillna(1.0)


def core_overlay(prices: pd.Series, core: float = 0.60, months: int = 10) -> pd.Series:
    """Never fully exit. Core is permanent so the train is never missed;
    the overlay is the only part timing can switch off."""
    return core + (1.0 - core) * sma_binary(prices, months)


def core_overlay_dvol(prices: pd.Series, core: float = 0.60, target: float = 0.15,
                      window: int = 60, cap: float = 1.0) -> pd.Series:
    return core + (1.0 - core) * downside_vol_target(
        prices, target=target, window=window, cap=cap
    )


REGISTRY = {
    "A_buy_hold":          buy_hold,
    "B_sma10_binary":      sma_binary,
    "C_vol_target":        vol_target,
    "D_downside_vol":      downside_vol_target,
    "E_asymmetric":        asymmetric,
    "F_core_overlay":      core_overlay,
    "G_core_dvol":         core_overlay_dvol,
}


def apply_rule(prices: pd.Series, name: str, cost_bps: float = 5.0, **params):
    """Return (net_returns, exposure). Turnover charged at cost_bps per unit traded."""
    expo = REGISTRY[name](prices, **params).reindex(prices.index).fillna(0.0)
    asset_r = prices.pct_change().fillna(0.0)
    gross = expo * asset_r
    turnover = expo.diff().abs().fillna(0.0)
    net = gross - turnover * cost_bps / 10_000.0
    return net, expo
