"""The insurance leg.

The stated objective -- full participation on the way up, limited loss on the
way down -- is the payoff of a put. A timing rule SYNTHESISES that payoff and
pays for it in whipsaw tax: hidden, variable, and until now unmeasured. A put
spread BUYS the same payoff and pays for it in premium: explicit, known in
advance, and requiring no forecast.

This module prices the bought version so the two can be compared on one axis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252


def bs_put(S, K, T, sigma, r=0.02, q=0.0):
    """Black-Scholes put. Vectorised over S/K/T/sigma."""
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    T = np.maximum(T, 1e-6)
    sigma = np.maximum(sigma, 1e-6)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def build_iv(prices: pd.Series, gvz: pd.Series | None = None,
             vrp: float = 0.02, rv_window: int = 60) -> pd.DataFrame:
    """Implied-vol input, with provenance.

    Uses GVZ (real 30-day ATM gold IV) wherever it exists; before 2008-06 falls
    back to realized vol plus a variance-risk-premium markup. The `source`
    column records which, so a result can never be read as fully market-priced
    when part of it is modelled.
    """
    rv = prices.pct_change().rolling(rv_window).std() * np.sqrt(TRADING_DAYS)
    fallback = rv + vrp
    if gvz is None:
        iv, src = fallback, pd.Series("rv+vrp", index=prices.index)
    else:
        g = gvz.reindex(prices.index)
        iv = g.combine_first(fallback)
        src = pd.Series(np.where(g.notna(), "gvz", "rv+vrp"), index=prices.index)
    return pd.DataFrame({"iv": iv.ffill(), "source": src})


def _leg_iv(atm_iv, moneyness, skew_per_10pct):
    """Crude skew adjustment: IV rises by `skew_per_10pct` per 10% out of the money.

    NOTE: gold skew is not equity skew. Gold carries genuine upside crash risk,
    so its put skew is far flatter than SPX and in some regimes calls are bid
    instead. The default is deliberately small and MUST be sensitivity-tested;
    e2 reports the result across a skew grid rather than trusting one value.
    """
    return atm_iv + skew_per_10pct * (moneyness / 0.10)


def put_spread_overlay(prices: pd.Series, iv: pd.Series,
                       otm_long: float = 0.10, otm_short: float = 0.25,
                       tenor_months: int = 12, r: float = 0.02,
                       skew_per_10pct: float = 0.01) -> pd.DataFrame:
    """Roll a protective put spread and mark it to market daily.

    Buy a put `otm_long` below spot, sell one `otm_short` below spot, both
    struck at the roll date and held to expiry, then re-struck at the
    prevailing spot.

    Accounting, per cycle [roll_k, roll_k+1]:
        day roll_k        : pay the premium                       -> pnl = -premium
        days in between   : mark to market                        -> pnl = d(value)
        day roll_k+1      : collect intrinsic value at expiry AND pay the new
                            premium

    Collecting the expiry payoff is not optional. Omitting it charges premium
    every year and never pays a claim, which makes any insurance look strictly
    worse than doing nothing.

    Returns daily P&L as a fraction of the insured notional, so it adds
    directly to the asset return.
    """
    idx = prices.index
    step = max(1, int(round(tenor_months / 12 * TRADING_DAYS)))
    roll_locs = list(range(0, len(idx), step))
    if roll_locs[-1] != len(idx) - 1:
        roll_locs.append(len(idx) - 1)          # close the final open cycle

    value = pd.Series(0.0, index=idx)           # spread value / entry spot
    pnl = pd.Series(0.0, index=idx)
    premium_paid = pd.Series(0.0, index=idx)
    payoff_received = pd.Series(0.0, index=idx)

    def price_spread(i, k_long, k_short, entry_spot, expiry_loc):
        T = max((expiry_loc - i) / TRADING_DAYS, 0.0)
        s = float(prices.iloc[i])
        v = float(iv.iloc[i]) if not np.isnan(iv.iloc[i]) else 0.15
        if T <= 0:                               # expiry: intrinsic only
            val = max(k_long - s, 0.0) - max(k_short - s, 0.0)
        else:
            val = (float(bs_put(s, k_long, T, _leg_iv(v, otm_long, skew_per_10pct), r))
                   - float(bs_put(s, k_short, T, _leg_iv(v, otm_short, skew_per_10pct), r)))
        return val / entry_spot

    for c in range(len(roll_locs) - 1):
        i0, i1 = roll_locs[c], roll_locs[c + 1]
        entry_spot = float(prices.iloc[i0])
        k_long = entry_spot * (1 - otm_long)
        k_short = entry_spot * (1 - otm_short)

        prem = price_spread(i0, k_long, k_short, entry_spot, i1)
        premium_paid.iloc[i0] = prem
        value.iloc[i0] = prem
        # No P&L on the entry day: cash of `prem` leaves and a position worth
        # `prem` arrives. Charging the premium here as well as letting it flow
        # through the mark-to-market diffs double-counts every roll.

        for i in range(i0 + 1, i1 + 1):
            value.iloc[i] = price_spread(i, k_long, k_short, entry_spot, i1)
            pnl.iloc[i] += value.iloc[i] - value.iloc[i - 1]

        payoff_received.iloc[i1] = value.iloc[i1]

    return pd.DataFrame({"value": value, "pnl": pnl,
                         "premium_paid": premium_paid,
                         "payoff_received": payoff_received})


def annual_premium_cost(overlay: pd.DataFrame, years: float) -> float:
    """Total premium outlay per year, as a fraction of insured notional.

    This is the number to put next to `whipsaw_ann` from decompose.attribution.
    """
    return float(overlay["premium_paid"].sum() / years)
