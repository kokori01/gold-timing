"""Split a timing rule's effect into protection earned and whipsaw tax paid.

Six years of research never measured the whipsaw tax. It is the hidden,
variable price of a timing rule, and it is the number that must be compared
against an option premium, which is the explicit, known price of the same payoff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def attribution(exposure: pd.Series, asset_returns: pd.Series) -> dict:
    """Daily attribution of (rule - buy_and_hold) into protection vs whipsaw.

    shortfall = 1 - exposure; the rule's contribution on day t is
    -shortfall_t * r_t. Under-invested on a down day earns protection;
    under-invested on an up day pays the whipsaw tax.
    """
    expo, r = exposure.align(asset_returns, join="inner")
    shortfall = 1.0 - expo
    contrib = -shortfall * r
    down, up = r < 0, r > 0
    years = len(r) / TRADING_DAYS
    protection, whipsaw = float(contrib[down].sum()), float(contrib[up].sum())
    return {
        "protection_total": protection,
        "whipsaw_total": whipsaw,
        "net_total": protection + whipsaw,
        "protection_ann": protection / years,
        "whipsaw_ann": whipsaw / years,
        "net_ann": (protection + whipsaw) / years,
        "years": years,
        "avg_exposure": float(expo.mean()),
        "days_derisked": int((expo < 0.999).sum()),
        "pct_time_derisked": float((expo < 0.999).mean()),
    }


def episodes(exposure: pd.Series, prices: pd.Series, threshold: float = 0.999) -> pd.DataFrame:
    """One row per contiguous risk-off episode.

    Answers the question directly: how many times did the rule step aside,
    how many of those were real declines, and how many were fake breakdowns?
    """
    expo, px = exposure.align(prices, join="inner")
    off = (expo < threshold).to_numpy()
    rows, i, n = [], 0, len(off)
    while i < n:
        if not off[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and off[j + 1]:
            j += 1
        seg_px, seg_ex = px.iloc[i:j + 1], expo.iloc[i:j + 1]
        asset_ret = float(seg_px.iloc[-1] / seg_px.iloc[0] - 1.0) if len(seg_px) > 1 else 0.0
        shortfall = float((1.0 - seg_ex).mean())
        rows.append({
            "start": px.index[i], "end": px.index[j], "days": j - i + 1,
            "asset_return": asset_ret,
            "avg_shortfall": shortfall,
            "contribution": -shortfall * asset_ret,
            "kind": "protective" if asset_ret < 0 else "whipsaw",
        })
        i = j + 1
    return pd.DataFrame(rows)


def episode_summary(ep: pd.DataFrame) -> dict:
    if ep.empty:
        return {"n_episodes": 0, "n_protective": 0, "n_whipsaw": 0,
                "hit_rate": float("nan"), "avg_protective_gain": float("nan"),
                "avg_whipsaw_cost": float("nan"), "payoff_ratio": float("nan")}
    prot, whip = ep[ep.kind == "protective"], ep[ep.kind == "whipsaw"]
    avg_p = float(prot.contribution.mean()) if len(prot) else 0.0
    avg_w = float(whip.contribution.mean()) if len(whip) else 0.0
    return {
        "n_episodes": len(ep),
        "n_protective": len(prot),
        "n_whipsaw": len(whip),
        "hit_rate": len(prot) / len(ep),
        "avg_protective_gain": avg_p,
        "avg_whipsaw_cost": avg_w,
        "payoff_ratio": abs(avg_p / avg_w) if avg_w else float("nan"),
    }
