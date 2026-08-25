"""Single data interface. Everything downstream goes through get_prices()."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATA_DIR.mkdir(exist_ok=True)

# Cross-section validation universe. We trade gold; we VALIDATE the rule here,
# so the drawdown-avoidance property is tested on ~30 crash events, not on the
# single 2011-2015 gold bear.
CROSS_SECTION = [
    "SPY", "EFA", "EEM", "EWJ",      # equity
    "TLT", "IEF", "LQD",             # rates / credit
    "UUP", "FXE", "FXY",             # fx
    "GLD", "SLV",                    # metals
    "DBC", "USO", "DBA",             # commodity
    "VNQ",                           # real estate
]

PRIMARY = "GLD"
IV_INDEX = "^GVZ"   # CBOE Gold ETF Volatility Index, 30-day ATM IV, from 2008-06


class DataUnavailable(RuntimeError):
    """Raised when a required series could not be obtained. Never silently substituted."""


def _cache_path(ticker: str) -> Path:
    return DATA_DIR / f"{ticker.replace('^', '_')}.parquet"


def _download(ticker: str, start: str, retries: int = 4) -> pd.Series:
    import yfinance as yf

    last_err = None
    for attempt in range(retries):
        try:
            df = yf.download(
                ticker, start=start, auto_adjust=True,
                progress=False, threads=False,
            )
            if df is None or len(df) == 0:
                raise DataUnavailable(f"empty frame for {ticker}")
            close = df["Close"]
            if isinstance(close, pd.DataFrame):     # yfinance MultiIndex quirk
                close = close.iloc[:, 0]
            s = close.dropna()
            s.name = ticker
            return s
        except Exception as exc:                     # noqa: BLE001
            last_err = exc
            time.sleep(2 * (attempt + 1))
    raise DataUnavailable(f"could not download {ticker}: {last_err}")


def get_prices(tickers, start: str = "2004-11-18", refresh: bool = False) -> pd.DataFrame:
    """Adjusted close prices. Cached to parquet; network only on cache miss.

    Raises DataUnavailable rather than returning a partial frame, so an
    experiment can never run on a silently truncated universe.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    out, missing = {}, []
    for t in tickers:
        p = _cache_path(t)
        if p.exists() and not refresh:
            out[t] = pd.read_parquet(p)[t]
            continue
        try:
            s = _download(t, start)
            s.to_frame().to_parquet(p)
            out[t] = s
        except DataUnavailable:
            missing.append(t)
    if missing:
        raise DataUnavailable(
            f"missing: {missing}. Re-run when reachable; do NOT substitute a proxy."
        )
    df = pd.DataFrame(out).sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def get_iv(start: str = "2008-06-03", refresh: bool = False) -> pd.Series:
    """GVZ implied vol in decimal form. Used to price the insurance leg."""
    df = get_prices([IV_INDEX], start=start, refresh=refresh)
    return (df[IV_INDEX] / 100.0).rename("gvz")


def tradable_returns(prices: pd.Series, roll_dates=None) -> pd.Series:
    """Return series that is safe to compound into P&L.

    A back-adjusted / spliced continuous series carries price jumps at roll
    boundaries that are NOT tradable. Booking them as P&L was the Phase 8X bug.
    Roll-boundary returns are zeroed here; the raw series stays signal-only.
    """
    r = prices.pct_change()
    if roll_dates is not None and len(roll_dates):
        r.loc[r.index.isin(pd.DatetimeIndex(roll_dates))] = 0.0
    return r.fillna(0.0)
