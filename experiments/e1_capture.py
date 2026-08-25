"""E1 - Does any timing rule beat holding gold on the objective we actually have?

Pre-registered question:
    Participate in rises, avoid giving back profits in falls. Measured as
    upside capture minus downside capture, and as CAGR at equal drawdown.

Pre-registered kill condition:
    If no rule clears the gate on GLD *and* reproduces on a majority of the
    cross-section, the timing path is closed. No filters are added.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from goldtiming import gate, metrics as M
from goldtiming.data import CROSS_SECTION, PRIMARY, DataUnavailable, get_prices
from goldtiming.decompose import attribution, episode_summary, episodes
from goldtiming.rules import REGISTRY, apply_rule
from goldtiming.trials import log_trial, trial_stats

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)
COST_BPS = 5.0


def run_one(prices: pd.Series, rule: str, asset: str, cost_bps: float = COST_BPS) -> dict:
    net, expo = apply_rule(prices, rule, cost_bps=cost_bps)
    bench = prices.pct_change().fillna(0.0)
    row = {"asset": asset, "rule": rule}
    row.update(M.summary(net, benchmark=bench))
    row.update(attribution(expo, bench))
    row.update(episode_summary(episodes(expo, prices)))
    log_trial("e1_capture", rule, {"cost_bps": cost_bps}, row, universe=asset)
    return row


def main() -> int:
    try:
        px = get_prices([PRIMARY])[PRIMARY]
    except DataUnavailable as exc:
        print(f"ABORT: {exc}")
        return 2

    print(f"{PRIMARY}: {px.index[0].date()} -> {px.index[-1].date()}  ({len(px)} bars)\n")

    rows = [run_one(px, r, PRIMARY) for r in REGISTRY]
    df = pd.DataFrame(rows)
    bench_row = df[df.rule == "A_buy_hold"].iloc[0].to_dict()

    cols = ["rule", "cagr", "max_dd", "calmar", "upside_capture", "downside_capture",
            "capture_spread", "cagr_at_equal_dd", "whipsaw_ann", "protection_ann",
            "net_ann", "n_episodes", "n_whipsaw", "n_protective"]
    print("=== GLD, all rules ===")
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    print("\n=== gate ===")
    for _, r in df.iterrows():
        if r.rule == "A_buy_hold":
            continue
        print(f"\n{r.rule}\n{gate.evaluate(r.to_dict(), bench_row)}")

    # Cross-section: the rule must be a property of the RULE, not of the single
    # 2011-2015 gold bear. One asset gives one crash; sixteen give ~thirty.
    print("\n=== cross-section validation ===")
    xs_rows = []
    for t in CROSS_SECTION:
        try:
            p = get_prices([t])[t]
        except DataUnavailable as exc:
            print(f"  skip {t}: {exc}")
            continue
        for rule in REGISTRY:
            if rule == "A_buy_hold":
                continue
            xs_rows.append(run_one(p, rule, t))

    if xs_rows:
        xs = pd.DataFrame(xs_rows)
        agg = xs.groupby("rule").agg(
            assets=("asset", "nunique"),
            median_spread=("capture_spread", "median"),
            share_spread_positive=("capture_spread", lambda s: float((s > 0).mean())),
            median_excess_eq_dd=("excess_at_equal_dd", "median"),
            share_excess_positive=("excess_at_equal_dd", lambda s: float((s > 0).mean())),
            median_whipsaw_ann=("whipsaw_ann", "median"),
        )
        print(agg.to_string(float_format=lambda v: f"{v:8.4f}"))
        xs.to_csv(RESULTS / "e1_cross_section.csv", index=False)

    df.to_csv(RESULTS / "e1_gld.csv", index=False)
    print(f"\ntrials: {trial_stats('e1_capture')}")
    print(f"written: {RESULTS}/e1_gld.csv, {RESULTS}/e1_cross_section.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
