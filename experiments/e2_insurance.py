"""E2 - Is buying the payoff cheaper than synthesising it?

The objective (full upside, limited downside) is a put payoff. A timing rule
synthesises it and pays in whipsaw tax; a put spread buys it and pays in
premium. E1 measured the tax. This measures the premium, on the same axis.

Pre-registered decision rule:
    premium_ann < |whipsaw_ann| of the best timing rule  ->  stop timing research,
                                                             price the insurance
    otherwise                                            ->  timing has finally
                                                             justified itself
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from goldtiming import metrics as M
from goldtiming.data import PRIMARY, DataUnavailable, get_iv, get_prices
from goldtiming.options import annual_premium_cost, build_iv, put_spread_overlay
from goldtiming.trials import log_trial

RESULTS = Path(__file__).resolve().parents[1] / "results"
TRADING_DAYS = 252

GRID = [
    # otm_long, otm_short, tenor_months, skew_per_10pct
    (0.10, 0.25, 12, 0.010),
    (0.10, 0.25, 12, 0.000),   # flat skew: gold is not equity
    (0.10, 0.25, 12, 0.025),   # steep skew: pessimistic pricing
    (0.10, 0.25,  6, 0.010),
    (0.15, 0.30, 12, 0.010),
    (0.05, 0.20, 12, 0.010),
    (0.10, 0.99, 12, 0.010),   # naked protective put, no short leg
]


def main() -> int:
    try:
        px = get_prices([PRIMARY])[PRIMARY]
    except DataUnavailable as exc:
        print(f"ABORT: {exc}")
        return 2
    try:
        gvz = get_iv()
        print(f"GVZ: {gvz.index[0].date()} -> {gvz.index[-1].date()}")
    except DataUnavailable as exc:
        print(f"WARNING: no GVZ ({exc}); pricing entirely on realized vol + VRP")
        gvz = None

    ivf = build_iv(px, gvz)
    cover = (ivf["source"] == "gvz").mean()
    print(f"IV provenance: {cover:.1%} market (GVZ), {1 - cover:.1%} modelled\n")

    bench = px.pct_change().fillna(0.0)
    years = len(px) / TRADING_DAYS
    bench_sum = M.summary(bench, benchmark=bench)

    rows = [{"config": "buy_hold (no insurance)", "premium_ann": 0.0,
             **{k: bench_sum[k] for k in
                ("cagr", "max_dd", "calmar", "upside_capture", "downside_capture",
                 "capture_spread")}}]

    for otm_l, otm_s, tenor, skew in GRID:
        ov = put_spread_overlay(px, ivf["iv"], otm_long=otm_l, otm_short=otm_s,
                                tenor_months=tenor, skew_per_10pct=skew)
        insured = bench + ov["pnl"]
        s = M.summary(insured, benchmark=bench)
        prem = annual_premium_cost(ov, years)
        name = f"put {int(otm_l*100)}/{int(otm_s*100) if otm_s < 0.9 else 'naked'} {tenor}m skew{skew:.3f}"
        row = {"config": name, "premium_ann": prem,
               **{k: s[k] for k in ("cagr", "max_dd", "calmar", "upside_capture",
                                    "downside_capture", "capture_spread")}}
        rows.append(row)
        log_trial("e2_insurance", "put_spread",
                  {"otm_long": otm_l, "otm_short": otm_s, "tenor_months": tenor,
                   "skew_per_10pct": skew},
                  {**s, "premium_ann": prem}, universe=PRIMARY)

    df = pd.DataFrame(rows)
    print("=== insurance overlay on GLD ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    e1 = RESULTS / "e1_gld.csv"
    if e1.exists():
        t = pd.read_csv(e1)
        bh_cagr, bh_dd = bench_sum["cagr"], abs(bench_sum["max_dd"])

        combined = []
        for _, r in t[t.rule != "A_buy_hold"].iterrows():
            combined.append({"approach": "synthesise", "name": r["rule"],
                             "cagr": r["cagr"], "max_dd": r["max_dd"],
                             "calmar": r["calmar"],
                             "explicit_cost_ann": float("nan"),
                             "hidden_cost_ann": abs(r["whipsaw_ann"])})
        for _, r in df.iloc[1:].iterrows():
            combined.append({"approach": "buy", "name": r["config"],
                             "cagr": r["cagr"], "max_dd": r["max_dd"],
                             "calmar": r["calmar"],
                             "explicit_cost_ann": r["premium_ann"],
                             "hidden_cost_ann": 0.0})

        c = pd.DataFrame(combined)
        c["cagr_given_up"] = bh_cagr - c["cagr"]
        c["dd_points_saved"] = bh_dd - c["max_dd"].abs()
        # The exchange rate. How much annual return does one point of drawdown
        # reduction cost? Negative dd_points_saved means it delivered nothing.
        c["cost_per_dd_point"] = c.apply(
            lambda r: r.cagr_given_up / r.dd_points_saved
            if r.dd_points_saved > 0.005 else float("nan"), axis=1)
        c = c.sort_values("cost_per_dd_point", na_position="last")

        print("\n=== the comparison this project exists to make ===")
        print(f"buy & hold: CAGR {bh_cagr:.2%}, MaxDD {-bh_dd:.2%}, "
              f"Calmar {bench_sum['calmar']:.3f}\n")
        print(c[["approach", "name", "cagr", "max_dd", "calmar", "cagr_given_up",
                 "dd_points_saved", "cost_per_dd_point", "explicit_cost_ann",
                 "hidden_cost_ann"]].to_string(
            index=False, float_format=lambda v: f"{v:8.4f}"))

        delivered = c[c.dd_points_saved > 0.005]
        if delivered.empty:
            print("\n  verdict: NEITHER approach reduced drawdown. Hold the asset.")
        else:
            best = delivered.iloc[0]
            print(f"\n  best exchange rate: [{best.approach}] {best['name']}")
            print(f"    gives up {best.cagr_given_up:.2%}/yr of CAGR")
            print(f"    to remove {best.dd_points_saved:.2%} of drawdown")
            print(f"    = {best.cost_per_dd_point:.3f} CAGR points per drawdown point")
            print(f"    Calmar {best.calmar:.3f} vs buy-and-hold {bench_sum['calmar']:.3f}")
            beats = delivered[delivered.calmar > bench_sum["calmar"]]
            print(f"\n  approaches beating buy-and-hold on Calmar: "
                  f"{len(beats)} of {len(c)}")
            if len(beats):
                print("    " + ", ".join(f"{r['name']} ({r.calmar:.3f})"
                                         for _, r in beats.iterrows()))
        c.to_csv(RESULTS / "e2_comparison.csv", index=False)

    df.to_csv(RESULTS / "e2_insurance.csv", index=False)
    print(f"\nwritten: {RESULTS}/e2_insurance.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
