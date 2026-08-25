# gold-timing

[![CI](https://github.com/kokori01/gold-timing/actions/workflows/ci.yml/badge.svg)](https://github.com/kokori01/gold-timing/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**One question:** for the objective *"participate when gold rises, don't give back
profits when it falls"* — does any timing rule beat simply holding gold, and is
buying that payoff as a put spread cheaper than synthesising it?

**Answer, as measured: no timing rule passes. Holding is hard to beat.**

Data: GLD, 2004-11-18 → 2026-08-24, 5474 daily bars (21.7y), adjusted close.
Costs: 5 bps per unit of turnover. Leverage financed at 5%/yr. IV from GVZ
(83.8% of days market-priced, 16.2% modelled as realized vol + 2% VRP).

## Why this repo exists rather than a restored AKOCT

Six years and eight phases produced no reproducible artifact — no repo, no data,
no reports on this machine. Phase 8X had found two real bugs, and both would
have silently returned. So they are the **first two tests**, written before any
strategy code:

| Test | Bug it prevents |
|---|---|
| `test_roll_gap_never_enters_pnl` | roll-boundary jumps in a spliced continuous series booked as tradable P&L |
| `test_gate_rejects_partial_pass` | benchmark predicate passing on MaxDD alone while failing CAGR |

Plus two the old work never had: `test_no_lookahead.py` proves every rule ignores
same-day and future prices by rewriting the future and checking exposure doesn't
move; `results/trials.jsonl` is written by the harness on every run, so the
Deflated Sharpe trial count can never again be unknowable.

## Results

Buy & hold: **CAGR 10.98%, MaxDD −45.56%, Calmar 0.241.**

### Every timing rule fails the pre-registered gate

Gate (conjunction, all binding): capture spread ≥ 0.10, positive excess CAGR at
equal drawdown *after financing*, Calmar ≥ benchmark.

| rule | CAGR | MaxDD | Calmar | up cap | down cap | spread | whipsaw/yr | protection/yr | net/yr |
|---|---|---|---|---|---|---|---|---|---|
| A buy & hold | 10.98% | −45.56% | 0.241 | 1.000 | 1.000 | — | — | — | — |
| B sma10 binary | 7.68% | −40.30% | 0.191 | 0.660 | 0.644 | 0.017 | −33.4% | +29.9% | **−3.46%** |
| C vol target | 11.34% | −45.59% | 0.249 | 0.936 | 0.886 | 0.050 | −9.2% | +9.2% | +0.06% |
| D downside vol | 11.05% | −48.07% | 0.230 | 0.945 | 0.914 | 0.031 | −6.9% | +6.8% | −0.09% |
| E asymmetric | 7.93% | −25.42% | **0.312** | 0.632 | 0.587 | 0.045 | −41.7% | +38.4% | −3.33% |
| F core+overlay | 9.79% | −40.57% | 0.241 | 0.866 | 0.855 | 0.011 | −13.4% | +12.0% | −1.38% |
| G core+dvol | 10.76% | −43.35% | 0.248 | 0.936 | 0.914 | 0.022 | −7.7% | +7.3% | −0.47% |

Best capture spread is 0.050 against a required 0.10. **Nothing is close.**

### The whipsaw objection, confirmed and quantified

Stepping aside in a fall also means stepping aside in fake breakdowns. Counted
as discrete risk-off episodes:

| rule | episodes | real declines | fake breakdowns | hit rate |
|---|---|---|---|---|
| B sma10 | 20 | 4 | 16 | 20% |
| E asymmetric | 141 | 18 | 123 | **13%** |
| G core+dvol | 23 | 10 | 13 | 43% |

Whipsaw tax exceeds protection earned in every rule. The asymmetric rule — built
specifically to re-enter on *price* rather than waiting for volatility to
normalise — still pays 41.7%/yr in whipsaw to earn 38.4%/yr in protection.

### Insurance is priced, and it mostly doesn't pay out

Rolling 12-month put spread, re-struck annually:

- premium paid, total: **50.5%**; payoff received, total: **16.4%** → 32 cents on the dollar
- payoff arrived in **2 of 22 years** (2013: +15.0%, 2015: +1.4%)

Gold's −45.6% drawdown was a four-year grind, not a crash. An annually re-struck
spread keeps resetting to the new lower spot, so a slow bear stays inside the
deductible year after year. **Put spreads are crash insurance; gold's big
drawdown was not a crash.**

Best config (15%/30% strikes, 12m): premium 1.58%/yr, CAGR 10.18%, MaxDD −41.06%,
Calmar 0.248.

### The exchange rate — CAGR given up per point of drawdown removed

| approach | name | gives up | removes | rate | cost visible in advance? |
|---|---|---|---|---|---|
| synthesise | G core+dvol | 0.22%/yr | 2.20pp | **0.102** | no |
| synthesise | E asymmetric | 3.05%/yr | 20.13pp | 0.151 | no |
| buy | put 15/30 12m | 0.80%/yr | 4.50pp | 0.179 | **yes** |
| synthesise | F core+overlay | 1.19%/yr | 4.99pp | 0.239 | no |
| synthesise | B sma10 | 3.30%/yr | 5.25pp | 0.628 | no |
| synthesise | C vol target | — | −0.04pp | n/a | no |
| synthesise | D downside vol | — | −2.51pp | n/a | no |

C and D **increased** drawdown. Total-volatility targeting is direction-blind, and
switching to downside semi-deviation did not rescue it on gold.

### Cross-section: the one good gold result does not generalise

The same rules on 16 assets, so the drawdown property is tested against ~30 crash
events instead of gold's single 2011-2015 bear:

| rule | assets with positive capture spread | assets beating buy & hold at equal DD (financed) |
|---|---|---|
| F core+overlay | 87.5% | 25.0% |
| G core+dvol | 75.0% | 31.3% |
| B sma10 | 68.8% | 6.3% |
| C vol target | 56.3% | 50.0% |
| D downside vol | 50.0% | 37.5% |
| **E asymmetric** | **43.8%** | **6.3%** |

E asymmetric has the best Calmar on gold (0.312) and the **worst** cross-section
generalisation. That is what a result fitted to one bear market looks like, and
catching it is exactly why the cross-section step exists.

## What this does NOT do

Deliberate exclusions, so no future session re-litigates them:

- **No microstructure / order flow.** OFI and queue imbalance predict over
  milliseconds-to-minutes; these rules hold for months. Wrong horizon, and the
  latency race is not winnable from a VPS.
- **No machine learning.** ~4,000 daily rows with regime-length autocorrelation is
  perhaps 40 effective observations. Not a dataset for 134 features.
- **No mixture-of-experts / dynamic allocator.** Allocation among experts cannot
  manufacture edge it does not have. Revisit only when ≥2 strategies each pass the
  gate independently.
- **No futures, no Databento, no roll construction yet.** ETFs answer the question
  for free. `tradable_returns()` and its test already exist for when futures return.
- **No LLM in the decision path.**
- **No parameter optimisation.** The grid is 7 rules at default parameters. Every
  run is logged to `results/trials.jsonl`; tuning without recording is how the
  trial count became unknowable last time.

## Layout

```
src/goldtiming/
  data.py        get_prices() + tradable_returns()   -- the only data interface
  metrics.py     Calmar, capture ratios, Sharpe SE, DSR, equal-drawdown CAGR
  rules.py       7 exposure rules, all .shift(1)
  decompose.py   protection vs whipsaw attribution, episode classification
  options.py     Black-Scholes, GVZ-based IV, rolling put spread overlay
  gate.py        pre-registered acceptance criteria (conjunction)
  trials.py      append-only run log -> DSR inputs
experiments/
  e1_capture.py    all rules on GLD + 16-asset cross-section
  e2_insurance.py  put spread pricing + the unified exchange-rate table
```

## Run

```bash
python3 -m venv .venv && ./.venv/bin/pip install -e .
./.venv/bin/python -m pytest -q
./.venv/bin/python experiments/e1_capture.py
./.venv/bin/python experiments/e2_insurance.py
```

31 tests. Results land in `results/`.
