# gold-timing — Project Memory

Compact resume context. Detail lives in `README.md` and in the vault at
`Brain/projects/trading-bot/`.

## Objective

Decide, with evidence, whether timing gold beats holding gold on the owner's
actual stated objective: participate in rises, avoid giving back profits in
falls. Measured as capture spread, Calmar, and CAGR at equal drawdown after
financing — not Sharpe.

## Status — 2026-08-25

First full run complete. **Every timing rule failed the pre-registered gate.**
Buy & hold: CAGR 10.98%, MaxDD −45.56%, Calmar 0.241 (GLD, 2004-11-18→2026-08-24).
Best capture spread across 6 rules: 0.050 against a required 0.10.

## Verified findings

1. Whipsaw tax exceeds protection earned in every rule. Risk-off episodes are
   dominated by fake breakdowns: sma10 4 real / 16 fake, asymmetric 18 / 123.
2. Total-vol targeting and downside-vol targeting both **increased** gold's
   drawdown (−45.59%, −48.07% vs −45.56%). Direction-blindness is real and
   semi-deviation did not fix it here.
3. Insurance pays 32 cents on the dollar: 50.5% premium vs 16.4% payoff over
   21.7y, with payoff in only 2 of 22 years. Gold's big drawdown was a 4-year
   grind; an annually re-struck spread keeps resetting inside the deductible.
4. Best exchange rate belongs to the "never fully exit" rules (core+downside-vol,
   0.102 CAGR points per drawdown point), then the put spread (0.179 — and the
   only one whose cost is known in advance).
5. E_asymmetric has the best gold Calmar (0.312) and the worst cross-section
   generalisation (43.8% of 16 assets). Fitted to the 2011-2015 bear.
6. Once leverage is financed at 5%, almost nothing beats buy & hold at equal
   drawdown on any asset (best: C vol_target, 50% of assets; sma10 and
   asymmetric: 1 of 16).

## Guardrails now enforced by code

- `test_roll_gap_never_enters_pnl` — Phase 8X bug 1 cannot return.
- `test_gate_rejects_partial_pass` — Phase 8X bug 2 cannot return; the gate is a
  conjunction and a missing metric counts as failure.
- `test_no_lookahead.py` — future prices are rewritten and exposure must not move.
- `results/trials.jsonl` — written by the harness, not by discipline. 103 runs,
  97 distinct configs, cross-trial Sharpe variance 0.0540.

## Open questions

- Is the capture-spread threshold of 0.10 right? E_asymmetric and C_vol_target
  clear every other criterion and fail only on this. Moving it is a **decision to
  record in the vault**, not a tuning knob — deciding it after seeing results is
  precisely the failure mode the gate exists to prevent.
- Does a longer-dated (24-36m) or non-re-struck put ladder fit gold's slow-grind
  drawdown better than annual re-striking?
- Would a portfolio benchmark (60/40 + gold sleeve) change the verdict, given
  gold's low equity correlation? Standalone comparison may be the wrong frame.

## Next actions

1. Decide the capture-spread threshold question in the vault before any re-run.
2. Test longer-tenor / laddered puts against the slow-grind drawdown shape.
3. If neither changes the verdict: close the gold timing question and record it.

## Do not re-open without a reason recorded in the vault

Microstructure/order flow, ML allocators, mixture-of-experts, genetic search,
futures + Databento, LLM in the decision path. Rationale in README "What this
does NOT do".
