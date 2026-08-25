"""The acceptance gate. Pre-registered, and every criterion is binding.

Phase 8F passed a candidate against 50/50 on MaxDD alone while it was failing
CAGR and Sharpe, because the predicate was a disjunction where it should have
been a conjunction. That bug survived a whole phase. Here every criterion must
pass, the breakdown is always returned, and tests/test_benchmark_gate.py fails
the build if a partial pass is ever accepted again.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Pre-registered thresholds for the timing-vs-holding question.
# Changing these is a decision to be recorded in the vault, not a tuning knob.
DEFAULT_CRITERIA = {
    "min_capture_spread": 0.10,   # upside capture must exceed downside by 10pp
    "min_excess_at_equal_dd": 0.0,  # must beat buy-and-hold at the SAME drawdown
    "min_calmar_ratio": 1.0,      # Calmar must be at least the benchmark's
    "max_whipsaw_ann": None,      # set by e2 once insurance premium is known
}


@dataclass
class GateResult:
    passed: bool
    checks: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)

    def __str__(self) -> str:
        head = "PASS" if self.passed else "FAIL"
        body = "\n".join(
            f"  {'ok ' if v else 'NO '} {k}" for k, v in self.checks.items()
        )
        tail = ("\n  reasons: " + "; ".join(self.reasons)) if self.reasons else ""
        return f"{head}\n{body}{tail}"


def evaluate(summary: dict, bench_summary: dict, criteria: dict | None = None) -> GateResult:
    c = {**DEFAULT_CRITERIA, **(criteria or {})}
    checks, reasons = {}, []

    spread = summary.get("capture_spread")
    checks["capture_spread"] = spread is not None and spread >= c["min_capture_spread"]
    if not checks["capture_spread"]:
        reasons.append(f"capture_spread={spread!r} < {c['min_capture_spread']}")

    excess = summary.get("excess_at_equal_dd")
    checks["excess_at_equal_dd"] = excess is not None and excess > c["min_excess_at_equal_dd"]
    if not checks["excess_at_equal_dd"]:
        reasons.append(f"excess_at_equal_dd={excess!r} <= {c['min_excess_at_equal_dd']}")

    cal, bcal = summary.get("calmar"), bench_summary.get("calmar")
    checks["calmar"] = (cal is not None and bcal not in (None, 0)
                        and cal / bcal >= c["min_calmar_ratio"])
    if not checks["calmar"]:
        reasons.append(f"calmar={cal!r} vs benchmark {bcal!r}")

    if c["max_whipsaw_ann"] is not None:
        w = summary.get("whipsaw_ann")
        checks["whipsaw_vs_insurance"] = w is not None and abs(w) <= c["max_whipsaw_ann"]
        if not checks["whipsaw_vs_insurance"]:
            reasons.append(
                f"whipsaw_ann={w!r} costs more than insurance ({c['max_whipsaw_ann']})"
            )

    # Conjunction. Never a disjunction. This line is the Phase 8F fix.
    return GateResult(passed=all(checks.values()), checks=checks, reasons=reasons)
