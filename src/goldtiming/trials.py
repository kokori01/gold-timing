"""Append-only trial log.

Every backtest run is recorded here by the harness itself, not by discipline.
The Deflated Sharpe Ratio needs the honest number of trials; six years of
undocumented experimentation made that number unknowable and left no way to
tell a real result from the best of many tries. This file is the fix.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

TRIALS_PATH = Path(__file__).resolve().parents[2] / "results" / "trials.jsonl"
TRIALS_PATH.parent.mkdir(exist_ok=True)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:      # noqa: BLE001
        return "nogit"


def _hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def log_trial(experiment: str, rule: str, params: dict, metrics: dict,
              universe=None, notes: str = "") -> str:
    cfg = {"rule": rule, "params": params, "universe": universe}
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "rule": rule,
        "params": params,
        "universe": universe,
        "config_hash": _hash(cfg),
        "git": _git_commit(),
        "metrics": {k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else v)
                    for k, v in metrics.items()},
        "notes": notes,
    }
    with TRIALS_PATH.open("a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    return rec["config_hash"]


def load_trials() -> list:
    if not TRIALS_PATH.exists():
        return []
    with TRIALS_PATH.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def trial_stats(experiment: str | None = None) -> dict:
    """Trial count and cross-trial Sharpe variance — the two DSR inputs."""
    recs = load_trials()
    if experiment:
        recs = [r for r in recs if r["experiment"] == experiment]
    srs = [r["metrics"].get("sharpe") for r in recs]
    srs = [s for s in srs if s is not None]
    return {
        "n_trials": len(recs),
        "n_distinct_configs": len({r["config_hash"] for r in recs}),
        "sharpe_var": float(np.var(srs, ddof=1)) if len(srs) > 1 else float("nan"),
    }
