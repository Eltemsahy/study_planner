"""Monotonicity check: predicted mark must not go DOWN when sessions (effort) go UP."""
import pandas as pd
from .. import config as C
from ..integration.planner import MarkOracle


def check(profiles, model, cps, cap=C.MAX_SESSIONS_PER_ASSIGNMENT, tol=1e-6):
    rows = []
    for p in profiles:
        o = MarkOracle(model, p, cps, envelope=False)      # RAW network output
        for i in range(p.n):
            vals = [o.raw(i, s) for s in range(cap + 1)]
            drops = [vals[s] - vals[s + 1] for s in range(cap) if vals[s + 1] < vals[s] - tol]
            rows.append({"profile": p.pid, "assignment": i + 1, "violations": len(drops),
                         "max_drop": max(drops) if drops else 0.0,
                         "net_change_0_to_cap": vals[-1] - vals[0]})
    df = pd.DataFrame(rows)
    summary = {"curves": len(df), "curves_with_violation": int((df.violations > 0).sum()),
               "share_curves_with_violation": float((df.violations > 0).mean()),
               "total_violating_steps": int(df.violations.sum()),
               "steps_checked": len(df) * cap,
               "max_drop_marks": float(df.max_drop.max())}
    return df, summary
