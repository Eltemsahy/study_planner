#!/usr/bin/env python
"""Planner output (allocation, timetable, predicted marks) for >= 30 student profiles."""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent / "src"))
from planner import config as C
from planner.data import oulad
from planner.data.profiles import make_profiles
from planner.integration.planner import plan
from planner.nn.model import MarkNN

df, meta = oulad.load()
nn = MarkNN.load(C.MODEL_DIR / "mark_nn.joblib")
profiles = make_profiles(df, 30, model=nn)
rows = []
for p in profiles:
    r = plan(p, nn, meta["clicks_per_session"], "ucs")
    rows.append({"profile": p.pid, "target": p.target, "slots_per_week": p.slots_per_week,
                 "deadline_weeks": p.deadlines, "compressed_deadlines": p.compressed,
                 "status": r.status, "allocation": r.allocation,
                 "total_sessions": r.total_sessions,
                 "predicted_marks": None if r.marks is None else [round(m, 1) for m in r.marks],
                 "timetable(week,slot,assignment,session)": r.timetable,
                 "message": r.message, "nodes_expanded": r.search.nodes_expanded,
                 "nn_enough_but_csp_rejected": len(r.nn_enough_csp_rejected)})
out = pd.DataFrame(rows)
C.RESULTS_DIR.mkdir(exist_ok=True)
out.to_csv(C.RESULTS_DIR / "profile_plans.csv", index=False)
print(out[["profile", "target", "slots_per_week", "status", "allocation", "predicted_marks"]]
      .to_string(index=False))
print(f"\n{(out.status=='OK').sum()} plans found, {(out.status!='OK').sum()} unreachable "
      f"(reported with best feasible marks). Saved results/profile_plans.csv")
