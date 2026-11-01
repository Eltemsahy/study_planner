#!/usr/bin/env python
"""Integrated demo: input -> search trace -> CSP pruning -> NN on/off -> NN-enough-but-CSP-rejects."""
import sys
from pathlib import Path
import copy
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).parent / "src"))
from planner import config as C
from planner.csp.model import build_csp
from planner.csp.solvers import solve
from planner.data import oulad
from planner.data.profiles import make_profiles
from planner.eval import plots
from planner.eval.experiments import random_csp
from planner.integration.planner import plan, MarkOracle
from planner.nn.model import MarkNN, PrevAverage

df, meta = oulad.load()
cps = meta["clicks_per_session"]
nn = MarkNN.load(C.MODEL_DIR / "mark_nn.joblib")
avg = PrevAverage().fit(df[df.code_presentation.isin(meta["train"])])
profiles = make_profiles(df, 30, model=nn)
H = lambda t: print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")
if meta.get("synthetic"):
    print("!! SYNTHETIC DATA: demo checks the pipeline, numbers are not real OULAD results !!")


def show_timetable(r, p):
    by = {(w, j): f"A{a+1}.{k}" for (w, j, a, k) in r.timetable}
    weeks = sorted({w for (w, _, _, _) in r.timetable})
    for w in weeks:
        print(f"  week {w:3d}: " + "  ".join(f"[{by.get((w, j), '  - ')}]" for j in range(p.slots_per_week)))


# ---------------------------------------------------------------- 1. INPUT
cands = [(p, plan(p, nn, cps, "ucs")) for p in profiles]
p, base = next(((p, r) for p, r in cands if r.status == "OK" and 4 <= (r.total_sessions or 0) <= 12),
               next(((p, r) for p, r in cands if r.status == "OK"), cands[0]))
H(f"1. INPUT  (profile {p.pid}, student {p.student})")
print(f"  target mark          : {p.target:g} on each of {p.n} assignments")
print(f"  deadlines (weeks)    : {p.deadlines}   (compressed what-if: {p.compressed})")
print(f"  slots per week       : {p.slots_per_week};  blocked (week,slot): {sorted(p.blocked)[:8]}...")
print(f"  clicks/session (G1)  : {cps}")
oracle0 = MarkOracle(nn, p, cps)
print(f"  predicted marks with 0 extra sessions: {[round(m, 1) for m in oracle0.marks((0,)*p.n)]}")
print(f"  real marks (reference only): {p.actual_scores}")

# ---------------------------------------------------------------- 2. SEARCH
H("2. SEARCH TRACE (UCS / Greedy / A*)")
traces = {}
for alg in ["ucs", "greedy", "astar"]:
    r = plan(p, nn, cps, alg, trace=True)
    traces[alg] = r.search.trace
    print(f"  {alg:7s}: plan={r.allocation}  cost={r.total_sessions}  expanded={r.search.nodes_expanded}"
          f"  generated={r.search.nodes_generated}  rejected={r.search.nodes_rejected}")
print("  first expansions of UCS  (step, state, g, h, verdict):")
for t in traces["ucs"][:10]:
    print("   ", t)
plots.plot_search_trace(traces)

# ---------------------------------------------------------------- 3. CSP pruning
H("3. CSP PRUNING  (same plan, switchable parts)")
plan_ = base.allocation
rows = {}
best_bt, hard = -1, None
for sd in range(40):                                   # pick an instance where plain BT really struggles
    cand_plan, cand = random_csp(np.random.default_rng(sd), 4)
    rr = solve(cand, max_assignments=20_000)
    if rr.status == "solved" and rr.backtracks > best_bt:
        best_bt, hard = rr.backtracks, cand
for label, csp in [("this student's plan", build_csp(plan_, p.deadlines, p.slots_per_week, p.blocked, C.WINDOW_WEEKS, p.first_week)),
                   ("harder random instance (4 slots/wk)", hard)]:
    print(f"  -- {label}: sessions={len(csp.variables)}")
    for name, (fc, ac) in {"BT": (0, 0), "BT+FC": (1, 0), "BT+FC+AC3": (1, 1)}.items():
        r = solve(csp, use_fc=bool(fc), use_ac3=bool(ac), var_order="static")
        print(f"     {name:11s} status={r.status:9s} assignments={r.assignments:6d} backtracks={r.backtracks:5d}"
              f" ac3_removed={r.ac3_removed:4d} fc_removed={r.fc_removed:5d}")
        if label.startswith("harder"):
            rows[name] = {"assignments": r.assignments, "backtracks": r.backtracks}
plots.plot_csp_pruning(rows)

# ---------------------------------------------------------------- 4. SOLUTION + NN ON/OFF
H("4. SOLUTION  (NN ON)  vs  NN OFF (average-mark predictor)")
print(f"  NN ON : {base.message}  allocation={base.allocation}  predicted marks={[round(m,1) for m in base.marks]}")
show_timetable(base, p)
off = plan(p, avg, cps, "ucs")
print(f"  NN OFF: status={off.status}  allocation={off.allocation}  -> {off.message}")
plots.plot_nn_on_off({"NN": base.total_sessions or 0, "Average-mark": off.total_sessions or 0})
cap = C.MAX_SESSIONS_PER_ASSIGNMENT
plots.plot_marks({f"assignment {i+1}": [oracle0.mark(i, s) for s in range(cap + 1)] for i in range(p.n)},
                 p.target, base.allocation)
plots.plot_timetable(base.timetable, p.slots_per_week, p.deadlines)

# ---------------------------------------------------------------- 5. NN says enough, CSP rejects
H("5. CASE: NN says 'enough' but the CSP rejects the schedule")
shown = False
for q in profiles:
    r = plan(q, nn, cps, "ucs", csp_precheck=False, max_csp_assignments=30_000)
    if r.nn_enough_csp_rejected:
        st, mk = r.nn_enough_csp_rejected[0]
        print(f"  profile {q.pid}: slots/week={q.slots_per_week}, deadlines={q.deadlines}, target={q.target:g}")
        print(f"  rejected plan {st}: NN predicted marks {[round(m,1) for m in mk]} (all >= target)")
        csp = build_csp(st, q.deadlines, q.slots_per_week, q.blocked, C.WINDOW_WEEKS, q.first_week)
        cr = solve(csp, use_fc=True, use_ac3=True, var_order="mrv", max_assignments=30_000)
        print(f"  CSP verdict: {cr.status} -> {cr.message}")
        print(f"  Search continued; final result: {r.status}; {r.message}")
        shown = True
        break
if not shown:
    print("  (none among the generated profiles; crafting a what-if: deadlines 1 week apart, 2 slots/week)")
    for q in profiles:
        d0 = q.deadlines[0]
        q2 = copy.copy(q)
        q2.deadlines, q2.slots_per_week, q2.blocked, q2.first_week = [d0, d0 + 1, d0 + 1], 2, frozenset(), max(0, d0 - 4)
        lo = int(min(oracle0.marks((0,) * 3)) if False else min(MarkOracle(nn, q2, cps).marks((0,) * 3)))
        for tgt in range(lo, 100):
            r = plan(q2, nn, cps, "ucs", target=tgt, csp_precheck=False, max_csp_assignments=30_000)
            if r.nn_enough_csp_rejected:
                st, mk = r.nn_enough_csp_rejected[0]
                print(f"  profile {q2.pid}: slots/week=2, deadlines={q2.deadlines}, target={tgt}")
                print(f"  rejected plan {st}: NN predicted marks {[round(m,1) for m in mk]} (all >= target)")
                csp = build_csp(st, q2.deadlines, 2, q2.blocked, C.WINDOW_WEEKS, q2.first_week)
                cr = solve(csp, use_fc=True, use_ac3=True, var_order="mrv", max_assignments=30_000)
                print(f"  CSP verdict: {cr.status} -> {cr.message}")
                print(f"  Search continued; final result: {r.status}; {r.message}")
                shown = True
                break
        if shown:
            break

# ---------------------------------------------------------------- 6. unreachable
H("6. IMPOSSIBLE TARGET -> clear 'no solution' message")
r = plan(p, nn, cps, "ucs", target=99)
print(" ", r.message)
print("\nFigures written to", C.FIG_DIR)
