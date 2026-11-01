"""Integrated planner: Search proposes -> NN judges -> CSP schedules (or rejects)."""
from __future__ import annotations
from dataclasses import dataclass, field
from math import ceil
import numpy as np
import pandas as pd
from .. import config as C
from ..csp.model import build_csp, hall_infeasible
from ..csp.solvers import solve
from ..search.algorithms import search

INF = 10 ** 6


class MarkOracle:
    """mark(i, s): predicted mark of assignment i with s extra sessions (memoised, counted).

    Rule G1b: a session adds clicks_per_session clicks, split over the 3 activity groups
    in proportion to the student's own click shares (60/20/20 if they had no clicks).
    envelope=True makes marks non-decreasing in s (running max) so the search is sound.
    """

    def __init__(self, model, profile, clicks_per_session, envelope=True):
        self.model, self.p, self.cps, self.envelope = model, profile, clicks_per_session, envelope
        self.cache, self.raw_cache, self.calls = {}, {}, 0
        self.shares = []
        for b in profile.base:
            g = np.array([b[f"clicks_{n}"] for n in C.GROUP_NAMES], float)
            self.shares.append(g / g.sum() if g.sum() > 0 else np.array([.6, .2, .2]))

    def raw(self, i, s):
        if (i, s) not in self.raw_cache:
            f = dict(self.p.base[i])
            if getattr(self.model, "uses_clicks", True):
                for n, sh in zip(C.GROUP_NAMES, self.shares[i]):
                    f[f"clicks_{n}"] += s * self.cps * sh
            self.raw_cache[(i, s)] = float(self.model.predict(pd.DataFrame([f]))[0])
            self.calls += 1
        return self.raw_cache[(i, s)]

    def mark(self, i, s):
        if (i, s) not in self.cache:
            self.cache[(i, s)] = (max(self.raw(i, t) for t in range(s + 1))
                                  if self.envelope else self.raw(i, s))
        return self.cache[(i, s)]

    def marks(self, state):
        return [self.mark(i, s) for i, s in enumerate(state)]

    def max_gain(self, i, cap):
        return max(self.mark(i, s + 1) - self.mark(i, s) for s in range(cap))


@dataclass
class PlanResult:
    status: str                       # "OK" | "UNREACHABLE"
    algorithm: str
    allocation: tuple | None
    marks: list | None
    timetable: list                   # (week, slot, assignment, session_no)
    message: str
    search: object = None
    nn_calls: int = 0
    csp_calls: int = 0
    csp_assignments: int = 0
    csp_backtracks: int = 0
    nn_enough_csp_rejected: list = field(default_factory=list)   # [(state, marks)]
    best_feasible: tuple | None = None                            # (state, marks)

    @property
    def total_sessions(self):
        return sum(self.allocation) if self.allocation else None


def plan(profile, model, clicks_per_session, algorithm="ucs", csp_fc=True, csp_ac3=True,
         csp_precheck=True, cap=C.MAX_SESSIONS_PER_ASSIGNMENT, envelope=True,
         trace=False, target=None, max_csp_assignments=200_000) -> PlanResult:
    target = profile.target if target is None else target
    n = profile.n
    oracle = MarkOracle(model, profile, clicks_per_session, envelope)
    csp_memo, stats = {}, {"calls": 0, "assign": 0, "back": 0}
    rejected_enough = []
    best = {"key": None, "state": None, "marks": None}

    def schedule(state):
        """CSP check for a plan (memoised). Returns solution dict or None."""
        if state in csp_memo:
            return csp_memo[state]
        csp = build_csp(state, profile.deadlines, profile.slots_per_week, profile.blocked,
                        C.WINDOW_WEEKS, profile.first_week)
        sol = None
        if csp_precheck and hall_infeasible(state, csp):
            sol = None
        else:
            r = solve(csp, use_fc=csp_fc, use_ac3=csp_ac3, var_order="mrv",
                      max_assignments=max_csp_assignments)
            stats["calls"] += 1
            stats["assign"] += r.assignments
            stats["back"] += r.backtracks
            sol = (r.solution, csp) if r.solved else None
        csp_memo[state] = sol
        return sol

    def short(state):
        return [max(0.0, target - m) for m in oracle.marks(state)]

    def feasible(state, enough):
        """NN-not-enough nodes only need a feasibility verdict: use the counting test when
        csp_precheck is on (exact for this model). NN-enough nodes always get the full CSP."""
        if enough or not csp_precheck:
            return schedule(state) is not None
        csp = build_csp(state, profile.deadlines, profile.slots_per_week, profile.blocked,
                        C.WINDOW_WEEKS, profile.first_week)
        return not hall_infeasible(state, csp)

    def evaluate(state):
        marks = oracle.marks(state)                       # 1) NN says how good the plan is
        enough = all(m >= target - 1e-9 for m in marks)
        if not feasible(state, enough):                   # 2) CSP says if it can be scheduled
            if enough:
                rejected_enough.append((state, list(marks)))
            return "reject"
        key = (-sum(short(state)), -sum(state))
        if best["key"] is None or key > best["key"]:
            best.update(key=key, state=state, marks=list(marks))
        return "goal" if enough else "continue"

    gains = [oracle.max_gain(i, cap) for i in range(n)]

    def heuristic(state):                                  # consistent: each session lowers
        h = 0                                              # ceil(short/gain) by at most 1
        for sh, g in zip(short(state), gains):
            if sh > 1e-9:
                h += ceil(sh / g - 1e-9) if g > 1e-12 else INF
        return h

    h_for = {"ucs": (lambda s: 0), "greedy": heuristic, "astar": heuristic}[algorithm]
    if algorithm == "greedy":                              # greedy: smooth shortfall guide
        h_for = lambda s: sum(short(s))
    res = search(n, cap, evaluate, h_for, algorithm, trace=trace)

    base = dict(algorithm=algorithm, search=res, nn_calls=oracle.calls, csp_calls=stats["calls"],
                csp_assignments=stats["assign"], csp_backtracks=stats["back"],
                nn_enough_csp_rejected=rejected_enough)
    if res.found:
        sol, csp = schedule(res.solution)
        tt = sorted((*csp.slot_label(v), a, k + 1) for (a, k), v in sol.items())
        marks = oracle.marks(res.solution)
        return PlanResult("OK", allocation=res.solution, marks=marks, timetable=tt,
                          message=f"Plan found: {sum(res.solution)} sessions.", **base)
    bm = best["marks"] or oracle.marks((0,) * n)
    msg = (f"No solution: target {target:g} is unreachable at this capacity "
           f"({profile.slots_per_week} slots/week). Best predicted marks possible: "
           f"{[round(m, 1) for m in bm]} with allocation {best['state']}.")
    return PlanResult("UNREACHABLE", allocation=None, marks=None, timetable=[], message=msg,
                      best_feasible=(best["state"], bm), **base)
