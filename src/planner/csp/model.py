"""Scheduling CSP.

Variables : one per study session, id = (assignment a, k) with k = 0..n_a-1.
Values    : integer slot ids = week * slots_per_week + slot_index.
Constraints (all binary):
    * all-different  : one session per slot
    * order          : (a,k) must be in an earlier slot than (a,k+1)
Unary (applied when building domains): session must lie in the allowed weeks
(before the assignment's deadline, inside the 4-week window the NN looks at) and
the slot must not be blocked.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations


@dataclass
class ScheduleCSP:
    variables: list                 # [(a, k), ...] in default (static) order
    domains: dict                   # var -> sorted list of slot ids
    slots_per_week: int

    def consistent(self, x, vx, y, vy) -> bool:
        if vx == vy:
            return False
        if x[0] == y[0]:
            return (vx < vy) if x[1] < y[1] else (vx > vy)
        return True

    def neighbours(self):
        nb = {v: [] for v in self.variables}
        for x, y in combinations(self.variables, 2):
            nb[x].append(y)
            nb[y].append(x)
        return nb

    def slot_label(self, slot_id):
        return slot_id // self.slots_per_week, slot_id % self.slots_per_week


def build_csp(plan, deadline_weeks, slots_per_week, blocked=frozenset(),
              window_weeks=4, first_week=0) -> ScheduleCSP:
    """plan = sessions per assignment; deadline_weeks[a] = week index of the deadline.
    Sessions may use weeks [deadline-window, deadline-1] (>= first_week)."""
    variables, domains = [], {}
    for a, n in enumerate(plan):
        lo = max(first_week, deadline_weeks[a] - window_weeks)
        hi = deadline_weeks[a] - 1
        dom = [w * slots_per_week + j
               for w in range(lo, hi + 1) for j in range(slots_per_week)
               if (w, j) not in blocked]
        for k in range(n):
            v = (a, k)
            variables.append(v)
            domains[v] = list(dom)
    return ScheduleCSP(variables, domains, slots_per_week)


def hall_infeasible(plan, csp: ScheduleCSP) -> bool:
    """Necessary (and, for this model, sufficient) counting test on every subset of
    assignments. Used only as an optional fast pre-check in the integrated planner."""
    n = len(plan)
    doms = []
    for a in range(n):
        doms.append(set(csp.domains.get((a, 0), [])) if plan[a] else set())
    for r in range(1, n + 1):
        for sub in combinations(range(n), r):
            need = sum(plan[a] for a in sub)
            have = set().union(*(doms[a] for a in sub))
            if need > len(have):
                return True
    return False
