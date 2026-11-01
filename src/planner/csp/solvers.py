"""Backtracking with switchable Forward Checking and AC-3, all with counters."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from .model import ScheduleCSP


@dataclass
class CSPResult:
    status: str                       # "solved" | "no_solution" | "budget_exceeded"
    solution: Optional[dict]
    assignments: int = 0              # every value tried for a variable
    backtracks: int = 0               # every time a variable runs out of values / FC wipe-out
    ac3_revisions: int = 0
    ac3_removed: int = 0
    fc_removed: int = 0
    trace: list = field(default_factory=list)

    @property
    def solved(self):
        return self.status == "solved"

    @property
    def message(self):
        return {"solved": "Schedule found.",
                "no_solution": "No solution: the sessions cannot be scheduled in the available slots.",
                "budget_exceeded": "Search budget exceeded before a decision."}[self.status]


def ac3(csp: ScheduleCSP, domains: dict, res: CSPResult, nb=None) -> bool:
    """Prune domains to arc consistency in place. False if some domain is wiped out."""
    nb = nb or csp.neighbours()
    queue = deque((x, y) for x in csp.variables for y in nb[x])
    while queue:
        x, y = queue.popleft()
        res.ac3_revisions += 1
        keep = [vx for vx in domains[x]
                if any(csp.consistent(x, vx, y, vy) for vy in domains[y])]
        removed = len(domains[x]) - len(keep)
        if removed:
            res.ac3_removed += removed
            domains[x] = keep
            if not keep:
                return False
            for z in nb[x]:
                if z != y:
                    queue.append((z, x))
    return True


def solve(csp: ScheduleCSP, use_fc: bool = False, use_ac3: bool = False,
          var_order: str = "static", max_assignments: int = 500_000,
          trace: bool = False) -> CSPResult:
    """Plain BT by default; use_fc adds forward checking; use_ac3 runs AC-3 once
    before search (and, with FC on, FC keeps domains consistent with assignments)."""
    res = CSPResult("no_solution", None)
    nb = csp.neighbours()
    domains = {v: list(d) for v, d in csp.domains.items()}
    if any(not d for d in domains.values()):
        return res
    if use_ac3 and not ac3(csp, domains, res, nb):
        return res
    assignment = {}

    def pick(doms):
        un = [v for v in csp.variables if v not in assignment]
        if var_order == "mrv":
            return min(un, key=lambda v: len(doms[v]))
        return un[0]

    def rec(doms) -> Optional[bool]:
        if len(assignment) == len(csp.variables):
            return True
        x = pick(doms)
        for vx in list(doms[x]):
            if res.assignments >= max_assignments:
                return None
            res.assignments += 1
            # consistency with already-assigned variables (plain BT check)
            if any(not csp.consistent(x, vx, y, assignment[y]) for y in assignment if y in nb[x]):
                continue
            assignment[x] = vx
            if trace:
                res.trace.append(("assign", x, vx))
            new = doms
            ok = True
            if use_fc:
                new = {v: list(d) for v, d in doms.items()}
                new[x] = [vx]
                for y in nb[x]:
                    if y in assignment:
                        continue
                    keep = [vy for vy in new[y] if csp.consistent(x, vx, y, vy)]
                    res.fc_removed += len(new[y]) - len(keep)
                    new[y] = keep
                    if not keep:
                        ok = False
                        break
            if ok:
                r = rec(new)
                if r is None:
                    return None
                if r:
                    return True
            del assignment[x]
            res.backtracks += 1
            if trace:
                res.trace.append(("backtrack", x, vx))
        return False

    r = rec(domains)
    if r is None:
        res.status = "budget_exceeded"
    elif r:
        res.status, res.solution = "solved", dict(assignment)
    return res
