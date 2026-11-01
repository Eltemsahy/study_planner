"""Generic graph search over study plans with node counters.

State  : tuple of sessions per assignment, e.g. (4, 5, 3). Start state = zeros.
Action : add ONE session to ONE assignment (step cost 1), so g(n) = total sessions.
Algorithms (same engine, different priority):
    ucs    : f = g
    greedy : f = h
    astar  : f = g + h   (h must be admissible/consistent for optimality)

`evaluate(state)` is supplied by the caller and returns (is_goal, info). The planner
uses it to run NN -> CSP. If it returns is_goal=None the node is *rejected*
(its subtree is pruned: more sessions can never repair an infeasible schedule).
"""
from __future__ import annotations
import heapq
import itertools
from dataclasses import dataclass, field
from typing import Callable, Optional

State = tuple


@dataclass
class SearchResult:
    algorithm: str
    solution: Optional[State]
    cost: Optional[int]
    nodes_expanded: int
    nodes_generated: int
    nodes_rejected: int = 0
    trace: list = field(default_factory=list)   # (step, state, g, h, verdict)

    @property
    def found(self):
        return self.solution is not None


def successors(state: State, cap: int):
    for i, s in enumerate(state):
        if s < cap:
            yield i, state[:i] + (s + 1,) + state[i + 1:]


def search(n_assign: int, cap: int,
           evaluate: Callable[[State], tuple],
           heuristic: Callable[[State], float] = lambda s: 0.0,
           algorithm: str = "ucs", trace: bool = False) -> SearchResult:
    """evaluate(state) -> (verdict, h_info) with verdict in {"goal","continue","reject"}."""
    if algorithm not in ("ucs", "greedy", "astar"):
        raise ValueError(f"unknown algorithm {algorithm}")
    start = (0,) * n_assign
    tie = itertools.count()

    def prio(g, h):
        return {"ucs": g, "greedy": h, "astar": g + h}[algorithm]

    h0 = heuristic(start)
    frontier = [(prio(0, h0), h0, next(tie), start, 0)]
    best_g = {start: 0}
    closed = set()
    res = SearchResult(algorithm, None, None, 0, 1)
    while frontier:
        _, h, _, state, g = heapq.heappop(frontier)
        if state in closed or g > best_g.get(state, g):
            continue
        closed.add(state)
        verdict = evaluate(state)
        res.nodes_expanded += 1          # a node counts as expanded once popped & tested
        if trace:
            res.trace.append((res.nodes_expanded, state, g, h, verdict))
        if verdict == "goal":
            res.solution, res.cost = state, g
            return res
        if verdict == "reject":
            res.nodes_rejected += 1
            continue
        for _, nxt in successors(state, cap):
            g2 = g + 1
            if nxt in closed or g2 >= best_g.get(nxt, 1 << 60):
                continue
            best_g[nxt] = g2
            h2 = heuristic(nxt)
            heapq.heappush(frontier, (prio(g2, h2), h2, next(tie), nxt, g2))
            res.nodes_generated += 1
    return res
