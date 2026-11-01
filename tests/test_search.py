import unittest
from helpers import *
from planner.search.algorithms import search, successors

# Toy world with a KNOWN answer: mark_i = 20 + 10*s_i, target 55 -> exactly 4 sessions each, cost 12.
def ev(s):
    return "goal" if all(20 + 10 * x >= 55 for x in s) else "continue"
def h(s):
    return sum(max(0, -(-(55 - (20 + 10 * x)) // 10)) for x in s)

class SearchTests(unittest.TestCase):
    def test_normal_ucs_known_answer(self):
        r = search(3, 12, ev, h, "ucs")
        self.assertEqual((r.solution, r.cost), ((4, 4, 4), 12))
        self.assertGreater(r.nodes_expanded, 0); self.assertGreaterEqual(r.nodes_generated, r.nodes_expanded)

    def test_normal_greedy_known_answer(self):
        r = search(3, 12, ev, h, "greedy"); self.assertEqual(r.solution, (4, 4, 4))

    def test_normal_astar_optimal_and_cheaper_than_ucs(self):
        a, u = search(3, 12, ev, h, "astar"), search(3, 12, ev, h, "ucs")
        self.assertEqual(a.cost, u.cost); self.assertLess(a.nodes_expanded, u.nodes_expanded)

    def test_normal_trace_recorded(self):
        r = search(2, 5, ev, h, "ucs", trace=True); self.assertEqual(len(r.trace), r.nodes_expanded)

    def test_normal_successors_add_one_session(self):
        self.assertEqual(sorted(s for _, s in successors((1, 0), 3)), [(1, 1), (2, 0)])

    def test_edge_start_is_goal(self):
        r = search(2, 5, lambda s: "goal"); self.assertEqual((r.solution, r.cost, r.nodes_expanded), ((0, 0), 0, 1))

    def test_edge_cap_zero(self):
        r = search(2, 0, ev, h, "ucs"); self.assertFalse(r.found); self.assertEqual(r.nodes_expanded, 1)

    def test_impossible_cap_too_small(self):
        r = search(3, 3, ev, h, "ucs"); self.assertFalse(r.found)
        self.assertEqual(r.nodes_expanded, 4 ** 3)          # whole finite space explored

    def test_edge_reject_prunes_subtree(self):
        r = search(2, 5, lambda s: "reject" if s != (0, 0) else "continue", algorithm="ucs")
        self.assertFalse(r.found); self.assertEqual(r.nodes_rejected, 2)

if __name__ == "__main__":
    unittest.main()
