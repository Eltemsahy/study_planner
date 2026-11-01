import unittest
from itertools import product
from helpers import *
from planner.csp.model import build_csp, hall_infeasible
from planner.csp.solvers import solve

MODES = [(0, 0), (1, 0), (0, 1), (1, 1)]

class CSPTests(unittest.TestCase):
    def test_normal_all_modes_solve_validly(self):
        csp = build_csp((3, 3, 3), (4, 5, 6), 2)
        for fc, ac in MODES:
            r = solve(csp, bool(fc), bool(ac)); self.assertTrue(r.solved)
            self.assertTrue(valid_schedule(csp, r.solution))

    def test_normal_fc_reduces_assignments(self):
        csp = build_csp((3, 3, 3), (4, 5, 6), 2)
        self.assertLess(solve(csp, True).assignments, solve(csp).assignments)

    def test_normal_ac3_prunes_domains(self):
        csp = build_csp((3, 1), (5, 5), 1)               # 4 sessions in exactly 4 slots force pruning
        r = solve(csp, use_ac3=True); self.assertGreater(r.ac3_removed, 0); self.assertTrue(r.solved)

    def test_normal_mrv_order_valid(self):
        csp = build_csp((2, 2, 2), (3, 3, 4), 2)
        r = solve(csp, True, True, var_order="mrv"); self.assertTrue(valid_schedule(csp, r.solution))

    def test_normal_ordering_respected(self):
        csp = build_csp((4,), (6,), 1); r = solve(csp)
        slots = [r.solution[(0, k)] for k in range(4)]; self.assertEqual(slots, sorted(slots))

    def test_edge_no_sessions(self):
        r = solve(build_csp((0, 0), (5, 6), 2)); self.assertTrue(r.solved); self.assertEqual(r.solution, {})

    def test_edge_exact_fit(self):
        csp = build_csp((4,), (4,), 1)                   # window weeks 0..3, 1 slot/week, 4 sessions
        self.assertEqual(len(csp.domains[(0, 0)]), 4)
        for fc, ac in MODES: self.assertTrue(solve(csp, bool(fc), bool(ac)).solved)

    def test_edge_blocked_slots_removed_from_domain(self):
        csp = build_csp((1,), (4,), 2, blocked=frozenset({(3, 0), (3, 1)}))
        self.assertTrue(all(v // 2 != 3 for v in csp.domains[(0, 0)]))

    def test_impossible_more_sessions_than_slots(self):
        csp = build_csp((3, 3), (3, 3), 1)               # 3 slots for 6 sessions
        for fc, ac in MODES:
            r = solve(csp, bool(fc), bool(ac))
            self.assertEqual(r.status, "no_solution"); self.assertIn("No solution", r.message)
        self.assertTrue(hall_infeasible((3, 3), csp))

    def test_impossible_deadline_leaves_no_slot(self):
        csp = build_csp((1,), (0,), 3)                   # deadline in week 0: nothing before it
        r = solve(csp); self.assertEqual(r.status, "no_solution"); self.assertIn("No solution", r.message)

    def test_normal_budget_is_reported(self):
        csp = build_csp((6, 6), (3, 3), 2)
        self.assertEqual(solve(csp, max_assignments=50).status, "budget_exceeded")

if __name__ == "__main__":
    unittest.main()
