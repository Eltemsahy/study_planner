import unittest
from helpers import *
from planner.integration.planner import plan, MarkOracle
from planner.eval.monotonicity import check

M, CPS = LinearModel(), 10      # +2 marks / session ; base mark 40

class PlannerTests(unittest.TestCase):
    def test_normal_known_answer_all_algorithms(self):
        p = toy_profile(target=55)                       # 40 + 2s >= 55  -> s = 8 each
        for alg in ["ucs", "greedy", "astar"]:
            r = plan(p, M, CPS, alg)
            self.assertEqual((r.status, r.allocation), ("OK", (8, 8, 8)), alg)
            self.assertTrue(all(m >= 55 for m in r.marks))

    def test_normal_timetable_matches_allocation(self):
        r = plan(toy_profile(target=48), M, CPS, "ucs")   # 4 sessions each
        self.assertEqual(r.allocation, (4, 4, 4)); self.assertEqual(len(r.timetable), 12)
        self.assertEqual(len({(w, j) for w, j, _, _ in r.timetable}), 12)   # one session per slot

    def test_normal_only_needy_assignment_gets_sessions(self):
        p = toy_profile(target=50); p.base[1]["clicks_content"] = 100   # assignment 2 already 60
        self.assertEqual(plan(p, M, CPS, "astar").allocation, (5, 0, 5))

    def test_normal_astar_matches_ucs_cost(self):
        p = toy_profile(target=52)
        self.assertEqual(plan(p, M, CPS, "astar").total_sessions, plan(p, M, CPS, "ucs").total_sessions)

    def test_normal_nn_enough_but_csp_rejects(self):
        p = toy_profile(target=50, slots=1, deadlines=(5,), n=1)   # NN needs 5 sessions, window has 4 slots
        r = plan(p, M, CPS, "ucs", csp_precheck=False, max_csp_assignments=5000)
        self.assertEqual(r.status, "UNREACHABLE"); self.assertTrue(r.nn_enough_csp_rejected)

    def test_edge_target_zero_needs_nothing(self):
        r = plan(toy_profile(target=0), M, CPS, "ucs"); self.assertEqual(r.allocation, (0, 0, 0))

    def test_edge_target_exactly_at_baseline(self):
        r = plan(toy_profile(target=40), M, CPS, "ucs"); self.assertEqual(r.total_sessions, 0)

    def test_edge_single_assignment(self):
        r = plan(toy_profile(target=46, n=1), M, CPS, "ucs"); self.assertEqual(r.allocation, (3,))

    def test_edge_monotone_envelope(self):
        class Wobbly(LinearModel):
            def predict(self, df):
                tot = df[[f"clicks_{g}" for g in C.GROUP_NAMES]].sum(axis=1).values
                return super().predict(df) - 3 * np.isclose(tot, 20)
        raw = MarkOracle(Wobbly(), toy_profile(), CPS, envelope=False)
        env = MarkOracle(Wobbly(), toy_profile(), CPS, envelope=True)
        self.assertLess(raw.raw(0, 2), raw.raw(0, 1))                       # violation exists
        self.assertTrue(all(env.mark(0, s + 1) >= env.mark(0, s) for s in range(10)))
        _, summ = check([toy_profile()], Wobbly(), CPS, cap=6); self.assertGreater(summ["total_violating_steps"], 0)

    def test_impossible_target_above_100(self):
        r = plan(toy_profile(target=101), M, CPS, "ucs")
        self.assertEqual(r.status, "UNREACHABLE"); self.assertIn("No solution", r.message)
        self.assertIn("Best predicted marks", r.message)

    def test_impossible_no_free_slots(self):
        p = toy_profile(target=60, blocked=frozenset((w, j) for w in range(0, 20) for j in range(3)))
        r = plan(p, M, CPS, "ucs"); self.assertEqual(r.status, "UNREACHABLE"); self.assertIn("No solution", r.message)

if __name__ == "__main__":
    unittest.main()
