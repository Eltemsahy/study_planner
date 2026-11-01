# 10-min talk + 5-min live demo
**Talk (10 min):** problem (1) · data & NN (2) · search + heuristic (2) · CSP (2) · integration (1) · experiments & limits (2).
**Live demo (5 min)** – `python run_demo.py` (sections map to the required items):
1. Section 2 – *search trace*: UCS vs Greedy vs A* expansions, `results/figures/demo_search_trace.png`.
2. Section 3 – *CSP pruning with and without FC/AC-3* on the same plan and on a hard instance (assignments/backtracks table + `demo_csp_pruning.png`).
3. Section 4 – *system with NN on vs off* (average-mark predictor), timetable `demo_timetable.png`, marks curves `demo_marks.png`.
4. Section 5 – *NN says enough, CSP rejects*, then search continues.
5. Section 6 – impossible target → clear "No solution" message with best predicted marks.
Backup: pre-rendered figures in `results/figures/`.
