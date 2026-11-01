# Report outline (max 12 pages)
1. **Problem & users** (0.5 p) – study planning as decision support; what-if, not causal.
2. **Data** (1 p) – OULAD, chosen module, train/test presentations, features, generated parts (cite DATA_DICTIONARY).
3. **Design choices** (2 p) – state/action/cost; goal test; why subtree pruning is valid; envelope; CSP model; windows.
4. **Neural network** (1.5 p) – architecture, training, table: NN vs average-of-previous-marks vs global mean (MAE/RMSE/R²).
5. **Search** (1.5 p) – UCS/Greedy/A*, heuristic admissibility+consistency argument, Exp. 1 table/plot.
6. **CSP** (1.5 p) – constraints, BT/FC/AC-3, counters, Exp. 2 table/plot (note: equal backtracks BT vs FC).
7. **Integration & demo cases** (1 p) – loop, "NN enough but CSP rejects" example, unreachable-target report.
8. **Experiments** (1.5 p) – Exp. 3 NN vs average predictor; monotonicity violations (share of curves, max drop).
9. **Limits** (1 p) – **cause vs correlation** (mandatory), synthetic session→clicks rule G1, snapshot history, one course, envelope hides NN non-monotonicity, Hall pre-check makes CSP mostly a scheduler.
10. **Conclusions** (0.5 p).
