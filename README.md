# P1 – Assessment-Aware Study Planner (Search + CSP + Neural Network)

Decision-support tool for a distance-learning student: how many study sessions per upcoming
assignment, and in which weekly slots, so that **predicted** marks reach a target with the fewest
sessions. Uses the Open University Learning Analytics Dataset (OULAD).

```
 Search (UCS / Greedy / A*)  --proposes plan-->  NN (predicted marks >= target?)
          ^                                              |  yes
          |   CSP says "no schedule": discard plan       v
          +---------------------------------------  CSP (BT / FC / AC-3) builds the timetable
```

> **Decision support only.** The link between effort (clicks) and marks in OULAD is
> *observational*. The planner is a what-if tool, not evidence that more sessions *cause* higher marks.

## 1. Environment
```bash
python3 -m venv .venv && source .venv/bin/activate        # Python 3.12 was used
pip install -r requirements.txt                            # versions are pinned
```

## 2. Exact run commands
```bash
python make_data.py --inspect      # Week-1 check: exact tables/columns, modules & presentations
python make_data.py                # download OULAD (or: --zip anonymisedData.zip), preprocess, seed 42
python train_nn.py                 # train on earlier presentations, test on the later one + baselines
python run_profiles.py             # allocation + timetable + predicted marks for 30 student profiles
python run_experiments.py          # 3 experiments x 5 seeds + monotonicity test -> results/
python run_demo.py                 # integrated demo (input, search trace, CSP pruning, NN on/off, rejection)
python -m unittest discover -s tests -v    # 37 tests (normal / edge / impossible)
```
Offline smoke test of the whole pipeline (NOT real data, flagged everywhere): `python make_data.py --synthetic`.

## 3. Layout (checklist item 1)
| Path | Purpose |
|---|---|
| `make_data.py` | download / preprocess, fixed seed (item 2) |
| `src/planner/data/` | `oulad.py` (chunked click aggregation, features), `profiles.py`, `synthetic.py` (test-only) |
| `src/planner/nn/` | `model.py`: MLP + baselines (item 4); top-level `train_nn.py` trains/evaluates |
| `src/planner/search/` | `algorithms.py`: UCS, Greedy, A* with counters (item 5) |
| `src/planner/csp/` | `model.py`, `solvers.py`: BT, FC, AC-3 switchable, counters (item 6) |
| `src/planner/integration/` | `planner.py`: Search -> NN -> CSP loop (item 7) |
| `src/planner/eval/` | `experiments.py`, `monotonicity.py`, `plots.py` (item 10) |
| `tests/` | 37 unit tests (item 9) |
| `DATA_DICTIONARY.md` | every column, real vs generated (item 3) |
| `docs/` | report outline, presentation/demo script, contribution statements (items 11-13) |

## 4. Key design decisions
* **State** = sessions per assignment, e.g. `(4,5,3)`; **action** = add one session; step cost 1.
* **Goal test** = NN predicts every mark >= target **and** the CSP finds a schedule.
  If the CSP fails the node is discarded and its subtree pruned (more sessions can't fix an infeasible schedule).
* **A\*** uses `h = sum_i ceil(shortfall_i / max_gain_i)`, where `max_gain_i` is the largest one-session
  gain of the NN curve for assignment *i*. It is admissible and consistent *with respect to that curve*.
* **Monotone envelope**: the raw NN can predict a lower mark after adding a session. The monotonicity
  test reports these violations on the raw network; the planner uses the running maximum so that
  search/heuristics stay sound (switch off with `envelope=False`).
* **CSP**: variables = sessions, values = weekly slots in the 4 weeks before the deadline (the same window the NN sees),
  all-different + in-order constraints, blocked slots removed from domains.
  `csp_precheck` (default on) uses a counting (Hall) test on non-goal nodes for speed; the demo/tests
  switch it off to show the real solver rejecting a plan. In this model the ordering constraint never
  makes feasibility harder, so the CSP mainly produces the timetable and the counters for Experiment 2.
* **Counters**: search = nodes expanded / generated / rejected; CSP = assignments (value tries),
  backtracks (undone assignments), AC-3 revisions/removed values, FC removed values. Note that BT and BT+FC
  often have the *same backtracks* but FC needs far fewer *assignments*.

## 5. Status / honest caveats
* The code was developed and tested against **synthetic OULAD-shaped tables** because the sandbox it was written in had no
  internet. **Run `make_data.py` yourself on the real data and re-check**: the download URL, the module/presentations
  in `config.py` (default FFF; train 2013B/2013J/2014B, test 2014J), and the activity-group mapping are unverified.
  Use `--inspect` to choose a module whose presentations share the same assessment structure.
* No result in `results/` may be quoted in the report unless `meta.json` says `"synthetic": false`.
* Planned assignments are the last 3 TMA/CMA of the test presentation; previous-score features are a snapshot taken before the
  first planned assignment (marks of later planned assignments are unknown at planning time).
* Sessions are assumed to add clicks (rule G1) — an assumption, not a measured effect.
