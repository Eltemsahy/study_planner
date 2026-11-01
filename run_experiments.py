#!/usr/bin/env python
"""Run the three experiments (5 seeds), the monotonicity test, tables and plots."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from planner import config as C
from planner.data import oulad
from planner.data.profiles import make_profiles
from planner.eval import experiments as E, plots, monotonicity
from planner.nn.model import MarkNN

C.RESULTS_DIR.mkdir(exist_ok=True)
df, meta = oulad.load()
if meta.get("synthetic"):
    print("!! WARNING: results below use SYNTHETIC data (pipeline test only) !!")


def save(name, raw, summ):
    raw.to_csv(C.RESULTS_DIR / f"{name}_raw.csv", index=False)
    summ.to_csv(C.RESULTS_DIR / f"{name}_summary.csv", index=False)
    (C.RESULTS_DIR / f"{name}_summary.md").write_text(summ.round(3).to_markdown(index=False)
                                                      if hasattr(summ, "to_markdown") else summ.round(3).to_string())
    print(f"\n== {name} ==\n{summ.round(3).to_string(index=False)}")


t = time.time()
raw, s = E.exp1_search(df, meta); save("exp1_search", raw, s); plots.plot_exp1(s)
raw, s = E.exp2_csp();            save("exp2_csp", raw, s);    plots.plot_exp2(s)
raw, s = E.exp3_nn_vs_avg(df, meta); save("exp3_nn_vs_avg", raw, s); plots.plot_exp3(s)

nn = MarkNN.load(C.MODEL_DIR / "mark_nn.joblib")
profs = make_profiles(df, 30, model=nn)
mono, summ = monotonicity.check(profs, nn, meta["clicks_per_session"])
mono.to_csv(C.RESULTS_DIR / "monotonicity_curves.csv", index=False)
(C.RESULTS_DIR / "monotonicity_summary.json").write_text(json.dumps(summ, indent=2))
plots.plot_monotonicity(mono)
print("\n== monotonicity (raw NN) ==", summ)
print(f"\nDone in {time.time()-t:.0f}s. Tables/plots in results/")
