"""All figures (matplotlib, Agg backend)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .. import config as C

COL = {"ucs": "#4C72B0", "greedy": "#DD8452", "astar": "#55A868",
       "BT": "#C44E52", "BT+FC": "#8172B3", "BT+FC+AC3": "#55A868",
       "NN": "#4C72B0", "Average-mark": "#999999"}


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = C.FIG_DIR / name
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    return p


def plot_exp1(s: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    for a, (col, title) in zip(ax, [("expanded", "Nodes expanded"), ("cost", "Plan cost (sessions)")]):
        a.bar(s.algorithm, s[f"{col}_mean"], yerr=s[f"{col}_sd"], capsize=4,
              color=[COL[x] for x in s.algorithm])
        a.set_title(title + " (mean ± sd over seeds)", fontsize=9)
    return _save(fig, "exp1_search.png")


def plot_exp2(s: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    modes = ["BT", "BT+FC", "BT+FC+AC3"]
    x = np.arange(3); w = 0.26
    for a, col in zip(ax, ["assignments", "backtracks"]):
        for i, m in enumerate(modes):
            d = s[s["mode"] == m].sort_values("slots_per_week")
            a.bar(x + (i - 1) * w, d[f"{col}_mean"], w, yerr=d[f"{col}_sd"], capsize=3,
                  label=m, color=COL[m])
        a.set_xticks(x); a.set_xticklabels(["2", "3", "4"]); a.set_xlabel("slots per week")
        a.set_yscale("log"); a.set_title(f"CSP {col} (log, mean ± sd)", fontsize=9)
    ax[0].legend(fontsize=8)
    return _save(fig, "exp2_csp.png")


def plot_exp3(s: pd.DataFrame):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for a, (col, t) in zip(ax, [("found", "Share of profiles with a plan"),
                                ("sessions", "Mean sessions (solved)"),
                                ("base_abs_err", "MAE of 0-session prediction vs real marks")]):
        a.bar(s.predictor, s[f"{col}_mean"], yerr=s[f"{col}_sd"].fillna(0), capsize=4,
              color=[COL[x] for x in s.predictor])
        a.set_title(t, fontsize=9)
    return _save(fig, "exp3_nn_vs_avg.png")


def plot_monotonicity(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.hist(df.max_drop, bins=15, color="#C44E52")
    ax.set_xlabel("largest drop in predicted mark when adding one session")
    ax.set_ylabel("curves"); ax.set_title("Raw NN monotonicity violations", fontsize=9)
    return _save(fig, "monotonicity.png")


# ------------------------------------------------------------------ demo figures
def plot_search_trace(traces: dict, name="demo_search_trace.png"):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for alg, tr in traces.items():
        ax.plot([t[0] for t in tr], [t[2] for t in tr], label=alg, color=COL[alg], lw=1.6)
    ax.set_xlabel("expansion #"); ax.set_ylabel("g = total sessions of expanded node")
    ax.set_title("Search trace: depth of expanded nodes", fontsize=9); ax.legend()
    return _save(fig, name)


def plot_csp_pruning(rows: dict, name="demo_csp_pruning.png"):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    labels = list(rows)
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w / 2, [rows[k]["assignments"] for k in labels], w, label="assignments", color="#4C72B0")
    ax.bar(x + w / 2, [rows[k]["backtracks"] for k in labels], w, label="backtracks", color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_yscale("symlog")
    ax.set_title("CSP effort on the same plan", fontsize=9); ax.legend()
    return _save(fig, name)


def plot_marks(curves: dict, target, chosen, name="demo_marks.png"):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, (lab, ys) in enumerate(curves.items()):
        ax.plot(range(len(ys)), ys, marker=".", label=lab)
        if chosen is not None:
            ax.scatter([chosen[i]], [ys[chosen[i]]], s=70, edgecolor="k", zorder=5)
    ax.axhline(target, color="k", ls="--", lw=1, label=f"target {target:g}")
    ax.set_xlabel("sessions"); ax.set_ylabel("predicted mark")
    ax.set_title("NN: predicted mark vs sessions (large dot = plan)", fontsize=9); ax.legend(fontsize=7)
    return _save(fig, name)


def plot_timetable(timetable, slots, deadlines, name="demo_timetable.png"):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    cols = ["#4C72B0", "#DD8452", "#55A868", "#8172B3"]
    if timetable:
        for (w, j, a, k) in timetable:
            ax.add_patch(plt.Rectangle((w, j), 0.9, 0.9, color=cols[a % 4]))
            ax.text(w + .45, j + .45, f"A{a+1}.{k}", ha="center", va="center", fontsize=7, color="w")
        w0 = min(t[0] for t in timetable) - 1; w1 = max(max(t[0] for t in timetable), max(deadlines)) + 1
    else:
        w0, w1 = 0, max(deadlines) + 1
    for a, d in enumerate(deadlines):
        ax.axvline(d, color=cols[a % 4], ls="--"); ax.text(d, slots + .05, f"A{a+1} due", fontsize=7)
    ax.set_xlim(w0, w1); ax.set_ylim(0, slots + .5)
    ax.set_xlabel("week"); ax.set_ylabel("slot"); ax.set_title("Week-by-week timetable", fontsize=9)
    return _save(fig, name)


def plot_nn_on_off(rows: dict, name="demo_nn_on_off.png"):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    labels = list(rows)
    ax.bar(labels, [rows[k] for k in labels], color=[COL.get(k, "#999") for k in labels])
    ax.set_ylabel("sessions in plan (0 = none found/needed)")
    ax.set_title("Planner with NN ON vs OFF (average-mark predictor)", fontsize=9)
    return _save(fig, name)
