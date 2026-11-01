"""The three experiments (>= 5 seeds, mean +- sd over seeds)."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from .. import config as C
from ..csp.model import build_csp, hall_infeasible
from ..csp.solvers import solve
from ..data.profiles import make_profiles
from ..integration.planner import plan
from ..nn.model import MarkNN, PrevAverage


def _train_nn(df, meta, seed):
    return MarkNN(seed).fit(df[df.code_presentation.isin(meta["train"])])


def agg(df, by, cols):
    """mean and sd across seeds (each seed contributes one number per group)."""
    per_seed = df.groupby(by + ["seed"])[cols].mean().reset_index()
    g = per_seed.groupby(by)[cols]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).add_suffix("_sd"))
    return out.reset_index()


# ----------------------------------------------------------------------------- Exp 1
def exp1_search(df, meta, seeds=C.SEEDS, n_profiles=C.N_EXP_PROFILES):
    rows = []
    cps = meta["clicks_per_session"]
    for sd in seeds:
        nn = _train_nn(df, meta, sd)
        for p in make_profiles(df, n_profiles, seed=sd, model=nn):
            for alg in ["ucs", "greedy", "astar"]:
                r = plan(p, nn, cps, alg)
                rows.append({"seed": sd, "profile": p.pid, "algorithm": alg,
                             "found": int(r.status == "OK"),
                             "expanded": r.search.nodes_expanded,
                             "generated": r.search.nodes_generated,
                             "cost": r.total_sessions if r.status == "OK" else np.nan})
    raw = pd.DataFrame(rows)
    # compare on the SAME instances: reachable by all three algorithms
    ok = raw.groupby(["seed", "profile"]).found.transform("min") == 1
    reach = raw[ok]
    summary = agg(reach, ["algorithm"], ["expanded", "generated", "cost"])
    opt = reach.pivot_table(index=["seed", "profile"], columns="algorithm", values="cost")
    summary["suboptimal_share"] = [float((opt[a] > opt["ucs"]).mean()) for a in summary.algorithm]
    summary["n_instances"] = int(len(opt))
    return raw, summary


# ----------------------------------------------------------------------------- Exp 2
MODES = {"BT": (False, False), "BT+FC": (True, False), "BT+FC+AC3": (True, True)}


def random_csp(rng, slots, n_assign=4, fill=0.7):
    for _ in range(100):
        dl = [int(x) for x in rng.permutation(np.arange(4, 4 + n_assign))]   # NOT sorted
        blocked = frozenset((w, j) for w in range(0, 4 + n_assign) for j in range(slots)
                            if rng.random() < C.BLOCK_PROB)
        empty = build_csp((1,) * n_assign, dl, slots, blocked)
        union = set().union(*(set(empty.domains[(a, 0)]) for a in range(n_assign)))
        total = max(n_assign, int(fill * len(union)))
        cuts = np.sort(rng.choice(np.arange(1, total), n_assign - 1, replace=False))
        plan_ = tuple(int(x) for x in np.diff(np.concatenate([[0], cuts, [total]])))
        csp = build_csp(plan_, dl, slots, blocked)
        if not hall_infeasible(plan_, csp):
            return plan_, csp
    raise RuntimeError("could not generate a feasible instance")


def exp2_csp(seeds=C.SEEDS, slots_list=(2, 3, 4), n_inst=C.N_CSP_INSTANCES, budget=100_000):
    rows = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        for slots in slots_list:
            for k in range(n_inst):
                plan_, csp = random_csp(rng, slots)
                for mode, (fc, ac) in MODES.items():
                    t = time.perf_counter()
                    r = solve(csp, use_fc=fc, use_ac3=ac, var_order="static", max_assignments=budget)
                    rows.append({"seed": sd, "slots_per_week": slots, "mode": mode,
                                 "sessions": sum(plan_), "status": r.status,
                                 "assignments": r.assignments, "backtracks": r.backtracks,
                                 "ac3_removed": r.ac3_removed, "fc_removed": r.fc_removed,
                                 "ms": 1000 * (time.perf_counter() - t),
                                 "budget_exceeded": int(r.status == "budget_exceeded")})
    raw = pd.DataFrame(rows)
    summary = agg(raw, ["slots_per_week", "mode"],
                  ["assignments", "backtracks", "ms", "budget_exceeded"])
    return raw, summary


# ----------------------------------------------------------------------------- Exp 3
def exp3_nn_vs_avg(df, meta, seeds=C.SEEDS, n_profiles=C.N_EXP_PROFILES):
    rows = []
    cps = meta["clicks_per_session"]
    train = df[df.code_presentation.isin(meta["train"])]
    for sd in seeds:
        nn = _train_nn(df, meta, sd)
        avg = PrevAverage().fit(train)
        for p in make_profiles(df, n_profiles, seed=sd, model=nn):      # same targets for both
            for name, model in [("NN", nn), ("Average-mark", avg)]:
                r = plan(p, model, cps, "ucs")
                base_pred = model.predict(pd.DataFrame(p.base))
                rows.append({"seed": sd, "profile": p.pid, "predictor": name,
                             "found": int(r.status == "OK"),
                             "sessions": r.total_sessions if r.status == "OK" else np.nan,
                             "needs_effort": int(r.status == "OK" and r.total_sessions > 0),
                             "base_abs_err": float(np.mean(np.abs(base_pred - p.actual_scores)))})
    raw = pd.DataFrame(rows)
    summary = agg(raw, ["predictor"], ["found", "sessions", "needs_effort", "base_abs_err"])
    return raw, summary
