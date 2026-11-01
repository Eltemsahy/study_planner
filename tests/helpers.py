import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from planner import config as C
from planner.data.profiles import Profile


class LinearModel:
    """Toy 'NN': mark = 40 + 0.2 * total clicks  (so +2 marks per session when cps = 10)."""
    uses_clicks = True

    def predict(self, df):
        tot = df[[f"clicks_{g}" for g in C.GROUP_NAMES]].sum(axis=1).values
        return np.clip(40 + 0.2 * tot, 0, 100)


def toy_profile(target=55, slots=3, deadlines=(8, 12, 16), blocked=frozenset(), n=3, base_clicks=0.0):
    base = [{**{f: 0.0 for f in C.FEATURES}, "clicks_content": base_clicks} for _ in range(n)]
    dl = list(deadlines)[:n]
    return Profile("T", 1, base, dl, dl, slots, blocked, float(target), [0.0] * n,
                   first_week=max(0, dl[0] - C.WINDOW_WEEKS))


def valid_schedule(csp, sol):
    vs = list(sol)
    ok = len(set(sol.values())) == len(vs)
    for v in vs:
        ok &= sol[v] in csp.domains[v]
        if (v[0], v[1] + 1) in sol:
            ok &= sol[v] < sol[(v[0], v[1] + 1)]
    return bool(ok)
