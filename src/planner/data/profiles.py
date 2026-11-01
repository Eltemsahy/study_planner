"""Student profiles for the planner (GENERATED combination of real history + rules G2-G4)."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .. import config as C


@dataclass
class Profile:
    pid: str
    student: int
    base: list                 # one feature dict per planned assignment (real data)
    real_deadlines: list       # absolute weeks of the real deadlines
    deadlines: list            # deadlines used by the CSP (may be compressed, rule G4)
    slots_per_week: int
    blocked: frozenset
    target: float
    actual_scores: list        # real marks (reference only, never used by the planner)
    first_week: int = 0
    compressed: bool = False

    @property
    def n(self):
        return len(self.base)


def make_profiles(df: pd.DataFrame, n=30, seed=C.SEED, presentations=None,
                  n_planned=C.N_PLANNED, model=None) -> list:
    rng = np.random.default_rng(seed)
    pres = presentations or C.TEST_PRESENTATIONS
    d = df[df.code_presentation.isin(pres)]
    last = sorted(d.assessment_pos.unique())[-n_planned:]
    rows = d[d.assessment_pos.isin(last)]
    counts = rows.groupby("id_student").assessment_pos.nunique()
    students = counts[counts == n_planned].index.values
    if len(students) == 0:
        raise ValueError("No student has marks for all planned assessments.")
    pick = rng.choice(students, size=min(n, len(students)), replace=False)
    out = []
    for i, sid in enumerate(pick):
        r = rows[rows.id_student == sid].sort_values("date")
        base = []
        first = r.iloc[0]
        for _, row in r.iterrows():
            f = {c: float(row[c]) for c in C.FEATURES}
            for c in ["prev_mean", "prev_last", "prev_min", "n_prev"]:   # history snapshot
                f[c] = float(first[c])
            base.append(f)
        real_dl = [int(x) for x in r.deadline_week.values]
        if model is not None:       # rule G3: target = lowest baseline predicted mark + uplift
            p0 = model.predict(pd.DataFrame(base))
            target = float(np.clip(round(p0.min() + rng.choice(C.TARGET_UPLIFTS)), 40, 95))
        else:                       # fallback without a model: earlier mean mark + uplift
            target = float(np.clip(round(first.prev_mean + rng.choice(C.TARGET_UPLIFTS)), 40, 95))
        slots = int(rng.choice([2, 3, 4]))
        compressed = bool(rng.random() < 0.5)                              # rule G4
        dl = list(real_dl)
        if compressed:
            dl[1] = dl[0] + int(rng.integers(1, 3))
            dl[2] = dl[1] + int(rng.integers(1, 3))
        fw = max(0, dl[0] - C.WINDOW_WEEKS)
        blocked = frozenset((w, j) for w in range(fw, max(dl) + 1) for j in range(slots)
                            if rng.random() < C.BLOCK_PROB)                # rule G2
        out.append(Profile(f"P{i+1:02d}", int(sid), base, real_dl, dl, slots, blocked,
                           target,
                           [float(x) for x in r.score.values], fw, compressed))
    return out
