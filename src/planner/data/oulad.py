"""OULAD loading, chunked click aggregation and assessment-level feature building."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from .. import config as C

REQUIRED = {
    "assessments.csv": ["code_module", "code_presentation", "id_assessment",
                        "assessment_type", "date", "weight"],
    "studentAssessment.csv": ["id_assessment", "id_student", "date_submitted",
                              "is_banked", "score"],
    "vle.csv": ["id_site", "code_module", "code_presentation", "activity_type"],
    "studentVle.csv": ["code_module", "code_presentation", "id_student",
                       "id_site", "date", "sum_click"],
}


def check_schema(raw_dir=C.RAW_DIR):
    """Week-1 check: exact table / column names must exist (clear error otherwise)."""
    for fname, cols in REQUIRED.items():
        p = raw_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run make_data.py (see README).")
        have = list(pd.read_csv(p, nrows=0).columns)
        miss = [c for c in cols if c not in have]
        if miss:
            raise ValueError(f"{fname}: missing columns {miss}; found {have}")


def group_of(activity_type: str) -> str:
    for g, types in C.CLICK_GROUPS.items():
        if activity_type in types:
            return g
    return "practice"


def weekly_clicks(raw_dir=C.RAW_DIR, module=C.MODULE, chunksize=500_000) -> pd.DataFrame:
    """Filter studentVle in chunks to one module, aggregate days -> weeks per click group."""
    vle = pd.read_csv(raw_dir / "vle.csv", usecols=["id_site", "code_module",
                      "code_presentation", "activity_type"])
    vle = vle[vle.code_module == module].copy()
    vle["group"] = vle.activity_type.map(group_of)
    site_group = vle[["id_site", "code_presentation", "group"]].drop_duplicates(
        ["id_site", "code_presentation"])
    parts = []
    for chunk in pd.read_csv(raw_dir / "studentVle.csv", chunksize=chunksize):
        chunk = chunk[chunk.code_module == module]
        if chunk.empty:
            continue
        chunk = chunk.merge(site_group, on=["id_site", "code_presentation"], how="left")
        chunk["group"] = chunk["group"].fillna("practice")
        chunk["week"] = chunk["date"] // 7
        parts.append(chunk.groupby(["code_presentation", "id_student", "week", "group"],
                                   as_index=False)["sum_click"].sum())
    wk = pd.concat(parts, ignore_index=True)
    return wk.groupby(["code_presentation", "id_student", "week", "group"],
                      as_index=False)["sum_click"].sum()


def build_assessment_table(raw_dir=C.RAW_DIR, wk: pd.DataFrame | None = None,
                           module=C.MODULE) -> pd.DataFrame:
    """One row per (presentation, student, assessment) with the 12 features + score."""
    wk = weekly_clicks(raw_dir, module) if wk is None else wk
    ass = pd.read_csv(raw_dir / "assessments.csv")
    ass = ass[(ass.code_module == module) & ass.assessment_type.isin(C.ASSESSMENT_TYPES)
              & ass.date.notna()].copy()
    ass["deadline_week"] = (ass["date"] // 7).astype(int)
    ass = ass.sort_values(["code_presentation", "date"])
    ass["assessment_pos"] = ass.groupby("code_presentation").cumcount() + 1
    sa = pd.read_csv(raw_dir / "studentAssessment.csv")
    sa = sa[(sa.is_banked == 0) & sa.score.notna()]
    df = sa.merge(ass[["id_assessment", "code_presentation", "assessment_type", "weight",
                       "deadline_week", "assessment_pos", "date"]], on="id_assessment")
    df["is_cma"] = (df.assessment_type == "CMA").astype(int)
    df["week_since_start"] = df.deadline_week
    # ---- clicks in windows (join weekly table, keep weeks inside the window) ---------
    piv = wk.pivot_table(index=["code_presentation", "id_student", "week"],
                         columns="group", values="sum_click", aggfunc="sum",
                         fill_value=0).reset_index()
    for g in C.GROUP_NAMES:
        if g not in piv:
            piv[g] = 0
    m = df[["code_presentation", "id_student", "id_assessment", "deadline_week"]].merge(
        piv, on=["code_presentation", "id_student"], how="left")
    rel = m.deadline_week - m.week
    win = m[(rel >= 1) & (rel <= C.WINDOW_WEEKS)]
    lag = m[(rel > C.WINDOW_WEEKS) & (rel <= C.WINDOW_WEEKS + C.LAG_WEEKS)]
    keys = ["code_presentation", "id_student", "id_assessment"]
    w = win.groupby(keys)[C.GROUP_NAMES].sum()
    w.columns = [f"clicks_{g}" for g in C.GROUP_NAMES]
    l = lag.groupby(keys)[C.GROUP_NAMES].sum().sum(axis=1).rename("clicks_lag_total")
    df = df.merge(w.reset_index(), on=keys, how="left").merge(l.reset_index(), on=keys, how="left")
    for c in [f"clicks_{g}" for g in C.GROUP_NAMES] + ["clicks_lag_total"]:
        df[c] = df[c].fillna(0.0)
    # ---- previous scores (same student, same presentation, earlier deadline) ---------
    df = df.sort_values(["code_presentation", "id_student", "date"])
    gb = df.groupby(["code_presentation", "id_student"])["score"]
    df["prev_last"] = gb.shift(1)
    df["prev_mean"] = gb.transform(lambda s: s.expanding().mean().shift(1))
    df["prev_min"] = gb.transform(lambda s: s.expanding().min().shift(1))
    df["n_prev"] = gb.cumcount()
    return df.reset_index(drop=True)


def finalize(df: pd.DataFrame, train_presentations=C.TRAIN_PRESENTATIONS):
    """Impute missing previous scores with the TRAIN mean score; compute generated rule G1."""
    tr = df[df.code_presentation.isin(train_presentations)]
    fill = float(tr.score.mean())
    for c in ["prev_last", "prev_mean", "prev_min"]:
        df[c] = df[c].fillna(fill)
    tot = tr[[f"clicks_{g}" for g in C.GROUP_NAMES]].sum(axis=1)
    med = float(tot[tot > 0].median())
    cps = max(1, int(round(med / C.SESSIONS_PER_WINDOW_REF)))
    meta = {"module": C.MODULE, "train": list(train_presentations),
            "test": C.TEST_PRESENTATIONS, "prev_fill": fill,
            "clicks_per_session": cps, "median_window_clicks_train": med}
    return df, meta


def save(df, meta):
    C.PROC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.PROC_DIR / "assessment_features.csv", index=False)
    (C.PROC_DIR / "meta.json").write_text(json.dumps(meta, indent=2))


def load():
    df = pd.read_csv(C.PROC_DIR / "assessment_features.csv")
    meta = json.loads((C.PROC_DIR / "meta.json").read_text())
    return df, meta


def inspect(raw_dir=C.RAW_DIR):
    """Week-1 helper: list modules/presentations/assessment structure."""
    ass = pd.read_csv(raw_dir / "assessments.csv")
    print(ass.groupby(["code_module", "code_presentation"]).assessment_type
          .value_counts().unstack(fill_value=0).to_string())
