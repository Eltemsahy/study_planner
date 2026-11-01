"""Central configuration: paths, constants, and the documented generation rules."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

SEED = 42

# ---- Dataset choice (confirm with `python make_data.py --inspect`) -------------
MODULE = "FFF"
TRAIN_PRESENTATIONS = ["2013B", "2013J", "2014B"]
TEST_PRESENTATIONS = ["2014J"]
ASSESSMENT_TYPES = ["TMA", "CMA"]          # exams are excluded

# ---- Click activity groups (3 groups, from vle.activity_type) -------------------
CLICK_GROUPS = {
    "content": ["resource", "oucontent", "url", "page", "subpage", "glossary",
                "htmlactivity", "folder", "sharedsubpage"],
    "interaction": ["forumng", "oucollaborate", "ouelluminate", "ouwiki",
                    "dualpane", "dataplus"],
    # everything else (quiz, homepage, questionnaire, repeatactivity, ...)
    "practice": [],
}
GROUP_NAMES = list(CLICK_GROUPS)

WINDOW_WEEKS = 4        # clicks counted in the 4 weeks before the deadline
LAG_WEEKS = 4           # extra feature: the 4 weeks before that window

FEATURES = [
    "clicks_content", "clicks_interaction", "clicks_practice",   # 3 activity groups
    "clicks_lag_total",                                          # earlier 4 weeks
    "prev_mean", "prev_last", "prev_min", "n_prev",              # previous scores
    "is_cma", "weight", "week_since_start", "assessment_pos",    # assessment info
]
TARGET = "score"

# ---- GENERATED quantities (documented rule, see DATA_DICTIONARY.md) --------------
# Rule G1: one study session = CLICKS_PER_SESSION clicks. It is set in make_data.py as
#   round(median positive 4-week click total of the training rows / SESSIONS_PER_WINDOW_REF)
# i.e. a typical active student's 4-week activity is assumed to be ~8 sessions.
SESSIONS_PER_WINDOW_REF = 8
# Rule G2: weekly slot capacity per profile in {2,3,4}; each (week, slot) is blocked
#   with probability BLOCK_PROB (job, family, ...). Seeded.
BLOCK_PROB = 0.15
# Rule G3: target mark = lowest baseline predicted mark (0 sessions) of the 3 assignments
#   + uplift drawn (seeded) from TARGET_UPLIFTS, rounded, clipped to [40, 95].
TARGET_UPLIFTS = [0, 2, 4, 6]
MAX_SESSIONS_PER_ASSIGNMENT = 12   # search cap (finite state space)
N_PLANNED = 3                      # assignments planned together
SEEDS = [11, 22, 33, 44, 55]      # >= 5 runs for every experiment
N_EXP_PROFILES = 24                # profiles per seed in experiments 1 and 3
N_CSP_INSTANCES = 20               # random CSP instances per (seed, slots) in experiment 2
