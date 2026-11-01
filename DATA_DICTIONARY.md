# Data dictionary

Legend: **REAL** = from OULAD; **DERIVED** = computed from real columns; **GENERATED** = produced by a documented rule (seeded);
**SYNTHETIC** = only when `make_data.py --synthetic` is used (offline test; not real, `data/raw/SYNTHETIC_DATA.txt` is written and `meta.json` has `"synthetic": true`).

## Raw tables used (REAL, CC BY 4.0)
| Table | Columns used | Meaning |
|---|---|---|
| `assessments.csv` | code_module, code_presentation, id_assessment, assessment_type, date, weight | assessment type (TMA/CMA/Exam), deadline (day), weight (%). Exams and rows without date are dropped |
| `studentAssessment.csv` | id_assessment, id_student, date_submitted, is_banked, score | mark 0-100 (**target**); banked rows dropped |
| `vle.csv` | id_site, code_module, code_presentation, activity_type | maps a VLE site to an activity type -> click group |
| `studentVle.csv` | code_module, code_presentation, id_student, id_site, date, sum_click | daily clicks (read in 500k-row chunks, one module only, aggregated to weeks) |

Not used (by rule): gender, age_band, region, disability, imd_band, highest_education (and any `studentInfo` demographics).

## Model dataset `data/processed/assessment_features.csv` (one row per student × assessment)
| Column | Kind | Definition |
|---|---|---|
| code_presentation, id_student, id_assessment | REAL ids | keys (not NN inputs) |
| `clicks_content` | DERIVED | clicks in the 4 weeks before the deadline on content sites (resource, oucontent, url, page, subpage, glossary, htmlactivity, folder, sharedsubpage) |
| `clicks_interaction` | DERIVED | same window, forum/collaboration/wiki/dualpane/dataplus |
| `clicks_practice` | DERIVED | same window, everything else (quiz, homepage, questionnaire, ...) |
| `clicks_lag_total` | DERIVED | all clicks in the 4 weeks before that window |
| `prev_mean`, `prev_last`, `prev_min` | DERIVED | mean / last / min of the student's earlier marks in the same presentation; missing -> mean training score (stored in meta.json) |
| `n_prev` | DERIVED | number of earlier marks |
| `is_cma` | DERIVED | 1 if assessment_type = CMA else 0 |
| `weight` | REAL | assessment weight |
| `week_since_start` | DERIVED | deadline day // 7 |
| `assessment_pos` | DERIVED | rank of the assessment by deadline within the presentation (not an NN input) |
| `score` | REAL | **target**, 0-100 |
NN inputs = the 12 columns in `config.FEATURES`; click columns are `log1p`-transformed and all inputs standardised.

## Generated quantities (rules)
| Id | Quantity | Rule |
|---|---|---|
| G1 | clicks per study session | `round(median positive 4-week click total of training rows / 8)` (a typical active student ≈ 8 sessions per 4 weeks); stored in `meta.json` |
| G1b | how a session changes NN inputs | adds `clicks_per_session` clicks, split over the 3 groups by the student's own click shares (60/20/20 if no clicks) |
| G2 | slot capacity and availability | slots per week ∈ {2,3,4} (uniform, seeded); each (week, slot) blocked with p = 0.15 (job, family, ...) |
| G3 | target mark | lowest baseline NN prediction (0 sessions) of the 3 assignments + uplift ∈ {0,2,4,6}, rounded, clipped to [40,95] |
| G4 | compressed deadlines (what-if) | with p = 0.5 the 2nd/3rd deadline is moved 1-2 weeks after the previous one **for the CSP only** (the NN's `week_since_start` keeps the real value) |
| – | search cap | at most 12 sessions per assignment |

## Profiles (`run_profiles.py` → `results/profile_plans.csv`)
Real students of the test presentation who have marks on all 3 planned assessments; real clicks, real earlier marks; G2-G4 generated. Real later marks are stored only for reference and are never planner inputs.
