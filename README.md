# Assessment-Aware Study Planner

A decision-support prototype that suggests how many study sessions to allocate to upcoming assessments and when to schedule them. It combines search, a neural-network mark predictor, and a constraint-satisfaction scheduler.

The project was developed for the Intelligent Systems course using the [Open University Learning Analytics Dataset (OULAD)](https://analyse.kmi.open.ac.uk/open_dataset).

> **Important:** This is a what-if tool, not a recommendation engine or evidence that studying more causes higher marks. The relationship between online activity and assessment scores is observational. Session lengths, timetable capacity, blocked slots, and target marks are generated assumptions.

## How it works

1. Search proposes study-session allocations using Uniform Cost Search, Greedy Search, or A*.
2. A neural network estimates assessment marks for each proposed allocation.
3. A constraint-satisfaction solver schedules the sessions into available weekly slots. Infeasible allocations are rejected.

## Requirements

- Python 3.12 recommended (Python 3.14 was also tested).
- OULAD data for the real-data workflow. The dataset is not included in this repository.

## Setup

Run these commands from the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell cannot find `py`, install Python 3.12 and ensure the Python launcher is available. If activation is blocked by your execution policy, use the environment's Python directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

When using the direct interpreter, replace `python` with `.\.venv\Scripts\python.exe` in the commands below.

## Prepare data and run

By default, the data-preparation script downloads OULAD and writes the required source tables to `data/raw/`, then creates processed data in `data/processed/`:

```powershell
python make_data.py
```

To use a dataset ZIP you already downloaded:

```powershell
python make_data.py --zip "C:\path\to\anonymisedData.zip"
```

To inspect available modules, presentations, and assessments:

```powershell
python make_data.py --inspect
```

Train the mark predictor before running the integrated demo:

```powershell
python train_nn.py
python run_demo.py
```

`run_demo.py` prints the search trace, timetable, and prediction details; it also writes plots under `results/figures/`.

### Other commands

```powershell
python run_profiles.py
python run_experiments.py
python -m unittest discover -s tests -v
```

- `run_profiles.py` plans timetables for generated student profiles.
- `run_experiments.py` runs the search, CSP, predictor-comparison, and monotonicity experiments.
- The test suite can be run independently of the data-preparation and training steps.

## Data and reproducibility

OULAD is distributed under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/); see the [dataset page](https://analyse.kmi.open.ac.uk/open_dataset) for details.

The configured course is module **FFF**. The model trains on presentations `2013B`, `2013J`, and `2014B`, and evaluates on `2014J`. It uses TMA and CMA assessments with deadlines. See [`src/planner/config.py`](src/planner/config.py) for the configuration and [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md) for the dataset fields and generated assumptions.

The random seed defaults to `42`. Check `data/processed/meta.json` and confirm `"synthetic": false` before interpreting results as real-data results.

For an offline pipeline smoke test, run:

```powershell
python make_data.py --synthetic
python train_nn.py
python run_demo.py
```

Synthetic data is only for testing the pipeline; do not report its results as OULAD results. Do not mix synthetic and real data in the same `data/raw/` and `data/processed/` directories.

## Outputs

- `models/mark_nn.joblib` — trained neural network.
- `results/nn_metrics.json` — test metrics and dataset metadata.
- `results/` — profile plans, experiment tables, and generated figures.

The data and model directories are generated locally and are not intended to be committed. See [`results/`](results/) for checked-in example outputs and [`docs/`](docs/) for the project report and presentation materials.

## Project structure

| Path | Purpose |
| --- | --- |
| `make_data.py` | Download or generate data and preprocess it |
| `train_nn.py` | Train and evaluate the mark predictor |
| `run_demo.py` | Run the integrated search, prediction, and scheduling demo |
| `src/planner/search/` | UCS, Greedy, and A* search |
| `src/planner/csp/` | Backtracking, forward checking, and AC-3 scheduling |
| `src/planner/integration/` | Connect search, mark prediction, and scheduling |
| `src/planner/eval/` | Experiments, metrics, and plots |
| `tests/` | Unit tests |

Authors: Ibrahim Eltemsahy and Moatassem Muhammed.
