#!/usr/bin/env python
"""Train the NN on earlier presentations, test on the later one, compare to baselines."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from planner import config as C
from planner.data import oulad
from planner.nn.model import evaluate_all

df, meta = oulad.load()
train = df[df.code_presentation.isin(meta["train"])]
test = df[df.code_presentation.isin(meta["test"])]
res, nn = evaluate_all(train, test, C.SEED)
C.MODEL_DIR.mkdir(exist_ok=True); C.RESULTS_DIR.mkdir(exist_ok=True)
nn.save(C.MODEL_DIR / "mark_nn.joblib")
(C.RESULTS_DIR / "nn_metrics.json").write_text(json.dumps(
    {"train_rows": len(train), "test_rows": len(test), "synthetic": meta.get("synthetic"),
     "metrics": res}, indent=2))
print(f"train={len(train)} test={len(test)}  synthetic={meta.get('synthetic')}")
for k, v in res.items():
    print(f"{k:40s} MAE={v['MAE']:.2f} RMSE={v['RMSE']:.2f} R2={v['R2']:.3f}")
