#!/usr/bin/env python
"""Download (or generate) and preprocess the data reproducibly.

  python make_data.py                    # download OULAD zip, preprocess
  python make_data.py --zip path.zip     # use a zip you already downloaded
  python make_data.py --inspect          # list modules/presentations/assessments (Week-1 check)
  python make_data.py --synthetic        # OFFLINE TEST DATA ONLY (not real, flagged)
"""
import argparse, sys, zipfile, urllib.request, random
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent / "src"))
from planner import config as C
from planner.data import oulad, synthetic

# NOTE: confirm this URL / that no registration is needed (project Week-1 task).
DEFAULT_URL = "https://analyse.kmi.open.ac.uk/open_dataset/download"
NEEDED = ["assessments.csv", "studentAssessment.csv", "vle.csv", "studentVle.csv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--zip", type=Path)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed)
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)

    if a.synthetic:
        synthetic.generate(seed=a.seed)
        print("!! SYNTHETIC data written (for pipeline testing only) !!")
    elif not all((C.RAW_DIR / f).exists() for f in NEEDED):
        z = a.zip
        if z is None:
            z = C.RAW_DIR / "oulad.zip"
            print(f"Downloading {a.url} ...")
            urllib.request.urlretrieve(a.url, z)
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if Path(n).name in NEEDED:
                    (C.RAW_DIR / Path(n).name).write_bytes(zf.read(n))
    oulad.check_schema()
    if a.inspect:
        oulad.inspect(); return
    df, meta = oulad.finalize(oulad.build_assessment_table())
    meta["synthetic"] = (C.RAW_DIR / "SYNTHETIC_DATA.txt").exists()
    oulad.save(df, meta)
    print(f"Saved {len(df)} rows; meta = {meta}")


if __name__ == "__main__":
    main()
