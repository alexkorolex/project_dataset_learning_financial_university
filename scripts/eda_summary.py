from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bank.utils import ensure_dir, load_config


def main(cfg_path: str):
    cfg = load_config(cfg_path)
    raw = Path(cfg["data"]["raw_dir"])
    train = pd.read_csv(raw / "train.csv")
    test = pd.read_csv(raw / "test.csv")

    summary = {
        "train_shape": train.shape,
        "test_shape": test.shape,
        "columns": train.columns.tolist(),
        "target_counts": train["y"].value_counts().to_dict(),
        "target_rate": float(train["y"].mean()),
        "dtypes": train.dtypes.astype(str).to_dict(),
        "missing_train": train.isna().sum().to_dict(),
        "missing_test": test.isna().sum().to_dict(),
    }
    out_dir = Path(cfg["artifacts"]["dir"])
    ensure_dir(out_dir)
    with open(out_dir / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
