from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from joblib import load
from bank.utils import load_config, ensure_dir
from bank.preprocess import prepare_frames

def main(cfg_path: str):
    cfg = load_config(cfg_path)
    raw = Path(cfg["data"]["raw_dir"])
    X, y, X_test, numeric, categorical = prepare_frames(raw / "train.csv", raw / "test.csv", cfg)

    art = Path(cfg["artifacts"]["dir"]); ensure_dir(art)

    for model_path in art.glob("model_*.joblib"):
        name = model_path.stem.replace("model_","")
        pipe = load(model_path)
        if hasattr(pipe[-1], "predict_proba"):
            p = pipe.predict_proba(X_test)[:,1]
        else:
            p = pipe.decision_function(X_test)

        sub = pd.DataFrame({"id": range(len(p)), "p": p})
        sub.to_csv(art / f"submission_{name}_test.csv", index=False)
        print(f"Wrote {art / f'submission_{name}_test.csv'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
