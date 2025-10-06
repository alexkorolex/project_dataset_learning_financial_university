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

    test_raw = pd.read_csv(raw / "test.csv")  # для настоящего id в сабмишене
    test_id = test_raw["id"] if "id" in test_raw.columns else pd.Series(range(len(test_raw)))

    X, y, X_test, numeric, categorical = prepare_frames(raw / "train.csv", raw / "test.csv", cfg)

    art = Path(cfg["artifacts"]["dir"]); ensure_dir(art)

    for model_path in art.glob("model_*.joblib"):
        name = model_path.stem.replace("model_","")
        pipe = load(model_path)
        if hasattr(pipe[-1], "predict_proba"):
            p = pipe.predict_proba(X_test)[:,1]
        else:
            p = pipe.decision_function(X_test)
        sub = pd.DataFrame({"id": test_id.values, "p": p})
        sub.to_csv(art / f"submission_{name}_test.csv", index=False)
        print(f"Wrote {art / f'submission_{name}_test.csv'}")

    from catboost import CatBoostClassifier
    for model_path in art.glob("model_*.cbm"):
        name = model_path.stem.replace("model_","")
        cb = CatBoostClassifier()
        cb.load_model(str(model_path))
        p = cb.predict_proba(X_test)[:,1]
        sub = pd.DataFrame({"id": test_id.values, "p": p})
        sub.to_csv(art / f"submission_{name}_test.csv", index=False)
        print(f"Wrote {art / f'submission_{name}_test.csv'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
