# scripts/train.py
from __future__ import annotations
import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from bank.metrics import compute_metrics, curves, find_best_threshold
from bank.pipelines import make_linear_preprocessor, make_tree_preprocessor
from bank.plotting import save_roc_pr_curves
from bank.preprocess import prepare_frames
from bank.utils import ensure_dir, load_config, save_json, set_seed
from bank.models import make_estimator


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _fit_predict_catboost(Xtr, ytr, Xva, yva, categorical_cols, params):
    """Train CatBoost with native categorical features and return (model, prob_valid)."""
    from catboost import CatBoostClassifier
    # индексы категориальных колонок по порядку столбцов
    cat_idx = [Xtr.columns.get_loc(c) for c in categorical_cols if c in Xtr.columns]

    model = CatBoostClassifier(**params)
    model.fit(
        Xtr, ytr,
        cat_features=cat_idx,
        eval_set=(Xva, yva),
        verbose=params.get("verbose", False),
    )
    prob = model.predict_proba(Xva)[:, 1]
    return model, prob


def main(cfg_path: str) -> None:
    setup_logger()
    t_start = time.perf_counter()

    cfg = load_config(cfg_path)
    set_seed(cfg["data"]["random_state"])

    cwd = Path.cwd().resolve()
    art_dir = Path(cfg["artifacts"]["dir"]).resolve()
    ensure_dir(art_dir)

    logging.info("CWD: %s", cwd)
    logging.info("Artifacts dir: %s", art_dir)

    # --- load & prepare ---
    raw = Path(cfg["data"]["raw_dir"]).resolve()
    logging.info("Loading data from: %s", raw)

    X, y, X_test, numeric, categorical = prepare_frames(
        raw / "train.csv", raw / "test.csv", cfg
    )
    logging.info("Prepared frames: X=%s y=%s | test=%s", X.shape, y.shape, X_test.shape)
    logging.info("Features -> numeric=%s | categorical=%s", numeric, categorical)

    # --- split ---
    logging.info(
        "Splitting train/valid (test_size=%s, random_state=%s)",
        cfg["data"]["val_size"],
        cfg["data"]["random_state"],
    )
    Xtr, Xva, ytr, yva = train_test_split(
        X,
        y,
        test_size=cfg["data"]["val_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )
    logging.info("Split done: Xtr=%s | Xva=%s", Xtr.shape, Xva.shape)

    # --- train each model ---
    metrics_valid: dict[str, dict] = {}
    for name, spec in cfg["models"].items():
        kind = spec["type"].lower()
        params = spec.get("params", {})

        if kind == "catboost":
            logging.info(">>> Training model: %s (catboost, native cats) ...", name)
            t0 = time.perf_counter()
            model, prob = _fit_predict_catboost(Xtr, ytr, Xva, yva, categorical, params)
            dt = time.perf_counter() - t0
            logging.info("Model %s trained in %.2fs", name, dt)

            # threshold tuning / metrics
            thr = find_best_threshold(yva, prob, metric="f1") if cfg["thresholds"].get("tune_on_valid", True) else cfg["thresholds"]["default"]
            m = compute_metrics(yva, prob, threshold=thr)
            metrics_valid[name] = m
            logging.info(
                "Metrics %s -> accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f thr=%.4f",
                name, m["accuracy"], m["precision"], m["recall"], m["f1"], m["roc_auc"], m["threshold"],
            )

            # save model (CatBoost native format) + preds + curves
            cbm_path = art_dir / f"model_{name}.cbm"
            model.save_model(cbm_path)
            logging.info("Saved CatBoost model: %s", cbm_path)

            preds_path = art_dir / f"preds_{name}_valid.csv"
            pd.DataFrame({"y_true": yva, "p": prob}).to_csv(preds_path, index=False)
            logging.info("Saved valid preds: %s", preds_path)

            roc_data, pr_data = curves(yva, prob)
            save_roc_pr_curves(name, roc_data, pr_data, art_dir)
            logging.info(
                "Saved curves: %s | %s",
                art_dir / f"curves_{name}_roc.png",
                art_dir / f"curves_{name}_pr.png",
            )

            continue  # к следующей модели

        # --- sklearn-путь (logreg, rf, hgbt) ---
        is_linear = (kind == "logreg")
        pre = make_linear_preprocessor(numeric, categorical) if is_linear else make_tree_preprocessor(numeric, categorical)

        logging.info(">>> Training model: %s (%s) ...", name, kind)
        t0 = time.perf_counter()
        pipe = Pipeline([("pre", pre), ("clf", make_estimator(kind, params))])
        pipe.fit(Xtr, ytr)
        dt = time.perf_counter() - t0
        logging.info("Model %s trained in %.2fs", name, dt)

        # proba/score
        if hasattr(pipe[-1], "predict_proba"):
            prob = pipe.predict_proba(Xva)[:, 1]
        else:
            prob = pipe.decision_function(Xva)

        # threshold tuning / metrics
        thr = find_best_threshold(yva, prob, metric="f1") if cfg["thresholds"].get("tune_on_valid", True) else cfg["thresholds"]["default"]
        m = compute_metrics(yva, prob, threshold=thr)
        metrics_valid[name] = m
        logging.info(
            "Metrics %s -> accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%.4f thr=%.4f",
            name, m["accuracy"], m["precision"], m["recall"], m["f1"], m["roc_auc"], m["threshold"],
        )

        # save model & preds & curves
        model_path = art_dir / f"model_{name}.joblib"
        dump(pipe, model_path)
        logging.info("Saved model: %s", model_path)

        preds_path = art_dir / f"preds_{name}_valid.csv"
        pd.DataFrame({"y_true": yva, "p": prob}).to_csv(preds_path, index=False)
        logging.info("Saved valid preds: %s", preds_path)

        roc_data, pr_data = curves(yva, prob)
        save_roc_pr_curves(name, roc_data, pr_data, art_dir)
        logging.info(
            "Saved curves: %s | %s",
            art_dir / f"curves_{name}_roc.png",
            art_dir / f"curves_{name}_pr.png",
        )

    # --- save metrics summary ---
    metrics_path = art_dir / "metrics_valid.json"
    save_json(metrics_valid, metrics_path)
    logging.info("Saved metrics summary: %s", metrics_path)
    logging.info("Done in %.2fs", time.perf_counter() - t_start)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
