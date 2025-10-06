from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_roc_pr_curves(name: str, roc_data, pr_data, out_dir: str | Path):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fpr, tpr = roc_data
    prec, rec = pr_data

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC — {name}")
    plt.tight_layout()
    plt.savefig(out / f"curves_{name}_roc.png")
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR — {name}")
    plt.tight_layout()
    plt.savefig(out / f"curves_{name}_pr.png")
    plt.close()
