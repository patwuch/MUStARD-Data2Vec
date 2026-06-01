"""Aggregate per-fold results into a final metrics summary.

Reads results/fold_{1..5}_results.json and writes
results/final_metrics.json with mean ± std across folds.

Usage:
    python src/aggregate_results.py --config config.yaml
"""

import argparse
import json
import os

import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    p = cfg["paths"]
    n_folds = cfg["training"]["n_folds"]

    f1_scores = []
    fold_summaries = []

    for fold_n in range(1, n_folds + 1):
        path = os.path.join(p["results_dir"], f"fold_{fold_n}_results.json")
        with open(path) as f:
            res = json.load(f)

        f1 = res["test_macro_f1"]
        f1_scores.append(f1)
        fold_summaries.append(
            {
                "fold": fold_n,
                "macro_f1": f1,
                "best_epoch": res["best_epoch"],
                "precision_macro": res["classification_report"]["macro avg"]["precision"],
                "recall_macro": res["classification_report"]["macro avg"]["recall"],
            }
        )
        print(
            f"Fold {fold_n}: macro_F1={f1:.4f}  "
            f"precision={fold_summaries[-1]['precision_macro']:.4f}  "
            f"recall={fold_summaries[-1]['recall_macro']:.4f}  "
            f"(epoch {res['best_epoch']})"
        )

    mean_f1 = float(np.mean(f1_scores))
    std_f1 = float(np.std(f1_scores))
    print(f"\n{'='*60}")
    print(f"5-Fold CV macro F1:  {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"{'='*60}")

    summary = {
        "mean_macro_f1": mean_f1,
        "std_macro_f1": std_f1,
        "folds": fold_summaries,
    }

    out_path = os.path.join(p["results_dir"], "final_metrics.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFinal metrics saved to {out_path}")


if __name__ == "__main__":
    main()
