"""Create stratified 5-fold splits of the paired dataset.

Reads dataset.pkl and writes fold_{1..5}.pkl to the folds directory.
The splits are stratified on the SAR (sarcasm) label and use a fixed
random seed for reproducibility.

Splitting strategy (replicates the original notebook, with the bug fixed):
  fold 1 = 20% of the full dataset
  fold 2–5 = four roughly equal shares of the remaining 80%

Usage:
    python src/create_folds.py --config config.yaml
"""

import argparse
import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
import yaml


def create_five_folds(df: pd.DataFrame, seed: int = 42):
    """Return five stratified splits of df.

    Splits are (fold1, fold2, fold3, fold4, fold5) — each approx 20%.
    """
    # fold 1 = 20%
    train_80, fold1 = train_test_split(
        df, test_size=0.2, stratify=df["SAR"], random_state=seed
    )
    # split the remaining 80% into four roughly equal quarters
    split23, split45 = train_test_split(
        train_80, test_size=0.5, stratify=train_80["SAR"], random_state=seed
    )
    fold2, fold3 = train_test_split(
        split23, test_size=0.5, stratify=split23["SAR"], random_state=seed
    )
    fold4, fold5 = train_test_split(
        split45, test_size=0.5, stratify=split45["SAR"], random_state=seed
    )
    return fold1, fold2, fold3, fold4, fold5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    p = cfg["paths"]

    with open(p["dataset_pkl"], "rb") as f:
        df = pickle.load(f)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    os.makedirs(p["folds_dir"], exist_ok=True)

    folds = create_five_folds(df)
    for i, fold in enumerate(folds, start=1):
        fold = fold.reset_index(drop=True)
        out_path = os.path.join(p["folds_dir"], f"fold_{i}.pkl")
        fold.to_pickle(out_path)
        n_sar = fold["SAR"].sum()
        print(f"fold_{i}: {len(fold)} rows  (sarcastic={n_sar}, non-sarcastic={len(fold)-n_sar})")

    print(f"\nFolds saved to {p['folds_dir']}/")


if __name__ == "__main__":
    main()
