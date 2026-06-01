"""Train and evaluate the model for one fold of 5-fold cross-validation.

For fold N: trains on all other folds, evaluates on fold N.
A small validation set (val_size rows) is randomly held out from the
training folds for early stopping.

Reads:
  data/folds/fold_{1..5}.pkl
  results/best_params.json

Writes:
  results/fold_{N}_results.json

Usage:
    python src/train_fold.py --config config.yaml --fold 1
"""

import argparse
import json
import os
import pickle
import sys

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import ConcatDataset, DataLoader, random_split

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.dataset import ContentDataset, build_scene_dataset
from src.train_utils import build_model, evaluation, seed, seed_worker, training

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def load_fold(folds_dir: str, fold_n: int) -> pd.DataFrame:
    path = os.path.join(folds_dir, f"fold_{fold_n}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fold", type=int, required=True, help="Test fold index (1–5)")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  Test fold: {args.fold}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    p = cfg["paths"]
    t = cfg["training"]
    n_folds = t["n_folds"]

    # Load best hyperparams from search
    with open(p["best_params"]) as f:
        best = json.load(f)
    dropout = best["dropout"]
    lr = best["lr"]
    batch_size = best["batch_size"]
    shared_emb = best["shared_emb_size"]
    proj_emb = best["proj_emb_size"]

    # Load speaker list from the full dataset for consistent one-hot encoding
    with open(p["dataset_pkl"], "rb") as f:
        full_df = pickle.load(f)
    if not isinstance(full_df, pd.DataFrame):
        full_df = pd.DataFrame(full_df)
    speaker_list = sorted(full_df["SPEAKER"].value_counts().keys().tolist())

    # Load all folds
    folds = {i: load_fold(p["folds_dir"], i) for i in range(1, n_folds + 1)}

    test_fold = folds[args.fold]
    train_folds = [folds[i] for i in range(1, n_folds + 1) if i != args.fold]

    mode = t["mode"]
    use_speaker = t["speaker"]
    use_context = t["context"]
    val_size = t["val_size"]

    # Build ContentDatasets for each training fold
    train_cds = []
    for fold_df in train_folds:
        ds = build_scene_dataset(fold_df)
        mp = fold_df[["SCENE", "SAR", "SPEAKER"]]
        train_cds.append(ContentDataset(mp, ds, speaker_list))

    train_combined = ConcatDataset(train_cds)

    # Hold out val_size samples for early stopping validation
    train_size = len(train_combined) - val_size
    seed()
    train_subset, val_subset = random_split(train_combined, [train_size, val_size])

    seed()
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, worker_init_fn=seed_worker,
    )
    seed()
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=False, worker_init_fn=seed_worker,
    )

    # Test DataLoader
    test_ds = build_scene_dataset(test_fold)
    test_map = test_fold[["SCENE", "SAR", "SPEAKER"]]
    test_cd = ContentDataset(test_map, test_ds, speaker_list)
    seed()
    test_loader = DataLoader(
        test_cd, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False, worker_init_fn=seed_worker,
    )

    # Build model
    seed()
    mod, input_modes = build_model(
        mode=mode,
        use_speaker=use_speaker,
        use_context=use_context,
        n_speaker=len(speaker_list),
        shared_dim=shared_emb,
        proj_dim=proj_emb,
        dropout=dropout,
    )
    mod.to(device)

    seed()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(mod.parameters(), betas=(0.5, 0.99), lr=lr)

    (true, pred), best_epoch = training(
        mod=mod,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=val_loader,
        input_modes=input_modes,
        use_context=use_context,
        use_speaker=use_speaker,
        device=device,
        fold=args.fold,
        max_epochs=t["epochs"],
        patience=t["patience"],
    )

    # Evaluate on held-out test fold
    print(f"\n=== Test results for fold {args.fold} ===")
    test_f1, test_loss = evaluation(
        test_loader, mod, input_modes, use_context, use_speaker,
        device, report=True,
    )
    print(f"Test macro F1: {test_f1:.4f}")

    true_test, pred_test = evaluation(
        test_loader, mod, input_modes, use_context, use_speaker,
        device, return_preds=True,
    )

    results = {
        "fold": args.fold,
        "best_epoch": best_epoch,
        "test_macro_f1": float(test_f1),
        "test_loss": float(test_loss),
        "classification_report": classification_report(
            true_test, pred_test, digits=3, output_dict=True
        ),
        "hyperparams": best,
    }

    os.makedirs(p["results_dir"], exist_ok=True)
    out_path = os.path.join(p["results_dir"], f"fold_{args.fold}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
