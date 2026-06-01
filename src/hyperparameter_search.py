"""Exhaustive hyperparameter grid search over a balanced 450-sample subset.

Reads dataset.pkl, samples a balanced subset, and grid-searches over
dropout × lr × batch_size × shared_emb_size × proj_emb_size.
Writes best_params.json with the winning combination.

Usage:
    python src/hyperparameter_search.py --config config.yaml
"""

import argparse
import itertools
import json
import os
import pickle
import sys

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.dataset import ContentDataset, build_scene_dataset
from src.train_utils import (
    build_model,
    evaluation,
    seed,
    seed_worker,
    training,
)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def sample_balanced(df: pd.DataFrame, n: int, random_state: int = 42) -> pd.DataFrame:
    """Return n rows balanced equally between sarcastic and non-sarcastic."""
    half = n // 2
    pos = df[df["SAR"] == 1].sample(half, random_state=random_state)
    neg = df[df["SAR"] == 0].sample(half, random_state=random_state)
    return pd.concat([pos, neg], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    p = cfg["paths"]
    t = cfg["training"]
    h = cfg["hyperparameter_search"]

    with open(p["dataset_pkl"], "rb") as f:
        df = pickle.load(f)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    speaker_list = sorted(df["SPEAKER"].value_counts().keys().tolist())

    # Balanced sample
    sample = sample_balanced(df, h["n_samples"])
    train_frac = h["train_fraction"]
    n_train = int(len(sample) * train_frac)
    n_val = len(sample) - n_train

    train_df = sample.iloc[:n_train].reset_index(drop=True)
    val_df = sample.iloc[n_train:].reset_index(drop=True)

    train_dataset = build_scene_dataset(train_df)
    val_dataset = build_scene_dataset(val_df)

    train_map = train_df[["SCENE", "SAR", "SPEAKER"]]
    val_map = val_df[["SCENE", "SAR", "SPEAKER"]]

    mode = t["mode"]
    use_speaker = t["speaker"]
    use_context = t["context"]

    input_modes_key = "".join(reversed(sorted(mode.upper())))

    best_params = None
    best_loss = float("inf")
    best_f1 = 0.0

    grid = list(
        itertools.product(
            h["dropout_values"],
            h["lr_values"],
            h["batch_size_values"],
            h["shared_emb_values"],
            h["proj_emb_values"],
        )
    )
    print(f"Grid size: {len(grid)} combinations")

    for dropout, lr, batch_size, shared_emb, proj_emb in grid:
        print(
            f"\ndropout={dropout}  lr={lr}  batch={batch_size}"
            f"  shared={shared_emb}  proj={proj_emb}"
        )
        seed()
        train_cd = ContentDataset(train_map, train_dataset, speaker_list)
        val_cd = ContentDataset(val_map, val_dataset, speaker_list)

        train_loader = DataLoader(
            train_cd, batch_size=batch_size, num_workers=0,
            pin_memory=False, worker_init_fn=seed_worker,
        )
        val_loader = DataLoader(
            val_cd, batch_size=batch_size, num_workers=0,
            pin_memory=False, worker_init_fn=seed_worker,
        )

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

        (true, pred), epo = training(
            mod=mod,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            valid_loader=val_loader,
            input_modes=input_modes,
            use_context=use_context,
            use_speaker=use_speaker,
            device=device,
            fold=0,
            max_epochs=t["epochs"],
            patience=t["patience"],
        )

        val_f1, val_loss = evaluation(
            val_loader, mod, input_modes, use_context, use_speaker, device
        )

        print(f"  Val F1={val_f1:.4f}  Val loss={val_loss.item():.4f}  Best epoch={epo}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_f1 = val_f1
            best_params = {
                "dropout": dropout,
                "lr": lr,
                "batch_size": batch_size,
                "shared_emb_size": shared_emb,
                "proj_emb_size": proj_emb,
                "val_f1": float(val_f1),
                "val_loss": float(val_loss),
                "best_epoch": epo,
            }
            print(f"  *** New best: {best_params}")

    os.makedirs(os.path.dirname(p["best_params"]), exist_ok=True)
    with open(p["best_params"], "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest params saved to {p['best_params']}")
    print(f"Best: {best_params}")


if __name__ == "__main__":
    main()
