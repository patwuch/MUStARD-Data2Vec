"""Prepare the paired utterance+context dataset from raw features.

Reads features.pkl (one row per clip — both context and utterance rows),
pairs each utterance row with its preceding context row, and writes
dataset.pkl containing only utterance rows with all six modality columns:
  uText, uAudio, uVideo  — utterance embeddings
  cText, cAudio, cVideo  — context embeddings (from the paired context row)

The pairing relies on the dataset structure where every two consecutive rows
with the same SCENE form a (context, utterance) pair.

Usage:
    python src/prepare_dataset.py --config config.yaml
"""

import argparse
import pickle

import pandas as pd
import yaml


def pair_context_utterance(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and add context columns from the preceding context rows.

    The raw DataFrame has columns text_embeddings / audio_embeddings /
    keyframe_embeddings.  After pairing, each utterance row gains cText,
    cAudio, cVideo populated from the corresponding context row.
    Rows without a paired context are dropped.
    """
    df = df.dropna(subset=["text_embeddings"]).reset_index(drop=True)

    df = df.rename(
        columns={
            "text_embeddings": "uText",
            "audio_embeddings": "uAudio",
            "keyframe_embeddings": "uVideo",
            "Sarcasm": "SAR",
        }
    )

    df["cAudio"] = None
    df["cVideo"] = None
    df["cText"] = None

    # Even-indexed rows are context rows; odd-indexed are utterance rows.
    # Copy context features into the next (utterance) row's context columns.
    for i in range(0, len(df) - 1, 2):
        df.at[i + 1, "cAudio"] = df.at[i, "uAudio"]
        df.at[i + 1, "cVideo"] = df.at[i, "uVideo"]
        df.at[i + 1, "cText"] = df.at[i, "uText"]

    # Keep only utterance rows that received a context partner
    df = df.dropna(subset=["cAudio"]).reset_index(drop=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    p = cfg["paths"]

    with open(p["features_pkl"], "rb") as f:
        raw = pickle.load(f)
    df = pd.DataFrame(raw) if not isinstance(raw, pd.DataFrame) else raw

    dataset = pair_context_utterance(df)
    print(f"Dataset rows after pairing: {len(dataset)}")
    print(f"Sarcastic: {dataset['SAR'].sum()}  Non-sarcastic: {(dataset['SAR']==0).sum()}")

    with open(p["dataset_pkl"], "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved paired dataset to {p['dataset_pkl']}")


if __name__ == "__main__":
    main()
