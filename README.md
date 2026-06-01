# mustard_sarcasm

Multimodal sarcasm detection on the MUStARD++ dataset using Data2Vec embeddings and a cross-modal collaborative attention classifier.

## Overview

This project reproduces and extends MUStARD++'s sarcasm detection approach by replacing its original feature extractors with [Data2Vec](https://arxiv.org/abs/2202.01855) models across all three modalities — text, audio, and video. The classifier uses a collaborative cross-modal attention mechanism to fuse utterance and context representations before making a binary sarcasm prediction. This project is finished originally in 2024 in fulfillment of U Antwerp's M.A. in Digital Text Analysis and reworked for clarity in 2026.

The full pipeline can be ran end-to-end via a single Snakemake command.

## Pipeline

```
extract_keyframes
      ↓
extract_features   →   data/features.pkl
      ↓
prepare_dataset    →   data/dataset.pkl
      ↓
  create_folds     →   data/folds/fold_{1-5}.pkl
      ↓
hyperparameter_search  →  results/best_params.json
      ↓
 train_fold[1-5]   →   results/fold_{N}_results.json   (parallelisable)
      ↓
aggregate_results  →   results/final_metrics.json
```

## Repository Structure

```
mustard_sarcasm/
├── Snakefile                         # Pipeline definition
├── config.yaml                       # All paths and hyperparameters
├── src/
│   ├── keyframe_extraction.py        # Katna keyframe extraction + fallback isolation
│   ├── feature_extraction.py         # Data2Vec text / audio / video embeddings
│   ├── prepare_dataset.py            # Pairs context and utterance rows
│   ├── create_folds.py               # Stratified 5-fold split
│   ├── hyperparameter_search.py      # Grid search on a balanced 450-sample subset
│   ├── train_fold.py                 # Trains and evaluates one fold (CLI: --fold N)
│   ├── aggregate_results.py          # Combines fold results into final metrics
│   ├── dataset.py                    # ContentDataset and attention pooling
│   ├── model.py                      # SarcasmClassifier (all ablation configs)
│   └── train_utils.py                # seed, training loop, evaluation, forward dispatch
└── notebooks/                        # Original exploratory notebooks (reference only)
    ├── keyframe_extraction.ipynb
    ├── preprocess_feature_extraction.py
    ├── hyperparameter_pretesting.ipynb
    └── final_evaluation.ipynb
```

## Setup

### Environment

```bash
conda env create -f environment.yaml
conda activate mustard_sarcasm
```

For a CPU-only installation, remove the `pytorch-cuda` line from `environment.yaml` before running the above.

### Data

Place the MUStARD++ data under `data/` matching the paths in `config.yaml`:

```
data/
├── mustardtext.csv
├── final_utterance_videos_0/     # raw utterance videos (pre-keyframe extraction)
├── final_context_videos_0/       # raw context videos
├── final_utterance_audios/       # utterance .wav files
└── final_context_audios/         # context .wav files
```

The keyframe extraction step populates the remaining directories automatically.

## Running the Pipeline

```bash
# Full pipeline, using up to 5 parallel workers (one per fold)
snakemake --cores 5

# Dry run to preview what will execute
snakemake --cores 5 -n

# Skip keyframe extraction (if keyframes already exist)
snakemake results/final_metrics.json --cores 5
```

## Configuration

All settings live in `config.yaml`. Key options:

| Key | Default | Description |
|-----|---------|-------------|
| `gpu` | `0` | GPU index. Set to `-1` to force CPU. |
| `training.mode` | `VTA` | Modality combination: `V`, `T`, `A`, `VT`, `VA`, `TA`, or `VTA` |
| `training.speaker` | `true` | Include speaker one-hot embedding |
| `training.context` | `true` | Include context modality embeddings |
| `training.epochs` | `500` | Maximum training epochs |
| `training.patience` | `5` | Early stopping patience |

### Ablation Studies

To run a different ablation configuration, change `training.mode`, `training.speaker`, and `training.context` in `config.yaml`, then re-run `snakemake --cores 5`. The model and forward dispatch in `src/model.py` and `src/train_utils.py` support all 28 combinations automatically.

## Model

`SarcasmClassifier` in `src/model.py` is a single flexible class that handles all ablation settings. It projects each available modality into a shared embedding space, applies collaborative cross-modal attention (each utterance representation is updated by attending to all other available features), and passes the concatenated result through a multi-layer prediction head.

Modality positions A/B/C are assigned by the `MODALITY_POSITIONS` map in `src/train_utils.py` based on the chosen mode.

## Output

After a full run, `results/` contains:

```
results/
├── best_params.json              # Winning hyperparameter combination
├── fold_1_results.json
├── fold_2_results.json
├── fold_3_results.json
├── fold_4_results.json
├── fold_5_results.json
└── final_metrics.json            # Mean ± std macro F1 across all folds
```

`final_metrics.json` reports the mean and standard deviation of macro F1 across the five folds.
