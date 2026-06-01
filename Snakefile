"""Snakemake pipeline for multimodal sarcasm detection on MUStARD++.

Full run (from raw videos to final metrics):
    snakemake --cores 1

Run only the training pipeline (assumes features already extracted):
    snakemake results/final_metrics.json --cores 5

Parallelise the five fold-training jobs across 5 cores:
    snakemake --cores 5

Skip keyframe extraction (if keyframes already exist):
    snakemake --cores 1 --allowed-rules extract_features prepare_dataset \
        create_folds hyperparameter_search train_fold aggregate_results

DAG:
    extract_keyframes
          |
    extract_features  →  data/features.pkl
          |
    prepare_dataset   →  data/dataset.pkl
          |
      create_folds    →  data/folds/fold_{1-5}.pkl
          |
  hyperparameter_search  →  results/best_params.json
          |
    train_fold[1-5]   →  results/fold_{i}_results.json   (run in parallel)
          |
    aggregate_results →  results/final_metrics.json
"""

configfile: "config.yaml"

N_FOLDS = config["training"]["n_folds"]
FOLD_RESULTS = expand(
    "{results_dir}/fold_{fold}_results.json",
    results_dir=config["paths"]["results_dir"],
    fold=range(1, N_FOLDS + 1),
)
FOLD_PKLS = expand(
    "{folds_dir}/fold_{fold}.pkl",
    folds_dir=config["paths"]["folds_dir"],
    fold=range(1, N_FOLDS + 1),
)


rule all:
    input:
        config["paths"]["results_dir"] + "/final_metrics.json"


# ---------------------------------------------------------------------------
# Step 1: Keyframe extraction
# ---------------------------------------------------------------------------
rule extract_keyframes:
    output:
        touch("data/.keyframes_done"),
    shell:
        "python src/keyframe_extraction.py --config config.yaml"


# ---------------------------------------------------------------------------
# Step 2: Feature extraction (text + visual + audio → features.pkl)
# ---------------------------------------------------------------------------
rule extract_features:
    input:
        "data/.keyframes_done",
    output:
        config["paths"]["features_pkl"],
    shell:
        "python src/feature_extraction.py --config config.yaml --gpu {config[gpu]}"


# ---------------------------------------------------------------------------
# Step 3: Pair context + utterance rows → dataset.pkl
# ---------------------------------------------------------------------------
rule prepare_dataset:
    input:
        config["paths"]["features_pkl"],
    output:
        config["paths"]["dataset_pkl"],
    shell:
        "python src/prepare_dataset.py --config config.yaml"


# ---------------------------------------------------------------------------
# Step 4: Stratified 5-fold split → data/folds/fold_{1..5}.pkl
# ---------------------------------------------------------------------------
rule create_folds:
    input:
        config["paths"]["dataset_pkl"],
    output:
        FOLD_PKLS,
    shell:
        "python src/create_folds.py --config config.yaml"


# ---------------------------------------------------------------------------
# Step 5: Hyperparameter grid search → results/best_params.json
# ---------------------------------------------------------------------------
rule hyperparameter_search:
    input:
        config["paths"]["dataset_pkl"],
    output:
        config["paths"]["best_params"],
    shell:
        "python src/hyperparameter_search.py --config config.yaml --gpu {config[gpu]}"


# ---------------------------------------------------------------------------
# Step 6: Train + evaluate each fold (parallelisable with --cores 5)
# ---------------------------------------------------------------------------
rule train_fold:
    input:
        folds=FOLD_PKLS,
        best_params=config["paths"]["best_params"],
        dataset=config["paths"]["dataset_pkl"],
    output:
        config["paths"]["results_dir"] + "/fold_{fold}_results.json",
    wildcard_constraints:
        fold=r"[0-9]+",
    shell:
        "python src/train_fold.py --config config.yaml --fold {wildcards.fold} --gpu {config[gpu]}"


# ---------------------------------------------------------------------------
# Step 7: Aggregate fold results → results/final_metrics.json
# ---------------------------------------------------------------------------
rule aggregate_results:
    input:
        FOLD_RESULTS,
    output:
        config["paths"]["results_dir"] + "/final_metrics.json",
    shell:
        "python src/aggregate_results.py --config config.yaml"
