# SAMF Osteosarcoma

SAMF maps frozen whole-slide tile embeddings to spatial gene-expression estimates, aggregates the estimates into Reactome and Hallmark pathway tokens, aligns pathway and histology tokens with bidirectional cross-attention, and predicts metastasis-free survival with gated attention pooling and a Cox head. The implementation also includes necrosis–viable interface discovery and patient-level statistical evaluation.

## Environment

Python 3.10, PyTorch 2.2.2, and CUDA 12.1 are pinned in the environment files.

```bash
conda env create -f environment.yml
conda activate samf-osteosarcoma
pip install -e .
```

A container can be built with:

```bash
docker build -t samf-osteosarcoma .
```

## Data inputs

Verified source addresses and access notes are collected in `datasets.txt`. The preparation pipeline expects a CSV spatial manifest containing:

```text
patient_id,slide_id,spot_id,image_path,matrix_path,x,y,scalefactors_path
```

Each spatial record must include Space Ranger coordinates and scale factors. The preparation command rejects expression-only records that lack paired histology alignment metadata.

```bash
samf-prepare --spatial-manifest data/spatial_manifest.csv
```

The manuscript names four GEO accessions as paired Visium–H&E sources. Official GEO metadata currently identifies GSE152048, GSE162454, and GSE217792 as single-cell RNA-seq. They cannot enter the spatial training split unless an independently obtained paired image and Space Ranger manifest is supplied and validated. This guard prevents modality substitution.

Slides are tessellated into non-overlapping 256 × 256 tiles at 20×. Tiles with mean HSV saturation below 0.05 or less than 50% tissue are removed. Spot counts require at least 500 detected genes and 1,000 UMIs, followed by library-size scaling, log2(x + 1), and selection of 2,000 highly variable genes.

## Model inputs

The model consumes frozen feature tensors with shape `batch × tiles × feature_dim` and a binary pathway-membership tensor with shape `331 × 2000`. UNI uses 1,024-dimensional features, CONCH uses 512, and Virchow uses 1,280. Foundation-model weights are not updated.

Pathway membership must contain 281 Reactome sets passing 90% overlap with the selected genes and 50 Hallmark sets. The exact gene ordering used for expression matrices and membership tensors must match.

## Training

The reported expression stage uses AdamW with learning rate 3e-4, weight decay 1e-2, batch size 256, and 100 epochs. Distillation uses temperature 4 and mixing coefficient 0.3. The survival stage uses AdamW with learning rate 2e-4, weight decay 1e-3, batch size one whole slide, and 50 epochs, with validation patience 15. Five patient-level folds use seeds 42, 123, 256, 512, and 1024.

```bash
bash scripts/train_folds.sh
```

The default configuration retains the 2,000-gene predictor, 331 pathways, four cross-attention layers, eight heads, hidden size 256, dropout 0.1, and Cox L2 coefficient 1e-4. Ablation files cover removal of distillation, MSE replacement, single-gene tokenization, removal of cross-attention, and backbone transfer.

## Evaluation

```bash
bash scripts/evaluate_target.sh
```

Expression evaluation reports per-gene Pearson and Spearman correlation. Survival evaluation reports patient-level concordance. Confidence intervals use 1,000 patient-level bootstrap resamples; pairwise comparisons use 1,000 permutations. Primary comparisons use Holm–Bonferroni adjustment, while exploratory pathway and subgroup analyses use Benjamini–Hochberg adjustment.

Expected manuscript targets are a TARGET-OS C-index of 0.638 ± 0.052, an external-cohort C-index of 0.611 ± 0.038, expression Pearson correlation of 0.342 ± 0.031, and metastasis AUROC of 0.724 ± 0.045. These values require the manuscript cohorts, frozen backbone features, pathway definitions, and patient splits.

## Interface analysis

Viable tile centroids within 200 µm of the nearest necrotic tile boundary form the interface set. Patient pathway signatures are averaged across interface tiles. Candidate cluster counts are 2 through 5, selected by silhouette score, and pathway comparisons use rank-sum tests with false-discovery-rate correction.

## Compute

The experimental configuration uses one NVIDIA A100 80GB GPU. Expression training takes about two hours per fold, pathway-classifier training about 30 minutes per fold, and frozen feature extraction about four hours per dataset. The full set of folds, baselines, and ablations is reported as approximately 120 GPU-hours. Disk use depends on retained whole-slide pyramids and feature caches; capacity should be sized after generating a manifest with measured file sizes.

## Quality checks

```bash
pytest -q
ruff check .
mypy --strict src/samf_osteosarcoma
```

The suite contains unit tests for expression transforms, tile selection, losses, pathway aggregation, survival concordance, interface geometry, multiple-testing correction, and an end-to-end model forward/backward integration test.

