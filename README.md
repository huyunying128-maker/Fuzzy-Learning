# Fuzzy Partition-Weighted Local Learning

This repository contains the code accompanying the article **Fuzzy Partition-Weighted Local Learning for Regression and Classification**. The code is organized by the same modeling path used in the article: data preparation, partition learning, local regression and classification, external feature-layer comparisons, appendix examples, and figure/table generation.

## Repository layout

```text
fuzzy-partition-weighted-learning/
├── dataset/
├── core/
├── regression/
├── classification/
├── appendix_examples/
├── results_and_figures/
├── README.md
├── requirements.txt
└── run_all.py
```

The `dataset/` folder prepares Concrete, Superconductivity, and MNIST data and creates reproducible train, validation, and test splits. The `core/` folder contains the shared implementation of distance tables, crisp and fuzzy memberships, truncation rules, local polynomial and logit models, modified k-means references, and the partition-weighted feature layer. The `regression/` folder reproduces the Concrete and Superconductivity regression experiments. The `classification/` folder reproduces the MNIST clustering-only, raw-logit, local-logit, feature-layer, external-classifier, and centroid-standardization experiments. The `appendix_examples/` folder reproduces the numerical examples for crisp and fuzzy clustering. The `results_and_figures/` folder exports the article tables and figures.

## Parameter grids

The partition parameters are searched in the same form across the main scripts.

- `k` is searched from `2` to `floor(n / 10)`, where `n` is the training-sample size.
- `p` is searched from `1.00` to `10.00` with step size `0.05`.
- `f` is searched from `1.00` to `10.00` with step size `0.05`.
- Polynomial or logit degree is evaluated as a fixed setting from `1` to `4`.

For large experiments, the scripts also accept reduced candidate lists so that a short reproducibility run can be performed before a full grid run.

## Main code groups

| Folder | Files | Main role |
|---|---:|---|
| `dataset/` | `01–05` | Dataset preparation, splitting, and dataset summary |
| `core/` | `01–11` | Shared partition, membership, truncation, local model, and feature-layer routines |
| `regression/` | `01–10` | Concrete and Superconductivity regression experiments |
| `classification/` | `01–09` | MNIST classification and membership-centroid experiments |
| `appendix_examples/` | `01–04` | Numerical crisp and fuzzy clustering examples |
| `results_and_figures/` | `01–06` | Article tables, plots, external-model figures, and cross-dataset summary |

## Typical workflow

The scripts can be run one file at a time. The top-level `run_all.py` file provides a single entry point for running selected stages. The staged format keeps the outputs inspectable and makes it possible to reproduce only one dataset or one article block when needed.

Prepared data are written under `data/processed/`. Split arrays are written under `data/splits/`. Experiment outputs are written under `outputs/`. Figures are written under `outputs/figures/`.

## Output files

The main regression runners write combined tables under:

```text
outputs/regression/concrete/concrete_regression_all_results.csv
outputs/regression/superconductivity/superconductivity_regression_all_results.csv
```

The MNIST runner writes combined classification tables under:

```text
outputs/classification/mnist/09_mnist_all_results_long.csv
outputs/classification/mnist/09_mnist_best_rows.csv
```

The final cross-dataset summary is written under:

```text
outputs/summary/06_cross_dataset_article_summary.csv
outputs/figures/cross_dataset/06_cross_dataset_best_rows.png
```

## Notes on reproducibility

The default split uses an 80/20 held-out test design with a validation portion inside the training data. Numerical variables in the tabular datasets are standardized from the training split and then transformed consistently across validation and test data. MNIST pixels are scaled to `[0, 1]` before partition learning and classification.
