# Fuzzy Clustering Induced Membership-Based Learning

This repository contains the implementation accompanying the manuscript on a
fuzzy clustering induced membership-based learning framework for regression and
classification. The original experiment scripts remain organized by the same
pipeline used in the manuscript: data preparation, partition learning, local
prediction, external feature-layer comparisons, and table/figure generation.

## Repository layout

```text
fuzzy-partition-weighted-learning/
├── dataset/                 # Dataset preparation and train/test split utilities
├── core/                    # Distances, memberships, truncation, local models
├── regression/              # Concrete and superconductivity experiments
├── classification/          # MNIST experiments
├── appendix_examples/      # Numerical crisp/fuzzy clustering examples
├── results_and_figures/     # Table and figure generation
├── reviewer_response/      # Repeated-split analysis, ablations, metric audit
├── README.md
├── requirements.txt
└── run_all.py
```

The three public benchmarks are prepared by the scripts in `dataset/`. The
prepared files are written under `data/processed/`; split arrays and legacy
experiment outputs are written under `data/splits/` and `outputs/`.

## Original experiment scripts

The legacy scripts reproduce the original single-split comparison. Their
validation set is used for candidate selection, and the held-out test set is
used for final evaluation. The legacy search space is configured in
`core/01_config.py`:

- `k`: integers from 2 to `floor(n / 10)`;
- `f`: 1.00 to 10.00 in increments of 0.05;
- `p`: 1.00 to 10.00 in increments of 0.05;
- local polynomial/logit degree: 1, 2, 3, and 4, evaluated as fixed settings;
- truncation rules: DTD, harmonic, SP, entropy, and HPD;
- regression ridge regularization: `alpha = 1e-4` by default;
- stopping tolerance: `1e-6` by default.

**Important:** the legacy scripts use a single pre-created validation split; they
do not implement K-fold cross-validation. Use the new
`reviewer_response/02_repeated_split_ablation.py` runner for inner K-fold model
selection and repeated outer test splits. When reporting revised results, use
the protocol that was actually run and describe it accurately in the manuscript.

The full theoretical `k × f × p` grid is extremely large. Use
`reviewer_response/04_export_search_protocol.py` to calculate and record its
size. In the legacy regression scripts, `--max-candidates` truncates the
ordered candidate list, which can favor early values in the list. For reviewer
analysis, provide explicit candidate lists to the new runner and preserve its
`run_config.json` and candidate log. If the new runner uses `--max-candidates`,
it takes a deterministic evenly spaced subset of the listed Cartesian grid.

## Reviewer-response analysis tools

The `reviewer_response/` folder adds analysis without replacing the original
experiment modules:

| File | Purpose |
|---|---|
| `01_metric_consistency_audit.py` | Check whether reported MSE and R² imply a consistent target variance; optionally recompute metrics from saved predictions |
| `02_repeated_split_ablation.py` | Repeated outer 80/20 splits, inner K-fold selection, component-wise ablations, confidence intervals, paired tests, and optional row-level predictions |
| `03_hpd_stability_analysis.py` | Test HPD stopping-threshold sensitivity and connect partition stability to downstream test metrics |
| `04_export_search_protocol.py` | Export legacy candidate ranges and estimate the theoretical tuning cost |
| `05_self_check.py` | Run small data-free checks for metrics, chunked distances, memberships, and local models |
| `_shared.py` | Shared data loading, preprocessing, partition fitting, model, and metric helpers |

`core/02_metrics.py` now also reports `target_variance`, computed from the same
`y_true` vector as MSE and R². The identity
`R² = 1 - MSE / Var(y_true)` can therefore be checked directly. The updated
`core/03_distance_functions.py` uses adaptive row chunking for large
sample/centroid/feature combinations to bound temporary memory use.

The component ablation runner compares the following views while holding the
selected partition fixed:

- global raw-input predictor;
- local raw-input model without membership-weighted fitting;
- Takagi–Sugeno-style weighted local predictor (raw local inputs, membership-weighted fitting, fuzzy aggregation);
- membership-gated local predictor without membership-weighted fitting;
- full gated/weighted model with hard winner-take-all aggregation;
- full partition-weighted model with fuzzy aggregation;
- a hard-membership local control;
- a lightweight mixture-of-experts-inspired soft gate trained to predict dominant partition labels; and
- the `[x, u, u*x]` feature layer when the dense feature matrix is below a configured size limit.

The Takagi–Sugeno-style row is a transparent weighted-local-model comparator; it
is **not** a full ANFIS implementation or a claim of reproducing every
Takagi–Sugeno system. Likewise, the learned-gate comparator is a cluster-supervised MoE-inspired control, not a canonical jointly optimized mixture-of-experts implementation, and the feature-layer model is a controlled external learner. These
comparisons isolate the operations implemented in this repository and should be
described with those boundaries.

## Setup

From the repository root:

```bash
pip install -r requirements.txt
```

Prepare the datasets if they are not already available:

```bash
python dataset/01_prepare_concrete.py
python dataset/02_prepare_superconductivity.py
python dataset/03_prepare_mnist.py
```

## Example commands

Run the data-free self-check first:

```bash
python reviewer_response/05_self_check.py
```

### 1. Audit existing result tables

```bash
python reviewer_response/01_metric_consistency_audit.py \
  --input-root outputs \
  --output-dir outputs/reviewer_response/metric_audit
```

This writes `aggregate_metric_audit.csv`, a grouped summary, and a text report.
The audit flags discrepancies; it does not silently rewrite paper tables. The
legacy aggregate CSV files do not contain every observation-level prediction,
so a definitive recomputation of an old result requires rerunning that model or
providing a prediction file.

### 2. Repeated-split ablation: Concrete

```bash
python reviewer_response/02_repeated_split_ablation.py \
  --input data/processed/concrete.csv \
  --target concrete_compressive_strength \
  --task regression \
  --dataset-name concrete \
  --seeds 42,43,44,45,46 \
  --cv-folds 3 \
  --k-values 5,10 \
  --f-values 1.5,2.0 \
  --p-values 2.0 \
  --truncations hpd \
  --tolerances 1e-6 \
  --regularization 0.0001,0.01 \
  --degree 1 \
  --max-iter 100 \
  --output-dir outputs/reviewer_response/concrete
```

### 3. Repeated-split ablation: Superconductivity

```bash
python reviewer_response/02_repeated_split_ablation.py \
  --input data/processed/superconductivity.csv \
  --target critical_temp \
  --task regression \
  --dataset-name superconductivity \
  --seeds 42,43,44,45,46 \
  --cv-folds 3 \
  --k-values 10,20 \
  --f-values 1.5,2.0 \
  --p-values 2.0 \
  --truncations hpd \
  --tolerances 1e-6 \
  --regularization 0.0001,0.01 \
  --degree 1 \
  --max-iter 100 \
  --partition-sample-size 5000 \
  --output-dir outputs/reviewer_response/superconductivity
```

The `--partition-sample-size 5000` option is an optional faster diagnostic
setting for centroid fitting. It is recorded in the run configuration and
should not be described as full-data partition fitting. Omit it for the full
outer/fold training set.

### 4. Repeated-split ablation: MNIST

```bash
python reviewer_response/02_repeated_split_ablation.py \
  --input data/processed/mnist_full.npz \
  --task classification \
  --dataset-name mnist \
  --seeds 42,43,44,45,46 \
  --cv-folds 3 \
  --k-values 5 \
  --f-values 1.5 \
  --p-values 2.0 \
  --truncations hpd \
  --tolerances 1e-6 \
  --regularization 0.1,1.0 \
  --degree 1 \
  --max-iter 50 \
  --partition-sample-size 5000 \
  --output-dir outputs/reviewer_response/mnist
```

For MNIST and other high-dimensional inputs, the full feature layer can be
skipped automatically if its estimated dense size exceeds
`--feature-layer-max-elements`. The output records whether it was evaluated or
skipped. Start with a small candidate grid to check the runtime, then expand the
grid for the final study. Use at least five outer seeds for an initial
robustness analysis; more repetitions are preferable when compute permits.

The runner writes:

- `run_config.json`: the exact split, CV, preprocessing, grid, and runtime settings;
- `inner_cv_candidate_log.csv`: candidate/fold losses, convergence, and runtime;
- `selected_hyperparameters.csv`: selected configuration for each outer seed;
- `run_metrics.csv`: one row per seed and ablation method;
- `summary_mean_sd_ci95.csv`: mean, standard deviation, and 95% t-based confidence intervals;
- `paired_significance_tests.csv`: paired t-tests, Wilcoxon tests, and Holm-adjusted Wilcoxon p-values;
- `test_predictions.csv`: optional observation-level predictions when `--save-predictions` is supplied.

The primary selection metric is MSE for regression and cross entropy for
classification. The test split is not used for hyperparameter selection.
Regularization is interpreted as Ridge `alpha` for regression and inverse
logistic-regression `C` for classification.

### 5. HPD threshold sensitivity

```bash
python reviewer_response/03_hpd_stability_analysis.py \
  --input data/processed/concrete.csv \
  --target concrete_compressive_strength \
  --task regression \
  --dataset-name concrete \
  --seeds 42,43,44,45,46 \
  --k 10 \
  --f 2.0 \
  --p 2.0 \
  --tolerances 0,1e-6,1e-4,1e-3 \
  --degree 1 \
  --regularization 0.0001 \
  --max-iter 100 \
  --output-dir outputs/reviewer_response/hpd_concrete
```

This is a sensitivity analysis at a fixed hyperparameter configuration, not a
replacement for CV-based selection. It compares each stopping threshold with a
zero-tolerance reference on the same outer training data and reports HPD,
adjusted Rand index, partition objective, iteration count, and downstream test
metrics.

### 6. Export the legacy search-space size

```bash
python reviewer_response/04_export_search_protocol.py \
  --n-train 824 \
  --output-dir outputs/reviewer_response/search_protocol
```

Replace `824` with the actual training-sample count used by the legacy search.
The output records theoretical grid size, fixed regularization/tolerance, and
the distinction between the legacy validation protocol and the new inner-CV
runner.

## Reading the statistical output

- Report mean and standard deviation across outer seeds; confidence intervals
  are also provided.
- Paired tests use the same outer split seed for each method. The
  `mean_delta_full_minus_comparator` sign depends on the metric: a negative
  value favors `full_pw` for MSE/MAE/cross entropy, while a positive value
  favors it for R²/accuracy.
- The number of seeds is the number of independent outer splits, not the
  number of observations. With only five seeds, significance tests can have
  low power and should be interpreted alongside effect sizes and intervals.
- The metric audit can detect inconsistent aggregate tables, but the final
  published values should come from recomputed predictions and a single,
  documented test protocol.
