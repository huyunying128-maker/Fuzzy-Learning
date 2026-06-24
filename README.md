# Fuzzy Partition-Weighted Local Learning

This repository contains reproducible code for the paper **Fuzzy Partition-Weighted Local Learning for Regression and Classification**.

The implementation covers partition learning, membership-weighted local inputs, local polynomial regression, local logit classification, partition-weighted feature layers, and five truncation rules: distance-table difference, harmonic distance-change control, square probability, Shannon entropy, and hereditary partition distance.

## Repository structure

```text
.
├── fpwl_core.py
├── truncation_rules.py
├── feature_layer.py
├── metrics_utils.py
├── run_all_experiments.py
├── dataset/
│   ├── download_datasets.py
│   ├── prepare_regression_data.py
│   └── prepare_mnist_data.py
├── regression/
│   ├── regression_config.py
│   ├── run_global_polynomial_baselines.py
│   ├── run_local_regression_models.py
│   ├── run_pw_regression_baselines.py
│   └── summarize_regression_results.py
└── classification/
    ├── classification_config.py
    ├── run_mnist_local_logit.py
    ├── run_mnist_pw_classifiers.py
    ├── run_mnist_centroid_visualization.py
    └── summarize_classification_results.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, the environment activation command is:

```powershell
.venv\Scripts\Activate.ps1
```

## Data

The regression experiments use the concrete compressive strength dataset and the superconductivity critical-temperature dataset. The MNIST classification scripts use the TensorFlow Keras or OpenML MNIST loader.

```bash
python dataset/download_datasets.py
python dataset/prepare_regression_data.py --dataset concrete --input data/raw/Concrete_Data.xls --output data/processed/concrete_regression.npz
python dataset/prepare_mnist_data.py --output data/processed/mnist_classification.npz
```

## Regression experiments

Global polynomial baselines:

```bash
python regression/run_global_polynomial_baselines.py --dataset concrete --degrees 1,2,3,4
```

Crisp and fuzzy local regression models:

```bash
python regression/run_local_regression_models.py --dataset concrete --partitions crisp,fuzzy --degrees 1,2,3,4 --truncations dtd,harmonic,sp,entropy,hpd --grid-mode quick
```

External regressors with the partition-weighted feature layer:

```bash
python regression/run_pw_regression_baselines.py --dataset concrete --models random_forest,xgboost
```

Regression summaries:

```bash
python regression/summarize_regression_results.py
```

## Classification experiments

MNIST local-logit models:

```bash
python classification/run_mnist_local_logit.py --degrees 1 --partitions crisp,fuzzy --truncations hpd --k-values 10
```

MNIST external classifiers with the partition-weighted feature layer:

```bash
python classification/run_mnist_pw_classifiers.py --classifiers ann,random_forest --k 10
```

MNIST centroid-standardization visualization:

```bash
python classification/run_mnist_centroid_visualization.py
```

Classification summaries:

```bash
python classification/summarize_classification_results.py
```

## Compact workflow

A compact workflow is available for quick reproducibility checks:

```bash
python run_all_experiments.py --dataset concrete --mnist-samples 2000
```

The compact workflow uses small default grids and a limited MNIST sample size. Larger paper-scale searches can be launched through the individual experiment scripts by changing the grids in the configuration files or command-line arguments.

## Outputs

Result tables are written under `results/`. Processed NumPy bundles are written under `data/processed/`. Public raw data are stored under `data/raw/`.

## Main modules

- `fpwl_core.py`: crisp and fuzzy partition learning.
- `truncation_rules.py`: DTD, harmonic, SP, entropy, and HPD stopping rules.
- `feature_layer.py`: membership-weighted local inputs and partition-weighted feature layers.
- `metrics_utils.py`: regression and classification metrics.

## License

The code is intended for academic research and reproducibility. A repository-level license file can be added according to the publication or institution policy.
