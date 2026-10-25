# Reviewer-response analysis utilities

These scripts provide supplementary analyses for the revised manuscript while
leaving the original experiment files intact.

## Recommended order

1. Run `05_self_check.py` to verify the local environment, then run `04_export_search_protocol.py` to document the actual legacy search-space
   size and current fixed settings.
2. Run `01_metric_consistency_audit.py` on the existing `outputs/` folder. Treat
   flags as items to investigate, not as automatically corrected results.
3. Run `02_repeated_split_ablation.py` on each dataset. It uses repeated outer
   train/test splits and inner K-fold selection, then evaluates matched
   component ablations on each held-out test set.
4. Run `03_hpd_stability_analysis.py` using the selected or otherwise
   pre-specified hyperparameters to assess threshold sensitivity and its
   relationship to downstream predictive performance.

## Interpretation boundaries

- The new repeated-split runner is a supplementary protocol; its results are
  not identical to the legacy single-split tables.
- The Takagi–Sugeno-style weighted local baseline is a controlled comparator,
  not a complete ANFIS implementation.
- The learned-gate baseline is a lightweight MoE-inspired control trained to
  predict dominant partition labels; it is not a canonical jointly optimized
  mixture-of-experts implementation.
- The feature-layer baseline is a linear external learner over `[x, u, u*x]`;
  it is not intended to represent every possible neural or tree-based model.
- A sampled centroid fit is marked in the configuration and output files. Do
  not report a sampled fit as full-data clustering.
- Preserve the exact output configuration, candidate logs, predictions, and
  environment/package versions with the results used in the manuscript.
