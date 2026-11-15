# Fraud Detection Laboratory v1.1

Standalone Streamlit research app for fraud anomaly detection, calibration-aware evaluation, cost-sensitive thresholding, and restricted model serialization. **An anomaly alert is not proof of fraud.**

## Current release: v1.1

The v1.1 validated research workflow uses a 60/20/20 train/calibration/test split and `skops.io` serialization. The earlier v0.3 comparison laboratory remains available for exploratory analysis.

## Features
- Synthetic transaction generator (with labels reserved for evaluation)
- CSV, XLSX, and Parquet uploads (20 MB, 100k rows maximum)
- Isolation Forest, novelty-mode Local Outlier Factor, and a reconstruction MLP (autoencoder-style)
- Exploratory v0.3 comparisons use a shared 75/25 train/test split; validated v1.1 research uses 60/20/20 train/calibration/test
- Training-only imputation, encoding, and scaling; calibration-based threshold selection in the validated workflow
- Precision, recall, F1, false-positive rate, average precision, ROC-AUC when binary labels are provided
- Unlabeled scoring with alert rates and downloadable test predictions
- Descriptive feature deviations (not causal explanations or model attributions)

## Quick start
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Caveats
Synthetic fraud examples are deliberately separable; high synthetic metrics are not representative of real-world detection. Fraud labels must be binary 0/1 and are never model features. Avoid future-derived columns, repeated-entity leakage, identifiers, and post-outcome information. The app rejects obvious identifier column names, but cannot detect all leakage. For real temporal transaction streams, use chronological and entity-aware evaluation rather than this demonstration's random split. The reconstruction MLP is a scikit-learn MLPRegressor trained to reproduce inputs, not a custom deep-learning framework. Its convergence warning can occur at the configured iteration limit. Feature deviations are descriptive, not faithful algorithm explanations.

For production use, further work is required on threshold calibration, drift monitoring, fairness, privacy, data retention, access control, and human review.

## v0.3: Evaluation and explainability laboratory

- Interactive alert threshold slider, precision/recall/F1 curves, and a **cost-sensitive** threshold analysis with configurable false-alarm and missed-fraud costs.
- Optional percentile-rank ensemble across the selected detectors; scores are normalized to ranks before averaging.
- Labeled score scatter plot and the original score histograms, transaction tables, CSV export, and descriptive feature deviations.
- Three-seed robustness experiment with per-seed metrics, mean/std summary, and CSV export.

**Research limitations:** The exploratory threshold optimum is calculated using held-out labels and **must not** be presented as unbiased test performance. Select thresholds on a separate calibration set, then evaluate on a new untouched test set before operational use. Detector scores and rank ensembles are not calibrated fraud probabilities. Feature deviations are descriptive, not exact model attribution or causation. Repeated random splits overlap; group- or time-based splitting is necessary for repeated customers or chronological transaction histories. The app is not a real-time fraud prevention service.

## v1.0 — validated research workflow

The sidebar now has **v1.0 validated research**, **v0.3 comparison laboratory**, and **Trusted model inference**. The previous v0.3 experiments are preserved.

### Three-way evaluation

The v1.0 workflow uses **60% training, 20% calibration, and 20% final test**. The preprocessing pipeline and detector are fitted on training rows only. The calibration partition selects a threshold by minimizing user-specified false-alert and missed-fraud costs *if both label classes occur in calibration*. Otherwise the threshold is the training anomaly-score quantile. The test partition is not used to choose the threshold, and is evaluated only after selection. This is an offline research demonstration, not a live fraud decision system.

Choose chronological split **only after sorting your file oldest to newest**. The random split is unsuitable for entity-linked data (e.g. repeat customers) or sequential forecasting; use a group/time-aware partition outside this demonstration. Leakage screening rejects identifier-like features and direct label copies but cannot detect all proxies, post-outcome variables, or correlated entities. No model is guaranteed to work on real fraud.

### Explainability

Transaction-level sensitivity replaces each input feature with its *training-set reference value* and measures the resulting anomaly-score difference. This is not a causal explanation, Shapley value, or probability of fraud. Categorical changes may produce counterfactual combinations that are unrealistic.

### Model export and inference — SECURITY WARNING

The ZIP contains a `model.skops` and report. **v1.1 uses skops serialization with a strict, explicit allowlist of model object types.** Legacy pickle/joblib ZIPs are rejected. The SHA-256 checksum detects corruption but is not authentication. The allowlist reduces arbitrary-code execution risk but does not guarantee that third-party models or their predictions are safe. Validate model provenance before loading. Inference rejects missing required features; extra columns are ignored. The anomaly score is not a calibrated probability.

### Research reporting

Download `fraud_experiment_report.json`, `fraud_test_predictions.csv`, and your own model bundle. Reports include the seed, dataset fingerprint, selected features, threshold selection method, calibration and test metrics, and partition sizes. For a final scientific assessment, add independent datasets, confidence intervals, entity-aware splitting, drift monitoring, and external validation.

### Local launch

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py
```


## v1.1 security changes

- Uses `skops.io` rather than pickle/joblib for saving and loading model bundles.
- Checks ZIP entries, size limits, format version, checksum, estimator classes, and an explicit allowed-type list.
- Legacy `trusted_model.joblib` bundles are intentionally incompatible. Retrain and export models in v1.1.
- The allowlist is intentionally restrictive; unsupported objects fail closed. Skops compatibility varies with installed scikit-learn/skops versions.
- Run `pytest -q` and test save/load for all three algorithms before publishing.

## Pre-publication checklist

- Install dependencies and run `python -m pytest -q -rs`. Serialization tests must run, not skip.
- Export, import, and score new rows for all three detectors (Isolation Forest, Local Outlier Factor, reconstruction MLP).
- Confirm predictions match before and after export, and malformed and legacy bundles are rejected.
- Inspect `git status --short` and `git ls-files` before pushing; do not commit private transaction data or trained models.
- This is research software, not an audited production fraud decision service.
