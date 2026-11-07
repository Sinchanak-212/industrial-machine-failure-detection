# Confidence Intervals and Threshold Sensitivity

Analysis run: 2026-11-07T14:48:40+05:30

## Data integrity and split reconstruction

The original train scripts (`train_sections.py`, `train_mfcc_sections.py`, and `train_isolation_forest.py`) each sort the 3,009 normal training recordings, select each section independently, then call `train_test_split(indices, test_size=0.10, random_state=42)`. Reproducing this exact operation yields 902 training and 101 normal validation recordings per section. Existing feature arrays align with the same sorted file list. Validation scores below were recomputed only by inference on those original validation indices, using frozen checkpoints, the checkpoint normalization values, and the saved Isolation Forest scaler/model. No validation labels beyond their normal-only membership were used to calculate thresholds.

The existing target prediction records contain split, experiment, section, filename, true label, anomaly score, locked 95th-percentile validation threshold, and prediction. The analysis verified uniqueness, score direction, score-threshold consistency, 100 normal and 100 anomalous target examples per section, and agreement between reproduced p95 thresholds and the existing evaluation report.

## Bootstrap confidence intervals

The analysis uses 2,000 percentile-bootstrap replicates with NumPy generator seed 20261107. In every replicate, recordings are sampled with replacement separately within each of the six section-by-label groups (three sections × two true labels), preserving 100 observations per label per section. The same sampled target filenames are used across experiments, and the MFCC CNN vs Isolation Forest comparison is paired on identical resamples. Section-level statistics are calculated first; macro intervals are percentile intervals of the equal-section mean metric in each replicate. Confidence bounds are the 2.5th and 97.5th percentiles of bootstrap replicates.

All target thresholded metrics use each experiment's original validation-derived p95 threshold. Precision uses `zero_division=0`, matching the original report convention; section-level bootstrap replicates with no predicted anomalies are assigned zero precision and counted explicitly. For the Isolation Forest, section 02 had no predicted anomalies in all 2,000 replicates; the macro precision is the mean of section precisions with this convention applied. This avoids undefined numeric values but should be considered when interpreting precision intervals.

### Macro ROC-AUC (target test)

| Experiment | Point estimate | 95% CI lower | 95% CI upper |
| --- | --- | --- | --- |
| Shared Mel CNN | 0.5217 | 0.4755 | 0.5653 |
| Section-specific Mel CNN | 0.5514 | 0.5057 | 0.5972 |
| Section-specific MFCC CNN | 0.6091 | 0.5651 | 0.6543 |
| MFCC frame-level P95 | 0.5779 | 0.5332 | 0.6237 |
| MFCC + Isolation Forest | 0.6105 | 0.5662 | 0.6557 |

### Paired target ROC-AUC difference

Difference is defined as **Section-specific MFCC CNN minus MFCC + Isolation Forest**; shared stratified resamples were used for both scores.

| Paired comparison | Point difference | 95% CI lower | 95% CI upper |
| --- | --- | --- | --- |
| Section-specific MFCC CNN - MFCC + Isolation Forest | -0.0014 | -0.0365 | 0.0338 |

The paired 95% interval includes zero; these data do not establish a difference between the two models' macro ROC-AUC. This is not a comparison of models selected using target results; the pair is emphasized as requested. Point-estimate ordering alone is not sufficient to claim superiority.

Section-wise intervals for all five metrics and all experiments, as well as the per-section paired AUC differences, are provided in `BOOTSTRAP_RESULTS.csv`.

## Validation-only thresholds and observed calibration false-positive rates

Thresholds are empirical NumPy linear percentiles of **normal validation anomaly scores**; a recording is anomalous only when `score > threshold`. The four nominal tail probabilities are 10%, 5%, 2.5%, and 1%. Because there are 101 validation recordings per section, observed rates are discrete and may differ from nominal rates. The Isolation Forest score is `-decision_function`, so larger scores consistently indicate greater anomaly.

The shared Mel model uses one pooled normal-validation threshold across all three sections (303 validation recordings); the section-specific models use the original 101 validation recordings for that section. The shared threshold is then applied to each section. Validation is normal-only, so validation false-positive rates can be measured, but validation precision, recall, F1, and anomaly detection rates cannot be estimated from these data.

### 95th-percentile thresholds for the primary MFCC and Isolation Forest models

| Experiment | Section | Threshold | Validation false positives | Validation normals | Observed validation FPR |
| --- | --- | --- | --- | --- | --- |
| Section-specific MFCC CNN | section_00 | 0.0035 | 5 | 101 | 0.0495 |
| Section-specific MFCC CNN | section_01 | 0.0045 | 5 | 101 | 0.0495 |
| Section-specific MFCC CNN | section_02 | 0.0032 | 5 | 101 | 0.0495 |
| MFCC + Isolation Forest | section_00 | 0.0086 | 5 | 101 | 0.0495 |
| MFCC + Isolation Forest | section_01 | 0.0103 | 5 | 101 | 0.0495 |
| MFCC + Isolation Forest | section_02 | 0.0136 | 5 | 101 | 0.0495 |

Complete threshold values and observed counts at all four percentiles for all five experiments are in `VALIDATION_THRESHOLD_DETAILS.csv`.

## Locked-threshold target-test sensitivity

The target test labels were used only to compute the following evaluation metrics after validation thresholds had been fixed. Every row below is a sensitivity-analysis operating point, not a threshold selection. ROC-AUC does not depend on the applied threshold and therefore repeats for each percentile. Macro thresholded scores are equal-section means; confusion counts are pooled.

| Experiment | Val percentile | ROC-AUC | Accuracy | Precision | Recall | F1 | FP | FN | TP | TN | Target FPR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Shared Mel CNN | 90.0000 | 0.5217 | 0.5017 | 0.5597 | 0.1267 | 0.1896 | 37 | 262 | 38 | 263 | 0.1233 |
| Shared Mel CNN | 95.0000 | 0.5217 | 0.4933 | 0.6284 | 0.0733 | 0.1191 | 26 | 278 | 22 | 274 | 0.0867 |
| Shared Mel CNN | 97.5000 | 0.5217 | 0.5017 | 0.3406 | 0.0600 | 0.1008 | 17 | 282 | 18 | 283 | 0.0567 |
| Shared Mel CNN | 99.0000 | 0.5217 | 0.5000 | 0.3111 | 0.0333 | 0.0591 | 10 | 290 | 10 | 290 | 0.0333 |
| Section-specific Mel CNN | 90.0000 | 0.5514 | 0.5183 | 0.6376 | 0.1567 | 0.2340 | 36 | 253 | 47 | 264 | 0.1200 |
| Section-specific Mel CNN | 95.0000 | 0.5514 | 0.5200 | 0.6429 | 0.1200 | 0.1896 | 24 | 264 | 36 | 276 | 0.0800 |
| Section-specific Mel CNN | 97.5000 | 0.5514 | 0.5050 | 0.5984 | 0.0667 | 0.1164 | 17 | 280 | 20 | 283 | 0.0567 |
| Section-specific Mel CNN | 99.0000 | 0.5514 | 0.5033 | 0.5778 | 0.0467 | 0.0851 | 12 | 286 | 14 | 288 | 0.0400 |
| Section-specific MFCC CNN | 90.0000 | 0.6091 | 0.5733 | 0.6604 | 0.3200 | 0.4295 | 52 | 204 | 96 | 248 | 0.1733 |
| Section-specific MFCC CNN | 95.0000 | 0.6091 | 0.5017 | 0.4981 | 0.0567 | 0.1016 | 16 | 283 | 17 | 284 | 0.0533 |
| Section-specific MFCC CNN | 97.5000 | 0.6091 | 0.5000 | 0.4778 | 0.0300 | 0.0558 | 9 | 291 | 9 | 291 | 0.0300 |
| Section-specific MFCC CNN | 99.0000 | 0.6091 | 0.5033 | 0.5833 | 0.0167 | 0.0323 | 3 | 295 | 5 | 297 | 0.0100 |
| MFCC frame-level P95 | 90.0000 | 0.5779 | 0.5433 | 0.6283 | 0.2300 | 0.3348 | 43 | 231 | 69 | 257 | 0.1433 |
| MFCC frame-level P95 | 95.0000 | 0.5779 | 0.5133 | 0.5625 | 0.0900 | 0.1541 | 19 | 273 | 27 | 281 | 0.0633 |
| MFCC frame-level P95 | 97.5000 | 0.5779 | 0.4983 | 0.3519 | 0.0333 | 0.0609 | 11 | 290 | 10 | 289 | 0.0367 |
| MFCC frame-level P95 | 99.0000 | 0.5779 | 0.4983 | 0.3667 | 0.0200 | 0.0379 | 7 | 294 | 6 | 293 | 0.0233 |
| MFCC + Isolation Forest | 90.0000 | 0.6105 | 0.5317 | 0.7443 | 0.1633 | 0.2339 | 30 | 251 | 49 | 270 | 0.1000 |
| MFCC + Isolation Forest | 95.0000 | 0.6105 | 0.5117 | 0.3765 | 0.0833 | 0.1352 | 18 | 275 | 25 | 282 | 0.0600 |
| MFCC + Isolation Forest | 97.5000 | 0.6105 | 0.5117 | 0.3706 | 0.0567 | 0.0969 | 10 | 283 | 17 | 290 | 0.0333 |
| MFCC + Isolation Forest | 99.0000 | 0.6105 | 0.4967 | 0.0833 | 0.0033 | 0.0064 | 3 | 299 | 1 | 297 | 0.0100 |

### Original 95th-percentile operating point

| Experiment | ROC-AUC | Accuracy | Precision | Recall | F1 | FP | FN | TP | TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Shared Mel CNN | 0.5217 | 0.4933 | 0.6284 | 0.0733 | 0.1191 | 26 | 278 | 22 | 274 |
| Section-specific Mel CNN | 0.5514 | 0.5200 | 0.6429 | 0.1200 | 0.1896 | 24 | 264 | 36 | 276 |
| Section-specific MFCC CNN | 0.6091 | 0.5017 | 0.4981 | 0.0567 | 0.1016 | 16 | 283 | 17 | 284 |
| MFCC frame-level P95 | 0.5779 | 0.5133 | 0.5625 | 0.0900 | 0.1541 | 19 | 273 | 27 | 281 |
| MFCC + Isolation Forest | 0.6105 | 0.5117 | 0.3765 | 0.0833 | 0.1352 | 18 | 275 | 25 | 282 |

The 95th-percentile threshold remains a defensible **locked baseline** when the operational requirement is to calibrate an approximately 5% false-positive rate using normal-only validation data. The section-specific models achieve 5/101 validation false positives (4.95%) per section. The shared Mel model's pooled threshold gives 16/303 overall (5.28%), but varies by section (12/101, 4/101, and 0/101). The observed validation rate is finite-sample dependent; the threshold does not ensure high target recall or precision. Whether this is an acceptable operating point depends on the cost of false alarms versus missed anomalies and must be specified before deployment. Target-test sensitivity does not justify moving to another percentile.

## Figures

- `roc_auc_confidence_intervals.png` — section and equal-section macro ROC-AUC intervals.
- `paired_auc_difference.png` — paired bootstrap distribution for the requested model difference.
- `validation_score_distributions.png` — normal validation score distributions and all four thresholds.
- `threshold_sensitivity.png` — target precision/recall/F1 and target FPR across validation percentiles.
- `target_recall_vs_threshold.png`
- `target_fpr_vs_threshold.png`
- `target_precision_recall_curves.png` — descriptive target-test curves only; not used for threshold selection.

## Limitations

Bootstrap intervals quantify uncertainty from resampling these held-out recordings under an independence-within-stratum assumption, conditional on the fixed validation-derived thresholds; they do not include uncertainty in threshold calibration itself. They also do not account for variation across machines, recording sessions, operating conditions, seeds, or test domains; the available test set has only 100 observations per class per section. The percentile bootstrap is not a substitute for independent replicated datasets or a prospective evaluation. Precision intervals also depend on the stated zero-division convention. A normal-only validation set supports threshold/FPR calibration, not validation estimates of anomaly recall or precision.
