# Final Experiment Analysis

Evaluation run: 2026-11-07T14:29:15+05:30

## Executive summary

Model comparison and the locked-threshold selection are based on source-test performance, not target-test results. The source-test macro ROC-AUC leader was **Section-specific MFCC CNN** (ROC-AUC 0.5698). It achieved target-test macro ROC-AUC 0.6091, accuracy 0.5017, precision 0.4981, recall 0.0567, and F1 0.1016. The highest descriptive target-test macro ROC-AUC was 0.6105 for MFCC + Isolation Forest; this target-only ranking is reported for transparency and was **not** used to select a model or threshold.

The target split contains 600 recordings (200 per section: 100 normal and 100 anomalous). All target scores use frozen checkpoints and normalization; threshold values are fixed using only the original normal validation split or saved normal-validation thresholds.

## Dataset and attribution

Audio is the fan subset of **MIMII DUE**, Zenodo record [4740355](https://doi.org/10.5281/zenodo.4740355), licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The project directory contains 3,009 normal training recordings (1,003 per section), 600 source-test recordings, and 600 target-test recordings. Each test split has 100 normal and 100 anomalous recordings for each of sections 00, 01, and 02.

## Preprocessing and evaluation methodology

- Mono audio is produced by averaging channels. Audio is resampled to 16 kHz when necessary.
- Mel CNN inputs use 128-bin power Mel spectrograms (`n_fft=1024`, `hop_length=256`) converted to dB with `ref=1.0`.
- MFCC CNN and Isolation Forest inputs use 40 MFCC coefficients (`n_fft=1024`, `hop_length=256`). Each Isolation Forest input contains the per-coefficient mean, standard deviation, 10th percentile, and 90th percentile (160 features).
- Model-specific normalization was retained from training checkpoints; Isolation Forest uses its saved `StandardScaler`.
- Each model uses the saved architecture and weights. CNN autoencoders have convolutional encoder channels 1-16-32-64 with stride 2, followed by a transposed-convolution decoder 64-32-16-1. The shared Mel checkpoint has one global model; Mel and MFCC section-specific checkpoints have one model per section.
- CNNs were trained for up to 30 epochs with batch size 8, Adam (`learning_rate=0.001`), mean-squared reconstruction loss, and seed 42; checkpoints retain the best validation-loss epoch. Each Isolation Forest uses 300 trees, `contamination="auto"`, and `random_state=42`.
- Training and validation use only normal recordings. The scripts use an independent, seeded 90/10 split within each section (`random_state=42`). Validation scores are computed from the exact validation indices in the original scripts. Reconstruction thresholds are the 95th percentile of normal validation recording-level reconstruction errors; MFCC frame-P95 uses the 95th percentile of each validation recording's frame-error P95. Isolation Forest uses its saved 5th-percentile `decision_function` threshold, equivalent to a 95th-percentile anomaly-score threshold after sign reversal.
- Saved Isolation Forest thresholds were used directly. Section-specific Mel and MFCC CNN validation thresholds were reproduced using the original evaluation procedures; the shared Mel and MFCC frame-P95 thresholds, which were not persisted as validation-threshold artifacts, were deterministically recomputed from the original normal validation split. The shared Mel checkpoint normalization was verified against `models/normalization.npz`.
- No model was trained, fine-tuned, normalized, thresholded, or tuned on either test split during this final evaluation. Target labels were used only to calculate evaluation metrics.
- Macro metrics are unweighted averages of the three section-level metrics. Confusion counts and score-distribution summaries in macro rows are pooled across all 600 recordings. Missing comparisons are represented as `N/A` in tables rather than as zero.

## Macro-average source/target comparison

Metrics below apply the same validation-derived threshold to each test set. ROC-AUC is threshold-independent.

| Experiment | Test set | ROC-AUC | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Shared Mel CNN | Source test | 0.5348 | 0.5133 | 0.6852 | 0.0933 | 0.1550 |
| Section-specific Mel CNN | Source test | 0.5545 | 0.5200 | 0.5742 | 0.1267 | 0.2069 |
| Section-specific MFCC CNN | Source test | 0.5698 | 0.5117 | 0.5868 | 0.0667 | 0.1175 |
| MFCC frame-level P95 | Source test | 0.5668 | 0.5117 | 0.5432 | 0.0967 | 0.1590 |
| MFCC + Isolation Forest | Source test | 0.5616 | 0.5233 | 0.6513 | 0.0933 | 0.1597 |
| Shared Mel CNN | Target test | 0.5217 | 0.4933 | 0.6284 | 0.0733 | 0.1191 |
| Section-specific Mel CNN | Target test | 0.5514 | 0.5200 | 0.6429 | 0.1200 | 0.1896 |
| Section-specific MFCC CNN | Target test | 0.6091 | 0.5017 | 0.4981 | 0.0567 | 0.1016 |
| MFCC frame-level P95 | Target test | 0.5779 | 0.5133 | 0.5625 | 0.0900 | 0.1541 |
| MFCC + Isolation Forest | Target test | 0.6105 | 0.5117 | 0.3765 | 0.0833 | 0.1352 |

## Target-test per-section performance

| Experiment | Section | ROC-AUC | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Shared Mel CNN | section_00 | 0.4725 | 0.4900 | 0.4643 | 0.1300 | 0.2031 |
| Shared Mel CNN | section_01 | 0.4666 | 0.4850 | 0.4211 | 0.0800 | 0.1345 |
| Shared Mel CNN | section_02 | 0.6261 | 0.5050 | 1.0000 | 0.0100 | 0.0198 |
| Section-specific Mel CNN | section_00 | 0.4551 | 0.5000 | 0.5000 | 0.1000 | 0.1667 |
| Section-specific Mel CNN | section_01 | 0.5529 | 0.5450 | 0.6286 | 0.2200 | 0.3259 |
| Section-specific Mel CNN | section_02 | 0.6463 | 0.5150 | 0.8000 | 0.0400 | 0.0762 |
| Section-specific MFCC CNN | section_00 | 0.7126 | 0.5150 | 0.6154 | 0.0800 | 0.1416 |
| Section-specific MFCC CNN | section_01 | 0.5323 | 0.4850 | 0.3333 | 0.0300 | 0.0550 |
| Section-specific MFCC CNN | section_02 | 0.5824 | 0.5050 | 0.5455 | 0.0600 | 0.1081 |
| MFCC frame-level P95 | section_00 | 0.6776 | 0.5300 | 0.6875 | 0.1100 | 0.1897 |
| MFCC frame-level P95 | section_01 | 0.5234 | 0.4900 | 0.4000 | 0.0400 | 0.0727 |
| MFCC frame-level P95 | section_02 | 0.5328 | 0.5200 | 0.6000 | 0.1200 | 0.2000 |
| MFCC + Isolation Forest | section_00 | 0.6080 | 0.5000 | 0.5000 | 0.0800 | 0.1379 |
| MFCC + Isolation Forest | section_01 | 0.5633 | 0.5350 | 0.6296 | 0.1700 | 0.2677 |
| MFCC + Isolation Forest | section_02 | 0.6603 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |

Detailed normal/anomaly score summaries, confusion counts, thresholds, and per-record scores are in `FINAL_RESULTS.csv` and `FINAL_TEST_PREDICTIONS.csv`.

## Interpretation and limitations

The source-to-target ROC-AUC change for the source-selected model is +0.0393. Its target performance is weak-to-moderate rank separation in this held-out domain; thresholded accuracy, recall, and F1 should be considered alongside AUC. This is anomaly detection trained only on normal recordings, not a demonstrated predictor of future failures or a classifier of specific failure types.

Several experiments may show low recall despite their validation-derived 95th-percentile normal thresholds. A threshold calibrated to keep most normal validation scores below cutoff does not guarantee separation of anomalous scores. Mean pooling can dilute brief or localized departures; frame P95 can also be sensitive to normal transient variation. MFCC summary statistics discard temporal ordering, limiting the Isolation Forest to recording-level distribution summaries. Differences between normal validation and either test domain may further shift score distributions. These are plausible explanations, not causal findings established by this evaluation.

The source-test results for the shared Mel model in the older `source_test_results.csv` used a provisional threshold calculated from source-test normal labels. This analysis retains its recorded source scores for ROC-AUC but recalculates its thresholded source metrics using the training-derived normal validation threshold, avoiding use of source-test labels for threshold selection. All other experiments are also compared from their per-record scores under the same validation-only threshold rule.

Limitations include three machine sections and one fan subset; one source and one target test partition; no uncertainty intervals or repeated seeds; possible recording/domain dependence; and no demonstrated mapping from sound anomaly scores to machine failure onset, severity, or failure class. A missing root `README.md` prevented verification of any project-specific documentation beyond the scripts, artifacts, and dataset attribution supplied for this evaluation.

## Future improvements

Use multiple machine IDs and operating conditions, repeated grouped splits and confidence intervals, calibration on a strictly separate normal validation set, and prospective monitoring data. Evaluate robustness to recording conditions and device variation, investigate temporal scoring/localization, and obtain independently verified fault labels before making failure-prediction or fault-type claims.

## Figures

- `source_target_roc_auc.png`
- `strongest_model_roc_curves.png` (models ranked using source-test macro ROC-AUC)
- `final_selected_confusion_matrices.png`
- `anomaly_score_distributions.png`
- `per_section_performance.png`
