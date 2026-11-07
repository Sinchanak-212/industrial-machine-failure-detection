# Experiment Log

Evaluation run: 2026-11-07T14:29:15+05:30

## Existing completed experiments

| Experiment | Existing evidence | Evaluation procedure |
|---|---|---|
| Shared Mel CNN autoencoder | `models/best_autoencoder.pth`, `models/normalization.npz`, `reports/source_test_results.csv` | One CNN autoencoder trained on normal fan recordings; shared training normalization; 128-bin Mel features. |
| Section-specific Mel CNN autoencoders | `models/section_models/`, `reports/section_model_comparison.csv`, `reports/section_test_predictions.csv` | Separate normal-only model and checkpoint normalization per section. |
| Section-specific MFCC CNN autoencoders | `models/mfcc_section_models/`, `reports/mfcc_section_model_comparison.csv`, `reports/mfcc_section_test_predictions.csv` | 40-MFCC inputs; 30 epochs, batch size 8, Adam learning rate 0.001; checkpoints include training normalization. |
| MFCC frame-level scoring | `reports/mfcc_frame_comparison.csv`, `reports/mfcc_frame_test_predictions.csv` | Mean frame reconstruction error and per-recording frame P95; the P95 threshold is recomputed from the original normal validation split, not test labels. |
| MFCC + Isolation Forest | `models/isolation_forest/`, `reports/isolation_forest_comparison.csv`, `reports/isolation_forest_predictions.csv` | 160 statistical MFCC features; saved StandardScaler and 300-tree Isolation Forest; saved threshold from the 5th percentile of normal validation decision scores. |

## Final evaluation

- Loaded existing checkpoints, scalers, and saved Isolation Forest thresholds; no weights were modified or saved.
- Reproduced per-section seeded training/validation indices with `random_state=42`.
- Existing CNN configuration: 30-epoch maximum, batch size 8, Adam learning rate 0.001, mean-squared error objective, seed 42. Existing Isolation Forest configuration: 300 trees, `contamination="auto"`, seed 42.
- Evaluated all 600 source-test and all 600 target-test recordings; target counts were checked as 100 normal and 100 anomalous per section.
- Evaluated all five experiment types at their training-derived normal-validation thresholds. Source scores were read from existing prediction reports; target scores were calculated one recording at a time.
- Computed per-section metrics, equal-section macro averages, confusion counts, score-distribution summaries, and ROC data.
- Plots were generated after scoring. The "strongest models" ROC figure ranks candidates by source-test macro ROC-AUC; target-test performance did not influence selection or thresholds.
- No data, training scripts, or model checkpoints were changed.

## Results

The source-test macro ROC-AUC leader is **Section-specific MFCC CNN** (0.5698). It has target-test macro ROC-AUC 0.6091. The descriptive target-test macro ROC-AUC leader is **MFCC + Isolation Forest** (0.6105); it was not selected using target results.

Detailed tables: `FINAL_RESULTS.csv`. Per-record scores: `FINAL_TEST_PREDICTIONS.csv`. Full interpretation and limitations: `FINAL_EXPERIMENT_ANALYSIS.md`.
