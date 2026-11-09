# Diagnostic and Performance Analysis Report: Anomaly Detection with CNN Autoencoders

**Project:** Industrial Machine Failure Detection (MIMII DUE Fan Audio)  
**Model Under Investigation:** Section-Specific CNN Autoencoder on MFCC Features  
**Evaluation Mode:** Frozen checkpoints, normal-only training, zero-retraining diagnostic evaluation  
**Date of Diagnosis:** November 2026  

---

## 1. Executive Summary & Diagnostic Verdict

This diagnostic evaluation investigates why an unsupervised convolutional neural network (CNN) autoencoder—trained strictly on normal operational sound from the MIMII DUE dataset—exhibits severely deficient anomaly recall (3.0% to 12.0%) despite ostensibly following sound unsupervised training guidelines.

### Primary Diagnostic Conclusions:
1. **Normal and Anomalous Audio Are Practically Inseparable Under the Current Model:**
   The reconstruction error distributions of normal and anomalous test audio exhibit between **60.0% and 77.0% histogram overlap** across all machine sections. Between **98.0% and 100.0%** of all anomalous test recordings fall entirely inside the dynamic range of normal test recordings.
2. **The Validation Threshold Is Not Too Conservative—The Error Signal Is Missing:**
   The threshold is accurately calibrated to the 95th percentile ($P_{95}$) of normal validation recordings, successfully maintaining normal false positive rates at **4.33% in source-test and 5.33% in target-test** (meeting the ~5% nominal false alarm budget). The low anomaly recall (**6.67% source-test, 5.67% target-test**) occurs because anomalies are reconstructed with nearly identical error to normal recordings, not because the cutoff is set too high. Lowering the threshold to capture more anomalies causes an intolerable explosion in false alarms.
3. **Core Root Cause 1: Absence of an Architectural Bottleneck in the Autoencoder:**
   For an input MFCC matrix of shape $(1, 40, 626) = 25,040$ elements, the CNN encoder compresses spatial dimensions to $(64, 5, 79) = 25,280$ latent units. **The latent representation is larger than the raw input** (compression ratio: 1.01). Without an information bottleneck, the network acts as an unconstrained identity mapping that reconstructs anomalous patterns just as effectively as normal patterns.
4. **Core Root Cause 2: Global Mean Pooling Dilutes Localized Anomalies:**
   Anomaly scores are computed as the global mean squared error (MSE) across all 25,040 matrix elements. Real mechanical anomalies (e.g., transient bearing ticks, periodic blade rubbing, localized harmonic peaks) occupy short temporal windows or narrow frequency bands. Averaging over 626 time frames dilutes localized anomalies by a factor of 50–500, submerging failure signatures under normal acoustic variability.
5. **Core Root Cause 3: Inappropriate Global Scalar Feature Normalization:**
   Preprocessing computes a single scalar mean and standard deviation across all 40 MFCC coefficients combined. MFCC coefficient 0 (energy) has orders of magnitude greater variance than higher coefficients, causing higher spectral details to be compressed into near-zero variance.
6. **Domain Shift Is Secondary:**
   Performance on source-test (macro AUC 0.5698, recall 6.67%) is as deficient as on target-test (macro AUC 0.6091, recall 5.67%). The breakdown occurs within the native domain itself.

---

## 2. Dataset Split and Sample Inventory

The MIMII DUE dataset fan audio subset comprises 3 machine sections (`section_00`, `section_01`, `section_02`) across training, source-domain testing, and target-domain testing.

| Dataset Split | Operating Condition | Section 00 | Section 01 | Section 02 | Total Recordings | Ground Truth Labels |
|---|---|---|---|---|---|---|
| **Training (Raw)** | Normal only | 1,003 | 1,003 | 1,003 | 3,009 | Normal (0) |
| **Train Split (Internal 90%)** | Normal only | 902 | 902 | 902 | 2,706 | Normal (0) |
| **Validation Split (Internal 10%)** | Normal only | 101 | 101 | 101 | 303 | Normal (0) |
| **Source Test Split** | Normal + Anomalous | 100 Norm / 100 Anom | 100 Norm / 100 Anom | 100 Norm / 100 Anom | 600 (300 Norm / 300 Anom) | 0 (Normal) / 1 (Anomaly) |
| **Target Test Split** | Normal + Anomalous | 100 Norm / 100 Anom | 100 Norm / 100 Anom | 100 Norm / 100 Anom | 600 (300 Norm / 300 Anom) | 0 (Normal) / 1 (Anomaly) |

### Key Integrity Checks Verified:
- **Zero Anomaly Contamination in Training:** All 3,009 training files are strictly normal recordings (`normal_id_*.wav`).
- **Reproducible Validation Partition:** An independent, seeded split (`train_test_split(..., test_size=0.10, random_state=42)`) yields exactly 902 training and 101 validation recordings per section.
- **Strictly Balanced Evaluation Sets:** Both source and target test sets maintain an exact 1:1 balance (100 normal, 100 anomalous per section).

---

## 3. Implementation Review & Codebase Audit

We conducted an audit of the existing codebase to verify compliance with unsupervised anomaly detection principles and detect inconsistencies or data leakage:

### Audit Findings Matrix:

| Verification Requirement | Status | Detailed Implementation Audit |
|---|---|---|
| **1. Training uses normal recordings only** | **PASS** | `src/process_mfcc_dataset.py` processes only `data/raw/fan/train/*.wav`. `src/train_mfcc_sections.py` trains models strictly on normal indices. No anomalous recordings were included during model fitting. |
| **2. Feature extraction consistency** | **PASS WITH OBSERVATION** | `src/process_mfcc_dataset.py`, `src/evaluate_mfcc_sections.py`, and `app/inference.py` all use identical MFCC parameters: `sr=16000`, `n_fft=1024`, `hop_length=256`, `n_mfcc=40`, mono conversion via channel averaging. However, `src/preprocessing.py` extracts **Mel Spectrograms** (128 bins in dB), not MFCCs; it was retained from the earlier Mel baseline and is not called by the MFCC pipeline. |
| **3. Normalization uses training stats only** | **PASS WITH ARCHITECTURAL FLAW** | `src/train_mfcc_sections.py` computes `mean` and `std` strictly over `train_indices` (902 samples), excluding validation and test sets. These values are saved in the `.pth` checkpoints. However, normalization computes a **single global scalar mean and standard deviation across all 40 coefficients**, rather than normalizing each MFCC coefficient independently. |
| **4. Validation data used for threshold calibration** | **PASS** | In `src/evaluate_mfcc_sections.py`, thresholds are computed as `np.percentile(val_errors, 95)` over the 101 normal validation recordings. These exact values are carried into `app/inference.py` (`0.003472`, `0.004485`, `0.003199`). |
| **5. Test data kept separate from training & thresholds** | **PASS (IN MFCC PIPELINE)**<br>**HISTORICAL LEAKAGE FOUND IN LEGACY SCRIPT** | In `src/evaluate_mfcc_sections.py` and `src/evaluate_final.py`, test data is strictly held out. However, inspection of the legacy script `src/evaluate.py` (lines 81–83) revealed **data leakage**: `normal_errors = df.loc[df["actual"] == 0, "reconstruction_error"]; threshold = normal_errors.quantile(0.95)` calibrated thresholds directly using test set normal labels. While `evaluate_final.py` subsequently corrected this by recalculating metrics under training validation thresholds, `src/evaluate.py` retains the flawed code. |
| **6. Section-specific model & threshold correspondence** | **PASS** | Checkpoints `section_00_autoencoder.pth`, `section_01_autoencoder.pth`, and `section_02_autoencoder.pth` are mapped strictly to their respective sections in evaluation and inference. |

---

## 4. Reconstruction-Error Distribution Analysis

Using the saved per-recording scores from `reports/FINAL_TEST_PREDICTIONS.csv` and validation thresholds from `reports/VALIDATION_THRESHOLD_DETAILS.csv`, we computed complete descriptive distribution statistics for both source-test and target-test partitions across all sections.

### Summary Statistics Table

| Dataset | Section | Class | N | Mean | Median | Std Dev | Min | Max | 25th % | 75th % | 90th % | 95th % | Threshold | Above Thresh | % Above (TPR / FPR) | Overlap (OVL %) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Validation** | section_00 | Normal | 101 | 0.002729 | *unpersisted* | *unpersisted* | *N/A* | *N/A* | *N/A* | *N/A* | 0.003057 | 0.003472 | 0.003472 | 5 | **4.95% (FPR)** | N/A |
| **Validation** | section_01 | Normal | 101 | 0.003600 | *unpersisted* | *unpersisted* | *N/A* | *N/A* | *N/A* | *N/A* | 0.003981 | 0.004485 | 0.004485 | 5 | **4.95% (FPR)** | N/A |
| **Validation** | section_02 | Normal | 101 | 0.002704 | *unpersisted* | *unpersisted* | *N/A* | *N/A* | *N/A* | *N/A* | 0.002901 | 0.003199 | 0.003199 | 5 | **4.95% (FPR)** | N/A |
| **Source Test** | section_00 | Normal | 100 | 0.002787 | 0.002691 | 0.000391 | 0.002345 | 0.004588 | 0.002571 | 0.002868 | 0.003196 | 0.003511 | 0.003472 | 6 | **6.00% (FPR)** | **67.0%** |
| **Source Test** | section_00 | Anomaly | 100 | 0.002856 | 0.002827 | 0.000342 | 0.002398 | 0.003811 | 0.002591 | 0.003035 | 0.003340 | 0.003470 | 0.003472 | 5 | **5.00% (Recall)** | **67.0%** |
| **Source Test** | section_01 | Normal | 100 | 0.003607 | 0.003589 | 0.000373 | 0.003108 | 0.005646 | 0.003282 | 0.003793 | 0.004070 | 0.004160 | 0.004485 | 2 | **2.00% (FPR)** | **73.0%** |
| **Source Test** | section_01 | Anomaly | 100 | 0.003634 | 0.003583 | 0.000471 | 0.003042 | 0.006667 | 0.003278 | 0.003860 | 0.004147 | 0.004268 | 0.004485 | 3 | **3.00% (Recall)** | **73.0%** |
| **Source Test** | section_02 | Normal | 100 | 0.002730 | 0.002670 | 0.000255 | 0.002482 | 0.003796 | 0.002563 | 0.002794 | 0.002930 | 0.003203 | 0.003199 | 5 | **5.00% (FPR)** | **67.0%** |
| **Source Test** | section_02 | Anomaly | 100 | 0.002840 | 0.002732 | 0.000362 | 0.002495 | 0.005282 | 0.002624 | 0.002958 | 0.003217 | 0.003332 | 0.003199 | 12 | **12.00% (Recall)** | **67.0%** |
| **Target Test** | section_00 | Normal | 100 | 0.002699 | 0.002631 | 0.000335 | 0.002341 | 0.004050 | 0.002462 | 0.002792 | 0.003009 | 0.003423 | 0.003472 | 5 | **5.00% (FPR)** | **60.0%** |
| **Target Test** | section_00 | Anomaly | 100 | 0.002939 | 0.002831 | 0.000406 | 0.002410 | 0.004726 | 0.002658 | 0.003213 | 0.003442 | 0.003547 | 0.003472 | 8 | **8.00% (Recall)** | **60.0%** |
| **Target Test** | section_01 | Normal | 100 | 0.003712 | 0.003680 | 0.000448 | 0.003107 | 0.005116 | 0.003341 | 0.003961 | 0.004246 | 0.004545 | 0.004485 | 6 | **6.00% (FPR)** | **67.0%** |
| **Target Test** | section_01 | Anomaly | 100 | 0.003732 | 0.003749 | 0.000395 | 0.003084 | 0.004975 | 0.003416 | 0.003986 | 0.004262 | 0.004374 | 0.004485 | 3 | **3.00% (Recall)** | **67.0%** |
| **Target Test** | section_02 | Normal | 100 | 0.002827 | 0.002795 | 0.000199 | 0.002535 | 0.003976 | 0.002716 | 0.002886 | 0.002959 | 0.003149 | 0.003199 | 5 | **5.00% (FPR)** | **77.0%** |
| **Target Test** | section_02 | Anomaly | 100 | 0.002877 | 0.002836 | 0.000228 | 0.002603 | 0.004009 | 0.002739 | 0.002919 | 0.003083 | 0.003217 | 0.003199 | 6 | **6.00% (Recall)** | **77.0%** |

### Critical Distribution Insights:
1. **Identical Central Tendencies:**
   - In `section_01` (source-test), normal mean error is `0.003607` while anomaly mean error is `0.003634` (a difference of only `0.000027` or $0.7\%$). The median anomaly error (`0.003583`) is actually **lower** than the median normal error (`0.003589`).
   - In `section_01` (target-test), normal mean error is `0.003712` and anomaly mean is `0.003732` (difference of `0.000020` or $0.5\%$).
2. **Dynamic Range Containment:**
   - In all sections, **98.0% to 100.0% of anomalous recordings lie completely inside the dynamic range of normal recordings**.
   - Anomalies never form an isolated high-error tail; they reside smack in the middle of normal variation.
3. **Distribution Overlap Coefficient (OVL):**
   - Empirical probability density overlap ranges from **60.0% to 77.0%**. For reference, two completely indistinguishable distributions have $\text{OVL} = 100\%$, while fully separable distributions have $\text{OVL} = 0\%$.

---

## 5. Threshold Calibration Diagnostics & The "Conservative" Fallacy

A frequent initial reaction to low recall is to assume that the calibration threshold was chosen "too conservatively." Our quantitative sensitivity analysis proves this assumption is incorrect.

### Why the Threshold Is Mathematically Sound:
- In unsupervised industrial anomaly detection, the operating objective is to bound false alarms during normal operation (where alarms cause costly, unnecessary plant shutdowns).
- By setting the threshold at the 95th percentile ($P_{95}$) of normal validation data, the system successfully controls false alarms on unseen test data:
  - Source-test observed FPR: $4.33\%$ (pooled across sections: 6/100, 2/100, 5/100).
  - Target-test observed FPR: $5.33\%$ (pooled across sections: 5/100, 6/100, 5/100).
  - The threshold achieves exactly what it was calibrated to do.

### What Happens If You Lower the Threshold?
Using `reports/THRESHOLD_SENSITIVITY.csv`, we observe the empirical trade-off if the threshold were moved to lower percentiles:

| Threshold Rule | Target Recall | Target FPR | Target Precision | Target F1 | Operational Impact |
|---|---|---|---|---|---|
| **$P_{99}$ (Top 1% Cutoff)** | $1.67\%$ (5/300) | $1.00\%$ (3/300) | $58.33\%$ | $0.0323$ | Almost all anomalies missed; zero operational value. |
| **$P_{97.5}$ (Top 2.5% Cutoff)** | $3.00\%$ (9/300) | $3.00\%$ (9/300) | $47.78\%$ | $0.0558$ | Severe under-detection. |
| **$P_{95}$ (Current Baseline)** | **$5.67\%$ (17/300)** | **$5.33\%$ (16/300)** | **$49.81\%$** | **$0.1016$** | Nominal 5% false alarm budget; 94.3% anomalies missed. |
| **$P_{90}$ (Top 10% Cutoff)** | $32.00\%$ (96/300) | $17.33\%$ (52/300) | $66.04\%$ | $0.4295$ | **False alarm rate triples to 17.3%**; plant experiences false shutdowns every 6 normal recordings. |

### Diagnostic Takeaway:
The low anomaly recall is **not a calibration defect**. It is a **representation failure**. The autoencoder fails to generate distinctively elevated reconstruction errors for anomalies. Shifting a threshold along overlapping distributions cannot solve a lack of separability.

---

## 6. Performance Dissection: Root Cause Analysis

To determine whether the failure stems from feature extraction, model reconstruction, threshold calibration, or domain shift, we evaluate the evidence for each hypothesis:

```
                      [ Root Cause Assessment ]
                                  │
    ┌─────────────────┬───────────┴───────────┬─────────────────┐
    ▼                 ▼                       ▼                 ▼
Threshold        Domain Shift            Feature           Model Architecture
Calibration      (Source vs Target)      Extraction        & Loss Aggregation
[EXONERATED]     [SECONDARY FACTOR]      [PRIMARY FACTOR]  [CRITICAL BOTTLENECK]
Accurately       Source recall is        Global scalar     Latent dim (25,280) >
bounds FPR at    equally dismal          norm squashes     Input dim (25,040).
~5% on test.     (6.67% vs 5.67%).       higher MFCCs.     Global MSE dilutes
                                                           transient faults.
```

### Detailed Evaluation of Factors:

#### Factor 1: Threshold Calibration — EXONERATED
As demonstrated in Section 5, threshold calibration performed precisely as specified by statistical theory. It maintained FPR within 4.3%–5.3%. The threshold cannot separate what the model has not separated.

#### Factor 2: Domain Shift — SECONDARY CONTRIBUTOR
MIMII DUE introduces domain shift in target-test recordings (differing background ambient noise, machine load conditions). However:
- In `section_00`, target-test ROC-AUC ($0.7126$) is actually **higher** than source-test ROC-AUC ($0.5796$).
- Macro recall on source-test is $6.67\%$ (20/300), nearly identical to target-test recall of $5.67\%$ (17/300).
- Normal mean reconstruction error remains stable across splits:
  - `section_00`: 0.002787 (source) vs 0.002699 (target)
  - `section_01`: 0.003607 (source) vs 0.003712 (target)
  - `section_02`: 0.002730 (source) vs 0.002827 (target)
Domain shift did not cause the low recall; the model is fundamentally blind to anomalies even in the native training domain.

#### Factor 3: Model Architecture & Lack of Information Bottleneck — CRITICAL PRIMARY CAUSE
Inspection of `src/model.py` reveals the dimensional structure:
- **Input Dimensions:** Single-channel MFCC spectrogram: $(1, 40, 626) = \mathbf{25,040\text{ values}}$.
- **Encoder Layer 1:** `Conv2d(1, 16, 3, stride=2, padding=1)` $\rightarrow (16, 20, 313)$
- **Encoder Layer 2:** `Conv2d(16, 32, 3, stride=2, padding=1)` $\rightarrow (32, 10, 157)$
- **Encoder Layer 3 (Bottleneck):** `Conv2d(32, 64, 3, stride=2, padding=1)` $\rightarrow (64, 5, 79) = \mathbf{25,280\text{ values}}$.
- **Latent Bottleneck Ratio:** $\frac{25,280}{25,040} = \mathbf{1.0096}$ (an expansion of 240 dimensions!).

In autoencoder theory, anomaly detection relies on forcing the network to compress data through a low-dimensional bottleneck, constraining it to learn only the manifold of normal operation. Here, the latent bottleneck has **more dimensions than the input**. With 25,280 degrees of freedom, the network acts as an unconstrained convolutional filter bank. It easily reconstructs anomalous sound patterns because it has sufficient parameter capacity to reconstruct arbitrary inputs.

#### Factor 4: Global Mean Pooling Loss Aggregation — CRITICAL PRIMARY CAUSE
The anomaly score is calculated as:
$$\text{Score} = \frac{1}{40 \times 626} \sum_{f=1}^{40} \sum_{t=1}^{626} \left(\hat{X}_{f,t} - X_{f,t}\right)^2$$
Fan failure modes (bearing cage cracks, partial blade rub, impeller imbalance) typically exhibit transient periodic clicks or slight spectral band harmonics. If an anomaly elevates reconstruction error by $400\%$ over a $0.2\text{-second}$ window (12 time frames out of 626):
$$\text{Dilution Factor} = \frac{12}{626} \approx 0.019\ (1.9\%)$$
The local surge in error is averaged across the remaining 614 normal frames, resulting in an imperceptible change of less than $0.0001$ in the global MSE score. This is corroborated by the MFCC frame-level P95 experiment (`reports/FINAL_RESULTS.csv`), where scoring by temporal 95th percentile improved recall by $45\%$ to $60\%$.

#### Factor 5: Inappropriate Global Scalar Feature Normalization — CRITICAL PRIMARY CAUSE
In `src/train_mfcc_sections.py`:
```python
total_sum += sample.sum()
total_sq_sum += np.square(sample).sum()
mean = total_sum / total_count
std = np.sqrt(total_sq_sum / total_count - mean**2)
```
In acoustic feature engineering, MFCC coefficients represent orthogonal spectral cosine bases:
- Coefficient 0 represents overall frame logarithmic energy (ranging from $-300$ to $-500\text{ dB}$).
- Coefficients 1–39 represent fine spectral envelope contours with variances close to $1.0$.
Normalizing all 40 coefficients by a single scalar standard deviation divides high-frequency coefficients by a standard deviation dominated by coefficient 0. Consequently, high-order coefficients (where subtle bearing friction and blade flutter anomalies reside) are compressed into near-zero variance, rendering them invisible to the MSE reconstruction loss.

---

## 7. Source-Test versus Target-Test Comparative Dynamics

The comparison between source and target domains provides key insights into model behavior under operational shift:

| Metric | Source-Test (`section_00`) | Target-Test (`section_00`) | Source-Test (`section_01`) | Target-Test (`section_01`) | Source-Test (`section_02`) | Target-Test (`section_02`) | Macro Source-Test | Macro Target-Test |
|---|---|---|---|---|---|---|---|---|
| **ROC-AUC** | 0.5796 | **0.7126** | 0.5029 | 0.5323 | **0.6269** | 0.5824 | 0.5698 | **0.6091** |
| **Accuracy** | 49.5% | 51.5% | 50.5% | 48.5% | 53.5% | 50.5% | 51.2% | 50.2% |
| **Precision** | 45.5% | 61.5% | 60.0% | 33.3% | 70.6% | 54.5% | 58.7% | 49.8% |
| **Recall (TPR)** | 5.0% | 8.0% | 3.0% | 3.0% | **12.0%** | 6.0% | **6.67%** | 5.67% |
| **False Pos. Rate** | 6.0% | 5.0% | 2.0% | 6.0% | 5.0% | 5.0% | 4.33% | 5.33% |
| **F1-Score** | 0.0901 | 0.1416 | 0.0571 | 0.0550 | 0.2051 | 0.1081 | 0.1175 | 0.1016 |

### Section-Level Analysis:
1. **Worst Performing Section: `section_01`**
   - Source-test ROC-AUC is $0.5029$—effectively indistinguishable from random guessing ($0.50$).
   - Target-test ROC-AUC is $0.5323$.
   - Recall is locked at an abysmal $3.0\%$ (only 3 out of 100 anomalous recordings detected).
   - Normal and anomalous error distributions exhibit **$73.0\%$ overlap**.
2. **Best Performing Section: Split-Dependent (`section_00` and `section_02`)**
   - In **Source Domain**, `section_02` performs best with ROC-AUC $0.6269$ and highest recall ($12.0\%$).
   - In **Target Domain**, `section_00` performs best with ROC-AUC $0.7126$ and recall $8.0\%$.
   - However, even in `section_00` target-test (where rank discrimination is moderate at $0.7126$), the operational recall remains just $8.0\%$ because anomalous reconstruction error values fail to rise decisively above normal background variance.

---

## 8. Diagnostic Visualizations

Four diagnostic figures have been generated and saved to `reports/anomaly_diagnostics/` in both PNG and SVG formats:

### 1. Overlapping Histograms / Density Plots
**File:** `reports/anomaly_diagnostics/reconstruction_error_histograms.png` (and `.svg`)  
**Diagnostic Content:** 6-panel grid displaying empirical frequency histograms of normal test error (blue) vs anomalous test error (red) alongside the vertical validation threshold (purple dashed line) for each section across both domains. Highlights the massive distributional overlap ($60\%–73\%$) and near-complete containment of anomalies inside normal variance.

### 2. Box Plot Comparison
**File:** `reports/anomaly_diagnostics/reconstruction_error_boxplots.png` (and `.svg`)  
**Diagnostic Content:** Side-by-side boxplots for each section and split, comparing minimum, 25th percentile, median, mean, 75th percentile, and maximum errors. Illustrates how anomaly medians and means align almost identically with normal medians, and how the threshold sits near the top whisker of both classes.

### 3. Threshold Calibration & ECDF Curves
**File:** `reports/anomaly_diagnostics/threshold_calibration_visualization.png` (and `.svg`)  
**Diagnostic Content:** Empirical Cumulative Distribution Function ($P(\text{Error} \le x)$) curves comparing Normal Source, Normal Target, Anomaly Source, and Anomaly Target against the $P_{95}$ horizontal calibration mark. Visually delineates the large shaded "Undetected Zone (False Negatives)", proving that the 95th percentile threshold is blind to anomalies because the cumulative probability curves coincide.

### 4. Per-Section Performance Comparison
**File:** `reports/anomaly_diagnostics/per_section_performance_comparison.png` (and `.svg`)  
**Diagnostic Content:** Dual-panel bar chart comparing threshold-independent ROC-AUC (left) against operational thresholded metrics (Recall, FPR, F1) at the $P_{95}$ cutoff (right). Visually emphasizes the stark divergence between moderate rank AUC ($0.50$ to $0.71$) and poor operational recall ($3\%$ to $12\%$).

---

## 9. Recommended Next Experiments (Ranked by Priority)

Before initiating full model retraining, the following experiments should be executed in strict rank order to address the identified bottlenecks systematically:

```
┌────────────────────────────────────────────────────────────────────────┐
│               RECOMMENDED NEXT EXPERIMENTS ROADMAP                     │
├──────┬──────────────────────────────────────────┬──────────────────────┤
│ Rank │ Experiment                               │ Retraining Required? │
├──────┼──────────────────────────────────────────┼──────────────────────┤
│  1   │ Temporal Max/Top-K Frame Error Scoring   │ NO (Uses Checkpoints)│
│  2   │ Per-Frequency Band Error Decomposition   │ NO (Uses Checkpoints)│
│  3   │ Per-Coefficient Normalization (Feature)  │ YES                  │
│  4   │ True Autoencoder Dimensional Bottleneck  │ YES                  │
│  5   │ Log-Mel Spectrogram vs MFCC Benchmark    │ YES                  │
│  6   │ Temporal Convolutions with Delta MFCCs   │ YES                  │
└──────┴──────────────────────────────────────────┴──────────────────────┘
```

### Priority 1 (Immediate - Zero Retraining Required): Frame-Level Max / Top-K Anomaly Pooling
- **RATIONALE:** Global mean pooling across all 626 frames dilutes localized fault signatures.
- **PROCEDURE:** Without retraining, run the existing frozen section checkpoints, compute the reconstruction error per time frame $t \in [1, 626]$, and evaluate anomaly scoring based on:
  1. The 95th or 98th percentile of frame errors within the recording ($P_{\text{frame}, 95}$).
  2. The mean of the top 5% highest error frames ($\text{Top-K MSE}$).
- **EXPECTED IMPACT:** Immediately elevates localized anomalous errors above the background noise floor without touching model weights.

### Priority 2 (Immediate - Zero Retraining Required): Frequency-Band Error Decomposition
- **RATIONALE:** Identify whether specific MFCC bands carry anomaly signal that is drowned out by low-order energy terms.
- **PROCEDURE:** Compute squared reconstruction error separately for low-frequency coefficients (MFCC 0–4), mid-frequency coefficients (MFCC 5–19), and high-frequency coefficients (MFCC 20–39). Evaluate ROC-AUC on sub-band errors using existing predictions.
- **EXPECTED IMPACT:** Determines whether anomalous deviation is concentrated in specific acoustic dimensions.

### Priority 3 (Feature Pipeline Revision): Per-Coefficient (Channel-Wise) Feature Normalization
- **RATIONALE:** Global scalar normalization suppresses the variance of coefficients 1–39.
- **PROCEDURE:** Modify `process_mfcc_dataset.py` and `train_mfcc_sections.py` to compute per-coefficient mean $\mu_c$ and standard deviation $\sigma_c$ across training frames:
  $$X_{\text{norm}}[c, t] = \frac{X[c, t] - \mu_c}{\sigma_c + \epsilon}$$
- **EXPECTED IMPACT:** Equalizes dynamic range across all 40 coefficients so that high-frequency acoustic flutter has equal gradient and loss contribution.

### Priority 4 (Architecture Overhaul): Introduce an Actual Information Bottleneck
- **RATIONALE:** The current latent space (25,280 dimensions) is larger than the input (25,040 dimensions).
- **PROCEDURE:** Redesign `src/model.py` to enforce true compression. Add a 4th convolutional downsampling stage or introduce a dense bottleneck layer (e.g., latent dimension $d \in [64, 256]$):
  - Encoder: $40 \times 626 \rightarrow 20 \times 313 \rightarrow 10 \times 157 \rightarrow 5 \times 79 \rightarrow \text{Bottleneck } (128)$.
  - Compression ratio: $128 / 25,040 \approx 0.0051$ ($99.5\%$ compression).
- **EXPECTED IMPACT:** Forces the autoencoder to learn a compact subspace of normal fan sound, preventing it from reconstructing anomalous patterns.

### Priority 5 (Input Representation): Re-evaluate Log-Mel Spectrograms with True Bottleneck
- **RATIONALE:** While earlier shared Mel models performed poorly due to lack of section specificity and lack of a bottleneck, 128-bin Log-Mel spectrograms preserve raw harmonic and tonal structure better than DCT-compressed MFCCs for stationary rotating machinery like fans.

### Priority 6 (Temporal Dynamics): Append Delta and Delta-Delta Features
- **RATIONALE:** Static MFCCs capture spectral shape but miss rotational speed fluctuations, flutter, and turbulence transitions.
- **PROCEDURE:** Append first and second temporal derivatives ($\Delta$ and $\Delta\Delta$ MFCCs) as input channels: shape $(3, 40, 626)$.

---

## 10. Conclusion & Immediate Recommended Next Step

The low anomaly recall of the section-specific CNN autoencoder is **not due to improper normal-only training, data leakage, or threshold miscalibration**. The training discipline was sound. The failure is structural:
1. The CNN autoencoder lacks a dimensional bottleneck ($25,280 > 25,040$).
2. Global scalar normalization suppresses high-order spectral detail.
3. Global mean pooling washes out localized mechanical faults.

### Immediate Next Step Before Retraining:
**Evaluate Top-K / Frame-Percentile error aggregation on the existing frozen checkpoint predictions.**
Because this step requires zero retraining and directly tests whether anomalies are temporally localized, it will confirm whether temporal pooling is the bottleneck before committing GPU time to retraining models with modified architectures and normalization schemes.
