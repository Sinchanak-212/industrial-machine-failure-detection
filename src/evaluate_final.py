from datetime import datetime
from pathlib import Path

import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from model import CNN_Autoencoder
from preprocessing import (
    HOP_LENGTH,
    N_FFT,
    SAMPLE_RATE,
    extract_mel_spectrogram,
)


ROOT = Path.cwd()
TRAIN_DIR = ROOT / "data" / "raw" / "fan" / "train"
SOURCE_DIR = ROOT / "data" / "raw" / "fan" / "source_test"
TARGET_DIR = ROOT / "data" / "raw" / "fan" / "target_test"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
SECTIONS = ["section_00", "section_01", "section_02"]
EXPERIMENTS = [
    "Shared Mel CNN",
    "Section-specific Mel CNN",
    "Section-specific MFCC CNN",
    "MFCC frame-level P95",
    "MFCC + Isolation Forest",
]
SEED = 42
VALIDATION_PERCENTILE = 95
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def require_files(paths):
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required existing evaluation artifacts are missing:\n"
            + "\n".join(missing)
        )


def check_split_files(directory):
    files = sorted(directory.glob("*.wav"))
    if len(files) != 600:
        raise ValueError(f"{directory} must contain 600 WAV files; found {len(files)}")

    for section in SECTIONS:
        section_files = [path for path in files if section in path.name]
        normal_count = sum("normal" in path.name for path in section_files)
        anomaly_count = sum("anomaly" in path.name for path in section_files)
        if (len(section_files), normal_count, anomaly_count) != (200, 100, 100):
            raise ValueError(
                f"Unexpected {directory.name} counts for {section}: "
                f"{len(section_files)} total, {normal_count} normal, "
                f"{anomaly_count} anomalous"
            )
    return files


def section_indices(files, section):
    return [i for i, path in enumerate(files) if section in path.name]


def validation_indices(files, section):
    indices = section_indices(files, section)
    _, val_indices = train_test_split(
        indices, test_size=0.10, random_state=SEED
    )
    return val_indices


def load_autoencoder(path):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    if not {"model_state_dict", "mean", "std"}.issubset(checkpoint):
        raise ValueError(f"Checkpoint has no model or normalization values: {path}")
    model = CNN_Autoencoder().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, float(checkpoint["mean"]), float(checkpoint["std"])


def batch_cnn_scores(model, feature_batch, mean, std):
    batch = np.asarray(feature_batch, dtype=np.float32)
    batch = (batch - mean) / (std + 1e-8)
    inputs = torch.from_numpy(batch).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        reconstruction = model(inputs)
        squared_error = (reconstruction - inputs) ** 2
        recording_mean = squared_error.mean(dim=(1, 2, 3)).cpu().numpy()
        frame_errors = squared_error.mean(dim=(1, 2)).cpu().numpy()
    frame_mean = frame_errors.mean(axis=1)
    frame_p95 = np.percentile(frame_errors, 95, axis=1)
    return recording_mean, frame_mean, frame_p95


def single_cnn_scores(model, feature, mean, std):
    feature = np.asarray(feature, dtype=np.float32)
    normalized = (feature - mean) / (std + 1e-8)
    inputs = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        reconstruction = model(inputs)
        squared_error = (reconstruction - inputs) ** 2
        recording_mean = float(squared_error.mean().item())
        frame_errors = squared_error.mean(dim=(0, 1, 2)).cpu().numpy()
    return recording_mean, float(np.percentile(frame_errors, 95))


def extract_mfcc(audio_path):
    audio, sample_rate = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(
            audio.astype(np.float32),
            orig_sr=sample_rate,
            target_sr=SAMPLE_RATE,
        )
    return librosa.feature.mfcc(
        y=audio.astype(np.float32),
        sr=SAMPLE_RATE,
        n_mfcc=40,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )


def isolation_features(mfcc):
    return np.concatenate(
        [
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.percentile(mfcc, 10, axis=1),
            np.percentile(mfcc, 90, axis=1),
        ]
    ).astype(np.float32)


def score_validation_cnn(model, features, files, section, mean, std):
    scores = []
    val_indices = validation_indices(files, section)
    for start in range(0, len(val_indices), 16):
        batch_indices = val_indices[start : start + 16]
        values, _, _ = batch_cnn_scores(
            model, features[batch_indices], mean, std
        )
        scores.extend(values.tolist())
    return np.asarray(scores)


def score_validation_frames(model, features, files, section, mean, std):
    mean_scores = []
    p95_scores = []
    val_indices = validation_indices(files, section)
    for start in range(0, len(val_indices), 16):
        batch_indices = val_indices[start : start + 16]
        _, means, p95s = batch_cnn_scores(
            model, features[batch_indices], mean, std
        )
        mean_scores.extend(means.tolist())
        p95_scores.extend(p95s.tolist())
    return np.asarray(mean_scores), np.asarray(p95_scores)


def load_source_scores():
    files = [
        REPORT_DIR / "source_test_results.csv",
        REPORT_DIR / "section_test_predictions.csv",
        REPORT_DIR / "mfcc_section_test_predictions.csv",
        REPORT_DIR / "mfcc_frame_test_predictions.csv",
        REPORT_DIR / "isolation_forest_predictions.csv",
    ]
    require_files(files)
    source_files = check_split_files(SOURCE_DIR)

    def normalize_rows(path, experiment, score_column):
        frame = pd.read_csv(path)
        if len(frame) != 600:
            raise ValueError(f"Expected 600 source-test rows in {path}")
        frame = frame.rename(
            columns={
                "section": "section",
                "Section": "section",
                "filename": "filename",
                "Filename": "filename",
                "actual": "actual",
                "Actual": "actual",
                score_column: "score",
                "Anomaly_Score": "score",
            }
        )
        if not {"section", "filename", "actual", "score"}.issubset(frame.columns):
            raise ValueError(f"Unexpected score columns in {path}")
        frame["experiment"] = experiment
        return frame[["experiment", "section", "filename", "actual", "score"]]

    rows = []
    shared = pd.read_csv(files[0])
    if len(shared) != 600:
        raise ValueError("Expected 600 shared-Mel source-test rows")
    shared["section"] = shared["filename"].map(
        lambda name: next((s for s in SECTIONS if s in name), None)
    )
    shared["experiment"] = EXPERIMENTS[0]
    shared = shared.rename(
        columns={"reconstruction_error": "score"}
    )
    rows.append(shared[["experiment", "section", "filename", "actual", "score"]])
    rows.append(
        normalize_rows(files[1], EXPERIMENTS[1], "reconstruction_error")
    )
    rows.append(
        normalize_rows(files[2], EXPERIMENTS[2], "reconstruction_error")
    )

    frame_scores = pd.read_csv(files[3])
    if len(frame_scores) != 600:
        raise ValueError("Expected 600 MFCC frame-level source-test rows")
    frame_scores = frame_scores.rename(
        columns={
            "section": "section",
            "filename": "filename",
            "actual": "actual",
            "p95_score": "score",
        }
    )
    frame_scores["experiment"] = EXPERIMENTS[3]
    rows.append(
        frame_scores[["experiment", "section", "filename", "actual", "score"]]
    )
    rows.append(normalize_rows(files[4], EXPERIMENTS[4], "Anomaly_Score"))

    source = pd.concat(rows, ignore_index=True)
    expected_names = {path.name for path in source_files}
    for experiment in EXPERIMENTS:
        subset = source[source["experiment"] == experiment]
        if len(subset) != 600 or set(subset["filename"]) != expected_names:
            raise ValueError(
                f"Existing source-test scores do not match {experiment} files"
            )
    source["dataset"] = "source_test"
    source["actual"] = source["actual"].astype(int)
    source["score"] = source["score"].astype(float)
    return source


def metric_row(dataset, experiment, section, subset, threshold):
    y_true = subset["actual"].to_numpy(dtype=int)
    scores = subset["score"].to_numpy(dtype=float)
    y_pred = (scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def distribution(label):
        values = scores[y_true == label]
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
        }

    normal = distribution(0)
    anomaly = distribution(1)
    return {
        "dataset": dataset,
        "experiment": experiment,
        "section": section,
        "n": int(len(subset)),
        "threshold": float(threshold),
        "threshold_direction": "anomaly if score > threshold",
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "normal_score_mean": normal["mean"],
        "normal_score_std": normal["std"],
        "normal_score_median": normal["median"],
        "normal_score_p05": normal["p05"],
        "normal_score_p95": normal["p95"],
        "anomaly_score_mean": anomaly["mean"],
        "anomaly_score_std": anomaly["std"],
        "anomaly_score_median": anomaly["median"],
        "anomaly_score_p05": anomaly["p05"],
        "anomaly_score_p95": anomaly["p95"],
    }


def summarize(results):
    metrics = []
    for dataset in ["source_test", "target_test"]:
        for experiment in EXPERIMENTS:
            experiment_rows = results[
                (results["dataset"] == dataset)
                & (results["experiment"] == experiment)
            ]
            section_metrics = []
            for section in SECTIONS:
                subset = experiment_rows[experiment_rows["section"] == section]
                if len(subset) != 200:
                    raise ValueError(
                        f"Expected 200 {dataset} rows for {experiment}/{section}"
                    )
                threshold = float(subset["threshold"].iloc[0])
                row = metric_row(dataset, experiment, section, subset, threshold)
                section_metrics.append(row)
                metrics.append(row)

            macro = {
                "dataset": dataset,
                "experiment": experiment,
                "section": "macro",
                "n": sum(row["n"] for row in section_metrics),
                "threshold": np.nan,
                "threshold_direction": "per-section thresholds; equal-section metric mean",
            }
            for column in ["roc_auc", "accuracy", "precision", "recall", "f1"]:
                macro[column] = float(
                    np.mean([row[column] for row in section_metrics])
                )
            for column in ["tn", "fp", "fn", "tp"]:
                macro[column] = int(sum(row[column] for row in section_metrics))
            pooled = metric_row(
                dataset,
                experiment,
                "macro",
                experiment_rows,
                float("nan"),
            )
            for column in [
                "normal_score_mean",
                "normal_score_std",
                "normal_score_median",
                "normal_score_p05",
                "normal_score_p95",
                "anomaly_score_mean",
                "anomaly_score_std",
                "anomaly_score_median",
                "anomaly_score_p05",
                "anomaly_score_p95",
            ]:
                macro[column] = pooled[column]
            metrics.append(macro)
    return pd.DataFrame(metrics)


def markdown_table(frame, columns, headers=None):
    headers = headers or columns
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                values.append("N/A")
            elif isinstance(value, (float, np.floating)):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def generate_plots(results, metrics, selected_experiment):
    macro = metrics[metrics["section"] == "macro"]
    source_macro = macro[macro["dataset"] == "source_test"].set_index("experiment")
    target_macro = macro[macro["dataset"] == "target_test"].set_index("experiment")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(EXPERIMENTS))
    width = 0.36
    ax.bar(
        x - width / 2,
        [source_macro.loc[name, "roc_auc"] for name in EXPERIMENTS],
        width,
        label="Source test",
        color="#4C78A8",
    )
    ax.bar(
        x + width / 2,
        [target_macro.loc[name, "roc_auc"] for name in EXPERIMENTS],
        width,
        label="Target test",
        color="#F58518",
    )
    ax.set_xticks(x, EXPERIMENTS, rotation=18, ha="right")
    ax.set_ylabel("Macro-average ROC-AUC")
    ax.set_ylim(0, 1)
    ax.set_title("Source-test vs untouched target-test ROC-AUC")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "source_target_roc_auc.png", dpi=180)
    plt.close(fig)

    source_auc = source_macro["roc_auc"].sort_values(ascending=False)
    strongest = list(source_auc.head(3).index)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    for ax, dataset in zip(axes, ["source_test", "target_test"]):
        for experiment in strongest:
            subset = results[
                (results["dataset"] == dataset)
                & (results["experiment"] == experiment)
            ]
            fpr, tpr, _ = roc_curve(subset["actual"], subset["score"])
            auc = roc_auc_score(subset["actual"], subset["score"])
            ax.plot(fpr, tpr, label=f"{experiment} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_title(dataset.replace("_", " ").title())
        ax.set_xlabel("False-positive rate")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("True-positive rate")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.suptitle(
        "Pooled ROC curves; top models ranked by source-test macro ROC-AUC"
    )
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "strongest_model_roc_curves.png", dpi=180)
    plt.close(fig)

    selected = results[
        (results["dataset"] == "target_test")
        & (results["experiment"] == selected_experiment)
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, section in zip(axes[:3], SECTIONS):
        subset = selected[selected["section"] == section]
        cm = confusion_matrix(
            subset["actual"], subset["predicted"], labels=[0, 1]
        )
        _plot_confusion(ax, cm, section)
    cm = confusion_matrix(
        selected["actual"], selected["predicted"], labels=[0, 1]
    )
    _plot_confusion(axes[3], cm, "All target sections")
    fig.suptitle(f"Target-test confusion matrices: {selected_experiment}")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "final_selected_confusion_matrices.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
    for ax, section in zip(axes, SECTIONS):
        subset = selected[selected["section"] == section]
        values = [
            subset.loc[subset["actual"] == label, "score"].to_numpy()
            for label in [0, 1]
        ]
        boxes = ax.boxplot(
            values,
            tick_labels=["Normal", "Anomaly"],
            patch_artist=True,
            showfliers=False,
        )
        for box, color in zip(boxes["boxes"], ["#54A24B", "#E45756"]):
            box.set_facecolor(color)
            box.set_alpha(0.65)
        ax.set_title(section)
        ax.set_ylabel("Anomaly score")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Target-test score distributions: {selected_experiment}")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "anomaly_score_distributions.png", dpi=180)
    plt.close(fig)

    target_section = metrics[
        (metrics["dataset"] == "target_test")
        & (metrics["section"] != "macro")
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plot_labels = [
        "Shared Mel",
        "Section Mel",
        "Section MFCC",
        "Frame P95",
        "Isolation Forest",
    ]
    for ax, metric in zip(
        axes.flat, ["roc_auc", "accuracy", "precision", "recall"]
    ):
        for section in SECTIONS:
            section_rows = target_section[
                target_section["section"] == section
            ].set_index("experiment")
            ax.plot(
                plot_labels,
                [section_rows.loc[name, metric] for name in EXPERIMENTS],
                marker="o",
                label=section,
            )
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", labelrotation=20, labelsize=9)
        ax.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Per-section target-test performance at locked thresholds")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "per_section_performance.png", dpi=180)
    plt.close(fig)

    return strongest


def _plot_confusion(ax, matrix, title):
    ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_title(title)
    ax.set_xticks([0, 1], ["Normal", "Anomaly"], rotation=30, ha="right")
    ax.set_yticks([0, 1], ["Normal", "Anomaly"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="black",
            )


def write_documentation(metrics, selected_experiment, strongest, run_time):
    macro = metrics[metrics["section"] == "macro"]
    main_table = macro[
        macro["experiment"].isin(EXPERIMENTS)
    ].copy()
    main_table["period"] = main_table["dataset"].map(
        {"source_test": "Source test", "target_test": "Target test"}
    )
    per_section = metrics[
        (metrics["dataset"] == "target_test")
        & (metrics["section"] != "macro")
    ]
    source_winner = (
        macro[macro["dataset"] == "source_test"]
        .sort_values("roc_auc", ascending=False)
        .iloc[0]
    )
    target_best_auc = (
        macro[macro["dataset"] == "target_test"]
        .sort_values("roc_auc", ascending=False)
        .iloc[0]
    )
    selected_target = macro[
        (macro["dataset"] == "target_test")
        & (macro["experiment"] == selected_experiment)
    ].iloc[0]

    target_table = per_section[
        [
            "experiment",
            "section",
            "roc_auc",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ]
    ]
    macro_table = main_table[
        [
            "experiment",
            "period",
            "roc_auc",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ]
    ]
    analysis = f"""# Final Experiment Analysis

Evaluation run: {run_time}

## Executive summary

Model comparison and the locked-threshold selection are based on source-test performance, not target-test results. The source-test macro ROC-AUC leader was **{source_winner['experiment']}** (ROC-AUC {source_winner['roc_auc']:.4f}). It achieved target-test macro ROC-AUC {selected_target['roc_auc']:.4f}, accuracy {selected_target['accuracy']:.4f}, precision {selected_target['precision']:.4f}, recall {selected_target['recall']:.4f}, and F1 {selected_target['f1']:.4f}. The highest descriptive target-test macro ROC-AUC was {target_best_auc['roc_auc']:.4f} for {target_best_auc['experiment']}; this target-only ranking is reported for transparency and was **not** used to select a model or threshold.

The target split contains 600 recordings (200 per section: 100 normal and 100 anomalous). All target scores use frozen checkpoints and normalization; threshold values are fixed using only the original normal validation split or saved normal-validation thresholds.

## Dataset and attribution

Audio is the fan subset of **MIMII DUE**, Zenodo record [4740355](https://doi.org/10.5281/zenodo.4740355), licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The project directory contains 3,009 normal training recordings (1,003 per section), 600 source-test recordings, and 600 target-test recordings. Each test split has 100 normal and 100 anomalous recordings for each of sections 00, 01, and 02.

## Preprocessing and evaluation methodology

- Mono audio is produced by averaging channels. Audio is resampled to 16 kHz when necessary.
- Mel CNN inputs use 128-bin power Mel spectrograms (`n_fft=1024`, `hop_length=256`) converted to dB with `ref=1.0`.
- MFCC CNN and Isolation Forest inputs use 40 MFCC coefficients (`n_fft=1024`, `hop_length=256`). Each Isolation Forest input contains the per-coefficient mean, standard deviation, 10th percentile, and 90th percentile (160 features).
- Model-specific normalization was retained from training checkpoints; Isolation Forest uses its saved `StandardScaler`.
- Each model uses the saved architecture and weights. CNN autoencoders have convolutional encoder channels 1-16-32-64 with stride 2, followed by a transposed-convolution decoder 64-32-16-1. The shared Mel checkpoint has one global model; Mel and MFCC section-specific checkpoints have one model per section.
- CNNs were trained for up to 30 epochs with batch size 8, Adam (`learning_rate=0.001`), mean-squared reconstruction loss, and seed 42; checkpoints retain the best validation-loss epoch. Each Isolation Forest uses 300 trees, `contamination=\"auto\"`, and `random_state=42`.
- Training and validation use only normal recordings. The scripts use an independent, seeded 90/10 split within each section (`random_state=42`). Validation scores are computed from the exact validation indices in the original scripts. Reconstruction thresholds are the 95th percentile of normal validation recording-level reconstruction errors; MFCC frame-P95 uses the 95th percentile of each validation recording's frame-error P95. Isolation Forest uses its saved 5th-percentile `decision_function` threshold, equivalent to a 95th-percentile anomaly-score threshold after sign reversal.
- Saved Isolation Forest thresholds were used directly. Section-specific Mel and MFCC CNN validation thresholds were reproduced using the original evaluation procedures; the shared Mel and MFCC frame-P95 thresholds, which were not persisted as validation-threshold artifacts, were deterministically recomputed from the original normal validation split. The shared Mel checkpoint normalization was verified against `models/normalization.npz`.
- No model was trained, fine-tuned, normalized, thresholded, or tuned on either test split during this final evaluation. Target labels were used only to calculate evaluation metrics.
- Macro metrics are unweighted averages of the three section-level metrics. Confusion counts and score-distribution summaries in macro rows are pooled across all 600 recordings. Missing comparisons are represented as `N/A` in tables rather than as zero.

## Macro-average source/target comparison

Metrics below apply the same validation-derived threshold to each test set. ROC-AUC is threshold-independent.

{markdown_table(macro_table, ['experiment', 'period', 'roc_auc', 'accuracy', 'precision', 'recall', 'f1'], ['Experiment', 'Test set', 'ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1'])}

## Target-test per-section performance

{markdown_table(target_table, ['experiment', 'section', 'roc_auc', 'accuracy', 'precision', 'recall', 'f1'], ['Experiment', 'Section', 'ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1'])}

Detailed normal/anomaly score summaries, confusion counts, thresholds, and per-record scores are in `FINAL_RESULTS.csv` and `FINAL_TEST_PREDICTIONS.csv`.

## Interpretation and limitations

The source-to-target ROC-AUC change for the source-selected model is {selected_target['roc_auc'] - source_winner['roc_auc']:+.4f}. Its target performance is {('consistent with useful rank separation in this held-out domain' if selected_target['roc_auc'] >= 0.70 else 'weak-to-moderate rank separation in this held-out domain')}; thresholded accuracy, recall, and F1 should be considered alongside AUC. This is anomaly detection trained only on normal recordings, not a demonstrated predictor of future failures or a classifier of specific failure types.

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
"""
    (REPORT_DIR / "FINAL_EXPERIMENT_ANALYSIS.md").write_text(
        analysis, encoding="utf-8"
    )

    log = f"""# Experiment Log

Evaluation run: {run_time}

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
- Existing CNN configuration: 30-epoch maximum, batch size 8, Adam learning rate 0.001, mean-squared error objective, seed 42. Existing Isolation Forest configuration: 300 trees, `contamination=\"auto\"`, seed 42.
- Evaluated all 600 source-test and all 600 target-test recordings; target counts were checked as 100 normal and 100 anomalous per section.
- Evaluated all five experiment types at their training-derived normal-validation thresholds. Source scores were read from existing prediction reports; target scores were calculated one recording at a time.
- Computed per-section metrics, equal-section macro averages, confusion counts, score-distribution summaries, and ROC data.
- Plots were generated after scoring. The "strongest models" ROC figure ranks candidates by source-test macro ROC-AUC; target-test performance did not influence selection or thresholds.
- No data, training scripts, or model checkpoints were changed.

## Results

The source-test macro ROC-AUC leader is **{source_winner['experiment']}** ({source_winner['roc_auc']:.4f}). It has target-test macro ROC-AUC {selected_target['roc_auc']:.4f}. The descriptive target-test macro ROC-AUC leader is **{target_best_auc['experiment']}** ({target_best_auc['roc_auc']:.4f}); it was not selected using target results.

Detailed tables: `FINAL_RESULTS.csv`. Per-record scores: `FINAL_TEST_PREDICTIONS.csv`. Full interpretation and limitations: `FINAL_EXPERIMENT_ANALYSIS.md`.
"""
    (REPORT_DIR / "EXPERIMENT_LOG.md").write_text(log, encoding="utf-8")


def main():
    if ROOT.name != "industrial-machine-failure-detection":
        raise RuntimeError(
            "Run this script from the industrial-machine-failure-detection root."
        )
    if not (ROOT / "README.md").is_file():
        print("WARNING: root README.md is absent; continuing with inspected sources.")

    feature_paths = [
        PROCESSED_DIR / "fan_train_mel.npy",
        PROCESSED_DIR / "fan_train_mfcc.npy",
        PROCESSED_DIR / "fan_train_mfcc_stats.npy",
        MODEL_DIR / "best_autoencoder.pth",
        MODEL_DIR / "normalization.npz",
        *[
            MODEL_DIR / "section_models" / f"{section}_autoencoder.pth"
            for section in SECTIONS
        ],
        *[
            MODEL_DIR / "mfcc_section_models" / f"{section}_autoencoder.pth"
            for section in SECTIONS
        ],
        *[
            MODEL_DIR / "isolation_forest" / f"{section}_{suffix}"
            for section in SECTIONS
            for suffix in (
                "model.joblib",
                "scaler.joblib",
                "threshold.npy",
            )
        ],
    ]
    require_files(feature_paths)
    REPORT_DIR.mkdir(exist_ok=True)

    train_files = sorted(TRAIN_DIR.glob("*.wav"))
    if len(train_files) != 3009:
        raise ValueError(f"Expected 3009 normal train files; found {len(train_files)}")
    mel_features = np.load(PROCESSED_DIR / "fan_train_mel.npy", mmap_mode="r")
    mfcc_features = np.load(PROCESSED_DIR / "fan_train_mfcc.npy", mmap_mode="r")
    stats_features = np.load(
        PROCESSED_DIR / "fan_train_mfcc_stats.npy", mmap_mode="r"
    )
    if (
        len(mel_features) != len(train_files)
        or len(mfcc_features) != len(train_files)
        or len(stats_features) != len(train_files)
    ):
        raise ValueError("Training feature arrays do not match sorted train files")
    if mel_features.shape[1:] != (128, 626):
        raise ValueError(f"Unexpected Mel feature shape: {mel_features.shape}")
    if mfcc_features.shape[1:] != (40, 626):
        raise ValueError(f"Unexpected MFCC feature shape: {mfcc_features.shape}")
    if stats_features.shape[1:] != (160,):
        raise ValueError(f"Unexpected Isolation Forest feature shape: {stats_features.shape}")

    source_files = check_split_files(SOURCE_DIR)
    target_files = check_split_files(TARGET_DIR)
    print(f"Device: {DEVICE}")
    print("Validated source/target counts: 600 each; 200 per section.")

    source = load_source_scores()
    thresholds = {}
    models = {
        "shared_mel": load_autoencoder(MODEL_DIR / "best_autoencoder.pth")
    }
    shared_model, shared_mean, shared_std = models["shared_mel"]
    with np.load(MODEL_DIR / "normalization.npz") as shared_normalization:
        if not (
            np.isclose(shared_mean, float(shared_normalization["mean"]))
            and np.isclose(shared_std, float(shared_normalization["std"]))
        ):
            raise ValueError(
                "Shared Mel checkpoint normalization does not match "
                "models/normalization.npz"
            )

    shared_val_indices = sorted(
        set(
            index
            for section in SECTIONS
            for index in validation_indices(train_files, section)
        )
    )
    shared_val_scores = []
    for start in range(0, len(shared_val_indices), 16):
        indices = shared_val_indices[start : start + 16]
        scores, _, _ = batch_cnn_scores(
            shared_model,
            mel_features[indices],
            shared_mean,
            shared_std,
        )
        shared_val_scores.extend(scores.tolist())
    thresholds[(EXPERIMENTS[0], "shared")] = float(
        np.percentile(shared_val_scores, VALIDATION_PERCENTILE)
    )

    isolation = {}
    for section in SECTIONS:
        mel_model = load_autoencoder(
            MODEL_DIR / "section_models" / f"{section}_autoencoder.pth"
        )
        mfcc_model = load_autoencoder(
            MODEL_DIR / "mfcc_section_models" / f"{section}_autoencoder.pth"
        )
        models[(EXPERIMENTS[1], section)] = mel_model
        models[(EXPERIMENTS[2], section)] = mfcc_model
        mel_val = score_validation_cnn(
            mel_model[0], mel_features, train_files, section, mel_model[1], mel_model[2]
        )
        mfcc_val = score_validation_cnn(
            mfcc_model[0], mfcc_features, train_files, section, mfcc_model[1], mfcc_model[2]
        )
        mean_val, p95_val = score_validation_frames(
            mfcc_model[0],
            mfcc_features,
            train_files,
            section,
            mfcc_model[1],
            mfcc_model[2],
        )
        thresholds[(EXPERIMENTS[1], section)] = float(
            np.percentile(mel_val, VALIDATION_PERCENTILE)
        )
        thresholds[(EXPERIMENTS[2], section)] = float(
            np.percentile(mfcc_val, VALIDATION_PERCENTILE)
        )
        thresholds[(EXPERIMENTS[3], section)] = float(
            np.percentile(p95_val, VALIDATION_PERCENTILE)
        )
        thresholds[("MFCC frame-level Mean", section)] = float(
            np.percentile(mean_val, VALIDATION_PERCENTILE)
        )

        model = joblib.load(
            MODEL_DIR / "isolation_forest" / f"{section}_model.joblib"
        )
        scaler = joblib.load(
            MODEL_DIR / "isolation_forest" / f"{section}_scaler.joblib"
        )
        saved_threshold = float(
            np.load(
                MODEL_DIR
                / "isolation_forest"
                / f"{section}_threshold.npy"
            )
        )
        if scaler.n_features_in_ != 160:
            raise ValueError(f"Unexpected Isolation Forest scaler for {section}")
        isolation[section] = (model, scaler, saved_threshold)
        thresholds[(EXPERIMENTS[4], section)] = -saved_threshold

    print("Validation thresholds reproduced from training split.")

    target_rows = []
    for index, path in enumerate(target_files, 1):
        section = next((name for name in SECTIONS if name in path.name), None)
        if section is None:
            raise ValueError(f"Unrecognized target-test filename: {path.name}")
        actual = int("anomaly" in path.name)

        mel = extract_mel_spectrogram(path)
        shared_score, _ = single_cnn_scores(
            shared_model, mel, shared_mean, shared_std
        )
        mel_model, mel_mean, mel_std = models[(EXPERIMENTS[1], section)]
        mel_score, _ = single_cnn_scores(mel_model, mel, mel_mean, mel_std)

        mfcc = extract_mfcc(path)
        mfcc_model, mfcc_mean, mfcc_std = models[(EXPERIMENTS[2], section)]
        mfcc_score, frame_p95_score = single_cnn_scores(
            mfcc_model, mfcc, mfcc_mean, mfcc_std
        )

        forest, scaler, _ = isolation[section]
        forest_features = isolation_features(mfcc).reshape(1, -1)
        anomaly_score = -float(
            forest.decision_function(scaler.transform(forest_features))[0]
        )

        scores_by_experiment = {
            EXPERIMENTS[0]: (shared_score, thresholds[(EXPERIMENTS[0], "shared")]),
            EXPERIMENTS[1]: (mel_score, thresholds[(EXPERIMENTS[1], section)]),
            EXPERIMENTS[2]: (mfcc_score, thresholds[(EXPERIMENTS[2], section)]),
            EXPERIMENTS[3]: (
                frame_p95_score,
                thresholds[(EXPERIMENTS[3], section)],
            ),
            EXPERIMENTS[4]: (
                anomaly_score,
                thresholds[(EXPERIMENTS[4], section)],
            ),
        }
        for experiment, (score, threshold) in scores_by_experiment.items():
            target_rows.append(
                {
                    "dataset": "target_test",
                    "experiment": experiment,
                    "section": section,
                    "filename": path.name,
                    "actual": actual,
                    "score": float(score),
                    "threshold": float(threshold),
                    "threshold_direction": "anomaly if score > threshold",
                    "predicted": int(score > threshold),
                }
            )
        if index % 100 == 0:
            print(f"Scored {index}/{len(target_files)} target recordings")

    target = pd.DataFrame(target_rows)
    if len(target) != 600 * len(EXPERIMENTS):
        raise ValueError("Target-test evaluation did not score every model/recording")
    for experiment in EXPERIMENTS:
        subset = target[target["experiment"] == experiment]
        if len(subset) != 600:
            raise ValueError(f"Incomplete target scores for {experiment}")

    source_thresholded = source.copy()
    source_thresholded["threshold"] = source_thresholded.apply(
        lambda row: thresholds[
            (row["experiment"], "shared")
            if row["experiment"] == EXPERIMENTS[0]
            else (row["experiment"], row["section"])
        ],
        axis=1,
    )
    source_thresholded["threshold_direction"] = "anomaly if score > threshold"
    source_thresholded["predicted"] = (
        source_thresholded["score"] > source_thresholded["threshold"]
    ).astype(int)
    results = pd.concat([source_thresholded, target], ignore_index=True)
    metrics = summarize(results)

    source_macro = metrics[
        (metrics["dataset"] == "source_test")
        & (metrics["section"] == "macro")
    ].sort_values("roc_auc", ascending=False)
    selected_experiment = str(source_macro.iloc[0]["experiment"])
    strongest = generate_plots(results, metrics, selected_experiment)
    run_time = datetime.now().astimezone().isoformat(timespec="seconds")

    metrics.to_csv(REPORT_DIR / "FINAL_RESULTS.csv", index=False)
    results.to_csv(REPORT_DIR / "FINAL_TEST_PREDICTIONS.csv", index=False)
    write_documentation(metrics, selected_experiment, strongest, run_time)

    print("\nTarget-test macro-average metrics:")
    print(
        metrics[
            (metrics["dataset"] == "target_test")
            & (metrics["section"] == "macro")
        ][
            [
                "experiment",
                "roc_auc",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print(f"\nSource-selected model: {selected_experiment}")
    print("Outputs written to reports/.")


if __name__ == "__main__":
    main()
