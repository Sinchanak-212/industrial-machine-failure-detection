from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from model import CNN_Autoencoder


ROOT = Path.cwd()
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_DIR = ROOT / "data" / "raw" / "fan" / "train"
SECTIONS = ["section_00", "section_01", "section_02"]
EXPERIMENTS = [
    "Shared Mel CNN",
    "Section-specific Mel CNN",
    "Section-specific MFCC CNN",
    "MFCC frame-level P95",
    "MFCC + Isolation Forest",
]
METRICS = ["roc_auc", "accuracy", "precision", "recall", "f1"]
PERCENTILES = [90.0, 95.0, 97.5, 99.0]
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 20261107
SPLIT_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def require_files(paths):
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required existing files are missing:\n" + "\n".join(missing)
        )


def get_validation_indices(train_files, section):
    section_indices = [
        index for index, path in enumerate(train_files)
        if section in path.name
    ]
    train_indices, validation_indices = train_test_split(
        section_indices,
        test_size=0.10,
        random_state=SPLIT_SEED,
    )
    if set(train_indices).intersection(validation_indices):
        raise ValueError(f"Overlapping train/validation indices for {section}")
    return validation_indices


def load_cnn(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path, map_location=DEVICE, weights_only=False
    )
    required = {"model_state_dict", "mean", "std"}
    if not required.issubset(checkpoint):
        raise ValueError(f"Checkpoint missing normalization/model values: {checkpoint_path}")
    model = CNN_Autoencoder().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, float(checkpoint["mean"]), float(checkpoint["std"])


def score_cnn_batch(model, feature_batch, mean, std):
    normalized = (
        np.asarray(feature_batch, dtype=np.float32) - mean
    ) / (std + 1e-8)
    inputs = torch.from_numpy(normalized).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        reconstruction = model(inputs)
        errors = (reconstruction - inputs) ** 2
        mean_scores = errors.mean(dim=(1, 2, 3)).cpu().numpy()
        frame_errors = errors.mean(dim=(1, 2)).cpu().numpy()
    p95_scores = np.percentile(frame_errors, 95, axis=1)
    return mean_scores, p95_scores


def score_cnn_validation(model, features, val_indices, mean, std):
    mean_scores = []
    p95_scores = []
    for start in range(0, len(val_indices), 16):
        batch_indices = val_indices[start : start + 16]
        batch_mean, batch_p95 = score_cnn_batch(
            model, features[batch_indices], mean, std
        )
        mean_scores.extend(batch_mean.tolist())
        p95_scores.extend(batch_p95.tolist())
    return np.asarray(mean_scores), np.asarray(p95_scores)


def build_validation_scores(train_files):
    mel_features = np.load(
        PROCESSED_DIR / "fan_train_mel.npy", mmap_mode="r"
    )
    mfcc_features = np.load(
        PROCESSED_DIR / "fan_train_mfcc.npy", mmap_mode="r"
    )
    stats_features = np.load(
        PROCESSED_DIR / "fan_train_mfcc_stats.npy", mmap_mode="r"
    )
    if not (
        len(mel_features) == len(mfcc_features)
        == len(stats_features) == len(train_files)
    ):
        raise ValueError("Training feature arrays do not match sorted train files")
    if mel_features.shape[1:] != (128, 626):
        raise ValueError(f"Unexpected Mel feature shape {mel_features.shape}")
    if mfcc_features.shape[1:] != (40, 626):
        raise ValueError(f"Unexpected MFCC feature shape {mfcc_features.shape}")
    if stats_features.shape[1:] != (160,):
        raise ValueError(f"Unexpected MFCC-stat feature shape {stats_features.shape}")

    validation = {}
    shared_model, shared_mean, shared_std = load_cnn(
        MODEL_DIR / "best_autoencoder.pth"
    )
    with np.load(MODEL_DIR / "normalization.npz") as normalization:
        if not (
            np.isclose(shared_mean, float(normalization["mean"]))
            and np.isclose(shared_std, float(normalization["std"]))
        ):
            raise ValueError(
                "Shared Mel checkpoint normalization differs from normalization.npz"
            )

    for section in SECTIONS:
        indices = get_validation_indices(train_files, section)
        val_names = [train_files[index].name for index in indices]
        if len(indices) != 101 or any("normal" not in name for name in val_names):
            raise ValueError(
                f"Expected 101 normal validation recordings for {section}"
            )

        shared_scores, _ = score_cnn_validation(
            shared_model, mel_features, indices, shared_mean, shared_std
        )

        mel_model, mel_mean, mel_std = load_cnn(
            MODEL_DIR / "section_models" / f"{section}_autoencoder.pth"
        )
        mel_scores, _ = score_cnn_validation(
            mel_model, mel_features, indices, mel_mean, mel_std
        )

        mfcc_model, mfcc_mean, mfcc_std = load_cnn(
            MODEL_DIR / "mfcc_section_models" / f"{section}_autoencoder.pth"
        )
        mfcc_scores, frame_p95_scores = score_cnn_validation(
            mfcc_model, mfcc_features, indices, mfcc_mean, mfcc_std
        )

        scaler_path = MODEL_DIR / "isolation_forest" / f"{section}_scaler.joblib"
        model_path = MODEL_DIR / "isolation_forest" / f"{section}_model.joblib"
        scaler = joblib.load(scaler_path)
        forest = joblib.load(model_path)
        if scaler.n_features_in_ != 160:
            raise ValueError(f"Unexpected Isolation Forest scaler for {section}")
        scaled_stats = scaler.transform(stats_features[indices])
        isolation_scores = -forest.decision_function(scaled_stats)

        validation[section] = {
            "n": len(indices),
            "filenames": val_names,
            "scores": {
                "Shared Mel CNN": shared_scores,
                "Section-specific Mel CNN": mel_scores,
                "Section-specific MFCC CNN": mfcc_scores,
                "MFCC frame-level P95": frame_p95_scores,
                "MFCC + Isolation Forest": isolation_scores,
            },
        }

    # Confirm that persisted Isolation Forest thresholds match the original
    # lower-tail decision_function calibration after reversing score direction.
    for section in SECTIONS:
        saved = float(
            np.load(
                MODEL_DIR
                / "isolation_forest"
                / f"{section}_threshold.npy"
            )
        )
        reproduced = float(
            np.percentile(
                validation[section]["scores"]["MFCC + Isolation Forest"], 95
            )
        )
        if not np.isclose(saved, -reproduced, rtol=1e-5, atol=1e-7):
            raise ValueError(
                f"Saved Isolation Forest threshold mismatch in {section}: "
                f"saved decision threshold={saved}, reproduced anomaly threshold={reproduced}"
            )
    return validation


def load_target_predictions():
    path = REPORT_DIR / "FINAL_TEST_PREDICTIONS.csv"
    frame = pd.read_csv(path)
    required = {
        "experiment",
        "section",
        "filename",
        "actual",
        "score",
        "dataset",
        "threshold",
        "threshold_direction",
        "predicted",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing prediction columns in {path}")
    target = frame[frame["dataset"] == "target_test"].copy()
    if len(target) != len(EXPERIMENTS) * 600:
        raise ValueError(f"Expected {len(EXPERIMENTS) * 600} target predictions")
    if target.duplicated(["experiment", "section", "filename"]).any():
        raise ValueError("Duplicate target prediction records found")
    if not (target["threshold_direction"] == "anomaly if score > threshold").all():
        raise ValueError("Unexpected anomaly score/threshold direction")

    for experiment in EXPERIMENTS:
        records = target[target["experiment"] == experiment]
        if len(records) != 600:
            raise ValueError(f"Incomplete target predictions for {experiment}")
        for section in SECTIONS:
            section_records = records[records["section"] == section]
            counts = section_records["actual"].value_counts().to_dict()
            if len(section_records) != 200 or counts != {0: 100, 1: 100}:
                raise ValueError(
                    f"Unexpected target split for {experiment}/{section}: {counts}"
                )
            calculated = (
                section_records["score"] > section_records["threshold"]
            ).astype(int)
            if not np.array_equal(
                calculated.to_numpy(),
                section_records["predicted"].astype(int).to_numpy(),
            ):
                raise ValueError(
                    f"Saved prediction does not match score/threshold for "
                    f"{experiment}/{section}"
                )
    return target


def check_baseline_thresholds(validation, target):
    existing = pd.read_csv(REPORT_DIR / "FINAL_RESULTS.csv")
    baseline = existing[
        (existing["dataset"] == "target_test")
        & (existing["section"] != "macro")
    ]
    shared_validation_scores = np.concatenate(
        [
            validation[section]["scores"]["Shared Mel CNN"]
            for section in SECTIONS
        ]
    )
    for experiment in EXPERIMENTS:
        for section in SECTIONS:
            result = baseline[
                (baseline["experiment"] == experiment)
                & (baseline["section"] == section)
            ]
            if len(result) != 1:
                raise ValueError(
                    f"Expected one saved 95th-percentile baseline for "
                    f"{experiment}/{section}"
                )
            scores = (
                shared_validation_scores
                if experiment == "Shared Mel CNN"
                else validation[section]["scores"][experiment]
            )
            reproduced = float(np.percentile(scores, 95))
            stored = float(result["threshold"].iloc[0])
            if not np.isclose(reproduced, stored, rtol=1e-5, atol=1e-7):
                raise ValueError(
                    f"Reproduced p95 does not match existing report for "
                    f"{experiment}/{section}: {reproduced} vs {stored}"
                )
            saved_scores = target[
                (target["experiment"] == experiment)
                & (target["section"] == section)
            ]["threshold"].to_numpy()
            if not np.allclose(saved_scores, stored, rtol=1e-5, atol=1e-7):
                raise ValueError(
                    f"Target prediction threshold does not match baseline for "
                    f"{experiment}/{section}"
                )


def compute_threshold_results(validation, target):
    details = []
    sensitivities = []
    thresholds = {}
    for experiment in EXPERIMENTS:
        pooled_validation = np.concatenate(
            [
                validation[section]["scores"][experiment]
                for section in SECTIONS
            ]
        )
        for percentile in PERCENTILES:
            pooled_threshold = float(np.percentile(pooled_validation, percentile))
            for section in SECTIONS:
                val_scores = validation[section]["scores"][experiment]
                threshold = (
                    pooled_threshold
                    if experiment == "Shared Mel CNN"
                    else float(np.percentile(val_scores, percentile))
                )
                thresholds[(experiment, section, percentile)] = threshold
                validation_fp = int(np.count_nonzero(val_scores > threshold))
                details.append(
                    {
                        "experiment": experiment,
                        "section": section,
                        "threshold_percentile": percentile,
                        "threshold": threshold,
                        "score_direction": "larger is more anomalous",
                        "threshold_rule": "anomaly if score > threshold",
                        "validation_threshold_scope": (
                            "pooled across all three sections"
                            if experiment == "Shared Mel CNN"
                            else "section-specific"
                        ),
                        "validation_normals_used_for_threshold": int(
                            len(pooled_validation)
                            if experiment == "Shared Mel CNN"
                            else len(val_scores)
                        ),
                        "validation_normals_scored_in_section": int(
                            len(val_scores)
                        ),
                        "validation_false_positives": validation_fp,
                        "validation_observed_fpr": (
                            validation_fp / len(val_scores)
                        ),
                        "expected_fpr": (100.0 - percentile) / 100.0,
                        "test_data_used_for_threshold": False,
                        "test_labels_used_for_threshold": False,
                    }
                )

            macro_val_fp = sum(
                row["validation_false_positives"]
                for row in details[-3:]
            )
            macro_val_n = sum(
                row["validation_normals_scored_in_section"]
                for row in details[-3:]
            )
            details.append(
                {
                    "experiment": experiment,
                    "section": "macro",
                    "threshold_percentile": percentile,
                    "threshold": (
                        pooled_threshold
                        if experiment == "Shared Mel CNN"
                        else np.nan
                    ),
                    "score_direction": "larger is more anomalous",
                    "threshold_rule": (
                        "same pooled threshold across sections"
                        if experiment == "Shared Mel CNN"
                        else "section-specific thresholds; macro summary"
                    ),
                    "validation_threshold_scope": (
                        "pooled across all three sections"
                        if experiment == "Shared Mel CNN"
                        else "section-specific"
                    ),
                    "validation_normals_used_for_threshold": int(
                        len(pooled_validation)
                    ),
                    "validation_normals_scored_in_section": macro_val_n,
                    "validation_false_positives": macro_val_fp,
                    "validation_observed_fpr": macro_val_fp / macro_val_n,
                    "expected_fpr": (100.0 - percentile) / 100.0,
                    "test_data_used_for_threshold": False,
                    "test_labels_used_for_threshold": False,
                }
            )

    detail_frame = pd.DataFrame(details)
    target_rows = target[target["experiment"].isin(EXPERIMENTS)]
    for experiment in EXPERIMENTS:
        for percentile in PERCENTILES:
            section_metrics = []
            for section in SECTIONS:
                records = target_rows[
                    (target_rows["experiment"] == experiment)
                    & (target_rows["section"] == section)
                ].copy()
                threshold = thresholds[(experiment, section, percentile)]
                scores = records["score"].to_numpy(dtype=float)
                labels = records["actual"].to_numpy(dtype=int)
                predicted = (scores > threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(
                    labels, predicted, labels=[0, 1]
                ).ravel()
                section_metrics.append(
                    {
                        "experiment": experiment,
                        "section": section,
                        "threshold_percentile": percentile,
                        "threshold": threshold,
                        "n": len(records),
                        "roc_auc": float(roc_auc_score(labels, scores)),
                        "accuracy": float(accuracy_score(labels, predicted)),
                        "precision": float(
                            precision_score(
                                labels, predicted, zero_division=0
                            )
                        ),
                        "recall": float(recall_score(labels, predicted)),
                        "f1": float(f1_score(labels, predicted, zero_division=0)),
                        "true_negatives": int(tn),
                        "false_positives": int(fp),
                        "false_negatives": int(fn),
                        "true_positives": int(tp),
                        "target_fpr": float(fp / (tn + fp)),
                        "target_labels_used_for_threshold": False,
                        "use": "sensitivity analysis only; not threshold selection",
                    }
                )
            macro = {
                "experiment": experiment,
                "section": "macro",
                "threshold_percentile": percentile,
                "threshold": np.nan,
                "n": sum(row["n"] for row in section_metrics),
                "true_negatives": sum(
                    row["true_negatives"] for row in section_metrics
                ),
                "false_positives": sum(
                    row["false_positives"] for row in section_metrics
                ),
                "false_negatives": sum(
                    row["false_negatives"] for row in section_metrics
                ),
                "true_positives": sum(
                    row["true_positives"] for row in section_metrics
                ),
                "target_labels_used_for_threshold": False,
                "use": "sensitivity analysis only; not threshold selection",
            }
            for metric in [
                "roc_auc",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "target_fpr",
            ]:
                macro[metric] = float(
                    np.mean([row[metric] for row in section_metrics])
                )
            sensitivities.extend(section_metrics)
            sensitivities.append(macro)

    return detail_frame, pd.DataFrame(sensitivities), thresholds


def metric_values(labels, scores, threshold):
    predicted = scores > threshold
    tn = int(np.count_nonzero((labels == 0) & ~predicted))
    fp = int(np.count_nonzero((labels == 0) & predicted))
    fn = int(np.count_nonzero((labels == 1) & ~predicted))
    tp = int(np.count_nonzero((labels == 1) & predicted))
    predicted_anomalies = int(np.count_nonzero(predicted))
    precision = tp / predicted_anomalies if predicted_anomalies else 0.0
    recall = tp / (tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "accuracy": float((tn + tp) / len(labels)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision_zero_division": int(predicted_anomalies == 0),
    }


def compute_bootstrap(target, repetitions, seed):
    score_data = {}
    filename_data = {}
    threshold_data = {}
    for experiment in EXPERIMENTS:
        for section in SECTIONS:
            subset = target[
                (target["experiment"] == experiment)
                & (target["section"] == section)
            ].sort_values("filename")
            normals = subset[subset["actual"] == 0]
            anomalies = subset[subset["actual"] == 1]
            if len(normals) != 100 or len(anomalies) != 100:
                raise ValueError(
                    f"Bootstrap requires 100 observations per class in "
                    f"{experiment}/{section}"
                )
            score_data[(experiment, section, 0)] = normals["score"].to_numpy(
                dtype=float
            )
            score_data[(experiment, section, 1)] = anomalies["score"].to_numpy(
                dtype=float
            )
            filename_data[(experiment, section, 0)] = normals[
                "filename"
            ].to_numpy()
            filename_data[(experiment, section, 1)] = anomalies[
                "filename"
            ].to_numpy()
            threshold_data[(experiment, section)] = float(
                subset["threshold"].iloc[0]
            )

    for section in SECTIONS:
        baseline_names = filename_data[(EXPERIMENTS[0], section, 0)]
        for label in [1]:
            expected_names = filename_data[(EXPERIMENTS[0], section, label)]
            for experiment in EXPERIMENTS[1:]:
                actual_names = filename_data[(experiment, section, label)]
                if not np.array_equal(actual_names, expected_names):
                    raise ValueError(
                        f"Prediction records not aligned for paired bootstrap "
                        f"in {section}, label {label}"
                    )
        for experiment in EXPERIMENTS[1:]:
            names = filename_data[(experiment, section, 0)]
            if not np.array_equal(names, baseline_names):
                raise ValueError(
                    f"Normal target records not aligned in {section}"
                )

    rng = np.random.default_rng(seed)
    draws = np.empty((repetitions, len(EXPERIMENTS), len(SECTIONS), len(METRICS)))
    paired_difference = np.empty((repetitions, len(SECTIONS)), dtype=float)
    zero_precision = np.zeros(
        (len(EXPERIMENTS), len(SECTIONS)), dtype=np.int64
    )
    zero_precision_by_draw = np.zeros(
        (repetitions, len(EXPERIMENTS), len(SECTIONS)), dtype=bool
    )
    point_by_section = {}

    for experiment_index, experiment in enumerate(EXPERIMENTS):
        for section_index, section in enumerate(SECTIONS):
            subset = target[
                (target["experiment"] == experiment)
                & (target["section"] == section)
            ]
            point_by_section[(experiment, section)] = metric_values(
                subset["actual"].to_numpy(dtype=int),
                subset["score"].to_numpy(dtype=float),
                float(subset["threshold"].iloc[0]),
            )

    for bootstrap_index in range(repetitions):
        sampled_scores = {}
        sampled_labels = {}
        for section_index, section in enumerate(SECTIONS):
            normal_indices = rng.integers(0, 100, size=100)
            anomaly_indices = rng.integers(0, 100, size=100)
            for experiment_index, experiment in enumerate(EXPERIMENTS):
                scores = np.concatenate(
                    [
                        score_data[(experiment, section, 0)][normal_indices],
                        score_data[(experiment, section, 1)][anomaly_indices],
                    ]
                )
                labels = np.concatenate(
                    [np.zeros(100, dtype=int), np.ones(100, dtype=int)]
                )
                threshold = threshold_data[(experiment, section)]
                values = metric_values(labels, scores, threshold)
                draws[
                    bootstrap_index, experiment_index, section_index, :
                ] = [values[metric] for metric in METRICS]
                if values["precision_zero_division"]:
                    zero_precision[experiment_index, section_index] += 1
                    zero_precision_by_draw[
                        bootstrap_index, experiment_index, section_index
                    ] = True
                sampled_scores[(experiment, section)] = scores
                sampled_labels[(experiment, section)] = labels

        for section_index, section in enumerate(SECTIONS):
            mfcc_scores = sampled_scores[
                ("Section-specific MFCC CNN", section)
            ]
            isolation_scores = sampled_scores[
                ("MFCC + Isolation Forest", section)
            ]
            labels = sampled_labels[("Section-specific MFCC CNN", section)]
            paired_difference[bootstrap_index, section_index] = (
                roc_auc_score(labels, mfcc_scores)
                - roc_auc_score(labels, isolation_scores)
            )

    result_rows = []
    for experiment_index, experiment in enumerate(EXPERIMENTS):
        for metric_index, metric in enumerate(METRICS):
            for section_index, section in enumerate(SECTIONS):
                sample = draws[:, experiment_index, section_index, metric_index]
                lower, upper = np.percentile(sample, [2.5, 97.5])
                result_rows.append(
                    {
                        "analysis": "target test stratified percentile bootstrap",
                        "experiment": experiment,
                        "section": section,
                        "metric": metric,
                        "point_estimate": point_by_section[
                            (experiment, section)
                        ][metric],
                        "ci_95_lower": float(lower),
                        "ci_95_upper": float(upper),
                        "bootstrap_repetitions": repetitions,
                        "bootstrap_seed": seed,
                        "resampling": (
                            "within section and true-label group; n=100 "
                            "sampled with replacement per label"
                        ),
                        "precision_zero_division_replicates": (
                            int(zero_precision[experiment_index, section_index])
                            if metric == "precision"
                            else 0
                        ),
                        "undefined_precision_convention": (
                            "zero_division=0; report count of no-positive-"
                            "prediction replicates"
                            if metric == "precision"
                            else ""
                        ),
                    }
                )
            macro_draws = draws[
                :, experiment_index, :, metric_index
            ].mean(axis=1)
            lower, upper = np.percentile(macro_draws, [2.5, 97.5])
            point = np.mean(
                [
                    point_by_section[(experiment, section)][metric]
                    for section in SECTIONS
                ]
            )
            result_rows.append(
                {
                    "analysis": "target test stratified percentile bootstrap",
                    "experiment": experiment,
                    "section": "macro",
                    "metric": metric,
                    "point_estimate": float(point),
                    "ci_95_lower": float(lower),
                    "ci_95_upper": float(upper),
                    "bootstrap_repetitions": repetitions,
                    "bootstrap_seed": seed,
                    "resampling": (
                        "within section and true-label group; equal-section "
                        "mean of section metrics per replicate"
                    ),
                    "precision_zero_division_replicates": (
                        int(
                            np.count_nonzero(
                                zero_precision_by_draw[
                                    :, experiment_index, :
                                ].any(axis=1)
                            )
                        )
                        if metric == "precision"
                        else 0
                    ),
                    "undefined_precision_convention": (
                        "zero_division=0; report count of no-positive-"
                        "prediction replicates"
                        if metric == "precision"
                        else ""
                    ),
                }
            )

    cnn_index = EXPERIMENTS.index("Section-specific MFCC CNN")
    isolation_index = EXPERIMENTS.index("MFCC + Isolation Forest")
    for section_index, section in enumerate(SECTIONS):
        sample = paired_difference[:, section_index]
        lower, upper = np.percentile(sample, [2.5, 97.5])
        point = (
            point_by_section[("Section-specific MFCC CNN", section)]["roc_auc"]
            - point_by_section[("MFCC + Isolation Forest", section)]["roc_auc"]
        )
        result_rows.append(
            {
                "analysis": "paired stratified bootstrap ROC-AUC difference",
                "experiment": (
                    "Section-specific MFCC CNN - MFCC + Isolation Forest"
                ),
                "section": section,
                "metric": "roc_auc_difference",
                "point_estimate": float(point),
                "ci_95_lower": float(lower),
                "ci_95_upper": float(upper),
                "bootstrap_repetitions": repetitions,
                "bootstrap_seed": seed,
                "resampling": (
                    "same sampled target recordings within section and "
                    "true-label groups for both models"
                ),
                "precision_zero_division_replicates": 0,
                "undefined_precision_convention": "",
            }
        )
    macro_difference = paired_difference.mean(axis=1)
    lower, upper = np.percentile(macro_difference, [2.5, 97.5])
    point = np.mean(
        [
            point_by_section[("Section-specific MFCC CNN", section)]["roc_auc"]
            - point_by_section[("MFCC + Isolation Forest", section)]["roc_auc"]
            for section in SECTIONS
        ]
    )
    result_rows.append(
        {
            "analysis": "paired stratified bootstrap ROC-AUC difference",
            "experiment": (
                "Section-specific MFCC CNN - MFCC + Isolation Forest"
            ),
            "section": "macro",
            "metric": "roc_auc_difference",
            "point_estimate": float(point),
            "ci_95_lower": float(lower),
            "ci_95_upper": float(upper),
            "bootstrap_repetitions": repetitions,
            "bootstrap_seed": seed,
            "resampling": (
                "same sampled target recordings within section and "
                "true-label groups for both models; equal-section mean"
            ),
            "precision_zero_division_replicates": 0,
            "undefined_precision_convention": "",
        }
    )
    return pd.DataFrame(result_rows), draws, paired_difference, point_by_section


def create_plots(validation, target, bootstrap_results, draws, paired_draws, thresholds, sensitivity):
    bootstrap_rows = bootstrap_results[
        bootstrap_results["analysis"]
        == "target test stratified percentile bootstrap"
    ]
    auc = bootstrap_rows[bootstrap_rows["metric"] == "roc_auc"].copy()
    fig, ax = plt.subplots(figsize=(13, 7))
    exp_positions = np.arange(len(EXPERIMENTS))
    offsets = np.linspace(-0.27, 0.27, 4)
    groups = ["section_00", "section_01", "section_02", "macro"]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#333333"]
    for group_index, (group, offset, color) in enumerate(
        zip(groups, offsets, colors)
    ):
        rows = auc[auc["section"] == group].set_index("experiment")
        point = [rows.loc[exp, "point_estimate"] for exp in EXPERIMENTS]
        low = [rows.loc[exp, "ci_95_lower"] for exp in EXPERIMENTS]
        high = [rows.loc[exp, "ci_95_upper"] for exp in EXPERIMENTS]
        ax.errorbar(
            exp_positions + offset,
            point,
            yerr=[
                np.asarray(point) - np.asarray(low),
                np.asarray(high) - np.asarray(point),
            ],
            fmt="o",
            capsize=3,
            label=group,
            color=color,
        )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(exp_positions, EXPERIMENTS, rotation=20, ha="right")
    ax.set_ylabel("Target-test ROC-AUC (95% bootstrap CI)")
    ax.set_ylim(0.25, 0.9)
    ax.set_title("Stratified bootstrap ROC-AUC confidence intervals")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Section")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "roc_auc_confidence_intervals.png", dpi=180)
    plt.close(fig)

    macro_diff = paired_draws.mean(axis=1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(macro_diff, bins=45, color="#4C78A8", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", label="No AUC difference")
    ax.axvline(
        np.mean(macro_diff),
        color="#E45756",
        linewidth=2,
        label=f"Bootstrap mean = {np.mean(macro_diff):.3f}",
    )
    ax.set_xlabel(
        "Paired macro ROC-AUC difference "
        "(MFCC CNN - Isolation Forest)"
    )
    ax.set_ylabel("Bootstrap replicates")
    ax.set_title(
        "Paired stratified bootstrap: target-test ROC-AUC difference"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "paired_auc_difference.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(
        len(EXPERIMENTS), len(SECTIONS), figsize=(15, 16), squeeze=False
    )
    threshold_colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
    for experiment_index, experiment in enumerate(EXPERIMENTS):
        for section_index, section in enumerate(SECTIONS):
            ax = axes[experiment_index, section_index]
            scores = validation[section]["scores"][experiment]
            ax.hist(
                scores,
                bins=24,
                density=True,
                color="#9ECAE9",
                alpha=0.75,
            )
            for percentile, color in zip(PERCENTILES, threshold_colors):
                value = thresholds[(experiment, section, percentile)]
                ax.axvline(
                    value,
                    color=color,
                    linewidth=1.5,
                    label=f"{percentile:g}th" if section_index == 0 else None,
                )
            if experiment_index == 0:
                ax.set_title(section)
            if section_index == 0:
                ax.set_ylabel(f"{experiment}\nDensity")
            ax.grid(axis="y", alpha=0.2)
            if experiment_index == 0 and section_index == 0:
                ax.legend(fontsize=8, title="Validation quantile")
    fig.suptitle(
        "Normal-only validation anomaly scores and validation-derived thresholds",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "validation_score_distributions.png", dpi=180)
    plt.close(fig)

    macro_sensitivity = sensitivity[sensitivity["section"] == "macro"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for experiment in EXPERIMENTS:
        rows = macro_sensitivity[
            macro_sensitivity["experiment"] == experiment
        ].sort_values("threshold_percentile")
        axes[0, 0].plot(
            rows["threshold_percentile"], rows["recall"], marker="o",
            label=experiment,
        )
        axes[0, 1].plot(
            rows["threshold_percentile"], rows["precision"], marker="o",
            label=experiment,
        )
        axes[1, 0].plot(
            rows["threshold_percentile"], rows["f1"], marker="o",
            label=experiment,
        )
        axes[1, 1].plot(
            rows["threshold_percentile"], rows["target_fpr"], marker="o",
            label=experiment,
        )
    titles = ["Target recall", "Target precision", "Target F1", "Target false-positive rate"]
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel("Normal-validation threshold percentile")
        ax.set_xticks(PERCENTILES)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel("Equal-section metric")
    axes[1, 0].set_ylabel("Equal-section metric")
    axes[0, 1].legend(fontsize=8)
    fig.suptitle(
        "Validation-only thresholds; target-test metrics are sensitivity analysis"
    )
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "threshold_sensitivity.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for experiment in EXPERIMENTS:
        rows = macro_sensitivity[
            macro_sensitivity["experiment"] == experiment
        ].sort_values("threshold_percentile")
        ax.plot(
            rows["threshold_percentile"],
            rows["recall"],
            marker="o",
            label=experiment,
        )
    ax.set_xlabel("Normal-validation threshold percentile")
    ax.set_ylabel("Equal-section target-test recall")
    ax.set_xticks(PERCENTILES)
    ax.set_ylim(0, 1)
    ax.set_title("Target-test recall at validation-derived thresholds")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "target_recall_vs_threshold.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for experiment in EXPERIMENTS:
        rows = macro_sensitivity[
            macro_sensitivity["experiment"] == experiment
        ].sort_values("threshold_percentile")
        ax.plot(
            rows["threshold_percentile"],
            rows["target_fpr"],
            marker="o",
            label=experiment,
        )
    ax.set_xlabel("Normal-validation threshold percentile")
    ax.set_ylabel("Equal-section target-test false-positive rate")
    ax.set_xticks(PERCENTILES)
    ax.set_ylim(0, 1)
    ax.set_title("Target-test false-positive rate at validation thresholds")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "target_fpr_vs_threshold.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    for experiment in EXPERIMENTS:
        rows = target[target["experiment"] == experiment]
        labels = rows["actual"].to_numpy(dtype=int)
        scores = rows["score"].to_numpy(dtype=float)
        precision, recall, _ = precision_recall_curve(labels, scores)
        average_precision = float(average_precision_score(labels, scores))
        ax.plot(
            recall,
            precision,
            label=f"{experiment} (average precision={average_precision:.3f})",
        )
    ax.axhline(0.5, color="gray", linestyle="--", label="Balanced prevalence (0.5)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(
        "Descriptive target-test precision-recall curves "
        "(not used for threshold selection)"
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "target_precision_recall_curves.png", dpi=180)
    plt.close(fig)


def fmt(value):
    return "N/A" if pd.isna(value) else f"{float(value):.4f}"


def make_markdown_table(frame, columns, headers):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                cells.append(fmt(value))
            elif pd.isna(value):
                cells.append("N/A")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_documentation(bootstrap, details, sensitivity, run_time):
    macro_bootstrap = bootstrap[
        (bootstrap["analysis"] == "target test stratified percentile bootstrap")
        & (bootstrap["section"] == "macro")
    ]
    macro_auc = macro_bootstrap[macro_bootstrap["metric"] == "roc_auc"]
    paired = bootstrap[
        (bootstrap["analysis"] == "paired stratified bootstrap ROC-AUC difference")
        & (bootstrap["section"] == "macro")
    ]
    target_summary = sensitivity[sensitivity["section"] == "macro"]
    baseline_targets = target_summary[target_summary["threshold_percentile"] == 95]
    baseline_thresholds = details[
        (details["section"] != "macro")
        & (details["threshold_percentile"] == 95)
    ]
    key_models = [
        "Section-specific MFCC CNN",
        "MFCC + Isolation Forest",
    ]
    key_thresholds = baseline_thresholds[
        baseline_thresholds["experiment"].isin(key_models)
    ][
        [
            "experiment",
            "section",
            "threshold",
            "validation_false_positives",
            "validation_normals_scored_in_section",
            "validation_observed_fpr",
        ]
    ]
    ci_table = macro_auc[
        [
            "experiment",
            "point_estimate",
            "ci_95_lower",
            "ci_95_upper",
        ]
    ].copy()
    paired_table = paired[
        [
            "experiment",
            "point_estimate",
            "ci_95_lower",
            "ci_95_upper",
        ]
    ]
    sensitivity_table = baseline_targets[
        [
            "experiment",
            "roc_auc",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "false_positives",
            "false_negatives",
            "true_positives",
            "true_negatives",
        ]
    ]
    full_sensitivity = target_summary[
        [
            "experiment",
            "threshold_percentile",
            "roc_auc",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "false_positives",
            "false_negatives",
            "true_positives",
            "true_negatives",
            "target_fpr",
        ]
    ]
    paired_row = paired.iloc[0]
    excludes_zero = paired_row["ci_95_lower"] > 0 or paired_row["ci_95_upper"] < 0
    comparison_interpretation = (
        "The paired 95% interval excludes zero, which is evidence of a difference "
        "for this dataset under the stated bootstrap assumptions; it is not "
        "evidence of external generalizability."
        if excludes_zero
        else "The paired 95% interval includes zero; these data do not establish "
        "a difference between the two models' macro ROC-AUC."
    )
    text = f"""# Confidence Intervals and Threshold Sensitivity

Analysis run: {run_time}

## Data integrity and split reconstruction

The original train scripts (`train_sections.py`, `train_mfcc_sections.py`, and `train_isolation_forest.py`) each sort the 3,009 normal training recordings, select each section independently, then call `train_test_split(indices, test_size=0.10, random_state=42)`. Reproducing this exact operation yields 902 training and 101 normal validation recordings per section. Existing feature arrays align with the same sorted file list. Validation scores below were recomputed only by inference on those original validation indices, using frozen checkpoints, the checkpoint normalization values, and the saved Isolation Forest scaler/model. No validation labels beyond their normal-only membership were used to calculate thresholds.

The existing target prediction records contain split, experiment, section, filename, true label, anomaly score, locked 95th-percentile validation threshold, and prediction. The analysis verified uniqueness, score direction, score-threshold consistency, 100 normal and 100 anomalous target examples per section, and agreement between reproduced p95 thresholds and the existing evaluation report.

## Bootstrap confidence intervals

The analysis uses {BOOTSTRAP_REPETITIONS:,} percentile-bootstrap replicates with NumPy generator seed {BOOTSTRAP_SEED}. In every replicate, recordings are sampled with replacement separately within each of the six section-by-label groups (three sections × two true labels), preserving 100 observations per label per section. The same sampled target filenames are used across experiments, and the MFCC CNN vs Isolation Forest comparison is paired on identical resamples. Section-level statistics are calculated first; macro intervals are percentile intervals of the equal-section mean metric in each replicate. Confidence bounds are the 2.5th and 97.5th percentiles of bootstrap replicates.

All target thresholded metrics use each experiment's original validation-derived p95 threshold. Precision uses `zero_division=0`, matching the original report convention; section-level bootstrap replicates with no predicted anomalies are assigned zero precision and counted explicitly. For the Isolation Forest, section 02 had no predicted anomalies in all {BOOTSTRAP_REPETITIONS:,} replicates; the macro precision is the mean of section precisions with this convention applied. This avoids undefined numeric values but should be considered when interpreting precision intervals.

### Macro ROC-AUC (target test)

{make_markdown_table(ci_table, ['experiment', 'point_estimate', 'ci_95_lower', 'ci_95_upper'], ['Experiment', 'Point estimate', '95% CI lower', '95% CI upper'])}

### Paired target ROC-AUC difference

Difference is defined as **Section-specific MFCC CNN minus MFCC + Isolation Forest**; shared stratified resamples were used for both scores.

{make_markdown_table(paired_table, ['experiment', 'point_estimate', 'ci_95_lower', 'ci_95_upper'], ['Paired comparison', 'Point difference', '95% CI lower', '95% CI upper'])}

{comparison_interpretation} This is not a comparison of models selected using target results; the pair is emphasized as requested. Point-estimate ordering alone is not sufficient to claim superiority.

Section-wise intervals for all five metrics and all experiments, as well as the per-section paired AUC differences, are provided in `BOOTSTRAP_RESULTS.csv`.

## Validation-only thresholds and observed calibration false-positive rates

Thresholds are empirical NumPy linear percentiles of **normal validation anomaly scores**; a recording is anomalous only when `score > threshold`. The four nominal tail probabilities are 10%, 5%, 2.5%, and 1%. Because there are 101 validation recordings per section, observed rates are discrete and may differ from nominal rates. The Isolation Forest score is `-decision_function`, so larger scores consistently indicate greater anomaly.

The shared Mel model uses one pooled normal-validation threshold across all three sections (303 validation recordings); the section-specific models use the original 101 validation recordings for that section. The shared threshold is then applied to each section. Validation is normal-only, so validation false-positive rates can be measured, but validation precision, recall, F1, and anomaly detection rates cannot be estimated from these data.

### 95th-percentile thresholds for the primary MFCC and Isolation Forest models

{make_markdown_table(key_thresholds, ['experiment', 'section', 'threshold', 'validation_false_positives', 'validation_normals_scored_in_section', 'validation_observed_fpr'], ['Experiment', 'Section', 'Threshold', 'Validation false positives', 'Validation normals', 'Observed validation FPR'])}

Complete threshold values and observed counts at all four percentiles for all five experiments are in `VALIDATION_THRESHOLD_DETAILS.csv`.

## Locked-threshold target-test sensitivity

The target test labels were used only to compute the following evaluation metrics after validation thresholds had been fixed. Every row below is a sensitivity-analysis operating point, not a threshold selection. ROC-AUC does not depend on the applied threshold and therefore repeats for each percentile. Macro thresholded scores are equal-section means; confusion counts are pooled.

{make_markdown_table(full_sensitivity, ['experiment', 'threshold_percentile', 'roc_auc', 'accuracy', 'precision', 'recall', 'f1', 'false_positives', 'false_negatives', 'true_positives', 'true_negatives', 'target_fpr'], ['Experiment', 'Val percentile', 'ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 'FP', 'FN', 'TP', 'TN', 'Target FPR'])}

### Original 95th-percentile operating point

{make_markdown_table(sensitivity_table, ['experiment', 'roc_auc', 'accuracy', 'precision', 'recall', 'f1', 'false_positives', 'false_negatives', 'true_positives', 'true_negatives'], ['Experiment', 'ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1', 'FP', 'FN', 'TP', 'TN'])}

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
"""
    (REPORT_DIR / "CONFIDENCE_INTERVAL_ANALYSIS.md").write_text(
        text, encoding="utf-8"
    )


def main():
    if ROOT.name != "industrial-machine-failure-detection":
        raise RuntimeError(
            "Run this script from the industrial-machine-failure-detection root."
        )
    required = [
        REPORT_DIR / "FINAL_RESULTS.csv",
        REPORT_DIR / "FINAL_TEST_PREDICTIONS.csv",
        REPORT_DIR / "FINAL_EXPERIMENT_ANALYSIS.md",
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
    require_files(required)

    train_files = sorted(TRAIN_DIR.glob("*.wav"))
    if len(train_files) != 3009:
        raise ValueError(f"Expected 3009 normal training files, got {len(train_files)}")
    if not all("normal" in path.name for path in train_files):
        raise ValueError("Training split must contain only normal recordings")
    validation = build_validation_scores(train_files)
    target = load_target_predictions()
    check_baseline_thresholds(validation, target)

    details, sensitivity, thresholds = compute_threshold_results(
        validation, target
    )
    bootstrap, draws, paired, _ = compute_bootstrap(
        target, BOOTSTRAP_REPETITIONS, BOOTSTRAP_SEED
    )
    run_time = datetime.now().astimezone().isoformat(timespec="seconds")

    details.to_csv(REPORT_DIR / "VALIDATION_THRESHOLD_DETAILS.csv", index=False)
    sensitivity.to_csv(REPORT_DIR / "THRESHOLD_SENSITIVITY.csv", index=False)
    bootstrap.to_csv(REPORT_DIR / "BOOTSTRAP_RESULTS.csv", index=False)
    create_plots(
        validation,
        target,
        bootstrap,
        draws,
        paired,
        thresholds,
        sensitivity,
    )
    write_documentation(bootstrap, details, sensitivity, run_time)

    print(f"Device: {DEVICE}")
    validation_counts = ", ".join(
        f"{section}={validation[section]['n']}" for section in SECTIONS
    )
    print(
        f"Validation split reconstructed: "
        f"{sum(validation[s]['n'] for s in SECTIONS)} normal recordings "
        f"({validation_counts})"
    )
    print(f"Target bootstrap repetitions: {BOOTSTRAP_REPETITIONS}")
    print("\nMacro target-test ROC-AUC confidence intervals:")
    print(
        bootstrap[
            (bootstrap["analysis"] == "target test stratified percentile bootstrap")
            & (bootstrap["section"] == "macro")
            & (bootstrap["metric"] == "roc_auc")
        ][
            ["experiment", "point_estimate", "ci_95_lower", "ci_95_upper"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\nPaired macro ROC-AUC difference confidence interval:")
    print(
        bootstrap[
            (bootstrap["analysis"] == "paired stratified bootstrap ROC-AUC difference")
            & (bootstrap["section"] == "macro")
        ][
            ["experiment", "point_estimate", "ci_95_lower", "ci_95_upper"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\n95th-percentile validation FPRs:")
    print(
        details[
            (details["threshold_percentile"] == 95)
            & (details["section"] != "macro")
        ][
            [
                "experiment",
                "section",
                "threshold",
                "validation_false_positives",
                "validation_normals_scored_in_section",
                "validation_observed_fpr",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    print("\nAnalysis files written to reports/.")


if __name__ == "__main__":
    main()
