"""Evaluation helpers shared by the training and comparison pipelines.

Every function here computes metrics from real predictions - no value is
hard-coded or estimated.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from ml.feature_engineering import FeatureSchema


def measure_latency(pipeline, X, repeats: int = 1) -> dict:
    """Measure single-sample prediction latency in milliseconds."""
    if len(X) == 0:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "total_samples": 0}
    durations = []
    for _ in range(max(1, repeats)):
        for row in X:
            started = time.perf_counter()
            pipeline.predict(np.asarray([row]))
            durations.append((time.perf_counter() - started) * 1000.0)
    array = np.asarray(durations)
    return {
        "mean_ms": round(float(array.mean()), 4),
        "median_ms": round(float(np.median(array)), 4),
        "p95_ms": round(float(np.percentile(array, 95)), 4),
        "max_ms": round(float(array.max()), 4),
        "total_samples": int(array.size),
    }


def probability_matrix(pipeline, X, classes) -> np.ndarray:
    """Return the class-probability matrix for ``X`` (zeros if unavailable)."""
    if not hasattr(pipeline, "predict_proba"):
        return np.zeros((len(X), len(classes)), dtype=np.float64)
    probabilities = np.asarray(pipeline.predict_proba(np.asarray(X)), dtype=np.float64)
    model_classes = [str(c) for c in getattr(pipeline, "classes_", classes)]
    aligned = np.zeros((len(X), len(classes)), dtype=np.float64)
    for index, label in enumerate(model_classes):
        if label in classes:
            aligned[:, classes.index(label)] = probabilities[:, index]
    return aligned


def select_rejection_threshold(
    y_true, probabilities, classes, min_true_acceptance: float = 0.90
) -> tuple:
    """Choose a rejection threshold from validation results.

    The threshold maximises the macro F1-score of the accepted predictions
    subject to retaining at least ``min_true_acceptance`` of the correctly
    classified validation samples.  The selected threshold also limits the
    false-acceptance rate: predictions that fall below it are reported as
    *"Gesture not recognised."* rather than being forced into a class.
    """
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    if len(y_true) == 0:
        return 0.60, {
            "threshold": 0.60,
            "reason": "No validation data was available; using the default.",
            "true_acceptance": 0.0,
            "false_acceptance": 0.0,
            "macro_f1_at_threshold": 0.0,
        }

    best_index = probabilities.argmax(axis=1)
    best_confidence = probabilities.max(axis=1)
    predicted = np.asarray([classes[index] for index in best_index])
    correct = predicted == y_true

    grid = np.arange(0.30, 0.96, 0.01)
    best = None
    for threshold in grid:
        accepted = best_confidence >= threshold
        if not accepted.any():
            continue
        true_acceptance = float(correct[accepted].mean())
        if true_acceptance < min_true_acceptance:
            continue
        _, _, f1, _ = precision_recall_fscore_support(
            y_true[accepted],
            predicted[accepted],
            average="macro",
            zero_division=0,
            labels=classes,
        )
        score = float(f1)
        if best is None or score > best[0]:
            best = (score, float(threshold), true_acceptance)

    if best is None:
        threshold = 0.60
        accepted = best_confidence >= threshold
        true_acceptance = float(correct[accepted].mean()) if accepted.any() else 0.0
        _, _, f1, _ = precision_recall_fscore_support(
            y_true[accepted],
            predicted[accepted],
            average="macro",
            zero_division=0,
            labels=classes,
        ) if accepted.any() else (0.0, 0.0, 0.0, 0.0)
        return threshold, {
            "threshold": threshold,
            "reason": "No threshold satisfied the acceptance constraint.",
            "true_acceptance": round(true_acceptance, 4),
            "false_acceptance": round(1.0 - true_acceptance, 4),
            "macro_f1_at_threshold": round(float(f1), 4),
        }

    _, threshold, true_acceptance = best
    accepted = best_confidence >= threshold
    if accepted.any():
        _, _, f1, _ = precision_recall_fscore_support(
            y_true[accepted],
            predicted[accepted],
            average="macro",
            zero_division=0,
            labels=classes,
        )
        false_acceptance = float((~correct[accepted]).mean())
    else:  # pragma: no cover - defensive
        f1 = 0.0
        false_acceptance = 1.0

    return round(threshold, 4), {
        "threshold": round(threshold, 4),
        "reason": "Maximised macro F1 subject to the acceptance constraint.",
        "true_acceptance": round(true_acceptance, 4),
        "false_acceptance": round(false_acceptance, 4),
        "macro_f1_at_threshold": round(float(f1), 4),
        "min_true_acceptance": min_true_acceptance,
    }


def evaluate_predictions(
    y_true,
    y_pred,
    classes,
    probabilities=None,
    pipeline=None,
    X_test=None,
) -> dict:
    """Compute the full metric set for a set of predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=classes
    )
    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0, labels=classes
        )
    )

    per_class = {}
    for index, label in enumerate(classes):
        per_class[str(label)] = {
            "precision": round(float(per_class_precision[index]), 4),
            "recall": round(float(per_class_recall[index]), 4),
            "f1_score": round(float(per_class_f1[index]), 4),
            "support": int(per_class_support[index]),
        }

    report = classification_report(
        y_true,
        y_pred,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4),
        "samples": int(len(y_true)),
    }

    latency = None
    if pipeline is not None and X_test is not None:
        latency = measure_latency(pipeline, X_test)
        metrics["latency_mean_ms"] = latency["mean_ms"]
        metrics["latency_p95_ms"] = latency["p95_ms"]

    return {
        "metrics": metrics,
        "per_class": per_class,
        "classification_report": report,
        "confusion_matrix": {"labels": [str(c) for c in classes], "matrix": matrix},
        "latency": latency,
    }


def save_confusion_matrix_png(matrix, labels, path: str) -> str:
    """Render the confusion matrix to a PNG file (best effort)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        return ""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    figure, axes = plt.subplots(figsize=(max(6, len(labels) * 0.8), max(5, len(labels) * 0.8)))
    image = axes.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axes)
    axes.set_xticks(range(len(labels)))
    axes.set_yticks(range(len(labels)))
    axes.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes.set_yticklabels(labels, fontsize=8)
    axes.set_xlabel("Predicted label")
    axes.set_ylabel("True label")
    axes.set_title("Confusion matrix")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def write_report(report: dict, path: str) -> str:
    """Write an evaluation report to disk as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    report = dict(report)
    report.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    return path


def schema_for(hand_count: int, order: str = "handedness") -> dict:
    return FeatureSchema(hand_count=hand_count, order=order).to_dict()
