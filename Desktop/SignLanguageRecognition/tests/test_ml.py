"""Machine-learning pipeline tests.

The datasets used here are **synthetic**: they are generated from the
parametric hand model in ``tests/conftest.py``.  They verify that the
pipeline runs, that the artefacts are written and reloaded correctly and that
the feature representation is consistent.  They are **not** evidence of
Malawi Sign Language recognition performance.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np
import pytest

from ml.evaluate import select_rejection_threshold
from ml.feature_engineering import (
    ONE_HAND_LANDMARK_DIMS,
    TWO_HAND_TOTAL_DIMS,
    FeatureError,
    build_feature_vector,
    feature_dimensions,
    normalize_landmarks,
    order_hands,
    validate_feature_vector,
)
from ml.preprocess import DatasetError, load_dataset, participant_split, validate_bundle
from tests.conftest import HAND_SHAPES, make_hand_observation

CLASSES = {
    "SIGN_A": "open",
    "SIGN_B": "fist",
    "SIGN_C": "point",
}
PARTICIPANTS = [f"P{i:02d}" for i in range(6)]


def build_feature_rows(classes=None, participants=None, per_class=12, noise=0.01):
    """Return ``(rows, y, groups)`` built from the synthetic hand model."""
    classes = classes or CLASSES
    participants = participants or PARTICIPANTS
    rows = []
    labels = []
    groups = []
    rng = np.random.default_rng(7)
    for label, shape in classes.items():
        for index in range(per_class):
            participant = participants[index % len(participants)]
            observation = make_hand_observation(
                shape=shape,
                handedness="Right",
                rotation=float(rng.normal(0, 0.06)),
                scale=1.0 + float(rng.normal(0, 0.04)),
                offset=(
                    0.5 + float(rng.normal(0, 0.02)),
                    0.5 + float(rng.normal(0, 0.02)),
                    float(rng.normal(0, 0.01)),
                ),
            )
            landmarks = [
                [value + float(rng.normal(0, noise)) for value in point]
                for point in observation.landmarks
            ]
            observation.landmarks = landmarks
            vector = build_feature_vector([observation], hand_count=1)
            rows.append(vector)
            labels.append(label)
            groups.append(participant)
    return np.asarray(rows), np.asarray(labels), np.asarray(groups)


def write_dataset_csv(path, rows, labels, groups, hand_count=1):
    columns = [f"f{index}" for index in range(feature_dimensions(hand_count))]
    header = (
        [
            "sample_id",
            "gesture_id",
            "gesture_label",
            "english_meaning",
            "participant_code",
            "session_code",
            "hand_count",
            "handedness",
            "feature_version",
            "quality_score",
            "created_at",
        ]
        + columns
    )
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for index, (vector, label, group) in enumerate(zip(rows, labels, groups)):
            row = {
                "sample_id": index + 1,
                "gesture_id": index + 1,
                "gesture_label": label,
                "english_meaning": label.replace("_", " ").title(),
                "participant_code": group,
                "session_code": "SES-TEST",
                "hand_count": hand_count,
                "handedness": "Right",
                "feature_version": "v1.0",
                "quality_score": 1.0,
                "created_at": "2026-01-01T00:00:00+00:00",
            }
            for position, value in enumerate(vector):
                row[columns[position]] = f"{float(value):.6f}"
            writer.writerow(row)
    return path


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def test_one_hand_features_have_63_dimensions():
    vector = build_feature_vector(
        [make_hand_observation(shape="open")], hand_count=1
    )
    assert vector.size == ONE_HAND_LANDMARK_DIMS == 63
    valid, reason = validate_feature_vector(vector, hand_count=1)
    assert valid, reason


def test_two_hand_features_have_128_dimensions_with_a_mask():
    vector = build_feature_vector(
        [
            make_hand_observation(shape="open", handedness="Left"),
            make_hand_observation(shape="fist", handedness="Right"),
        ],
        hand_count=2,
    )
    assert vector.size == TWO_HAND_TOTAL_DIMS == 128
    # The last two entries are the presence flags for the two hand slots.
    assert vector[-2] == 1.0
    assert vector[-1] == 1.0


def test_a_missing_hand_is_masked():
    vector = build_feature_vector(
        [make_hand_observation(shape="open", handedness="Right")], hand_count=2
    )
    assert vector.size == TWO_HAND_TOTAL_DIMS
    # Only the right hand was detected, so the left slot is empty.
    assert vector[-2] == 0.0
    assert vector[-1] == 1.0


def test_normalisation_places_the_wrist_at_the_origin():
    landmarks = make_hand_observation(shape="open").landmarks
    normalized = normalize_landmarks(landmarks)
    assert normalized[0] == [0.0, 0.0, 0.0]


def test_normalisation_is_scale_invariant():
    small = make_hand_observation(shape="open", scale=0.5).landmarks
    large = make_hand_observation(shape="open", scale=2.0).landmarks
    assert np.allclose(
        np.asarray(normalize_landmarks(small)), np.asarray(normalize_landmarks(large))
    )


def test_the_two_hand_order_is_stable():
    left = make_hand_observation(shape="open", handedness="Left")
    right = make_hand_observation(shape="fist", handedness="Right")
    forward = order_hands([right, left])
    backward = order_hands([left, right])
    assert [hand.handedness for hand in forward] == ["Left", "Right"]
    assert [hand.handedness for hand in backward] == ["Left", "Right"]


def test_invalid_feature_vectors_are_rejected():
    valid, reason = validate_feature_vector([0.0] * 10, hand_count=1)
    assert valid is False
    assert "63" in reason

    vector = np.ones(63)
    vector[3] = np.nan
    valid, reason = validate_feature_vector(vector, hand_count=1)
    assert valid is False

    valid, reason = validate_feature_vector(np.zeros(128), hand_count=2)
    assert valid is False
    assert "present" in reason


def test_an_unsupported_hand_count_raises():
    with pytest.raises(FeatureError):
        build_feature_vector([make_hand_observation()], hand_count=3)
    with pytest.raises(FeatureError):
        feature_dimensions(5)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def test_the_split_is_participant_independent():
    rows, labels, groups = build_feature_rows()
    train, validation, test, info = participant_split(groups, 0.7, 0.15, random_state=42)

    train_groups = set(groups[train])
    validation_groups = set(groups[validation])
    test_groups = set(groups[test])

    assert not train_groups & validation_groups
    assert not train_groups & test_groups
    assert not validation_groups & test_groups
    assert info["train_participants"]
    assert info["test_participants"]


def test_the_split_is_deterministic():
    rows, labels, groups = build_feature_rows()
    first = participant_split(groups, 0.7, 0.15, random_state=42)[3]
    second = participant_split(groups, 0.7, 0.15, random_state=42)[3]
    assert first == second


def test_a_single_participant_cannot_be_split():
    with pytest.raises(DatasetError):
        participant_split(np.array(["P1"] * 10), 0.7, 0.15, random_state=42)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def test_load_and_validate_a_dataset(tmp_path):
    rows, labels, groups = build_feature_rows()
    path = write_dataset_csv(tmp_path / "dataset_v1.csv", rows, labels, groups)

    bundle = load_dataset(str(path))
    assert bundle.sample_count == len(labels)
    assert bundle.feature_count == 63
    assert set(bundle.classes) == set(CLASSES)
    assert len(bundle.participants) == len(PARTICIPANTS)

    report = validate_bundle(bundle)
    assert report["samples"] == len(labels)
    assert report["classes"] == len(CLASSES)


def test_loading_a_missing_dataset_raises(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(str(tmp_path / "missing.csv"))


def test_a_dataset_with_one_class_is_rejected(tmp_path):
    rows, labels, groups = build_feature_rows()
    path = write_dataset_csv(tmp_path / "dataset_v1.csv", rows[:12], labels[:12], groups[:12])
    with pytest.raises(DatasetError):
        load_dataset(str(path))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def test_training_produces_saved_artifacts_and_metrics(app, tmp_path):
    from ml.train import train_single_schema

    rows, labels, groups = build_feature_rows(per_class=20)
    path = write_dataset_csv(tmp_path / "dataset_v1.csv", rows, labels, groups)

    result = train_single_schema(
        dataset_path=str(path),
        hand_count=1,
        algorithm="random_forest",
        parameters={"n_estimators": 40},
        tune=False,
        random_state=42,
        train_ratio=0.6,
        validation_ratio=0.2,
    )

    assert result["metrics"]["accuracy"] > 0.5
    assert 0.0 <= result["metrics"]["precision_macro"] <= 1.0
    assert 0.0 <= result["metrics"]["f1_macro"] <= 1.0
    assert result["schema"]["total_dims"] == 63
    assert len(result["classes"]) == len(CLASSES)
    assert result["latency"]["mean_ms"] > 0
    assert result["confusion_matrix"]["labels"] == sorted(CLASSES)
    assert result["rejection_threshold"] > 0

    # The test participants were never used for training or tuning.
    split = result["split"]
    assert not set(split["train_participants"]) & set(split["test_participants"])


def test_the_svm_and_knn_models_can_be_trained(app, tmp_path):
    from ml.train import train_single_schema

    rows, labels, groups = build_feature_rows(per_class=16)
    path = write_dataset_csv(tmp_path / "dataset_v1.csv", rows, labels, groups)

    for algorithm in ("svm", "knn"):
        result = train_single_schema(
            dataset_path=str(path),
            hand_count=1,
            algorithm=algorithm,
            parameters={},
            tune=False,
            random_state=42,
            train_ratio=0.6,
            validation_ratio=0.2,
        )
        assert result["metrics"]["accuracy"] > 0.5
        assert result["algorithm"] == algorithm


def test_a_model_can_be_saved_and_reloaded(app, tmp_path):
    import joblib

    from app.services import model_service
    from ml.train import train_single_schema

    rows, labels, groups = build_feature_rows(per_class=16)
    path = write_dataset_csv(tmp_path / "dataset_v1.csv", rows, labels, groups)

    result = train_single_schema(
        dataset_path=str(path),
        hand_count=1,
        algorithm="random_forest",
        parameters={"n_estimators": 30},
        tune=False,
        random_state=42,
        train_ratio=0.6,
        validation_ratio=0.2,
    )

    app.config["MODELS_DIR"] = str(tmp_path / "models")
    app.config["ENCODERS_DIR"] = str(tmp_path / "encoders")
    os.makedirs(app.config["MODELS_DIR"], exist_ok=True)
    os.makedirs(app.config["ENCODERS_DIR"], exist_ok=True)

    model_path, encoder_path = model_service.save_artifacts(
        pipeline=result["pipeline"],
        encoder=result["encoder"],
        schema=result["schema"],
        metrics=result["metrics"],
        algorithm="random_forest",
        version="9.9.9",
        rejection_threshold=result["rejection_threshold"],
    )
    assert os.path.isfile(model_path)
    assert os.path.isfile(encoder_path)

    reloaded = model_service.load_artifacts(
        model_path, encoder_path, (app.config["MODELS_DIR"], app.config["ENCODERS_DIR"])
    )
    assert reloaded.version == "9.9.9"
    assert reloaded.hand_count == 1
    assert reloaded.expected_dims == 63
    assert sorted(reloaded.classes) == sorted(CLASSES)

    sample = rows[0]
    prediction = reloaded.pipeline.predict(np.asarray([sample]))
    assert prediction[0] in CLASSES


def test_models_are_not_loaded_from_outside_the_artifact_directories(app, tmp_path):
    from app.services import model_service

    outside = tmp_path / "evil.joblib"
    outside.write_bytes(b"not a model")
    with pytest.raises(model_service.ModelArtifactError):
        model_service.load_artifacts(
            str(outside),
            str(outside),
            (str(tmp_path / "models"), str(tmp_path / "encoders")),
        )


def test_label_consistency_between_the_encoder_and_the_model(app, tmp_path):
    from ml.train import train_single_schema

    rows, labels, groups = build_feature_rows(per_class=16)
    path = write_dataset_csv(tmp_path / "dataset_v1.csv", rows, labels, groups)
    result = train_single_schema(
        dataset_path=str(path),
        hand_count=1,
        algorithm="random_forest",
        parameters={"n_estimators": 20},
        tune=False,
        random_state=42,
        train_ratio=0.6,
        validation_ratio=0.2,
    )
    encoder_classes = [str(value) for value in result["encoder"].classes_]
    pipeline_classes = [str(value) for value in result["pipeline"].classes_]
    assert encoder_classes == pipeline_classes
    assert sorted(encoder_classes) == sorted(CLASSES)


def test_the_rejection_threshold_limits_false_acceptance():
    rng = np.random.default_rng(3)
    y_true = np.array(["A"] * 50 + ["B"] * 50)
    # A well separated but imperfect prediction set.
    probabilities = np.zeros((100, 2))
    for index in range(100):
        if index < 50:
            probabilities[index] = [0.9, 0.1]
        else:
            probabilities[index] = [0.55, 0.45]

    threshold, info = select_rejection_threshold(y_true, probabilities, ["A", "B"])
    assert 0.0 < threshold <= 1.0
    assert "true_acceptance" in info
    assert "false_acceptance" in info


def test_the_evaluation_report_is_written(app, tmp_path):
    from ml.evaluate import write_report

    report = {"model_version": "1.0.0", "metrics": {"accuracy": 0.9}}
    path = write_report(report, str(tmp_path / "evaluation_1.0.0.json"))
    assert os.path.isfile(path)
    with open(path, encoding="utf-8") as handle:
        assert json.load(handle)["model_version"] == "1.0.0"
