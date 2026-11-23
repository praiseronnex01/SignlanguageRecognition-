"""Recognition pipeline and API tests.

The hand detector is replaced by a stub so that the tests can exercise every
branch of the recognition logic deterministically.  The images used are
synthetic; no real participant data is involved.
"""

from __future__ import annotations

import io
import os

import numpy as np
import pytest

from app.services.gesture_predictor import predict_from_detection
from app.services.model_service import ModelArtifactError
from tests.conftest import make_detection, make_hand_observation


class StubDetector:
    """Stands in for :class:`app.services.hand_detector.HandDetector`."""

    hands_to_return = []
    error = None

    def __init__(self, *args, **kwargs):
        pass

    def detect(self, frame, source="image"):
        if StubDetector.error:
            raise StubDetector.error
        return make_detection(StubDetector.hands_to_return)

    def detect_from_bytes(self, payload, source="image"):
        return self.detect(None, source)


@pytest.fixture()
def stub_detector(monkeypatch):
    StubDetector.hands_to_return = []
    StubDetector.error = None
    import app.services.gesture_predictor as predictor

    monkeypatch.setattr(predictor, "HandDetector", StubDetector, raising=False)
    monkeypatch.setattr(
        "app.services.hand_detector.HandDetector", StubDetector, raising=False
    )
    yield StubDetector
    StubDetector.hands_to_return = []
    StubDetector.error = None


# ---------------------------------------------------------------------------
# Feature extraction through the service layer
# ---------------------------------------------------------------------------
def test_features_are_extracted_for_one_hand(app):
    from app.services import feature_extractor

    detection = make_detection([make_hand_observation(shape="open")])
    vector = feature_extractor.extract_features(detection, hand_count=1)
    assert vector.size == 63


def test_features_are_extracted_for_two_hands(app):
    from app.services import feature_extractor

    detection = make_detection(
        [
            make_hand_observation(shape="open", handedness="Left"),
            make_hand_observation(shape="fist", handedness="Right"),
        ]
    )
    vector = feature_extractor.extract_features(detection, hand_count=2)
    assert vector.size == 128


def test_features_fail_when_no_hand_was_detected(app):
    from app.services import feature_extractor

    with pytest.raises(feature_extractor.FeatureExtractionError):
        feature_extractor.extract_features(make_detection([]), hand_count=1)


def test_the_stored_payload_round_trips(app):
    from app.services import feature_extractor

    detection = make_detection([make_hand_observation(shape="point")])
    payload = feature_extractor.build_payload(detection)
    hands = feature_extractor.payload_to_hands(payload)
    assert len(hands) == 1
    assert len(hands[0].landmarks) == 21


# ---------------------------------------------------------------------------
# Prediction outcomes
# ---------------------------------------------------------------------------
def test_no_hand_is_reported_as_not_detected(app):
    outcome = predict_from_detection(make_detection([]))
    assert outcome.recognized is False
    assert outcome.hand_detected is False
    assert outcome.message == "No hand detected"
    assert outcome.gesture is None
    assert outcome.translation is None


def test_no_model_means_no_prediction(app):
    outcome = predict_from_detection(
        make_detection([make_hand_observation(shape="open")])
    )
    assert outcome.recognized is False
    assert outcome.hand_detected is True
    assert outcome.gesture is None
    assert "model" in (outcome.message or "").lower()


def test_a_published_model_produces_a_prediction(app, gesture, tmp_path, monkeypatch):
    """Train a tiny model on synthetic data and publish it."""
    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    from app.extensions import db
    from app.models.model_version import ModelVersion
    from app.services import feature_extractor, model_service

    labels = ["OPEN", "FIST"]
    encoder = LabelEncoder().fit(labels)
    pipeline = RandomForestClassifier(n_estimators=20, random_state=42)

    vectors = []
    targets = []
    for label, shape in (("OPEN", "open"), ("FIST", "fist")):
        for index in range(12):
            observation = make_hand_observation(
                shape=shape,
                rotation=float(np.random.default_rng(index).normal(0, 0.05)),
            )
            detection = make_detection([observation])
            vectors.append(feature_extractor.extract_features(detection, hand_count=1))
            targets.append(label)
    pipeline.fit(np.asarray(vectors), encoder.transform(targets))

    models_dir = tmp_path / "models"
    encoders_dir = tmp_path / "encoders"
    models_dir.mkdir()
    encoders_dir.mkdir()
    app.config["MODELS_DIR"] = str(models_dir)
    app.config["ENCODERS_DIR"] = str(encoders_dir)

    schema = feature_extractor.schema_for(1)
    model_path, encoder_path = model_service.save_artifacts(
        pipeline=pipeline,
        encoder=encoder,
        schema=schema,
        metrics={"accuracy": 1.0},
        algorithm="random_forest",
        version="1.2.3",
        rejection_threshold=0.5,
    )

    model_version = ModelVersion(
        version="1.2.3",
        algorithm="random_forest",
        model_path=model_path,
        encoder_path=encoder_path,
        feature_schema=__import__("json").dumps(schema),
        accuracy=1.0,
        precision=1.0,
        recall=1.0,
        f1_score=1.0,
        status=ModelVersion.STATUS_PUBLISHED,
        rejection_threshold=0.5,
        feature_version="v1.0",
    )
    db.session.add(model_version)
    db.session.commit()
    model_service.clear_cache()

    outcome = predict_from_detection(
        make_detection([make_hand_observation(shape="open")])
    )
    assert outcome.recognized is True
    assert outcome.gesture in labels
    assert outcome.confidence > 0
    assert outcome.model_version == "1.2.3"
    assert outcome.hand_detected is True


def test_a_low_confidence_prediction_is_rejected(app, tmp_path):
    """A model that is unsure must not report a gesture."""
    import json

    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    from app.extensions import db
    from app.models.model_version import ModelVersion
    from app.services import feature_extractor, model_service

    encoder = LabelEncoder().fit(["OPEN", "FIST"])
    # A model trained on random data produces an almost uniform distribution.
    pipeline = RandomForestClassifier(n_estimators=10, random_state=1)
    pipeline.fit(
        np.random.default_rng(0).normal(size=(40, 63)),
        encoder.transform(["OPEN"] * 20 + ["FIST"] * 20),
    )

    models_dir = tmp_path / "models"
    encoders_dir = tmp_path / "encoders"
    models_dir.mkdir()
    encoders_dir.mkdir()
    app.config["MODELS_DIR"] = str(models_dir)
    app.config["ENCODERS_DIR"] = str(encoders_dir)

    schema = feature_extractor.schema_for(1)
    model_path, encoder_path = model_service.save_artifacts(
        pipeline=pipeline,
        encoder=encoder,
        schema=schema,
        metrics={"accuracy": 0.0},
        algorithm="random_forest",
        version="0.0.1",
        rejection_threshold=0.99,
    )
    db.session.add(
        ModelVersion(
            version="0.0.1",
            algorithm="random_forest",
            model_path=model_path,
            encoder_path=encoder_path,
            feature_schema=json.dumps(schema),
            status=ModelVersion.STATUS_PUBLISHED,
            rejection_threshold=0.99,
            feature_version="v1.0",
        )
    )
    db.session.commit()
    model_service.clear_cache()

    outcome = predict_from_detection(
        make_detection([make_hand_observation(shape="open")])
    )
    assert outcome.recognized is False
    assert outcome.translation == "Gesture not recognised"
    assert outcome.confidence < 0.99


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------
def test_predict_with_a_valid_image(client, login_user, stub_detector, sample_jpeg):
    stub_detector.hands_to_return = [make_hand_observation(shape="open")]
    response = client.post(
        "/recognition/api/predict",
        data={"file": (io.BytesIO(sample_jpeg), "frame.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["success"] is True
    assert "recognized" in body
    assert "hand_detected" in body


def test_predict_with_an_invalid_image(client, login_user, stub_detector):
    response = client.post(
        "/recognition/api/predict",
        data={"file": (io.BytesIO(b"this is not an image"), "frame.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert response.get_json()["success"] is False


def test_predict_without_any_image(client, login_user):
    response = client.post("/recognition/api/predict", json={})
    assert response.status_code == 400


def test_predict_when_the_detector_is_unavailable(
    client, login_user, stub_detector, sample_jpeg
):
    from app.services.hand_detector import HandDetectorError

    stub_detector.error = HandDetectorError("The MediaPipe model asset is missing.")
    response = client.post(
        "/recognition/api/predict",
        data={"file": (io.BytesIO(sample_jpeg), "frame.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 503
    assert response.get_json()["success"] is False


def test_predict_requires_a_signed_in_user(client, sample_jpeg):
    response = client.post(
        "/recognition/api/predict",
        data={"file": (io.BytesIO(sample_jpeg), "frame.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code in (401, 302)


def test_recognition_status_reports_the_configuration(client, login_user):
    response = client.get("/recognition/api/status")
    assert response.status_code == 200
    body = response.get_json()
    assert body["model_available"] is False
    assert body["supported_gesture_count"] >= 0
    assert 0 < body["confidence_threshold"] <= 1
