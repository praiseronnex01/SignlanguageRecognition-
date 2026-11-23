"""Integration tests covering the full request flow."""

from __future__ import annotations

import io
import json

import pytest

from app.extensions import db
from app.models.recognition_history import RecognitionHistory
from app.services import feature_extractor, model_service
from tests.conftest import make_detection, make_hand_observation


@pytest.fixture()
def published_model(app, tmp_path):
    """Publish a small but genuine model trained on synthetic hand shapes."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    from app.models.model_version import ModelVersion

    labels = ["OPEN", "FIST"]
    encoder = LabelEncoder().fit(labels)
    pipeline = RandomForestClassifier(n_estimators=30, random_state=42)

    vectors = []
    targets = []
    for label, shape in (("OPEN", "open"), ("FIST", "fist")):
        for index in range(15):
            rng = np.random.default_rng(index)
            observation = make_hand_observation(
                shape=shape, rotation=float(rng.normal(0, 0.05))
            )
            vectors.append(
                feature_extractor.extract_features(
                    make_detection([observation]), hand_count=1
                )
            )
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
        metrics={"accuracy": 0.95, "f1_macro": 0.95},
        algorithm="random_forest",
        version="2.0.0",
        rejection_threshold=0.55,
    )

    row = ModelVersion(
        version="2.0.0",
        algorithm="random_forest",
        model_path=model_path,
        encoder_path=encoder_path,
        feature_schema=json.dumps(schema),
        accuracy=0.95,
        precision=0.95,
        recall=0.95,
        f1_score=0.95,
        status=ModelVersion.STATUS_PUBLISHED,
        rejection_threshold=0.55,
        feature_version="v1.0",
    )
    db.session.add(row)
    db.session.commit()
    model_service.clear_cache()
    return row


def test_the_prediction_endpoint_then_saves_to_history(
    client, login_user, published_model, gesture, monkeypatch, sample_jpeg
):
    from app.services import gesture_predictor
    from tests.test_recognition import StubDetector

    StubDetector.hands_to_return = [make_hand_observation(shape="open")]
    monkeypatch.setattr(
        "app.services.hand_detector.HandDetector", StubDetector, raising=False
    )

    response = client.post(
        "/recognition/api/predict",
        data={"file": (io.BytesIO(sample_jpeg), "frame.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["recognized"] is True
    assert body["gesture"] in {"OPEN", "FIST"}
    assert 0 < body["confidence"] <= 1
    assert body["model_version"] == "2.0.0"

    # The recognised label must exist in the vocabulary to be stored.
    label = body["gesture"]
    from app.models.gesture import Gesture

    if not Gesture.query.filter_by(label=label).first():
        db.session.add(
            Gesture(
                label=label,
                english_meaning=label.title(),
                is_validated=True,
                is_active=True,
                hand_count=1,
            )
        )
        db.session.commit()

    response = client.post(
        "/recognition/api/save",
        json={
            "gesture": label,
            "confidence": body["confidence"],
            "model_version": body["model_version"],
        },
    )
    assert response.status_code == 201
    entry = response.get_json()["entry"]
    assert entry["gesture_label"] == label
    assert entry["model_version"] == "2.0.0"

    response = client.get("/history/api/history")
    assert response.status_code == 200
    assert response.get_json()["total"] == 1


def test_history_is_private_to_the_signed_in_user(app, client, login_user, admin_user):
    from app.models.recognition_history import RecognitionHistory

    db.session.add(
        RecognitionHistory(
            user_id=admin_user.id,
            gesture_id=None,
            confidence=0.9,
            translation="Some other user's sign",
        )
    )
    db.session.commit()

    response = client.get("/history/api/history")
    assert response.status_code == 200
    assert response.get_json()["total"] == 0


def test_a_user_cannot_delete_another_users_history(
    app, client, login_user, admin_user
):
    from app.models.recognition_history import RecognitionHistory

    entry = RecognitionHistory(
        user_id=admin_user.id, gesture_id=None, confidence=0.8, translation="Admin sign"
    )
    db.session.add(entry)
    db.session.commit()

    response = client.delete(f"/history/api/history/{entry.id}")
    assert response.status_code == 403
    assert RecognitionHistory.query.count() == 1


def test_history_can_be_cleared_and_exported(app, client, login_user, gesture):
    from app.models.recognition_history import RecognitionHistory

    for index in range(3):
        db.session.add(
            RecognitionHistory(
                user_id=login_user.id,
                gesture_id=gesture.id,
                confidence=0.7 + index * 0.05,
                translation=gesture.english_meaning,
            )
        )
    db.session.commit()

    response = client.get("/history/api/history/export")
    assert response.status_code == 200
    assert "text/csv" in response.headers["Content-Type"]
    body = response.get_data(as_text=True)
    assert "gesture_label" in body
    assert gesture.label in body

    response = client.delete("/history/api/history")
    assert response.status_code == 200
    assert response.get_json()["deleted"] == 3
    assert RecognitionHistory.query.count() == 0


def test_history_search_and_pagination(app, client, login_user, gesture):
    from app.models.recognition_history import RecognitionHistory

    for index in range(7):
        db.session.add(
            RecognitionHistory(
                user_id=login_user.id,
                gesture_id=gesture.id,
                confidence=0.8,
                translation=gesture.english_meaning,
            )
        )
    db.session.commit()

    response = client.get("/history/api/history?per_page=3&page=1")
    body = response.get_json()
    assert body["total"] == 7
    assert len(body["entries"]) == 3
    assert body["pages"] == 3

    response = client.get("/history/api/history?search=TEST_SIGN")
    assert response.get_json()["total"] == 7

    response = client.get("/history/api/history?search=nothing-matches")
    assert response.get_json()["total"] == 0


def test_administrator_can_inspect_another_users_history(app, client, login_admin, standard_user):
    from app.models.recognition_history import RecognitionHistory

    db.session.add(
        RecognitionHistory(
            user_id=standard_user.id, gesture_id=None, confidence=0.6, translation="User sign"
        )
    )
    db.session.commit()

    response = client.get(f"/history/api/history?user_id={standard_user.id}")
    assert response.status_code == 200
    assert response.get_json()["total"] == 1


def test_the_dashboard_statistics_reflect_the_database(app, client, login_admin, gesture):
    from app.models.dataset_sample import DatasetSample
    from app.models.participant import Participant

    participant = Participant(participant_code="INT01", consent_status="granted")
    db.session.add(participant)
    db.session.commit()

    response = client.get("/api/dashboard/stats")
    assert response.status_code == 200
    stats = response.get_json()["stats"]

    assert stats["gestures"]["total"] >= 1
    assert stats["gestures"]["validated"] >= 1
    assert stats["users"]["total"] >= 1
    assert stats["participants"]["total"] >= 1
    assert stats["dataset"]["total_samples"] == 0
    assert stats["model"]["published"] is None
    assert stats["recognition"]["total"] == 0


def test_the_dashboard_page_renders(app, client, login_admin):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert b"Dashboard" in response.data


def test_administrator_can_change_a_user_role(app, client, login_admin, standard_user):
    response = client.put(
        f"/admin/api/admin/users/{standard_user.id}", json={"role": "admin"}
    )
    assert response.status_code == 200
    assert response.get_json()["user"]["role"] == "admin"

    response = client.put(
        f"/admin/api/admin/users/{standard_user.id}", json={"is_active": False}
    )
    assert response.status_code == 200
    assert response.get_json()["user"]["is_active"] is False


def test_an_administrator_cannot_demote_themselves(app, client, login_admin, admin_user):
    response = client.put(
        f"/admin/api/admin/users/{admin_user.id}", json={"role": "user"}
    )
    assert response.status_code == 400


def test_audit_logs_are_recorded(app, client, login_admin):
    client.post(
        "/vocabulary/api/vocabulary",
        json={"label": "AUDITED", "english_meaning": "Audited sign"},
    )
    response = client.get("/admin/api/admin/audit-logs")
    assert response.status_code == 200
    logs = response.get_json()["logs"]
    assert any(entry["action"] == "vocabulary.create" for entry in logs)


def test_the_landing_page_is_public(app, client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Malawi Sign Language" in response.data


def test_every_navigation_link_resolves(app, client, login_admin):
    links = [
        "/dashboard",
        "/recognition",
        "/vocabulary",
        "/history",
        "/dataset",
        "/training",
        "/evaluation",
        "/admin",
        "/auth/profile",
    ]
    for path in links:
        response = client.get(path)
        assert response.status_code == 200, f"{path} -> {response.status_code}"
