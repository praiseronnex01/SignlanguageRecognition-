"""Dataset collection, validation and export tests."""

from __future__ import annotations

import io
import json

import pytest

from app.extensions import db
from app.models.collection_session import CollectionSession
from app.models.dataset_sample import DatasetSample
from app.models.participant import Participant
from app.services import dataset_service
from tests.conftest import make_detection, make_hand_observation


def _payload(handedness="Right", shape="open"):
    from app.services import feature_extractor

    detection = type(
        "Detection",
        (),
        {"hands": [make_hand_observation(shape=shape, handedness=handedness)]},
    )()
    return feature_extractor.build_payload(detection)


def _two_hand_payload():
    from app.services import feature_extractor

    detection = type(
        "Detection",
        (),
        {
            "hands": [
                make_hand_observation(shape="open", handedness="Left"),
                make_hand_observation(shape="fist", handedness="Right"),
            ]
        },
    )()
    return feature_extractor.build_payload(detection)


def test_participant_is_created_with_the_expected_consent_state(app):
    participant = dataset_service.get_or_create_participant("P001")
    assert participant.participant_code == "P001"
    assert participant.consent_status == Participant.CONSENT_PENDING
    assert participant.has_consent is False


def test_consent_can_be_recorded(app):
    participant = dataset_service.get_or_create_participant("P002")
    dataset_service.set_consent(participant, Participant.CONSENT_GRANTED)
    assert participant.has_consent is True


def test_a_session_requires_consent(app):
    participant = dataset_service.get_or_create_participant("P003")
    with pytest.raises(dataset_service.DatasetError):
        dataset_service.create_session(participant)


def test_a_session_is_created_once_consent_is_given(app):
    participant = dataset_service.get_or_create_participant("P004")
    dataset_service.set_consent(participant, Participant.CONSENT_GRANTED)
    session = dataset_service.create_session(participant)
    assert session.status == CollectionSession.STATUS_ACTIVE
    assert session.session_code.startswith("SES-")

    again = dataset_service.create_session(participant)
    assert again.id == session.id


def test_store_a_valid_sample(app, gesture):
    participant = dataset_service.get_or_create_participant("P010")
    dataset_service.set_consent(participant, Participant.CONSENT_GRANTED)
    session = dataset_service.create_session(participant)

    sample = dataset_service.store_sample(
        gesture=gesture,
        participant=participant,
        session=session,
        payload=_payload(),
        quality_score=0.9,
    )
    assert sample.id is not None
    assert sample.hand_count == 1
    assert sample.gesture_id == gesture.id
    assert sample.participant_id == participant.id
    assert sample.session_id == session.id
    assert sample.feature_version == "v1.0"

    stored = json.loads(sample.landmarks_json)
    assert len(stored["hands"]) == 1
    assert len(stored["hands"][0]["landmarks"]) == 21


def test_a_sample_with_the_wrong_number_of_hands_is_rejected(app, gesture):
    participant = dataset_service.get_or_create_participant("P011")
    dataset_service.set_consent(participant, Participant.CONSENT_GRANTED)
    session = dataset_service.create_session(participant)

    with pytest.raises(dataset_service.DatasetError):
        dataset_service.store_sample(
            gesture=gesture,
            participant=participant,
            session=session,
            payload=_two_hand_payload(),
        )


def test_a_two_hand_gesture_accepts_two_hands(app, two_hand_gesture):
    participant = dataset_service.get_or_create_participant("P012")
    dataset_service.set_consent(participant, Participant.CONSENT_GRANTED)
    session = dataset_service.create_session(participant)

    sample = dataset_service.store_sample(
        gesture=two_hand_gesture,
        participant=participant,
        session=session,
        payload=_two_hand_payload(),
    )
    assert sample.hand_count == 2


def test_missing_landmarks_are_rejected(app, gesture):
    participant = dataset_service.get_or_create_participant("P013")
    dataset_service.set_consent(participant, Participant.CONSENT_GRANTED)
    session = dataset_service.create_session(participant)

    broken = {"feature_version": "v1.0", "order": "handedness", "hands": [{"landmarks": []}]}
    with pytest.raises(dataset_service.DatasetError):
        dataset_service.store_sample(
            gesture=gesture,
            participant=participant,
            session=session,
            payload=broken,
        )


def test_a_sample_cannot_be_stored_without_consent(app, gesture):
    participant = dataset_service.get_or_create_participant("P014")
    session = CollectionSession(
        session_code="SES-FORCED", participant_id=participant.id, status="active"
    )
    db.session.add(session)
    db.session.commit()

    with pytest.raises(dataset_service.DatasetError):
        dataset_service.store_sample(
            gesture=gesture,
            participant=participant,
            session=session,
            payload=_payload(),
        )


def test_duplicates_are_detected(app, gesture):
    participant = dataset_service.get_or_create_participant("P015")
    dataset_service.set_consent(participant, Participant.CONSENT_GRANTED)
    session = dataset_service.create_session(participant)

    dataset_service.store_sample(
        gesture=gesture,
        participant=participant,
        session=session,
        payload=_payload(),
    )
    with pytest.raises(dataset_service.DatasetError):
        dataset_service.store_sample(
            gesture=gesture,
            participant=participant,
            session=session,
            payload=_payload(),
        )


def test_dataset_statistics_are_computed_from_the_database(app, gesture):
    participant = dataset_service.get_or_create_participant("P016")
    dataset_service.set_consent(participant, Participant.CONSENT_GRANTED)
    session = dataset_service.create_session(participant)
    dataset_service.store_sample(
        gesture=gesture,
        participant=participant,
        session=session,
        payload=_payload(),
    )

    stats = dataset_service.dataset_stats()
    assert stats["total_samples"] == 1
    assert stats["participant_count"] == 1
    assert stats["total_sessions"] == 1
    assert any(entry["label"] == gesture.label for entry in stats["per_gesture"])


def test_dataset_validation_reports_problems(app, gesture):
    participant = dataset_service.get_or_create_participant("P017")
    dataset_service.set_consent(participant, Participant.CONSENT_GRANTED)
    session = dataset_service.create_session(participant)
    dataset_service.store_sample(
        gesture=gesture,
        participant=participant,
        session=session,
        payload=_payload(),
    )

    report = dataset_service.validate_dataset()
    assert report["checked"] == 1
    assert report["invalid"] == 0

    broken = DatasetSample(
        gesture_id=gesture.id,
        participant_id=participant.id,
        session_id=session.id,
        landmarks_json='{"hands": [{"landmarks": [[0,0,0]]}]}',
        hand_count=1,
        feature_version="v1.0",
    )
    db.session.add(broken)
    db.session.commit()

    report = dataset_service.validate_dataset()
    assert report["invalid"] == 1


def test_export_produces_a_training_ready_csv(app, gesture, tmp_path):
    app.config["DATASET_DIR"] = str(tmp_path)

    participant = dataset_service.get_or_create_participant("P018")
    dataset_service.set_consent(participant, Participant.CONSENT_GRANTED)
    session = dataset_service.create_session(participant)
    dataset_service.store_sample(
        gesture=gesture,
        participant=participant,
        session=session,
        payload=_payload(),
    )

    metadata = dataset_service.export_dataset()
    assert metadata["sample_count"] == 1
    assert metadata["class_count"] == 1
    assert (tmp_path / "processed" / "dataset_v1.csv").exists()

    import csv

    with open(tmp_path / "processed" / "dataset_v1.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["gesture_label"] == gesture.label
    assert rows[0]["participant_code"] == "P018"
    assert rows[0]["hand_count"] == "1"
    assert float(rows[0]["f0"]) == 0.0


def test_export_refuses_when_there_is_nothing_to_export(app):
    with pytest.raises(dataset_service.DatasetError):
        dataset_service.export_dataset()


def test_delete_a_sample(app, gesture):
    participant = dataset_service.get_or_create_participant("P019")
    dataset_service.set_consent(participant, Participant.CONSENT_GRANTED)
    session = dataset_service.create_session(participant)
    sample = dataset_service.store_sample(
        gesture=gesture,
        participant=participant,
        session=session,
        payload=_payload(),
    )
    assert dataset_service.delete_sample(sample.id) is True
    assert dataset_service.delete_sample(sample.id) is False


def test_dataset_api_endpoints(client, login_admin, gesture):
    response = client.get("/dataset/api/dataset/stats")
    assert response.status_code == 200
    assert response.get_json()["success"] is True

    response = client.post(
        "/dataset/api/dataset/participants",
        json={"participant_code": "API01", "consent_status": "granted"},
    )
    assert response.status_code == 201

    response = client.post(
        "/dataset/api/dataset/sessions", json={"participant_code": "API01"}
    )
    assert response.status_code == 201

    response = client.get("/dataset/api/dataset/samples")
    assert response.status_code == 200
    assert "samples" in response.get_json()

    response = client.get("/dataset/api/dataset/validate")
    assert response.status_code == 200
    assert "report" in response.get_json()


def test_collection_response_includes_hand_landmarks(
    client, login_admin, gesture, sample_jpeg, monkeypatch
):
    class StubDatasetDetector:
        def __init__(self, *args, **kwargs):
            pass

        def detect(self, frame):
            return make_detection([make_hand_observation(shape="open")])

    monkeypatch.setattr(
        "app.blueprints.dataset.routes.HandDetector",
        StubDatasetDetector,
    )
    client.post(
        "/dataset/api/dataset/participants",
        json={"participant_code": "LANDMARK01", "consent_status": "granted"},
    )
    session = client.post(
        "/dataset/api/dataset/sessions",
        json={"participant_code": "LANDMARK01"},
    ).get_json()["session"]

    response = client.post(
        "/dataset/api/dataset/collect",
        data={
            "file": (io.BytesIO(sample_jpeg), "frame.jpg"),
            "gesture_id": gesture.id,
            "session_id": session["id"],
            "participant_code": "LANDMARK01",
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 201
    body = response.get_json()
    assert body["stored"] is True
    assert len(body["landmarks"]["hands"]) == 1
    assert len(body["landmarks"]["hands"][0]["landmarks"]) == 21
