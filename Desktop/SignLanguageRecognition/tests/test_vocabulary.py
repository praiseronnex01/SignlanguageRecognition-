"""Vocabulary management tests."""

from __future__ import annotations

from app.models.gesture import Gesture


def test_create_gesture_as_administrator(client, login_admin):
    response = client.post(
        "/vocabulary/api/vocabulary",
        json={
            "label": "greeting",
            "english_meaning": "Greeting",
            "category": "Basic Conversation",
            "description": "A salutation.",
            "hand_count": 1,
        },
    )
    assert response.status_code == 201
    gesture = response.get_json()["gesture"]
    assert gesture["label"] == "GREETING"
    assert gesture["is_validated"] is False
    assert gesture["is_active"] is True
    assert gesture["hand_count"] == 1


def test_a_new_gesture_is_never_validated_automatically(client, login_admin):
    response = client.post(
        "/vocabulary/api/vocabulary",
        json={
            "label": "AUTO_VALIDATED",
            "english_meaning": "Should not be validated",
            "is_validated": True,
        },
    )
    assert response.status_code == 201
    assert response.get_json()["gesture"]["is_validated"] is False


def test_duplicate_labels_are_rejected(client, login_admin, gesture):
    response = client.post(
        "/vocabulary/api/vocabulary",
        json={"label": gesture.label, "english_meaning": "Duplicate"},
    )
    assert response.status_code == 409


def test_missing_english_meaning_is_rejected(client, login_admin):
    response = client.post(
        "/vocabulary/api/vocabulary", json={"label": "NO_MEANING"}
    )
    assert response.status_code == 400
    assert "english_meaning" in response.get_json()["errors"]


def test_list_and_search_gestures(client, login_admin, gesture):
    response = client.get("/vocabulary/api/vocabulary")
    assert response.status_code == 200
    body = response.get_json()
    assert body["success"] is True
    assert any(entry["id"] == gesture.id for entry in body["gestures"])

    response = client.get("/vocabulary/api/vocabulary?search=TEST_SIGN")
    body = response.get_json()
    assert body["total"] == 1
    assert body["gestures"][0]["label"] == "TEST_SIGN"

    response = client.get("/vocabulary/api/vocabulary?search=does-not-exist")
    assert response.get_json()["total"] == 0


def test_category_and_hand_count_filters(client, login_admin, gesture, two_hand_gesture):
    response = client.get("/vocabulary/api/vocabulary?category=Testing")
    assert response.get_json()["total"] == 2

    response = client.get("/vocabulary/api/vocabulary?hand_count=2")
    body = response.get_json()
    assert body["total"] == 1
    assert body["gestures"][0]["label"] == "TEST_TWO_HAND"


def test_retrieve_a_single_gesture(client, login_admin, gesture):
    response = client.get(f"/vocabulary/api/vocabulary/{gesture.id}")
    assert response.status_code == 200
    assert response.get_json()["gesture"]["label"] == "TEST_SIGN"

    response = client.get("/vocabulary/api/vocabulary/999999")
    assert response.status_code == 404


def test_update_a_gesture(client, login_admin, gesture):
    response = client.put(
        f"/vocabulary/api/vocabulary/{gesture.id}",
        json={
            "english_meaning": "Updated meaning",
            "description": "Updated description",
            "category": "Updated category",
        },
    )
    assert response.status_code == 200
    body = response.get_json()["gesture"]
    assert body["english_meaning"] == "Updated meaning"
    assert body["category"] == "Updated category"


def test_validation_is_an_explicit_decision(client, login_admin, unvalidated_gesture):
    assert unvalidated_gesture.is_validated is False

    response = client.put(
        f"/vocabulary/api/vocabulary/{unvalidated_gesture.id}", json={"is_validated": True}
    )
    assert response.status_code == 200
    body = response.get_json()["gesture"]
    assert body["is_validated"] is True
    assert body["validation_status"] == "validated"
    assert body["recognition_available"] is True

    response = client.put(
        f"/vocabulary/api/vocabulary/{unvalidated_gesture.id}", json={"is_validated": False}
    )
    assert response.get_json()["gesture"]["is_validated"] is False


def test_deactivate_and_reactivate(client, login_admin, gesture):
    response = client.delete(f"/vocabulary/api/vocabulary/{gesture.id}")
    assert response.status_code == 200
    assert response.get_json()["gesture"]["is_active"] is False

    response = client.put(
        f"/vocabulary/api/vocabulary/{gesture.id}", json={"is_active": True}
    )
    assert response.get_json()["gesture"]["is_active"] is True


def test_permanent_delete(client, login_admin, gesture):
    gesture_id = gesture.id
    response = client.delete(
        f"/vocabulary/api/vocabulary/{gesture_id}", json={"permanent": True}
    )
    assert response.status_code == 200
    with client.application.app_context():
        from app.extensions import db

        assert db.session.get(Gesture, gesture_id) is None


def test_standard_users_cannot_change_the_vocabulary(client, login_user, gesture):
    response = client.post(
        "/vocabulary/api/vocabulary",
        json={"label": "NOPE", "english_meaning": "Nope"},
    )
    assert response.status_code == 403

    response = client.put(
        f"/vocabulary/api/vocabulary/{gesture.id}", json={"english_meaning": "Changed"}
    )
    assert response.status_code == 403

    response = client.delete(f"/vocabulary/api/vocabulary/{gesture.id}")
    assert response.status_code == 403
