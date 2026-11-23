"""Authentication, authorisation and session tests."""

from __future__ import annotations

import pytest

from app.models.user import User


def test_register_creates_a_hashed_account(client, app):
    response = client.post(
        "/auth/api/auth/register",
        json={
            "full_name": "New User",
            "email": "new.user@test.local",
            "password": "Str0ngPass",
            "confirm_password": "Str0ngPass",
        },
    )
    assert response.status_code == 201
    body = response.get_json()
    assert body["success"] is True
    assert body["user"]["role"] == User.ROLE_USER

    with app.app_context():
        user = User.query.filter_by(email="new.user@test.local").first()
        assert user is not None
        assert user.password_hash != "Str0ngPass"
        assert "Str0ngPass" not in user.password_hash
        assert user.check_password("Str0ngPass") is True
        assert user.check_password("wrong") is False


def test_register_rejects_a_duplicate_email(client, standard_user):
    response = client.post(
        "/auth/api/auth/register",
        json={
            "full_name": "Duplicate",
            "email": standard_user.email,
            "password": "Str0ngPass",
            "confirm_password": "Str0ngPass",
        },
    )
    assert response.status_code == 409
    assert response.get_json()["success"] is False


def test_register_validates_its_input(client):
    response = client.post(
        "/auth/api/auth/register",
        json={
            "full_name": "X",
            "email": "not-an-email",
            "password": "short",
            "confirm_password": "different",
        },
    )
    assert response.status_code == 400
    errors = response.get_json()["errors"]
    assert "email" in errors
    assert "password" in errors
    assert "confirm_password" in errors


def test_login_and_logout(client, standard_user):
    response = client.post(
        "/auth/api/auth/login",
        json={"email": standard_user.email, "password": "User12345"},
    )
    assert response.status_code == 200
    assert response.get_json()["user"]["email"] == standard_user.email

    response = client.get("/auth/api/auth/me")
    assert response.status_code == 200

    response = client.post("/auth/api/auth/logout")
    assert response.status_code == 200

    response = client.get("/auth/api/auth/me")
    assert response.status_code == 401


def test_login_with_invalid_credentials(client, standard_user):
    response = client.post(
        "/auth/api/auth/login",
        json={"email": standard_user.email, "password": "nope"},
    )
    assert response.status_code == 401
    assert response.get_json()["error"]


def test_login_rejects_an_inactive_account(client, app, standard_user):
    standard_user.is_active = False
    from app.extensions import db

    db.session.commit()

    response = client.post(
        "/auth/api/auth/login",
        json={"email": standard_user.email, "password": "User12345"},
    )
    assert response.status_code == 403


def test_protected_pages_redirect_anonymous_users(client):
    for path in ("/dashboard", "/recognition", "/vocabulary", "/history"):
        response = client.get(path)
        assert response.status_code in (302, 401)


@pytest.mark.parametrize(
    "path",
    [
        "/dataset",
        "/training",
        "/evaluation",
        "/admin",
        "/dataset/api/dataset/stats",
        "/training/api/training/runs",
        "/admin/api/admin/users",
    ],
)
def test_administrator_routes_are_protected(client, login_user, path):
    response = client.get(path)
    assert response.status_code == 403


@pytest.mark.parametrize(
    "path",
    ["/dataset", "/training", "/evaluation", "/admin"],
)
def test_administrator_routes_are_available_to_administrators(
    client, login_admin, path
):
    response = client.get(path)
    assert response.status_code == 200


def test_password_hash_is_salted(app, admin_user):
    other = User(full_name="Second", email="second@test.local")
    other.set_password("Admin12345")
    assert other.password_hash != admin_user.password_hash


def test_profile_can_be_updated(client, login_user):
    response = client.post(
        "/auth/profile",
        data={"full_name": "Renamed User"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Renamed User" in response.data


def test_csrf_protection_is_active(app):
    """The testing configuration disables CSRF; this asserts the toggle."""
    assert app.config["WTF_CSRF_ENABLED"] is False
    app.config["WTF_CSRF_ENABLED"] = True
    try:
        with app.test_client() as client:
            response = client.post(
                "/auth/api/auth/login",
                json={"email": "someone@test.local", "password": "whatever"},
            )
            assert response.status_code == 400
    finally:
        app.config["WTF_CSRF_ENABLED"] = False
