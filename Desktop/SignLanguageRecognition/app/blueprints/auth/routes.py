"""Authentication routes: pages and JSON API."""

from __future__ import annotations

from urllib.parse import urljoin, urlsplit

from flask import (
    Blueprint,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required, login_user, logout_user

from app.blueprints.auth.forms import LoginForm, ProfileForm, RegistrationForm
from app.extensions import db
from app.models.user import User
from app.utils import json_error, json_success, log_audit

bp = Blueprint("auth", __name__, url_prefix="/auth")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_safe_redirect(target: str | None) -> bool:
    """Only allow redirects that stay on this host."""
    if not target:
        return False
    parts = urlsplit(target)
    return not parts.netloc and not parts.scheme


def _normalise_email(email: str) -> str:
    return (email or "").strip().lower()


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@bp.route("/login", methods=["GET", "POST"])
def login_page():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.index"))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=_normalise_email(form.email.data)).first()
        if user is None or not user.check_password(form.password.data):
            flash("Invalid email address or password.", "error")
            log_audit("auth.login_failed", f"Failed sign-in attempt for {form.email.data}")
            return render_template("auth/login.html", form=form), 401
        if not user.is_active:
            flash("This account has been deactivated. Contact an administrator.", "error")
            return render_template("auth/login.html", form=form), 403

        login_user(user, remember=bool(form.remember.data))
        log_audit("auth.login", f"{user.email} signed in", user)
        flash("Welcome back.", "success")
        target = request.args.get("next")
        if _is_safe_redirect(target):
            return redirect(target)
        return redirect(url_for("dashboard.index"))

    return render_template("auth/login.html", form=form)


@bp.route("/register", methods=["GET", "POST"])
def register_page():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.index"))

    form = RegistrationForm()
    if form.validate_on_submit():
        if not form.accept_terms.data:
            flash("You must confirm the data handling statement.", "error")
            return render_template("auth/register.html", form=form), 400

        email = _normalise_email(form.email.data)
        if User.query.filter_by(email=email).first() is not None:
            flash("An account with that email address already exists.", "error")
            return render_template("auth/register.html", form=form), 409

        user = User(
            full_name=form.full_name.data.strip(),
            email=email,
            role=User.ROLE_USER,
        )
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        log_audit("auth.register", f"New account created for {email}", user)
        login_user(user)
        flash("Your account has been created.", "success")
        return redirect(url_for("dashboard.index"))

    return render_template("auth/register.html", form=form)


@bp.route("/logout", methods=["GET", "POST"])
@login_required
def logout():
    log_audit("auth.logout", f"{current_user.email} signed out", current_user)
    logout_user()
    flash("You have been signed out.", "success")
    return redirect(url_for("auth.login_page"))


@bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    form = ProfileForm(obj=current_user)
    if form.validate_on_submit():
        current_user.full_name = form.full_name.data.strip()
        db.session.commit()
        log_audit("profile.update", "Profile updated", current_user)
        flash("Your profile has been updated.", "success")
        return redirect(url_for("auth.profile"))
    return render_template("auth/profile.html", form=form)


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------
@bp.post("/api/auth/register")
def api_register():
    payload = request.get_json(silent=True) or request.form.to_dict()

    full_name = (payload.get("full_name") or "").strip()
    email = _normalise_email(payload.get("email"))
    password = payload.get("password") or ""
    confirm_password = payload.get("confirm_password") or password

    errors: dict = {}
    if len(full_name) < 2:
        errors["full_name"] = "Full name must be at least two characters."
    if not email or "@" not in email:
        errors["email"] = "A valid email address is required."
    if len(password) < 8:
        errors["password"] = "Password must be at least eight characters."
    if password != confirm_password:
        errors["confirm_password"] = "The passwords do not match."
    if errors:
        return json_error("Please correct the highlighted fields.", 400, errors=errors)

    if User.query.filter_by(email=email).first() is not None:
        return json_error("An account with that email address already exists.", 409)

    user = User(full_name=full_name, email=email, role=User.ROLE_USER)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    log_audit("auth.register", f"New account created for {email}", user)
    login_user(user)
    return json_success(user=user.to_dict()), 201


@bp.post("/api/auth/login")
def api_login():
    payload = request.get_json(silent=True) or request.form.to_dict()
    email = _normalise_email(payload.get("email"))
    password = payload.get("password") or ""

    if not email or not password:
        return json_error("Email address and password are required.", 400)

    user = User.query.filter_by(email=email).first()
    if user is None or not user.check_password(password):
        log_audit("auth.login_failed", f"Failed sign-in attempt for {email}")
        return json_error("Invalid email address or password.", 401)
    if not user.is_active:
        return json_error("This account has been deactivated.", 403)

    login_user(user, remember=bool(payload.get("remember")))
    log_audit("auth.login", f"{user.email} signed in", user)
    return json_success(user=user.to_dict())


@bp.post("/api/auth/logout")
def api_logout():
    if current_user.is_authenticated:
        log_audit("auth.logout", f"{current_user.email} signed out", current_user)
    logout_user()
    return json_success(message="Signed out.")


@bp.get("/api/auth/me")
def api_me():
    if not current_user.is_authenticated:
        return json_error("Authentication is required.", 401)
    data = current_user.to_dict()
    data["recognition_count"] = _recognition_count()
    return json_success(user=data)


def _recognition_count() -> int:
    from app.models.recognition_history import RecognitionHistory

    return RecognitionHistory.query.filter_by(user_id=current_user.id).count()


__all__ = ["bp", "jsonify", "urljoin"]
