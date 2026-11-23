"""Administration routes (administrator only)."""

from __future__ import annotations

from datetime import datetime, timezone

from flask import Blueprint, render_template, request
from flask_login import current_user, login_required
from sqlalchemy import func

from app.extensions import db
from app.models.audit_log import AuditLog
from app.models.collection_session import CollectionSession
from app.models.dataset_sample import DatasetSample
from app.models.gesture import Gesture
from app.models.model_version import ModelVersion
from app.models.participant import Participant
from app.models.recognition_history import RecognitionHistory
from app.models.training_run import TrainingRun
from app.models.user import User
from app.utils import admin_required, json_error, json_success, log_audit, parse_pagination

bp = Blueprint("admin", __name__, url_prefix="/admin")


@bp.route("")
@bp.route("/")
@login_required
@admin_required
def index():
    return render_template(
        "admin/index.html",
        user_count=User.query.count(),
        gesture_count=Gesture.query.count(),
        sample_count=DatasetSample.query.count(),
        participant_count=Participant.query.count(),
        model_count=ModelVersion.query.count(),
        run_count=TrainingRun.query.count(),
        recent_logs=AuditLog.query.order_by(AuditLog.created_at.desc()).limit(15).all(),
    )


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------
@bp.get("/api/admin/users")
@login_required
@admin_required
def api_users():
    page, per_page = parse_pagination(default_per_page=25, max_per_page=200)
    query = User.query
    search = (request.args.get("search") or "").strip()
    role = (request.args.get("role") or "").strip()
    if search:
        pattern = f"%{search}%"
        query = query.filter(
            db.or_(User.full_name.ilike(pattern), User.email.ilike(pattern))
        )
    if role in {User.ROLE_ADMIN, User.ROLE_USER}:
        query = query.filter(User.role == role)

    pagination = query.order_by(User.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    return json_success(
        users=[user.to_dict() for user in pagination.items],
        page=page,
        per_page=per_page,
        total=pagination.total,
        pages=pagination.pages,
    )


@bp.put("/api/admin/users/<int:user_id>")
@login_required
@admin_required
def api_update_user(user_id: int):
    """Change a user's role or activation state."""
    user = db.session.get(User, user_id)
    if user is None:
        return json_error("User not found.", 404)

    payload = request.get_json(silent=True) or request.form.to_dict()
    changed: list = []

    if "role" in payload:
        role = str(payload["role"]).strip().lower()
        if role not in User.VALID_ROLES:
            return json_error("Invalid role. Expected 'admin' or 'user'.", 400)
        if user.id == current_user.id and role != User.ROLE_ADMIN:
            return json_error("You cannot remove your own administrator role.", 400)
        if user.role != role:
            user.role = role
            changed.append(f"role -> {role}")

    if "is_active" in payload:
        active = _parse_bool(payload["is_active"], user.is_active)
        if user.id == current_user.id and not active:
            return json_error("You cannot deactivate your own account.", 400)
        if user.is_active != active:
            user.is_active = active
            changed.append("active" if active else "inactive")

    if "full_name" in payload and payload["full_name"]:
        name = str(payload["full_name"]).strip()
        if len(name) >= 2:
            user.full_name = name
            changed.append("name")

    if "password" in payload and payload["password"]:
        password = str(payload["password"])
        if len(password) < 8:
            return json_error("The password must be at least eight characters.", 400)
        user.set_password(password)
        changed.append("password")

    db.session.commit()
    log_audit(
        "admin.user.update",
        f"Updated user #{user.id} ({', '.join(changed) if changed else 'no changes'})",
        current_user,
    )
    return json_success(user=user.to_dict())


@bp.delete("/api/admin/users/<int:user_id>")
@login_required
@admin_required
def api_delete_user(user_id: int):
    user = db.session.get(User, user_id)
    if user is None:
        return json_error("User not found.", 404)
    if user.id == current_user.id:
        return json_error("You cannot delete your own account.", 400)

    email = user.email
    db.session.delete(user)
    db.session.commit()
    log_audit("admin.user.delete", f"Deleted user {email}", current_user)
    return json_success(deleted=True, message=f"User {email} was deleted.")


# ---------------------------------------------------------------------------
# Audit logs
# ---------------------------------------------------------------------------
@bp.get("/api/admin/audit-logs")
@login_required
@admin_required
def api_audit_logs():
    page, per_page = parse_pagination(default_per_page=30, max_per_page=200)
    query = AuditLog.query
    action = (request.args.get("action") or "").strip()
    if action:
        query = query.filter(AuditLog.action.ilike(f"%{action}%"))
    pagination = query.order_by(AuditLog.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    return json_success(
        logs=[entry.to_dict() for entry in pagination.items],
        page=page,
        per_page=per_page,
        total=pagination.total,
        pages=pagination.pages,
    )


# ---------------------------------------------------------------------------
# System overview
# ---------------------------------------------------------------------------
@bp.get("/api/admin/overview")
@login_required
@admin_required
def api_overview():
    """A single call that powers the administration overview page."""
    by_role = (
        db.session.query(User.role, func.count(User.id)).group_by(User.role).all()
    )
    by_status = (
        db.session.query(Gesture.validation_status, func.count(Gesture.id))
        .group_by(Gesture.validation_status)
        .all()
    )
    return json_success(
        overview={
            "users": {
                "total": User.query.count(),
                "active": User.query.filter_by(is_active=True).count(),
                "by_role": {role: count for role, count in by_role},
            },
            "gestures": {
                "total": Gesture.query.count(),
                "active": Gesture.query.filter_by(is_active=True).count(),
                "validated": Gesture.query.filter_by(is_validated=True).count(),
                "by_status": {status: count for status, count in by_status},
            },
            "dataset": {
                "samples": DatasetSample.query.count(),
                "participants": Participant.query.count(),
                "sessions": CollectionSession.query.count(),
            },
            "models": {
                "versions": ModelVersion.query.count(),
                "published": ModelVersion.query.filter_by(
                    status=ModelVersion.STATUS_PUBLISHED
                ).count(),
            },
            "training": {
                "runs": TrainingRun.query.count(),
                "active": TrainingRun.query.filter(
                    TrainingRun.status.in_(
                        [TrainingRun.STATUS_QUEUED, TrainingRun.STATUS_RUNNING]
                    )
                ).count(),
            },
            "recognition": {
                "total": RecognitionHistory.query.count(),
                "today": RecognitionHistory.query.filter(
                    RecognitionHistory.created_at
                    >= datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
                ).count(),
            },
        }
    )


def _parse_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
