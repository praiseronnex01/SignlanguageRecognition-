"""Shared helpers: consistent JSON responses, auditing and pagination."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import wraps

from flask import jsonify, request
from flask_login import current_user

from app.extensions import db


def json_success(**payload):
    """Return a consistent success envelope."""
    body = {"success": True}
    body.update(payload)
    return jsonify(body)


def json_error(message: str, status: int = 400, **extra):
    """Return a consistent error envelope with the correct HTTP status."""
    body = {"success": False, "error": message}
    body.update(extra)
    return jsonify(body), status


def client_ip() -> str | None:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr


def log_audit(action: str, description: str, user=None) -> None:
    """Record an action in the audit trail.

    Audit logging must never break the request that triggered it, therefore
    any database problem is swallowed after being reported to the app logger.
    """
    from app.models.audit_log import AuditLog

    try:
        actor = user
        if actor is None and getattr(current_user, "is_authenticated", False):
            actor = current_user
        entry = AuditLog(
            user_id=getattr(actor, "id", None),
            action=action,
            description=description,
            ip_address=client_ip(),
        )
        db.session.add(entry)
        db.session.commit()
    except Exception:  # pragma: no cover - defensive
        db.session.rollback()


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_pagination(default_per_page: int = 15, max_per_page: int = 200) -> tuple:
    """Return ``(page, per_page)`` from the query string."""
    try:
        page = max(1, int(request.args.get("page", 1)))
    except (TypeError, ValueError):
        page = 1
    try:
        per_page = int(request.args.get("per_page", default_per_page))
    except (TypeError, ValueError):
        per_page = default_per_page
    per_page = max(1, min(per_page, max_per_page))
    return page, per_page


def admin_required(view):
    """Route decorator restricting a view to administrators."""

    @wraps(view)
    def wrapper(*args, **kwargs):
        if not getattr(current_user, "is_authenticated", False):
            from flask import redirect, url_for

            return redirect(url_for("auth.login_page", next=request.path))
        if not current_user.is_admin:
            from flask import abort

            abort(403)
        return view(*args, **kwargs)

    return wrapper
