"""Personal recognition history routes.

A normal user only ever sees their own history.  Administrators may inspect
any user's history by supplying ``user_id`` explicitly.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime

from flask import Blueprint, Response, render_template, request
from flask_login import current_user, login_required

from app.extensions import db
from app.models.gesture import Gesture
from app.models.recognition_history import RecognitionHistory
from app.models.user import User
from app.utils import json_error, json_success, log_audit, parse_pagination

bp = Blueprint("history", __name__, url_prefix="/history")


def _target_user() -> User:
    """Resolve whose history is being requested."""
    requested = request.args.get("user_id")
    if requested and current_user.is_admin:
        return db.session.get(User, int(requested)) or current_user
    return current_user


def _apply_filters(query, user_id: int):
    search = (request.args.get("search") or "").strip()
    date_from = (request.args.get("date_from") or "").strip()
    date_to = (request.args.get("date_to") or "").strip()

    query = query.filter(RecognitionHistory.user_id == user_id)

    if search:
        pattern = f"%{search}%"
        query = query.join(Gesture, RecognitionHistory.gesture_id == Gesture.id).filter(
            db.or_(
                Gesture.label.ilike(pattern),
                Gesture.english_meaning.ilike(pattern),
                RecognitionHistory.translation.ilike(pattern),
            )
        )
    if date_from:
        try:
            start = datetime.strptime(date_from, "%Y-%m-%d")
            query = query.filter(RecognitionHistory.created_at >= start)
        except ValueError:
            pass
    if date_to:
        try:
            end = datetime.strptime(date_to, "%Y-%m-%d")
            query = query.filter(RecognitionHistory.created_at < end.replace(hour=23, minute=59, second=59))
        except ValueError:
            pass
    return query


@bp.route("")
@bp.route("/")
@login_required
def index():
    page, per_page = parse_pagination(default_per_page=20, max_per_page=100)
    query = _apply_filters(RecognitionHistory.query, current_user.id)
    pagination = query.order_by(RecognitionHistory.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    return render_template(
        "history/index.html",
        pagination=pagination,
        entries=pagination.items,
        search=request.args.get("search", ""),
        date_from=request.args.get("date_from", ""),
        date_to=request.args.get("date_to", ""),
    )


@bp.get("/api/history")
@login_required
def api_list():
    page, per_page = parse_pagination(default_per_page=20, max_per_page=100)
    target = _target_user()
    query = _apply_filters(RecognitionHistory.query, target.id)
    pagination = query.order_by(RecognitionHistory.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    return json_success(
        entries=[entry.to_dict() for entry in pagination.items],
        page=page,
        per_page=per_page,
        total=pagination.total,
        pages=pagination.pages,
    )


@bp.delete("/api/history/<int:entry_id>")
@login_required
def api_delete(entry_id: int):
    entry = db.session.get(RecognitionHistory, entry_id)
    if entry is None:
        return json_error("History entry not found.", 404)
    if entry.user_id != current_user.id and not current_user.is_admin:
        return json_error("You cannot delete another user's history entry.", 403)
    db.session.delete(entry)
    db.session.commit()
    log_audit("history.delete", f"Deleted recognition history entry #{entry_id}", current_user)
    return json_success(deleted=True)


@bp.delete("/api/history")
@login_required
def api_clear():
    """Clear the signed-in user's history (or another user's, for admins)."""
    target = _target_user()
    count = RecognitionHistory.query.filter_by(user_id=target.id).delete(
        synchronize_session=False
    )
    db.session.commit()
    log_audit(
        "history.clear",
        f"Cleared {count} history entries for user #{target.id}",
        current_user,
    )
    return json_success(deleted=count, message=f"{count} entries were removed.")


@bp.get("/api/history/export")
@login_required
def api_export():
    """Export the history as a CSV download."""
    target = _target_user()
    query = _apply_filters(RecognitionHistory.query, target.id)
    entries = query.order_by(RecognitionHistory.created_at.desc()).all()

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "id",
            "gesture_label",
            "english_meaning",
            "confidence",
            "model_version",
            "hand_count",
            "date",
            "time",
        ]
    )
    for entry in entries:
        created = entry.created_at
        writer.writerow(
            [
                entry.id,
                entry.gesture.label if entry.gesture else "",
                entry.translation or "",
                f"{entry.confidence:.4f}",
                entry.model_version.version if entry.model_version else "",
                entry.hand_count or "",
                created.strftime("%Y-%m-%d") if created else "",
                created.strftime("%H:%M:%S") if created else "",
            ]
        )

    filename = f"msl-recognition-history-{target.id}.csv"
    return Response(
        buffer.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
