"""Vocabulary management routes.

The vocabulary is the authoritative list of signs the system can recognise.
A sign only becomes recognisable after an administrator has verified it
(``is_validated``) against the Malawian Sign Language Dictionary or with a
competent MSL signer.  Nothing in this module marks a sign as validated
automatically.
"""

from __future__ import annotations

import os
import re
import uuid

from flask import Blueprint, current_app, render_template, request, url_for
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename

from app.extensions import db
from app.models.gesture import (
    HAND_ONE,
    HAND_TWO,
    STATUS_CANDIDATE,
    STATUS_REJECTED,
    STATUS_VALIDATED,
    VALID_HAND_COUNTS,
    Gesture,
)
from app.utils import admin_required, json_error, json_success, log_audit

bp = Blueprint("vocabulary", __name__, url_prefix="/vocabulary")

LABEL_PATTERN = re.compile(r"^[A-Za-z0-9 _\-]{2,80}$")


def _upload_folder() -> str:
    folder = os.path.join(current_app.instance_path, "uploads", "gestures")
    os.makedirs(folder, exist_ok=True)
    return folder


def _save_reference_image(file_storage) -> str | None:
    """Persist an uploaded reference image and return its public URL."""
    if not file_storage or not file_storage.filename:
        return None
    extension = os.path.splitext(secure_filename(file_storage.filename))[1].lower()
    if extension not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError("Reference images must be PNG, JPEG or WebP files.")

    filename = f"{uuid.uuid4().hex}{extension}"
    destination = os.path.join(_upload_folder(), filename)
    file_storage.save(destination)
    return url_for("vocabulary.reference_image", filename=filename)


@bp.route("")
@bp.route("/")
@login_required
def index():
    categories = (
        db.session.query(Gesture.category)
        .filter(Gesture.category.isnot(None))
        .distinct()
        .order_by(Gesture.category)
        .all()
    )
    gestures = Gesture.query.order_by(Gesture.label).all()
    return render_template(
        "vocabulary/index.html",
        gestures=gestures,
        categories=[row[0] for row in categories],
        hand_one=HAND_ONE,
        hand_two=HAND_TWO,
    )


@bp.get("/reference-image/<path:filename>")
def reference_image(filename: str):
    """Serve a stored gesture reference image."""
    from flask import send_from_directory

    safe_name = secure_filename(filename)
    return send_from_directory(_upload_folder(), safe_name)


@bp.get("/api/vocabulary")
@login_required
def api_list():
    query = Gesture.query
    search = (request.args.get("search") or "").strip()
    category = (request.args.get("category") or "").strip()
    hand_count = request.args.get("hand_count")
    include_inactive = request.args.get("include_inactive") in {"1", "true", "yes"}

    if search:
        pattern = f"%{search}%"
        query = query.filter(
            db.or_(
                Gesture.label.ilike(pattern),
                Gesture.english_meaning.ilike(pattern),
                Gesture.description.ilike(pattern),
            )
        )
    if category and category.lower() != "all":
        query = query.filter(Gesture.category == category)
    if hand_count in {"1", "2"}:
        query = query.filter(Gesture.hand_count == int(hand_count))
    if not include_inactive:
        query = query.filter(Gesture.is_active.is_(True))

    gestures = query.order_by(Gesture.label).all()
    categories = (
        db.session.query(Gesture.category)
        .filter(Gesture.category.isnot(None))
        .distinct()
        .order_by(Gesture.category)
        .all()
    )
    return json_success(
        gestures=[gesture.to_dict(include_sample_count=True) for gesture in gestures],
        total=len(gestures),
        categories=[row[0] for row in categories],
    )


@bp.get("/api/vocabulary/<int:gesture_id>")
@login_required
def api_detail(gesture_id: int):
    gesture = db.session.get(Gesture, gesture_id)
    if gesture is None:
        return json_error("Gesture not found.", 404)
    return json_success(gesture=gesture.to_dict(include_sample_count=True))


@bp.post("/api/vocabulary")
@admin_required
def api_create():
    payload = request.get_json(silent=True) or request.form.to_dict()

    label = (payload.get("label") or "").strip().upper()
    english_meaning = (payload.get("english_meaning") or "").strip()
    category = (payload.get("category") or "General").strip() or "General"

    errors: dict = {}
    if not LABEL_PATTERN.match(label):
        errors["label"] = (
            "Label must be 2-80 characters using letters, digits, spaces, "
            "hyphens or underscores."
        )
    if not english_meaning:
        errors["english_meaning"] = "An English meaning is required."
    if errors:
        return json_error("Please correct the highlighted fields.", 400, errors=errors)

    if Gesture.query.filter_by(label=label).first() is not None:
        return json_error(
            f"A gesture with the label '{label}' already exists.", 409
        )

    hand_count = _parse_hand_count(payload.get("hand_count"), HAND_ONE)

    gesture = Gesture(
        label=label,
        english_meaning=english_meaning,
        category=category,
        description=(payload.get("description") or "").strip() or None,
        reference_video=(payload.get("reference_video") or "").strip() or None,
        is_validated=False,  # never validated automatically
        is_active=bool(payload.get("is_active", True)),
        hand_count=hand_count,
        validation_status=STATUS_CANDIDATE,
        validation_notes=(payload.get("validation_notes") or "").strip() or None,
        source_reference=(payload.get("source_reference") or "").strip()
        or current_app.config.get("MSL_REFERENCE_URL"),
    )

    if "reference_image" in request.files:
        try:
            gesture.reference_image = _save_reference_image(request.files["reference_image"])
        except ValueError as exc:
            return json_error(str(exc), 400)
    elif payload.get("reference_image"):
        gesture.reference_image = str(payload["reference_image"]).strip()

    db.session.add(gesture)
    db.session.commit()
    log_audit("vocabulary.create", f"Added gesture '{gesture.label}'", current_user)
    return json_success(gesture=gesture.to_dict(include_sample_count=True)), 201


@bp.put("/api/vocabulary/<int:gesture_id>")
@admin_required
def api_update(gesture_id: int):
    gesture = db.session.get(Gesture, gesture_id)
    if gesture is None:
        return json_error("Gesture not found.", 404)

    payload = request.get_json(silent=True) or request.form.to_dict()

    if "label" in payload:
        label = (payload.get("label") or "").strip().upper()
        if not LABEL_PATTERN.match(label):
            return json_error("The gesture label is not valid.", 400)
        clash = Gesture.query.filter(Gesture.label == label, Gesture.id != gesture.id).first()
        if clash is not None:
            return json_error(f"A gesture with the label '{label}' already exists.", 409)
        gesture.label = label

    if "english_meaning" in payload:
        meaning = (payload.get("english_meaning") or "").strip()
        if not meaning:
            return json_error("An English meaning is required.", 400)
        gesture.english_meaning = meaning

    if "category" in payload and payload.get("category"):
        gesture.category = str(payload["category"]).strip()
    if "description" in payload:
        gesture.description = (payload.get("description") or "").strip() or None
    if "reference_video" in payload:
        gesture.reference_video = (payload.get("reference_video") or "").strip() or None
    if "hand_count" in payload:
        gesture.hand_count = _parse_hand_count(payload.get("hand_count"), gesture.hand_count)
    if "source_reference" in payload:
        gesture.source_reference = (payload.get("source_reference") or "").strip() or None
    if "validation_notes" in payload:
        gesture.validation_notes = (payload.get("validation_notes") or "").strip() or None
    if "is_active" in payload:
        gesture.is_active = _parse_bool(payload.get("is_active"), gesture.is_active)

    if "is_validated" in payload:
        requested = _parse_bool(payload.get("is_validated"), gesture.is_validated)
        if requested and not gesture.is_validated:
            # Validation is an explicit, audited administrator decision.
            gesture.is_validated = True
            gesture.validation_status = STATUS_VALIDATED
            gesture.validation_notes = (
                gesture.validation_notes
                or "Marked as validated by an administrator."
            )
        elif not requested and gesture.is_validated:
            gesture.is_validated = False
            gesture.validation_status = STATUS_CANDIDATE
    if "validation_status" in payload:
        status = str(payload["validation_status"]).strip().lower()
        if status not in {STATUS_CANDIDATE, STATUS_VALIDATED, STATUS_REJECTED}:
            return json_error("Invalid validation status.", 400)
        if status == STATUS_VALIDATED:
            gesture.is_validated = True
        elif status == STATUS_CANDIDATE:
            gesture.is_validated = False
        gesture.validation_status = status

    if "reference_image" in request.files:
        try:
            gesture.reference_image = _save_reference_image(request.files["reference_image"])
        except ValueError as exc:
            return json_error(str(exc), 400)
    elif payload.get("reference_image"):
        gesture.reference_image = str(payload["reference_image"]).strip() or None

    db.session.commit()
    log_audit(
        "vocabulary.update",
        f"Updated gesture '{gesture.label}'",
        current_user,
    )
    return json_success(gesture=gesture.to_dict(include_sample_count=True))


@bp.delete("/api/vocabulary/<int:gesture_id>")
@admin_required
def api_delete(gesture_id: int):
    """Deactivate a gesture (and optionally remove it permanently)."""
    gesture = db.session.get(Gesture, gesture_id)
    if gesture is None:
        return json_error("Gesture not found.", 404)

    payload = request.get_json(silent=True) or {}
    permanent = _parse_bool(payload.get("permanent"), False)

    if permanent:
        label = gesture.label
        db.session.delete(gesture)
        db.session.commit()
        log_audit("vocabulary.delete", f"Deleted gesture '{label}'", current_user)
        return json_success(message=f"Gesture '{label}' was deleted.")

    gesture.is_active = False
    gesture.validation_status = STATUS_REJECTED if payload.get("reject") else gesture.validation_status
    db.session.commit()
    log_audit(
        "vocabulary.deactivate",
        f"Deactivated gesture '{gesture.label}'",
        current_user,
    )
    return json_success(
        message=f"Gesture '{gesture.label}' was deactivated.",
        gesture=gesture.to_dict(include_sample_count=True),
    )


def _parse_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_hand_count(value, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed in VALID_HAND_COUNTS else default
