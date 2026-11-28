"""Dataset collection, validation and export routes (administrator only)."""

from __future__ import annotations

import base64
import binascii
import re

from flask import Blueprint, current_app, render_template, request, send_file
from flask_login import current_user, login_required

from app.extensions import db
from app.models.collection_session import CollectionSession
from app.models.dataset_sample import DatasetSample
from app.models.gesture import Gesture
from app.models.participant import Participant
from app.services import dataset_service, feature_extractor
from app.services.hand_detector import HandDetector, HandDetectorError, decode_image
from app.utils import admin_required, json_error, json_success, log_audit, parse_pagination

bp = Blueprint("dataset", __name__, url_prefix="/dataset")

_DATA_URL_PATTERN = re.compile(r"^data:image/(png|jpe?g|webp);base64,", re.IGNORECASE)


@bp.route("")
@bp.route("/")
@login_required
@admin_required
def index():
    gestures = Gesture.query.order_by(Gesture.label).all()
    participants = dataset_service.participant_summary()
    sessions = (
        CollectionSession.query.order_by(CollectionSession.started_at.desc())
        .limit(25)
        .all()
    )
    return render_template(
        "dataset/index.html",
        gestures=gestures,
        participants=participants,
        sessions=sessions,
        stats=dataset_service.dataset_stats(),
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
@bp.get("/api/dataset/stats")
@login_required
@admin_required
def api_stats():
    return json_success(stats=dataset_service.dataset_stats())


@bp.get("/api/dataset/validate")
@login_required
@admin_required
def api_validate():
    return json_success(report=dataset_service.validate_dataset())


@bp.get("/api/dataset/duplicates")
@login_required
@admin_required
def api_duplicates():
    return json_success(report=dataset_service.duplicate_report())


# ---------------------------------------------------------------------------
# Participants and sessions
# ---------------------------------------------------------------------------
@bp.get("/api/dataset/participants")
@login_required
@admin_required
def api_participants():
    return json_success(participants=dataset_service.participant_summary())


@bp.post("/api/dataset/participants")
@login_required
@admin_required
def api_create_participant():
    payload = request.get_json(silent=True) or request.form.to_dict()
    code = (payload.get("participant_code") or "").strip().upper()
    consent = (payload.get("consent_status") or Participant.CONSENT_PENDING).strip()

    if not code:
        return json_error("A participant code is required.", 400)
    if consent not in Participant.CONSENT_STATES:
        return json_error("Invalid consent status.", 400)

    participant = dataset_service.get_or_create_participant(code, consent)
    log_audit(
        "dataset.participant",
        f"Registered participant {participant.participant_code}",
        current_user,
    )
    return json_success(participant=participant.to_dict()), 201


@bp.put("/api/dataset/participants/<int:participant_id>/consent")
@login_required
@admin_required
def api_set_consent(participant_id: int):
    participant = db.session.get(Participant, participant_id)
    if participant is None:
        return json_error("Participant not found.", 404)
    payload = request.get_json(silent=True) or request.form.to_dict()
    try:
        dataset_service.set_consent(participant, (payload.get("consent_status") or "").strip())
    except dataset_service.DatasetError as exc:
        return json_error(str(exc), 400)
    log_audit(
        "dataset.consent",
        f"Consent for {participant.participant_code} set to {participant.consent_status}",
        current_user,
    )
    return json_success(participant=participant.to_dict())


@bp.post("/api/dataset/sessions")
@login_required
@admin_required
def api_create_session():
    payload = request.get_json(silent=True) or request.form.to_dict()
    code = (payload.get("participant_code") or "").strip().upper()
    if not code:
        return json_error("A participant code is required.", 400)

    participant = dataset_service.get_or_create_participant(code)
    try:
        session = dataset_service.create_session(
            participant, administrator=current_user, notes=payload.get("notes")
        )
    except dataset_service.DatasetError as exc:
        return json_error(str(exc), 400)
    log_audit(
        "dataset.session.start",
        f"Started collection session {session.session_code}",
        current_user,
    )
    return json_success(session=session.to_dict()), 201


@bp.post("/api/dataset/sessions/<int:session_id>/close")
@login_required
@admin_required
def api_close_session(session_id: int):
    session = db.session.get(CollectionSession, session_id)
    if session is None:
        return json_error("Collection session not found.", 404)
    status = (request.get_json(silent=True) or {}).get(
        "status", CollectionSession.STATUS_COMPLETED
    )
    if status not in CollectionSession.STATUSES:
        return json_error("Invalid session status.", 400)
    dataset_service.close_session(session, status=status)
    log_audit(
        "dataset.session.close",
        f"Closed collection session {session.session_code}",
        current_user,
    )
    return json_success(session=session.to_dict())


# ---------------------------------------------------------------------------
# Sample collection
# ---------------------------------------------------------------------------
def _decode_payload() -> bytes:
    if "file" in request.files:
        uploaded = request.files["file"]
        if uploaded and uploaded.filename:
            data = uploaded.read()
            if data:
                return data
    payload = request.get_json(silent=True) or {}
    encoded = payload.get("image") or payload.get("frame")
    if not encoded:
        raise ValueError("No image was supplied")
    if isinstance(encoded, str):
        encoded = _DATA_URL_PATTERN.sub("", encoded.strip())
    try:
        return base64.b64decode(encoded, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("The supplied image is not valid base64 data") from exc


@bp.post("/api/dataset/collect")
@login_required
@admin_required
def api_collect():
    """Detect hands in a frame and store the extracted landmarks as a sample."""
    payload = request.get_json(silent=True) or request.form.to_dict()

    gesture_id = payload.get("gesture_id")
    session_id = payload.get("session_id")
    participant_code = (payload.get("participant_code") or "").strip().upper()

    if not gesture_id or not session_id or not participant_code:
        return json_error(
            "gesture_id, session_id and participant_code are required.", 400
        )

    gesture = db.session.get(Gesture, int(gesture_id))
    if gesture is None:
        return json_error("Gesture not found.", 404)

    session = db.session.get(CollectionSession, int(session_id))
    if session is None:
        return json_error("Collection session not found.", 404)

    participant = db.session.get(Participant, session.participant_id)
    if participant is None:
        return json_error("Participant not found for this session.", 404)
    if participant.participant_code != participant_code:
        return json_error("The participant code does not match the session.", 400)

    try:
        image_bytes = _decode_payload()
    except ValueError as exc:
        return json_error(str(exc), 400)

    try:
        frame = decode_image(image_bytes)
    except ValueError as exc:
        return json_error(str(exc), 400)

    try:
        detection = HandDetector(
            max_hands=int(current_app.config.get("MAX_HANDS", 2))
        ).detect(frame)
    except HandDetectorError as exc:
        return json_error(str(exc), 503)

    landmarks = detection.to_dict(include_landmarks=True)

    if not detection.hands:
        return json_success(
            stored=False,
            message="No hand detected - the frame was not saved.",
            hand_detected=False,
            hand_count=0,
            landmarks=landmarks,
        )

    if len(detection.hands) != int(gesture.hand_count):
        return json_success(
            stored=False,
            message=(
                f"Gesture '{gesture.label}' requires {gesture.hand_count} hand(s) "
                f"but {len(detection.hands)} were detected."
            ),
            hand_detected=True,
            hand_count=len(detection.hands),
            landmarks=landmarks,
        )

    try:
        landmark_payload = feature_extractor.build_payload(detection)
        sample = dataset_service.store_sample(
            gesture=gesture,
            participant=participant,
            session=session,
            payload=landmark_payload,
            quality_score=feature_extractor.quality_score(detection),
        )
    except feature_extractor.FeatureExtractionError as exc:
        return json_error(str(exc), 400, landmarks=landmarks)
    except dataset_service.DatasetError as exc:
        return json_success(
            stored=False,
            message=str(exc),
            hand_detected=True,
            hand_count=len(detection.hands),
            landmarks=landmarks,
        )

    return json_success(
        stored=True,
        message="Sample saved.",
        sample=sample.to_dict(),
        hand_detected=True,
        hand_count=len(detection.hands),
        landmarks=landmarks,
    ), 201


@bp.get("/api/dataset/samples")
@login_required
@admin_required
def api_samples():
    page, per_page = parse_pagination(default_per_page=25, max_per_page=200)
    query = DatasetSample.query
    gesture_id = request.args.get("gesture_id")
    participant_code = request.args.get("participant_code")
    session_id = request.args.get("session_id")

    if gesture_id:
        query = query.filter(DatasetSample.gesture_id == int(gesture_id))
    if participant_code:
        query = query.join(Participant).filter(
            Participant.participant_code == participant_code.upper()
        )
    if session_id:
        query = query.filter(DatasetSample.session_id == int(session_id))

    pagination = query.order_by(DatasetSample.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    return json_success(
        samples=[sample.to_dict() for sample in pagination.items],
        page=page,
        per_page=per_page,
        total=pagination.total,
        pages=pagination.pages,
    )


@bp.delete("/api/dataset/samples/<int:sample_id>")
@login_required
@admin_required
def api_delete_sample(sample_id: int):
    sample = db.session.get(DatasetSample, sample_id)
    if sample is None:
        return json_error("Sample not found.", 404)
    label = sample.gesture.label if sample.gesture else str(sample.gesture_id)
    deleted = dataset_service.delete_sample(sample_id)
    log_audit("dataset.sample.delete", f"Deleted sample #{sample_id} ({label})", current_user)
    return json_success(deleted=deleted)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
@bp.get("/api/dataset/export")
@login_required
@admin_required
def api_export():
    """Export the dataset and return the generated files."""
    include_unvalidated = request.args.get("include_unvalidated") in {"1", "true", "yes"}
    try:
        metadata = dataset_service.export_dataset(
            include_unvalidated=include_unvalidated
        )
    except dataset_service.DatasetError as exc:
        return json_error(str(exc), 400)
    log_audit("dataset.export", f"Exported dataset {metadata['version']}", current_user)
    return json_success(dataset=metadata)


@bp.get("/api/dataset/export/download")
@login_required
@admin_required
def api_export_download():
    from app.config import DATASET_DIR

    version = request.args.get("version")
    metadata_dir = DATASET_DIR / "metadata"
    path = None
    if version:
        candidate = DATASET_DIR / "processed" / f"dataset_{version}.csv"
        if candidate.is_file():
            path = candidate
    if path is None:
        processed = sorted(
            (DATASET_DIR / "processed").glob("dataset_v*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        path = processed[0] if processed else None
    if path is None:
        return json_error("No exported dataset is available yet.", 404)
    return send_file(
        path,
        mimetype="text/csv",
        as_attachment=True,
        download_name=path.name,
    )


@bp.get("/api/dataset/versions")
@login_required
@admin_required
def api_versions():
    return json_success(versions=dataset_service.list_dataset_versions())
