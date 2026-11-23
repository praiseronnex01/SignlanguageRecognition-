"""Live gesture recognition routes (pages + JSON API)."""

from __future__ import annotations

import base64
import binascii
import re

from flask import Blueprint, current_app, render_template, request
from flask_login import current_user, login_required

from app.extensions import db
from app.models.gesture import Gesture
from app.models.model_version import ModelVersion
from app.models.recognition_history import RecognitionHistory
from app.services import model_service
from app.services.gesture_predictor import PredictionError, predict_frame
from app.services.hand_detector import HandDetectorError, decode_image
from app.utils import json_error, json_success, log_audit

bp = Blueprint("recognition", __name__, url_prefix="/recognition")

_DATA_URL_PATTERN = re.compile(r"^data:image/(png|jpe?g|webp);base64,", re.IGNORECASE)


def _extract_image_payload() -> bytes:
    """Return the image bytes from either a multipart upload or a JSON body."""
    if "file" in request.files:
        uploaded = request.files["file"]
        if uploaded and uploaded.filename:
            payload = uploaded.read()
            if payload:
                return payload

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


@bp.route("")
@bp.route("/")
@login_required
def index():
    models = model_service.published_models()
    supported = (
        Gesture.query.filter_by(is_validated=True, is_active=True)
        .order_by(Gesture.label)
        .all()
    )
    return render_template(
        "recognition/index.html",
        supported_gestures=supported,
        has_model=bool(models),
        model_versions=[model.to_dict() for model in models.values()],
        confidence_threshold=float(
            current_app.config.get("RECOGNITION_CONFIDENCE_THRESHOLD", 0.60)
        ),
    )


@bp.post("/api/predict")
@login_required
def api_predict():
    """Classify a single camera frame.

    Accepts a multipart ``file`` upload or a JSON body with a base64 image.
    """
    try:
        payload = _extract_image_payload()
    except ValueError as exc:
        return json_error(str(exc), 400)

    try:
        frame = decode_image(payload)
    except ValueError as exc:
        return json_error(str(exc), 400)

    try:
        outcome = predict_frame(frame)
    except (HandDetectorError, PredictionError) as exc:
        current_app.logger.warning("Prediction failed: %s", exc)
        return json_error(str(exc), 503)
    except Exception as exc:  # pragma: no cover - defensive
        current_app.logger.exception("Unexpected prediction error")
        return json_error(f"Prediction failed: {exc}", 500)

    result = outcome.to_dict()

    if request.args.get("include_landmarks") in {"1", "true", "yes"}:
        from app.services.hand_detector import HandDetector

        try:
            detection = HandDetector(
                max_hands=int(current_app.config.get("MAX_HANDS", 2))
            ).detect(frame)
            result["landmarks"] = detection.to_dict(include_landmarks=True)
        except Exception:  # pragma: no cover - overlay is optional
            result["landmarks"] = {"hands": []}

    return json_success(**result)


@bp.post("/api/save")
@login_required
def api_save():
    """Store a recognised gesture in the signed-in user's history."""
    payload = request.get_json(silent=True) or request.form.to_dict()

    gesture_label = (payload.get("gesture") or "").strip()
    if not gesture_label:
        return json_error("A gesture label is required.", 400)

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        return json_error("Confidence must be a number.", 400)
    if not 0.0 <= confidence <= 1.0:
        return json_error("Confidence must be between 0 and 1.", 400)

    gesture = Gesture.query.filter_by(label=gesture_label).first()
    if gesture is None:
        return json_error(f"Unknown gesture '{gesture_label}'.", 404)

    model_version_id = None
    version = payload.get("model_version")
    if version:
        row = ModelVersion.query.filter_by(version=str(version)).first()
        model_version_id = row.id if row else None

    entry = RecognitionHistory(
        user_id=current_user.id,
        gesture_id=gesture.id,
        confidence=confidence,
        model_version_id=model_version_id,
        translation=gesture.english_meaning,
        hand_count=gesture.hand_count,
    )
    db.session.add(entry)
    db.session.commit()
    log_audit(
        "recognition.save",
        f"Saved '{gesture.label}' at {confidence:.2f} confidence",
        current_user,
    )
    return json_success(entry=entry.to_dict()), 201


@bp.get("/api/status")
@login_required
def api_status():
    """Report the recognition subsystem status."""
    models = model_service.published_models()
    published = [model.to_dict() for model in models.values()]

    threshold = float(current_app.config.get("RECOGNITION_CONFIDENCE_THRESHOLD", 0.60))
    for model in published:
        if model.get("rejection_threshold") is not None:
            threshold = float(model["rejection_threshold"])

    validated = Gesture.query.filter_by(is_validated=True, is_active=True).count()
    candidates = Gesture.query.filter_by(is_validated=True, is_active=True).order_by(
        Gesture.label
    ).all()

    return json_success(
        model_available=bool(models),
        model_versions=published,
        confidence_threshold=round(threshold, 4),
        supported_gesture_count=validated,
        supported_gestures=[
            {"label": gesture.label, "translation": gesture.english_meaning}
            for gesture in candidates
        ],
        max_hands=int(current_app.config.get("MAX_HANDS", 2)),
        max_upload_bytes=int(current_app.config.get("MAX_CONTENT_LENGTH", 0)),
    )
