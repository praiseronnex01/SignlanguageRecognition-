"""Dashboard routes: an operational overview built entirely from live queries."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from flask import Blueprint, render_template
from flask_login import login_required
from sqlalchemy import func

from app.extensions import db
from app.models.collection_session import CollectionSession
from app.models.dataset_sample import DatasetSample
from app.models.gesture import Gesture
from app.models.model_version import ModelVersion
from app.models.participant import Participant
from app.models.recognition_history import RecognitionHistory
from app.models.training_run import TrainingRun
from app.models.user import User
from app.services import model_service
from app.utils import json_success

bp = Blueprint("dashboard", __name__)


def _collect_statistics() -> dict:
    """Compute every dashboard statistic from the database."""
    now = datetime.now(timezone.utc)
    week_ago = now - timedelta(days=7)

    total_gestures = Gesture.query.count()
    active_gestures = Gesture.query.filter_by(is_active=True).count()
    validated_gestures = Gesture.query.filter_by(is_validated=True).count()

    total_samples = DatasetSample.query.count()
    total_participants = Participant.query.count()
    consented_participants = Participant.query.filter_by(
        consent_status=Participant.CONSENT_GRANTED
    ).count()
    total_users = User.query.count()
    active_users = User.query.filter_by(is_active=True).count()
    total_sessions = CollectionSession.query.count()
    total_recognitions = RecognitionHistory.query.count()

    recent_recognitions = (
        RecognitionHistory.query.order_by(RecognitionHistory.created_at.desc())
        .limit(8)
        .all()
    )
    recognitions_week = RecognitionHistory.query.filter(
        RecognitionHistory.created_at >= week_ago
    ).count()

    samples_week = DatasetSample.query.filter(
        DatasetSample.created_at >= week_ago
    ).count()

    per_gesture = (
        db.session.query(Gesture.label, func.count(DatasetSample.id))
        .outerjoin(DatasetSample, DatasetSample.gesture_id == Gesture.id)
        .group_by(Gesture.id)
        .order_by(func.count(DatasetSample.id).desc())
        .limit(8)
        .all()
    )

    published_rows = model_service.published_model_rows()
    published_model = None
    if published_rows:
        row = published_rows[0]
        published_model = {
            "version": row.version,
            "algorithm": row.algorithm,
            "accuracy": row.accuracy,
            "f1_score": row.f1_score,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "status": row.status,
        }

    latest_model = (
        ModelVersion.query.order_by(ModelVersion.created_at.desc()).first()
    )

    active_runs = (
        TrainingRun.query.filter(
            TrainingRun.status.in_([TrainingRun.STATUS_QUEUED, TrainingRun.STATUS_RUNNING])
        )
        .order_by(TrainingRun.created_at.desc())
        .all()
    )

    recent_runs = (
        TrainingRun.query.order_by(TrainingRun.created_at.desc()).limit(5).all()
    )

    dataset_summary = {
        "total_samples": total_samples,
        "samples_this_week": samples_week,
        "participant_count": total_participants,
        "consented_participants": consented_participants,
        "session_count": total_sessions,
        "gestures_with_samples": Gesture.query.filter(Gesture.id.in_(
            db.session.query(DatasetSample.gesture_id).distinct()
        )).count(),
        "per_gesture": [
            {"label": label, "sample_count": count} for label, count in per_gesture
        ],
    }

    return {
        "gestures": {
            "total": total_gestures,
            "active": active_gestures,
            "validated": validated_gestures,
            "awaiting_validation": total_gestures - validated_gestures,
        },
        "dataset": dataset_summary,
        "participants": {
            "total": total_participants,
            "consented": consented_participants,
        },
        "users": {
            "total": total_users,
            "active": active_users,
            "admins": User.query.filter_by(role=User.ROLE_ADMIN).count(),
        },
        "recognition": {
            "total": total_recognitions,
            "last_seven_days": recognitions_week,
            "recent": [entry.to_dict() for entry in recent_recognitions],
        },
        "model": {
            "published": published_model,
            "published_count": len(published_rows),
            "latest": latest_model.to_dict() if latest_model else None,
            "total_versions": ModelVersion.query.count(),
        },
        "training": {
            "active_runs": [run.to_dict() for run in active_runs],
            "recent_runs": [run.to_dict() for run in recent_runs],
        },
    }


@bp.route("/dashboard")
@login_required
def index():
    stats = _collect_statistics()
    return render_template(
        "dashboard/index.html",
        stats=stats,
        user_stats=stats["users"],
        dataset_stats=stats["dataset"],
        model_stats=stats["model"],
        activity=stats["recognition"]["recent"],
        training_runs=stats["training"]["recent_runs"],
    )


@bp.get("/api/dashboard/stats")
@login_required
def api_stats():
    return json_success(stats=_collect_statistics())
