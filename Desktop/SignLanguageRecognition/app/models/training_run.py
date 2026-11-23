"""A managed background training run."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from app.extensions import db


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TrainingRun(db.Model):
    __tablename__ = "training_runs"

    STATUS_QUEUED = "queued"
    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"
    STATUS_CANCELLED = "cancelled"
    STATUSES = {
        STATUS_QUEUED,
        STATUS_RUNNING,
        STATUS_COMPLETED,
        STATUS_FAILED,
        STATUS_CANCELLED,
    }

    id = db.Column(db.Integer, primary_key=True)
    model_version_id = db.Column(
        db.Integer, db.ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True
    )
    status = db.Column(
        db.String(20), nullable=False, default=STATUS_QUEUED, index=True
    )
    configuration_json = db.Column(db.Text, nullable=False, default="{}")
    progress = db.Column(db.Float, nullable=False, default=0.0)
    started_at = db.Column(db.DateTime(timezone=True), nullable=True)
    completed_at = db.Column(db.DateTime(timezone=True), nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    duration_seconds = db.Column(db.Float, nullable=True)
    dataset_version = db.Column(db.String(40), nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow, index=True)

    model_versions = db.relationship(
        "ModelVersion",
        back_populates="training_run",
        foreign_keys="ModelVersion.training_run_id",
        lazy="select",
    )
    model_version = db.relationship(
        "ModelVersion",
        foreign_keys=[model_version_id],
        lazy="select",
        post_update=True,
    )
    metrics = db.relationship(
        "EvaluationMetric",
        back_populates="training_run",
        lazy="select",
        cascade="all, delete-orphan",
    )

    @property
    def configuration(self) -> dict:
        try:
            return json.loads(self.configuration_json or "{}")
        except (TypeError, ValueError):
            return {}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "model_version_id": self.model_version_id,
            "status": self.status,
            "configuration": self.configuration,
            "progress": self.progress,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "dataset_version": self.dataset_version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "model_versions": [mv.to_dict() for mv in self.model_versions],
        }

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<TrainingRun {self.id} {self.status} progress={self.progress}>"
