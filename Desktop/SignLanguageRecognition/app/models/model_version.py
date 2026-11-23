"""Versioned machine-learning model artifacts and their evaluation scores."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from app.extensions import db


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ModelVersion(db.Model):
    __tablename__ = "model_versions"

    STATUS_TRAINING = "training"
    STATUS_CANDIDATE = "candidate"
    STATUS_PUBLISHED = "published"
    STATUS_ARCHIVED = "archived"
    STATUS_FAILED = "failed"
    STATUSES = {
        STATUS_TRAINING,
        STATUS_CANDIDATE,
        STATUS_PUBLISHED,
        STATUS_ARCHIVED,
        STATUS_FAILED,
    }

    id = db.Column(db.Integer, primary_key=True)
    version = db.Column(db.String(40), nullable=False, unique=True, index=True)
    algorithm = db.Column(db.String(60), nullable=False, index=True)
    model_path = db.Column(db.String(255), nullable=False)
    encoder_path = db.Column(db.String(255), nullable=False)
    feature_schema = db.Column(db.Text, nullable=False)  # JSON document
    accuracy = db.Column(db.Float, nullable=True)
    precision = db.Column(db.Float, nullable=True)
    recall = db.Column(db.Float, nullable=True)
    f1_score = db.Column(db.Float, nullable=True)
    status = db.Column(
        db.String(20), nullable=False, default=STATUS_CANDIDATE, index=True
    )
    rejection_threshold = db.Column(db.Float, nullable=True)
    dataset_version = db.Column(db.String(40), nullable=True, index=True)
    feature_version = db.Column(db.String(20), nullable=False, default="v1.0")
    training_run_id = db.Column(
        db.Integer,
        db.ForeignKey(
            "training_runs.id",
            ondelete="SET NULL",
            name="fk_model_version_training_run",
            use_alter=True,
        ),
        nullable=True,
    )
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow, index=True)

    training_run = db.relationship(
        "TrainingRun",
        back_populates="model_versions",
        foreign_keys=[training_run_id],
    )

    @property
    def schema(self) -> dict:
        try:
            return json.loads(self.feature_schema or "{}")
        except (TypeError, ValueError):
            return {}

    @property
    def hand_count(self) -> int:
        return int(self.schema.get("hand_count", 1) or 1)

    @property
    def is_published(self) -> bool:
        return self.status == self.STATUS_PUBLISHED

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "version": self.version,
            "algorithm": self.algorithm,
            "model_path": self.model_path,
            "encoder_path": self.encoder_path,
            "feature_schema": self.schema,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "status": self.status,
            "rejection_threshold": self.rejection_threshold,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "training_run_id": self.training_run_id,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<ModelVersion {self.id} {self.version!r} {self.algorithm!r} {self.status}>"
