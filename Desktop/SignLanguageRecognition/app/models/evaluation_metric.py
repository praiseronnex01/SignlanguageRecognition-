"""Individual evaluation metrics attached to a training run."""

from __future__ import annotations

from datetime import datetime, timezone

from app.extensions import db


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EvaluationMetric(db.Model):
    __tablename__ = "evaluation_metrics"

    id = db.Column(db.Integer, primary_key=True)
    training_run_id = db.Column(
        db.Integer,
        db.ForeignKey("training_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    metric_name = db.Column(db.String(120), nullable=False, index=True)
    metric_value = db.Column(db.Float, nullable=True)
    metric_group = db.Column(db.String(60), nullable=True, index=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)

    training_run = db.relationship("TrainingRun", back_populates="metrics")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "training_run_id": self.training_run_id,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "metric_group": self.metric_group,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<EvaluationMetric {self.metric_name}={self.metric_value}>"
