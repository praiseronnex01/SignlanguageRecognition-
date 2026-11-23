"""Per-user recognition history."""

from __future__ import annotations

from datetime import datetime, timezone

from app.extensions import db


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RecognitionHistory(db.Model):
    __tablename__ = "recognition_history"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    gesture_id = db.Column(
        db.Integer, db.ForeignKey("gestures.id", ondelete="SET NULL"), nullable=True, index=True
    )
    confidence = db.Column(db.Float, nullable=False)
    model_version_id = db.Column(
        db.Integer, db.ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True, index=True
    )
    translation = db.Column(db.String(160), nullable=True)
    hand_count = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow, index=True)

    user = db.relationship("User")
    gesture = db.relationship("Gesture")
    model_version = db.relationship("ModelVersion")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "gesture_id": self.gesture_id,
            "gesture_label": self.gesture.label if self.gesture else None,
            "english_meaning": self.gesture.english_meaning if self.gesture else None,
            "translation": self.translation
            or (self.gesture.english_meaning if self.gesture else None),
            "confidence": round(float(self.confidence), 4) if self.confidence is not None else None,
            "model_version_id": self.model_version_id,
            "model_version": self.model_version.version if self.model_version else None,
            "hand_count": self.hand_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<RecognitionHistory {self.id} user={self.user_id} conf={self.confidence}>"
