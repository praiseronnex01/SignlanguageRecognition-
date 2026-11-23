"""A single labelled landmark sample captured from the webcam.

Only extracted hand landmarks are persisted - never raw video frames.  The
``landmarks_json`` column holds the normalised landmark payload; the
``feature_hash`` column is a stable digest used for duplicate detection.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from app.extensions import db


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DatasetSample(db.Model):
    __tablename__ = "dataset_samples"

    id = db.Column(db.Integer, primary_key=True)
    gesture_id = db.Column(
        db.Integer, db.ForeignKey("gestures.id", ondelete="CASCADE"), nullable=False, index=True
    )
    participant_id = db.Column(
        db.Integer, db.ForeignKey("participants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    session_id = db.Column(
        db.Integer,
        db.ForeignKey("collection_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    landmarks_json = db.Column(db.Text, nullable=False)
    handedness = db.Column(db.String(16), nullable=True)
    hand_count = db.Column(db.Integer, nullable=False, default=1, index=True)
    feature_version = db.Column(db.String(20), nullable=False, default="v1.0", index=True)
    feature_hash = db.Column(db.String(64), nullable=True, index=True)
    quality_score = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow, index=True)

    gesture = db.relationship("Gesture", back_populates="samples")
    participant = db.relationship("Participant", back_populates="samples")
    session = db.relationship("CollectionSession", back_populates="samples")

    @staticmethod
    def compute_hash(landmarks: list) -> str:
        """Stable digest of a landmark payload for duplicate detection."""
        payload = json.dumps(landmarks, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self, include_landmarks: bool = False) -> dict:
        data = {
            "id": self.id,
            "gesture_id": self.gesture_id,
            "gesture_label": self.gesture.label if self.gesture else None,
            "participant_id": self.participant_id,
            "participant_code": self.participant.participant_code
            if self.participant
            else None,
            "session_id": self.session_id,
            "session_code": self.session.session_code if self.session else None,
            "handedness": self.handedness,
            "hand_count": self.hand_count,
            "feature_version": self.feature_version,
            "quality_score": self.quality_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_landmarks:
            data["landmarks"] = json.loads(self.landmarks_json)
        return data

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<DatasetSample {self.id} gesture={self.gesture_id} p={self.participant_id}>"
