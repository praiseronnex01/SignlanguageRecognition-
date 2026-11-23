"""Collection session: one administrator capturing samples from one participant."""

from __future__ import annotations

from datetime import datetime, timezone

from app.extensions import db


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CollectionSession(db.Model):
    __tablename__ = "collection_sessions"

    STATUS_ACTIVE = "active"
    STATUS_COMPLETED = "completed"
    STATUS_ABORTED = "aborted"
    STATUSES = {STATUS_ACTIVE, STATUS_COMPLETED, STATUS_ABORTED}

    id = db.Column(db.Integer, primary_key=True)
    session_code = db.Column(db.String(64), nullable=False, unique=True, index=True)
    participant_id = db.Column(
        db.Integer, db.ForeignKey("participants.id", ondelete="CASCADE"), nullable=False, index=True
    )
    administrator_id = db.Column(
        db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    started_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    completed_at = db.Column(db.DateTime(timezone=True), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    status = db.Column(
        db.String(20), nullable=False, default=STATUS_ACTIVE, index=True
    )

    participant = db.relationship("Participant", back_populates="sessions")
    administrator = db.relationship("User")
    samples = db.relationship(
        "DatasetSample",
        back_populates="session",
        lazy="select",
        cascade="all, delete-orphan",
    )

    @property
    def is_active(self) -> bool:
        return self.status == self.STATUS_ACTIVE

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_code": self.session_code,
            "participant_id": self.participant_id,
            "participant_code": self.participant.participant_code
            if self.participant
            else None,
            "administrator_id": self.administrator_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "notes": self.notes,
            "status": self.status,
            "sample_count": len(self.samples),
        }

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<CollectionSession {self.id} {self.session_code!r} {self.status}>"
