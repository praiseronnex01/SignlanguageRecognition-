"""Anonymous dataset participant record.

Participants are identified only by an anonymous code.  No names, contact
details or biometric data are stored, and raw camera footage is never kept.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.extensions import db


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Participant(db.Model):
    __tablename__ = "participants"

    CONSENT_PENDING = "pending"
    CONSENT_GRANTED = "granted"
    CONSENT_DECLINED = "declined"
    CONSENT_WITHDRAWN = "withdrawn"
    CONSENT_STATES = {
        CONSENT_PENDING,
        CONSENT_GRANTED,
        CONSENT_DECLINED,
        CONSENT_WITHDRAWN,
    }

    id = db.Column(db.Integer, primary_key=True)
    participant_code = db.Column(
        db.String(64), nullable=False, unique=True, index=True
    )
    consent_status = db.Column(
        db.String(20), nullable=False, default=CONSENT_PENDING, index=True
    )
    consent_recorded_at = db.Column(db.DateTime(timezone=True), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)

    sessions = db.relationship(
        "CollectionSession",
        back_populates="participant",
        lazy="select",
        cascade="all, delete-orphan",
    )
    samples = db.relationship(
        "DatasetSample",
        back_populates="participant",
        lazy="select",
        cascade="all, delete-orphan",
    )

    @property
    def has_consent(self) -> bool:
        return self.consent_status == self.CONSENT_GRANTED

    def to_dict(self, include_sample_count: bool = False) -> dict:
        data = {
            "id": self.id,
            "participant_code": self.participant_code,
            "consent_status": self.consent_status,
            "consent_recorded_at": (
                self.consent_recorded_at.isoformat()
                if self.consent_recorded_at
                else None
            ),
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_sample_count:
            data["sample_count"] = len(self.samples)
        return data

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Participant {self.id} {self.participant_code!r}>"
