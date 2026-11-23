"""Gesture vocabulary model.

A ``Gesture`` row is the authoritative record for one static Malawi Sign
Language sign.  Signs are only usable for recognition once an administrator
has verified them against the Malawian Sign Language Dictionary or with a
competent MSL signer and flipped ``is_validated`` to ``True``.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.extensions import db

# Hand configurations supported by the feature extractor.
HAND_ONE = 1
HAND_TWO = 2
VALID_HAND_COUNTS = {HAND_ONE, HAND_TWO}

# Review states for the vocabulary.
STATUS_CANDIDATE = "candidate"
STATUS_VALIDATED = "validated"
STATUS_REJECTED = "rejected"
VALID_STATUSES = {STATUS_CANDIDATE, STATUS_VALIDATED, STATUS_REJECTED}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Gesture(db.Model):
    __tablename__ = "gestures"

    id = db.Column(db.Integer, primary_key=True)
    label = db.Column(db.String(80), nullable=False, unique=True, index=True)
    english_meaning = db.Column(db.String(160), nullable=False)
    category = db.Column(db.String(80), nullable=False, default="General", index=True)
    description = db.Column(db.Text, nullable=True)
    reference_image = db.Column(db.String(255), nullable=True)
    reference_video = db.Column(db.String(255), nullable=True)
    is_validated = db.Column(db.Boolean, nullable=False, default=False, index=True)
    is_active = db.Column(db.Boolean, nullable=False, default=True, index=True)
    # Number of hands required to perform the sign (1 or 2).  This selects the
    # feature schema used both at collection time and at inference time.
    hand_count = db.Column(db.Integer, nullable=False, default=HAND_ONE)
    validation_status = db.Column(
        db.String(20), nullable=False, default=STATUS_CANDIDATE, index=True
    )
    validation_notes = db.Column(db.Text, nullable=True)
    source_reference = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utcnow)
    updated_at = db.Column(
        db.DateTime(timezone=True), nullable=False, default=utcnow, onupdate=utcnow
    )

    samples = db.relationship(
        "DatasetSample",
        back_populates="gesture",
        lazy="select",
        cascade="all, delete-orphan",
    )

    @property
    def is_available(self) -> bool:
        """A gesture is recognisable only when validated and enabled."""
        return bool(self.is_validated and self.is_active)

    def to_dict(self, include_sample_count: bool = False) -> dict:
        data = {
            "id": self.id,
            "label": self.label,
            "english_meaning": self.english_meaning,
            "category": self.category,
            "description": self.description,
            "reference_image": self.reference_image,
            "reference_video": self.reference_video,
            "is_validated": self.is_validated,
            "is_active": self.is_active,
            "hand_count": self.hand_count,
            "validation_status": self.validation_status,
            "validation_notes": self.validation_notes,
            "source_reference": self.source_reference,
            "recognition_available": self.is_available,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_sample_count:
            data["sample_count"] = len(self.samples)
        return data

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Gesture {self.id} {self.label!r} validated={self.is_validated}>"
