"""Dataset management service.

Handles participant registration, collection sessions, sample persistence,
data validation, class-balance reporting and the export of a training-ready
dataset file.

Raw video is **never** stored.  Only extracted hand landmarks and the
metadata required to train and audit the classifier are persisted.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import secrets
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone

from sqlalchemy import func

from app.extensions import db
from app.models.collection_session import CollectionSession
from app.models.dataset_sample import DatasetSample
from app.models.gesture import Gesture
from app.models.participant import Participant
from app.services import feature_extractor
from ml.feature_engineering import (
    ONE_HAND_LANDMARK_DIMS,
    TWO_HAND_TOTAL_DIMS,
    FeatureError,
    build_feature_vector,
    validate_feature_vector,
)

MAX_FEATURE_DIMS = max(ONE_HAND_LANDMARK_DIMS, TWO_HAND_TOTAL_DIMS)

BASE_COLUMNS = [
    "sample_id",
    "gesture_id",
    "gesture_label",
    "english_meaning",
    "participant_code",
    "session_code",
    "hand_count",
    "handedness",
    "feature_version",
    "quality_score",
    "created_at",
]
FEATURE_COLUMNS = [f"f{index}" for index in range(MAX_FEATURE_DIMS)]


class DatasetError(RuntimeError):
    """Raised for invalid dataset operations."""


# ---------------------------------------------------------------------------
# Participants and sessions
# ---------------------------------------------------------------------------
def get_or_create_participant(participant_code: str, consent_status: str | None = None):
    """Return the participant for ``participant_code``, creating it if needed."""
    code = (participant_code or "").strip()
    if not code:
        raise DatasetError("A participant code is required")
    if len(code) > 64:
        raise DatasetError("The participant code is too long (max 64 characters)")

    participant = Participant.query.filter_by(participant_code=code).first()
    if participant is None:
        participant = Participant(
            participant_code=code,
            consent_status=consent_status or Participant.CONSENT_PENDING,
        )
        db.session.add(participant)
        db.session.commit()
    elif consent_status and participant.consent_status != consent_status:
        participant.consent_status = consent_status
        participant.consent_recorded_at = datetime.now(timezone.utc)
        db.session.commit()
    return participant


def set_consent(participant: Participant, consent_status: str) -> Participant:
    if consent_status not in Participant.CONSENT_STATES:
        raise DatasetError(
            f"Invalid consent status '{consent_status}'. Expected one of: "
            f"{', '.join(sorted(Participant.CONSENT_STATES))}"
        )
    participant.consent_status = consent_status
    participant.consent_recorded_at = datetime.now(timezone.utc)
    db.session.commit()
    return participant


def create_session(participant: Participant, administrator=None, notes: str | None = None):
    """Open a collection session for a participant."""
    if participant is None:
        raise DatasetError("A participant is required to start a session")
    if not participant.has_consent:
        raise DatasetError(
            f"Participant '{participant.participant_code}' has not given consent "
            "for data collection."
        )

    active = CollectionSession.query.filter_by(
        participant_id=participant.id, status=CollectionSession.STATUS_ACTIVE
    ).first()
    if active is not None:
        return active

    code = f"SES-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{secrets.token_hex(3).upper()}"
    session = CollectionSession(
        session_code=code,
        participant_id=participant.id,
        administrator_id=getattr(administrator, "id", None),
        notes=notes,
        status=CollectionSession.STATUS_ACTIVE,
    )
    db.session.add(session)
    db.session.commit()
    return session


def close_session(session: CollectionSession, status: str = CollectionSession.STATUS_COMPLETED):
    if session.status != CollectionSession.STATUS_ACTIVE:
        return session
    session.status = status
    session.completed_at = datetime.now(timezone.utc)
    db.session.commit()
    return session


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------
def store_sample(
    gesture: Gesture,
    participant: Participant,
    session: CollectionSession,
    payload: dict,
    handedness: str | None = None,
    quality_score: float | None = None,
    allow_duplicate: bool = False,
) -> DatasetSample:
    """Persist one labelled landmark sample.

    The sample is validated before it is written: the payload must contain the
    expected number of hands and the derived feature vector must be finite and
    correctly sized.
    """
    if gesture is None:
        raise DatasetError("A gesture label is required")
    if participant is None:
        raise DatasetError("A participant is required")
    if session is None:
        raise DatasetError("A collection session is required")
    if not participant.has_consent:
        raise DatasetError(
            f"Participant '{participant.participant_code}' has not given consent "
            "for data collection."
        )
    if session.status != CollectionSession.STATUS_ACTIVE:
        raise DatasetError("The collection session is not active")

    hands = payload.get("hands") or []
    expected_hands = int(gesture.hand_count)
    if len(hands) != expected_hands:
        raise DatasetError(
            f"Gesture '{gesture.label}' requires {expected_hands} hand(s) but "
            f"{len(hands)} were detected"
        )

    try:
        vector = build_feature_vector(
            feature_extractor.payload_to_hands(payload),
            hand_count=expected_hands,
            order=payload.get("order", "handedness"),
        )
    except (FeatureError, feature_extractor.FeatureExtractionError) as exc:
        raise DatasetError(f"Invalid landmark payload: {exc}") from exc

    valid, reason = validate_feature_vector(vector, hand_count=expected_hands)
    if not valid:
        raise DatasetError(f"Invalid landmark payload: {reason}")

    feature_hash = DatasetSample.compute_hash(payload.get("hands"))
    if not allow_duplicate:
        existing = DatasetSample.query.filter_by(
            gesture_id=gesture.id, feature_hash=feature_hash
        ).first()
        if existing is not None:
            raise DatasetError("This sample duplicates an existing sample")

    sample = DatasetSample(
        gesture_id=gesture.id,
        participant_id=participant.id,
        session_id=session.id,
        landmarks_json=feature_extractor.dumps(payload),
        handedness=(handedness or ", ".join(h.get("handedness", "Unknown") for h in hands)),
        hand_count=len(hands),
        feature_version=payload.get("feature_version", "v1.0"),
        feature_hash=feature_hash,
        quality_score=quality_score,
    )
    db.session.add(sample)
    db.session.commit()
    return sample


def delete_sample(sample_id: int) -> bool:
    sample = db.session.get(DatasetSample, sample_id)
    if sample is None:
        return False
    db.session.delete(sample)
    db.session.commit()
    return True


# ---------------------------------------------------------------------------
# Statistics and validation
# ---------------------------------------------------------------------------
def dataset_stats() -> dict:
    """Aggregate statistics computed from the database."""
    total_samples = DatasetSample.query.count()

    per_gesture_rows = (
        db.session.query(
            Gesture.id,
            Gesture.label,
            Gesture.english_meaning,
            Gesture.is_validated,
            Gesture.is_active,
            Gesture.hand_count,
            func.count(DatasetSample.id),
        )
        .outerjoin(DatasetSample, DatasetSample.gesture_id == Gesture.id)
        .group_by(Gesture.id)
        .order_by(Gesture.label)
        .all()
    )

    per_gesture = [
        {
            "gesture_id": row[0],
            "label": row[1],
            "english_meaning": row[2],
            "is_validated": row[3],
            "is_active": row[4],
            "hand_count": row[5],
            "sample_count": row[6],
        }
        for row in per_gesture_rows
    ]

    per_participant_rows = (
        db.session.query(
            Participant.participant_code,
            func.count(DatasetSample.id),
        )
        .outerjoin(DatasetSample, DatasetSample.participant_id == Participant.id)
        .group_by(Participant.id)
        .order_by(func.count(DatasetSample.id).desc())
        .all()
    )

    per_participant = [
        {"participant_code": row[0], "sample_count": row[1]} for row in per_participant_rows
    ]

    counts = [entry["sample_count"] for entry in per_gesture]
    non_zero = [c for c in counts if c > 0]
    balance = None
    if non_zero:
        balance = round(min(non_zero) / max(non_zero), 4) if max(non_zero) else None

    duplicates = (
        db.session.query(
            DatasetSample.feature_hash,
            func.count(DatasetSample.id),
        )
        .filter(DatasetSample.feature_hash.isnot(None))
        .group_by(DatasetSample.feature_hash)
        .having(func.count(DatasetSample.id) > 1)
        .all()
    )

    hand_counts = (
        db.session.query(DatasetSample.hand_count, func.count(DatasetSample.id))
        .group_by(DatasetSample.hand_count)
        .all()
    )

    validated_classes = sum(
        1 for entry in per_gesture if entry["is_validated"] and entry["sample_count"] > 0
    )

    return {
        "total_samples": total_samples,
        "total_gestures": len(per_gesture),
        "gestures_with_samples": len(non_zero),
        "validated_classes": validated_classes,
        "per_gesture": per_gesture,
        "per_participant": per_participant,
        "participant_count": len(per_participant),
        "class_balance_ratio": balance,
        "duplicate_groups": len(duplicates),
        "duplicate_samples": sum(count - 1 for _, count in duplicates),
        "by_hand_count": {str(row[0]): row[1] for row in hand_counts},
        "total_sessions": CollectionSession.query.count(),
        "active_sessions": CollectionSession.query.filter_by(
            status=CollectionSession.STATUS_ACTIVE
        ).count(),
        "participants": Participant.query.count(),
    }


def validate_dataset() -> dict:
    """Check every stored sample for structural problems."""
    problems: list = []
    checked = 0
    for sample in DatasetSample.query.all():
        checked += 1
        gesture = sample.gesture
        try:
            payload = feature_extractor.loads(sample.landmarks_json)
            hands = payload.get("hands") or []
            if gesture is None:
                problems.append(
                    {"sample_id": sample.id, "issue": "Sample has no gesture"}
                )
                continue
            if len(hands) != int(gesture.hand_count):
                problems.append(
                    {
                        "sample_id": sample.id,
                        "issue": (
                            f"Hand count mismatch: expected {gesture.hand_count}, "
                            f"found {len(hands)}"
                        ),
                    }
                )
                continue
            vector = build_feature_vector(
                feature_extractor.payload_to_hands(payload),
                hand_count=int(gesture.hand_count),
                order=payload.get("order", "handedness"),
            )
            valid, reason = validate_feature_vector(
                vector, hand_count=int(gesture.hand_count)
            )
            if not valid:
                problems.append({"sample_id": sample.id, "issue": reason})
        except (FeatureError, feature_extractor.FeatureExtractionError, ValueError) as exc:
            problems.append({"sample_id": sample.id, "issue": str(exc)})

    return {
        "checked": checked,
        "valid": checked - len(problems),
        "invalid": len(problems),
        "problems": problems[:200],
    }


def class_distribution() -> dict:
    """Return the number of samples per gesture label."""
    rows = (
        db.session.query(Gesture.label, func.count(DatasetSample.id))
        .join(DatasetSample, DatasetSample.gesture_id == Gesture.id)
        .group_by(Gesture.label)
        .order_by(func.count(DatasetSample.id).desc())
        .all()
    )
    return {row[0]: row[1] for row in rows}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def _dataset_dir():
    """Return the configured dataset directory."""
    try:
        from flask import current_app

        return Path(current_app.config["DATASET_DIR"])
    except (RuntimeError, KeyError):  # pragma: no cover - outside app context
        from app.config import DATASET_DIR

        return Path(DATASET_DIR)


def _next_dataset_version() -> int:
    metadata_dir = _dataset_dir() / "metadata"
    versions = []
    if metadata_dir.is_dir():
        for entry in metadata_dir.glob("dataset_v*.json"):
            try:
                versions.append(int(entry.stem.split("_v")[1]))
            except (IndexError, ValueError):
                continue
    return (max(versions) + 1) if versions else 1


def export_dataset(include_unvalidated: bool = False) -> dict:
    """Write a training-ready CSV plus a metadata document.

    Only gestures that are enabled (and, unless ``include_unvalidated`` is
    set, validated) are exported.
    """
    dataset_root = _dataset_dir()
    query = db.session.query(DatasetSample).join(Gesture)
    if not include_unvalidated:
        query = query.filter(Gesture.is_validated.is_(True))
    samples = query.order_by(DatasetSample.id).all()

    if not samples:
        raise DatasetError(
            "There are no samples to export. Capture samples first, and make "
            "sure the gestures have been validated by an administrator."
        )

    version = _next_dataset_version()
    processed_dir = dataset_root / "processed"
    exports_dir = dataset_root / "exports"
    metadata_dir = dataset_root / "metadata"
    for directory in (processed_dir, exports_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)

    csv_path = processed_dir / f"dataset_v{version}.csv"
    export_path = exports_dir / f"dataset_v{version}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"

    header = BASE_COLUMNS + FEATURE_COLUMNS
    per_gesture: Counter = Counter()
    per_participant: Counter = Counter()
    per_hand_count: Counter = Counter()
    written = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for sample in samples:
            gesture = sample.gesture
            participant = sample.participant
            session = sample.session
            if gesture is None:
                continue
            try:
                payload = feature_extractor.loads(sample.landmarks_json)
                vector = build_feature_vector(
                    feature_extractor.payload_to_hands(payload),
                    hand_count=int(gesture.hand_count),
                    order=payload.get("order", "handedness"),
                )
            except (FeatureError, feature_extractor.FeatureExtractionError, ValueError):
                continue

            row = {
                "sample_id": sample.id,
                "gesture_id": gesture.id,
                "gesture_label": gesture.label,
                "english_meaning": gesture.english_meaning,
                "participant_code": participant.participant_code if participant else "",
                "session_code": session.session_code if session else "",
                "hand_count": int(gesture.hand_count),
                "handedness": sample.handedness or "",
                "feature_version": sample.feature_version,
                "quality_score": sample.quality_score if sample.quality_score is not None else "",
                "created_at": sample.created_at.isoformat() if sample.created_at else "",
            }
            values = list(vector) + [0.0] * (MAX_FEATURE_DIMS - len(vector))
            for index, value in enumerate(values):
                row[FEATURE_COLUMNS[index]] = f"{float(value):.6f}"
            writer.writerow(row)
            written += 1
            per_gesture[gesture.label] += 1
            if participant:
                per_participant[participant.participant_code] += 1
            per_hand_count[str(int(gesture.hand_count))] += 1

    with open(csv_path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()

    metadata = {
        "version": f"v{version}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": written,
        "class_count": len(per_gesture),
        "participant_count": len(per_participant),
        "samples_per_gesture": dict(per_gesture),
        "samples_per_participant": dict(per_participant),
        "samples_per_hand_count": dict(per_hand_count),
        "include_unvalidated": include_unvalidated,
        "feature_columns": FEATURE_COLUMNS,
        "feature_dims": {
            "1": ONE_HAND_LANDMARK_DIMS,
            "2": TWO_HAND_TOTAL_DIMS,
        },
        "sha256": digest,
        "csv_path": str(csv_path),
        "export_path": str(export_path),
    }
    metadata_path = metadata_dir / f"dataset_v{version}.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    # Keep a timestamped copy in the exports folder.
    with open(csv_path, "rb") as source, open(export_path, "wb") as target:
        target.write(source.read())

    metadata["metadata_path"] = str(metadata_path)
    metadata["export_path"] = str(export_path)
    return metadata


def list_dataset_versions() -> list:
    """Return the exported dataset versions, newest first."""
    dataset_root = _dataset_dir()
    metadata_dir = dataset_root / "metadata"
    versions: list = []
    if metadata_dir.is_dir():
        for path in sorted(metadata_dir.glob("dataset_v*.json")):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                data["metadata_path"] = str(path)
                data["csv_path"] = data.get(
                    "csv_path",
                    str(dataset_root / "processed" / f"dataset_{data.get('version')}.csv"),
                )
                versions.append(data)
            except (OSError, ValueError):
                continue
    versions.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return versions


def duplicate_report() -> dict:
    """Group samples by their feature hash to expose duplicates."""
    rows = (
        db.session.query(
            DatasetSample.feature_hash,
            func.count(DatasetSample.id),
            func.min(DatasetSample.id),
        )
        .filter(DatasetSample.feature_hash.isnot(None))
        .group_by(DatasetSample.feature_hash)
        .having(func.count(DatasetSample.id) > 1)
        .all()
    )
    return {
        "duplicate_groups": len(rows),
        "duplicate_samples": sum(count - 1 for _, count, _ in rows),
        "groups": [
            {"feature_hash": digest, "count": count, "first_sample_id": first}
            for digest, count, first in rows[:100]
        ],
    }


def participant_summary() -> list:
    """Per-participant contribution summary used by the admin dashboard."""
    rows = (
        db.session.query(
            Participant.id,
            Participant.participant_code,
            Participant.consent_status,
            func.count(DatasetSample.id),
            func.count(func.distinct(DatasetSample.gesture_id)),
            func.count(func.distinct(DatasetSample.session_id)),
        )
        .outerjoin(DatasetSample, DatasetSample.participant_id == Participant.id)
        .group_by(Participant.id)
        .order_by(func.count(DatasetSample.id).desc())
        .all()
    )
    return [
        {
            "id": row[0],
            "participant_code": row[1],
            "consent_status": row[2],
            "sample_count": row[3],
            "gesture_count": row[4],
            "session_count": row[5],
        }
        for row in rows
    ]


def remove_invalid_samples() -> int:
    """Delete samples that fail structural validation."""
    report = validate_dataset()
    invalid_ids = [problem["sample_id"] for problem in report["problems"]]
    removed = 0
    for sample_id in invalid_ids:
        sample = db.session.get(DatasetSample, sample_id)
        if sample is not None:
            db.session.delete(sample)
            removed += 1
    if removed:
        db.session.commit()
    return removed


def os_removed_paths(paths) -> None:
    """Utility used by tests to clean generated dataset files."""
    for path in paths:
        if path and os.path.isfile(path):
            os.remove(path)
