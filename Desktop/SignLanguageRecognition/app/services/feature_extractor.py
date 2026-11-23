"""Feature extraction service for the Flask application.

All heavy lifting lives in :mod:`ml.feature_engineering`; this module adapts
the detector output to the structures the web application persists and the
prediction service consumes.
"""

from __future__ import annotations

import json

import numpy as np

from ml.feature_engineering import (
    FEATURE_VERSION,
    FeatureError,
    FeatureSchema,
    build_feature_vector,
    feature_dimensions,
    landmark_quality_score,
    normalize_landmarks,
    order_hands,
    validate_feature_vector,
)


class FeatureExtractionError(ValueError):
    """Raised when features cannot be produced for a detection result."""


def build_payload(detection, order: str = "handedness") -> dict:
    """Create the serialisable landmark payload stored in the database.

    Raw (image-normalised) landmarks are kept so that a future feature
    version can be re-derived without re-capturing the data.
    """
    if detection is None or not getattr(detection, "hands", None):
        raise FeatureExtractionError("No hand was detected in the supplied frame")

    hands = order_hands(detection.hands, mode=order)
    return {
        "feature_version": FEATURE_VERSION,
        "order": order,
        "hands": [
            {
                "handedness": hand.handedness,
                "handedness_score": round(float(hand.handedness_score), 4),
                "landmarks": [[float(c) for c in point] for point in hand.landmarks],
            }
            for hand in hands
        ],
    }


def payload_to_hands(payload: dict):
    """Rebuild ``HandObservation`` objects from a stored payload."""
    from ml.feature_engineering import HandObservation

    if not payload:
        raise FeatureExtractionError("Empty landmark payload")
    hands = payload.get("hands") if isinstance(payload, dict) else None
    if not hands:
        raise FeatureExtractionError("Landmark payload contains no hands")
    observations = []
    for entry in hands:
        landmarks = entry.get("landmarks")
        if not landmarks:
            raise FeatureExtractionError("Landmark payload entry has no landmarks")
        observations.append(
            HandObservation(
                landmarks=landmarks,
                handedness=entry.get("handedness", "Unknown"),
                handedness_score=float(entry.get("handedness_score", 1.0) or 1.0),
            )
        )
    return observations


def extract_features(
    detection,
    hand_count: int = 1,
    order: str = "handedness",
    require_present: bool = True,
) -> np.ndarray:
    """Build the numeric feature vector for a detection result.

    ``hand_count`` selects the feature schema (1 -> 63 features, 2 -> 128).
    """
    if hand_count not in (1, 2):
        raise FeatureExtractionError(f"Unsupported hand count '{hand_count}'")

    hands = list(getattr(detection, "hands", []) or [])
    if not hands:
        raise FeatureExtractionError("No hand was detected in the supplied frame")

    try:
        vector = build_feature_vector(hands, hand_count=hand_count, order=order)
    except FeatureError as exc:
        raise FeatureExtractionError(str(exc)) from exc

    valid, reason = validate_feature_vector(
        vector, hand_count=hand_count, require_present=require_present
    )
    if not valid:
        raise FeatureExtractionError(reason)
    return vector


def features_from_payload(payload: dict, hand_count: int, order: str = "handedness"):
    """Rebuild the feature vector for a stored dataset payload."""
    hands = payload_to_hands(payload)
    try:
        vector = build_feature_vector(hands, hand_count=hand_count, order=order)
    except FeatureError as exc:
        raise FeatureExtractionError(str(exc)) from exc
    return vector


def quality_score(detection) -> float:
    """Return the mean quality score across the detected hands."""
    hands = list(getattr(detection, "hands", []) or [])
    if not hands:
        return 0.0
    scores = [landmark_quality_score(hand.landmarks) for hand in hands]
    return round(sum(scores) / len(scores), 4)


def schema_for(hand_count: int, order: str = "handedness") -> dict:
    return FeatureSchema(hand_count=hand_count, order=order).to_dict()


def expected_dimensions(hand_count: int) -> int:
    return feature_dimensions(hand_count)


def dumps(payload: dict) -> str:
    return json.dumps(payload, separators=(",", ":"))


def loads(raw: str) -> dict:
    try:
        return json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise FeatureExtractionError("Stored landmark payload is not valid JSON") from exc


__all__ = [
    "FEATURE_VERSION",
    "FeatureExtractionError",
    "build_payload",
    "dumps",
    "expected_dimensions",
    "extract_features",
    "features_from_payload",
    "loads",
    "normalize_landmarks",
    "payload_to_hands",
    "quality_score",
    "schema_for",
]
