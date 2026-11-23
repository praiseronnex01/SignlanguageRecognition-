"""Canonical feature engineering for Malawi Sign Language recognition.

This module is the **single source of truth** for the feature representation.
Both the offline training pipeline (:mod:`ml.train`) and the online prediction
service (:mod:`app.services.gesture_predictor`) import the functions defined
here, which guarantees that training and inference always agree on:

* how landmarks are normalised,
* the dimensionality of the feature vector,
* the ordering of the two hand slots,
* how a missing hand is represented.

Feature layout
--------------
One-hand gestures
    21 landmarks x 3 coordinates = **63** features.

Two-hand gestures
    2 x 21 landmarks x 3 coordinates = **126** coordinate features plus
    **2** presence-mask features (one per hand slot) = **128** in total.

Normalisation
-------------
Landmarks are translated so that the wrist is the origin and scaled by a palm
reference measurement (wrist -> middle-finger MCP) so that the distance of the
hand from the camera does not change the representation.  The hand is *not*
rotated onto a canonical axis: in-plane orientation carries meaning for many
static signs and must be preserved.

Handedness convention
---------------------
MediaPipe reports handedness for the *image* that was analysed.  The browser
pipeline never mirrors the captured frame before it is sent to the server, so
the handedness labels produced during dataset collection and during live
recognition are directly comparable.  Two-hand slots are therefore ordered
``Left`` first and ``Right`` second (see :func:`order_hands`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

FEATURE_VERSION = "v1.0"

LANDMARK_COUNT = 21
COORD_DIMS = 3

ONE_HAND_LANDMARK_DIMS = LANDMARK_COUNT * COORD_DIMS          # 63
TWO_HAND_LANDMARK_DIMS = 2 * ONE_HAND_LANDMARK_DIMS           # 126
TWO_HAND_MASK_DIMS = 2                                        # presence flags
TWO_HAND_TOTAL_DIMS = TWO_HAND_LANDMARK_DIMS + TWO_HAND_MASK_DIMS  # 128

#: Ordering strategies for the two-hand feature slots.
ORDER_BY_HANDEDNESS = "handedness"
ORDER_BY_SPATIAL = "spatial"
VALID_ORDERS = {ORDER_BY_HANDEDNESS, ORDER_BY_SPATIAL}

# MediaPipe hand-landmark indices (relative to the raw landmark list).
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

#: Anatomical chain used for the palm reference measurement.
PALM_FINGER_MCPS = (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)


class FeatureError(ValueError):
    """Raised when a feature vector cannot be built or fails validation."""


@dataclass
class HandObservation:
    """A single detected hand with its normalised landmarks."""

    landmarks: list  # 21 x 3 sequence of (x, y, z)
    handedness: str = "Unknown"
    handedness_score: float = 1.0

    def __post_init__(self) -> None:
        if len(self.landmarks) != LANDMARK_COUNT:
            raise FeatureError(
                f"Expected {LANDMARK_COUNT} landmarks, received {len(self.landmarks)}"
            )

    @property
    def is_left(self) -> bool:
        return self.handedness.lower().startswith("l")

    @property
    def is_right(self) -> bool:
        return self.handedness.lower().startswith("r")


@dataclass
class FeatureSchema:
    """Describes the feature representation used by a model."""

    hand_count: int = 1
    feature_version: str = FEATURE_VERSION
    order: str = ORDER_BY_HANDEDNESS
    extra: dict = field(default_factory=dict)

    @property
    def landmark_dims(self) -> int:
        return ONE_HAND_LANDMARK_DIMS * self.hand_count

    @property
    def mask_dims(self) -> int:
        return TWO_HAND_MASK_DIMS if self.hand_count == 2 else 0

    @property
    def total_dims(self) -> int:
        return self.landmark_dims + self.mask_dims

    def to_dict(self) -> dict:
        return {
            "feature_version": self.feature_version,
            "hand_count": self.hand_count,
            "landmark_count": LANDMARK_COUNT,
            "coord_dims": COORD_DIMS,
            "landmark_dims": self.landmark_dims,
            "mask_dims": self.mask_dims,
            "total_dims": self.total_dims,
            "order": self.order,
            "normalisation": "wrist-translated, palm-scaled, orientation preserved",
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FeatureSchema":
        data = data or {}
        return cls(
            hand_count=int(data.get("hand_count", 1) or 1),
            feature_version=data.get("feature_version", FEATURE_VERSION),
            order=data.get("order", ORDER_BY_HANDEDNESS),
            extra={
                k: v
                for k, v in data.items()
                if k
                not in {
                    "feature_version",
                    "hand_count",
                    "order",
                }
            },
        )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(
        (float(a[0]) - float(b[0])) ** 2
        + (float(a[1]) - float(b[1])) ** 2
        + (float(a[2]) - float(b[2])) ** 2
    )


def palm_reference_size(landmarks: Sequence[Sequence[float]]) -> float:
    """Return a stable length scale for a hand.

    The primary reference is the wrist -> middle-finger MCP distance.  If that
    measurement degenerates (e.g. the hand is cropped) the mean of the four
    finger MCP distances is used, and finally the bounding-box diagonal.
    """
    wrist = landmarks[WRIST]
    primary = _distance(wrist, landmarks[MIDDLE_MCP])
    if primary > 1e-6:
        return primary

    distances = [_distance(wrist, landmarks[i]) for i in PALM_FINGER_MCPS]
    distances = [d for d in distances if d > 1e-6]
    if distances:
        return sum(distances) / len(distances)

    xs = [float(p[0]) for p in landmarks]
    ys = [float(p[1]) for p in landmarks]
    diagonal = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    return diagonal if diagonal > 1e-6 else 1.0


def normalize_landmarks(
    landmarks: Sequence[Sequence[float]], scale: float | None = None
) -> list:
    """Translate and scale a hand's landmarks into the canonical frame.

    The wrist becomes the origin and the palm reference length becomes 1.0.
    The orientation of the hand is intentionally preserved (no rotation is
    applied) because many static signs are distinguished by wrist rotation.
    """
    if landmarks is None or len(landmarks) != LANDMARK_COUNT:
        raise FeatureError(
            f"Expected {LANDMARK_COUNT} landmarks, received "
            f"{0 if landmarks is None else len(landmarks)}"
        )

    points = []
    for point in landmarks:
        try:
            x, y, z = float(point[0]), float(point[1]), float(point[2])
        except (TypeError, IndexError, ValueError) as exc:
            raise FeatureError("Landmark contains non-numeric coordinates") from exc
        if not all(math.isfinite(v) for v in (x, y, z)):
            raise FeatureError("Landmark contains non-finite coordinates")
        points.append((x, y, z))

    wrist = points[WRIST]
    reference = float(scale) if scale else palm_reference_size(points)
    if not math.isfinite(reference) or reference <= 1e-6:
        reference = 1.0

    normalized = []
    for x, y, z in points:
        normalized.append(
            [
                (x - wrist[0]) / reference,
                (y - wrist[1]) / reference,
                (z - wrist[2]) / reference,
            ]
        )
    return normalized


def landmark_quality_score(landmarks: Sequence[Sequence[float]]) -> float:
    """Heuristic 0..1 quality score used to flag poor captures."""
    try:
        points = [(float(p[0]), float(p[1]), float(p[2])) for p in landmarks]
    except (TypeError, IndexError, ValueError):
        return 0.0
    if len(points) != LANDMARK_COUNT:
        return 0.0
    if not all(math.isfinite(v) for p in points for v in p):
        return 0.0

    spread = palm_reference_size(points)
    if spread <= 1e-6:
        return 0.0

    # A hand whose fingertips collapse onto the wrist is a degenerate capture.
    tips = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
    mean_tip_distance = sum(_distance(points[WRIST], points[t]) for t in tips) / len(tips)
    ratio = mean_tip_distance / spread
    if ratio < 0.8:
        return round(max(0.0, ratio / 0.8), 4)
    if ratio > 4.0:
        return 0.5
    return 1.0


# ---------------------------------------------------------------------------
# Hand ordering
# ---------------------------------------------------------------------------
def order_hands(
    hands: Sequence[HandObservation], mode: str = ORDER_BY_HANDEDNESS
) -> list:
    """Return the hands in a deterministic, repeatable slot order.

    ``handedness`` (default)
        The anatomically *Left* hand occupies slot 0 and the *Right* hand
        occupies slot 1.  When MediaPipe reports the same label for both
        hands the ordering falls back to the horizontal position in the image.

    ``spatial``
        Slot 0 is always the hand that appears furthest to the left in the
        image.  This is mirror-sensitive and only used when a deployment
        deliberately mirrors the capture pipeline.
    """
    if mode not in VALID_ORDERS:
        raise FeatureError(
            f"Unknown hand ordering '{mode}'. Expected one of: {sorted(VALID_ORDERS)}"
        )

    hands = list(hands)
    if len(hands) <= 1:
        return hands

    if mode == ORDER_BY_SPATIAL:
        return sorted(hands, key=lambda h: float(h.landmarks[WRIST][0]))

    def key(hand: HandObservation):
        if hand.is_right:
            rank = 1
        elif hand.is_left:
            rank = 0
        else:
            rank = 1  # unknown hands are placed in the secondary slot
        return (rank, float(hand.landmarks[WRIST][0]))

    return sorted(hands, key=key)


# ---------------------------------------------------------------------------
# Feature vector construction
# ---------------------------------------------------------------------------
def feature_dimensions(hand_count: int) -> int:
    if hand_count == 1:
        return ONE_HAND_LANDMARK_DIMS
    if hand_count == 2:
        return TWO_HAND_TOTAL_DIMS
    raise FeatureError(f"Unsupported hand count '{hand_count}'. Expected 1 or 2.")


def build_feature_vector(
    hands: Sequence[HandObservation],
    hand_count: int = 1,
    order: str = ORDER_BY_HANDEDNESS,
) -> np.ndarray:
    """Build the numeric feature vector for one observation.

    ``hands`` may contain zero, one or two detected hands.  Missing hands are
    represented by zeroed slots and a ``0`` presence flag, so the vector length
    is always exactly ``feature_dimensions(hand_count)``.
    """
    if hand_count not in (1, 2):
        raise FeatureError(f"Unsupported hand count '{hand_count}'. Expected 1 or 2.")

    ordered = order_hands(hands, mode=order)
    expected = feature_dimensions(hand_count)
    vector = np.zeros(expected, dtype=np.float64)

    if not ordered:
        raise FeatureError("No hands available to build a feature vector.")

    if hand_count == 1:
        hand = ordered[0]
        values = normalize_landmarks(hand.landmarks)
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        vector[:flat.size] = flat
        return vector

    # Two-hand layout: slot 0 = Left, slot 1 = Right, followed by the mask.
    left_hand = None
    right_hand = None
    unlabelled: list = []
    for hand in ordered:
        if hand.is_left and left_hand is None:
            left_hand = hand
        elif hand.is_right and right_hand is None:
            right_hand = hand
        else:
            unlabelled.append(hand)

    if left_hand is None and right_hand is None and ordered:
        # Both hands carry the same / unknown label: keep them in the order
        # returned by ``order_hands`` and mark the first slot as present.
        left_hand = ordered[0]
        if len(ordered) > 1:
            right_hand = ordered[1]
    else:
        for hand in unlabelled:
            if left_hand is None:
                left_hand = hand
            elif right_hand is None:
                right_hand = hand

    for slot, hand in enumerate((left_hand, right_hand)):
        if hand is None:
            continue
        values = normalize_landmarks(hand.landmarks)
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        start = slot * ONE_HAND_LANDMARK_DIMS
        vector[start : start + flat.size] = flat
        vector[TWO_HAND_LANDMARK_DIMS + slot] = 1.0

    return vector


def validate_feature_vector(
    vector: Iterable[float], hand_count: int = 1, require_present: bool = True
) -> tuple:
    """Validate a feature vector.

    Returns ``(is_valid, reason)``.  A vector is invalid when it has the
    wrong length, contains non-finite values, or when no hand slot is marked
    as present (which means the hand detector did not find a usable hand).
    """
    try:
        array = np.asarray(list(vector), dtype=np.float64)
    except (TypeError, ValueError):
        return False, "Feature vector contains non-numeric values"

    expected = feature_dimensions(hand_count)
    if array.size != expected:
        return False, f"Expected {expected} features, received {array.size}"
    if not np.all(np.isfinite(array)):
        return False, "Feature vector contains non-finite values"

    if require_present:
        if hand_count == 1:
            if not np.any(array != 0):
                return False, "Feature vector is empty"
        else:
            if array[TWO_HAND_LANDMARK_DIMS:].sum() <= 0:
                return False, "No hand slot is marked as present"

    return True, "ok"


def describe_schema(hand_count: int, order: str = ORDER_BY_HANDEDNESS) -> dict:
    """Return a JSON-serialisable description of the feature schema."""
    return FeatureSchema(hand_count=hand_count, order=order).to_dict()
