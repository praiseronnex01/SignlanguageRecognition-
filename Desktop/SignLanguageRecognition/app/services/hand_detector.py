"""MediaPipe Hand Landmarker service.

Responsibilities
----------------
* Load the official ``hand_landmarker.task`` model asset once and reuse it.
* Decode an incoming JPEG/PNG/WebP payload into an OpenCV BGR frame.
* Convert BGR -> RGB and run the hand landmarker.
* Return up to two hands, each with 21 landmarks and a handedness label.

The model asset is **not** bundled with the repository.  Download it with::

    curl -L -o assets/models/hand_landmarker.task \\
      https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

on Windows PowerShell::

    Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" `
      -OutFile "assets/models/hand_landmarker.task"

The expected location can be overridden with the ``MEDIAPIPE_MODEL_PATH``
environment variable.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np

from ml.feature_engineering import HandObservation

_LOCK = threading.Lock()
_LANDMARKER = None
_MEDIAPIPE = None
_LOAD_ERROR: str | None = None


class HandDetectorError(RuntimeError):
    """Raised when the detector cannot be initialised."""


@dataclass
class DetectionResult:
    """Structured output of :meth:`HandDetector.detect`."""

    hands: list = field(default_factory=list)
    frame_width: int = 0
    frame_height: int = 0
    source: str = "image"

    @property
    def hand_count(self) -> int:
        return len(self.hands)

    @property
    def handedness_labels(self) -> list:
        return [hand.handedness for hand in self.hands]

    def to_dict(self, include_landmarks: bool = True) -> dict:
        payload = {
            "hand_count": self.hand_count,
            "handedness": self.handedness_labels,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
        }
        if include_landmarks:
            payload["hands"] = [
                {
                    "landmarks": hand.landmarks,
                    "handedness": hand.handedness,
                    "handedness_score": hand.handedness_score,
                }
                for hand in self.hands
            ]
        return payload


def decode_image(payload: bytes) -> np.ndarray:
    """Decode raw image bytes into a BGR numpy array."""
    if not payload:
        raise ValueError("Empty image payload")
    buffer = np.frombuffer(payload, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("The uploaded data is not a readable image")
    return frame


def model_asset_path() -> str:
    """Return the configured path of the MediaPipe task model."""
    from flask import current_app

    try:
        configured = current_app.config.get("MEDIAPIPE_MODEL_PATH")
    except RuntimeError:  # outside of an application context
        configured = None
    return configured or os.getenv(
        "MEDIAPIPE_MODEL_PATH",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "assets",
            "models",
            "hand_landmarker.task",
        ),
    )


def is_model_available() -> bool:
    """Return ``True`` when the MediaPipe model asset is present on disk."""
    return os.path.isfile(model_asset_path())


def _build_landmarker():
    """Instantiate the MediaPipe HandLandmarker (expensive, called once)."""
    path = model_asset_path()
    if not os.path.isfile(path):
        raise HandDetectorError(
            "The MediaPipe hand landmarker model asset was not found at "
            f"'{path}'. Download 'hand_landmarker.task' into the assets/models "
            "directory or set MEDIAPIPE_MODEL_PATH to its location."
        )

    try:
        import mediapipe as mp
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )
    except ImportError as exc:  # pragma: no cover - environment specific
        raise HandDetectorError(
            "The 'mediapipe' package is not installed. Run "
            "'pip install -r requirements.txt' to install it."
        ) from exc

    from flask import current_app

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=path),
        running_mode=RunningMode.IMAGE,
        num_hands=int(current_app.config.get("MAX_HANDS", 2)),
        min_hand_detection_confidence=float(
            current_app.config.get("HAND_DETECTION_CONFIDENCE", 0.5)
        ),
        min_hand_presence_confidence=float(
            current_app.config.get("HAND_PRESENCE_CONFIDENCE", 0.5)
        ),
        min_tracking_confidence=float(
            current_app.config.get("HAND_TRACKING_CONFIDENCE", 0.5)
        ),
    )
    return HandLandmarker.create_from_options(options), mp


def get_landmarker():
    """Return (and lazily create) the shared HandLandmarker instance."""
    global _LANDMARKER, _MEDIAPIPE, _LOAD_ERROR
    if _LANDMARKER is not None:
        return _LANDMARKER, _MEDIAPIPE
    with _LOCK:
        if _LANDMARKER is None:
            _LANDMARKER, _MEDIAPIPE = _build_landmarker()
    return _LANDMARKER, _MEDIAPIPE


def reset_landmarker() -> None:
    """Drop the cached landmarker (used by tests)."""
    global _LANDMARKER, _MEDIAPIPE, _LOAD_ERROR
    with _LOCK:
        _LANDMARKER = None
        _MEDIAPIPE = None
        _LOAD_ERROR = None


class HandDetector:
    """Reusable hand detection service."""

    def __init__(self, max_hands: int = 2):
        self.max_hands = max_hands

    def detect(self, frame: np.ndarray, source: str = "image") -> DetectionResult:
        """Run hand detection on a BGR frame and return structured results."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Cannot run hand detection on an empty frame")

        landmarker, mp = get_landmarker()
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        with _LOCK:
            result = landmarker.detect(mp_image)

        hands: list = []
        # MediaPipe renamed the result attributes between releases; support both.
        raw_landmarks = (
            getattr(result, "hand_landmarks", None)
            or getattr(result, "landmarks", None)
            or []
        )
        handedness_entries = (
            getattr(result, "handednesses", None)
            or getattr(result, "handedness", None)
            or []
        )

        for index, landmark_set in enumerate(raw_landmarks):
            points = [
                [float(p.x), float(p.y), float(p.z)] for p in landmark_set
            ]
            label = "Unknown"
            score = 1.0
            if index < len(handedness_entries):
                entry = handedness_entries[index]
                # ``handedness`` is a list of category lists (one per hand).
                if isinstance(entry, (list, tuple)):
                    entry = entry[0] if entry else None
                if entry is not None:
                    category = getattr(entry, "category_name", None)
                    if category:
                        label = str(category)
                    score = float(getattr(entry, "score", 1.0) or 1.0)
            hands.append(
                HandObservation(
                    landmarks=points,
                    handedness=label,
                    handedness_score=round(score, 4),
                )
            )

        return DetectionResult(
            hands=hands[: self.max_hands],
            frame_width=width,
            frame_height=height,
            source=source,
        )

    def detect_from_bytes(self, payload: bytes, source: str = "image") -> DetectionResult:
        """Decode ``payload`` and run detection on the resulting frame."""
        return self.detect(decode_image(payload), source=source)
