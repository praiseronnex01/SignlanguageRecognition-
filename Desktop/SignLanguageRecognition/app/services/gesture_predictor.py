"""Gesture prediction service.

Combines the computer-vision pipeline (MediaPipe hand landmarks) with the
published machine-learning model to turn a single camera frame into an
interpretable recognition result.

Guarantees enforced here
------------------------
* A gesture is **never** reported when hand detection failed.
* A gesture is **never** reported when the model is unavailable.
* A prediction whose confidence is below the rejection threshold is reported
  as *"Gesture not recognised."*
* The response always carries the model version that produced it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from app.services import feature_extractor, model_service
from ml.feature_engineering import FeatureError


class PredictionError(RuntimeError):
    """Raised when a prediction cannot be produced."""


@dataclass
class PredictionOutcome:
    """The outcome of a single prediction attempt."""

    recognized: bool = False
    gesture: str | None = None
    translation: str | None = None
    confidence: float = 0.0
    hand_detected: bool = False
    hand_count: int = 0
    model_version: str | None = None
    message: str | None = None
    hand_count_expected: int | None = None
    latency_ms: float = 0.0
    alternatives: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "success": True,
            "recognized": self.recognized,
            "gesture": self.gesture,
            "translation": self.translation,
            "confidence": round(float(self.confidence), 4),
            "hand_detected": self.hand_detected,
            "hand_count": self.hand_count,
            "hand_count_expected": self.hand_count_expected,
            "model_version": self.model_version,
            "message": self.message,
            "latency_ms": round(self.latency_ms, 2),
            "alternatives": self.alternatives,
        }


def _lookup_gesture(label: str):
    from app.models.gesture import Gesture

    if not label:
        return None
    return Gesture.query.filter_by(label=label).first()


def _effective_threshold(model, config) -> float:
    threshold = config.get("RECOGNITION_CONFIDENCE_THRESHOLD", 0.60)
    if model is not None and model.rejection_threshold is not None:
        try:
            threshold = float(model.rejection_threshold)
        except (TypeError, ValueError):
            pass
    return float(threshold)


def _predict_with_model(model, vector: np.ndarray) -> tuple:
    """Return ``(label, confidence)`` for a feature vector."""
    pipeline = model.pipeline
    probabilities = None

    if hasattr(pipeline, "predict_proba"):
        try:
            probabilities = pipeline.predict_proba(np.asarray([vector]))
        except (ValueError, TypeError):
            probabilities = None

    if probabilities is not None and len(probabilities):
        probabilities = probabilities[0]
        classes = [str(c) for c in getattr(model.encoder, "classes_", model.classes)]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        label = classes[best_index] if best_index < len(classes) else str(pipeline.classes_[best_index])
        alternatives = [
            {"gesture": classes[i] if i < len(classes) else str(i), "confidence": round(float(probabilities[i]), 4)}
            for i in np.argsort(probabilities)[::-1][1:4]
        ]
        return label, confidence, alternatives

    prediction = pipeline.predict(np.asarray([vector]))
    label = str(prediction[0])
    return label, 1.0, []


def predict_frame(frame, config=None) -> PredictionOutcome:
    """Run the full recognition pipeline on a BGR frame."""
    from flask import current_app

    config = config or current_app.config
    started = time.perf_counter()

    from app.services.hand_detector import HandDetector, HandDetectorError

    try:
        detection = HandDetector(max_hands=int(config.get("MAX_HANDS", 2))).detect(frame)
    except HandDetectorError as exc:
        raise PredictionError(str(exc)) from exc
    except ValueError as exc:
        raise PredictionError(str(exc)) from exc

    outcome = predict_from_detection(detection, config=config)
    outcome.latency_ms = (time.perf_counter() - started) * 1000.0
    return outcome


def predict_from_detection(detection, config=None) -> PredictionOutcome:
    """Run the model on an already computed detection result."""
    from flask import current_app

    config = config or current_app.config

    hands = list(getattr(detection, "hands", []) or [])
    if not hands:
        return PredictionOutcome(
            recognized=False,
            hand_detected=False,
            message="No hand detected",
        )

    models = model_service.published_models(config)
    if not models:
        return PredictionOutcome(
            recognized=False,
            hand_detected=True,
            hand_count=len(hands),
            message=(
                "No published model is available. Train and publish a model "
                "before using live recognition."
            ),
        )

    candidates = _candidate_models(models, len(hands))
    if not candidates:
        return PredictionOutcome(
            recognized=False,
            hand_detected=True,
            hand_count=len(hands),
            message=(
                f"{len(hands)} hand(s) detected but no published model matches "
                "this hand configuration."
            ),
        )

    best: tuple | None = None
    for model, hand_count, use_single_hand in candidates:
        try:
            vector = feature_extractor.extract_features(
                detection, hand_count=hand_count, require_present=True
            )
        except feature_extractor.FeatureExtractionError:
            continue

        if model.expected_dims and vector.size != model.expected_dims:
            continue

        try:
            label, confidence, alternatives = _predict_with_model(model, vector)
        except (ValueError, FeatureError, IndexError) as exc:
            current_app.logger.warning(
                "Model %s failed to score a frame: %s", model.version, exc
            )
            continue

        candidate = (confidence, model, label, hand_count, alternatives)
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        return PredictionOutcome(
            recognized=False,
            hand_detected=True,
            hand_count=len(hands),
            message="The model could not be evaluated for this frame.",
        )

    confidence, model, label, hand_count, alternatives = best
    threshold = _effective_threshold(model, config)

    if confidence < threshold:
        return PredictionOutcome(
            recognized=False,
            hand_detected=True,
            hand_count=len(hands),
            hand_count_expected=hand_count,
            confidence=confidence,
            model_version=model.version,
            translation="Gesture not recognised",
            message=(
                f"Confidence {confidence:.0%} is below the acceptance "
                f"threshold of {threshold:.0%}."
            ),
            alternatives=alternatives,
        )

    gesture = _lookup_gesture(label)
    translation = gesture.english_meaning if gesture else label

    return PredictionOutcome(
        recognized=True,
        gesture=label,
        translation=translation,
        confidence=confidence,
        hand_detected=True,
        hand_count=len(hands),
        hand_count_expected=hand_count,
        model_version=model.version,
        alternatives=alternatives,
    )


def _candidate_models(models: dict, detected_hands: int) -> list:
    """Choose the (model, hand_count) combinations to evaluate.

    * Two hands detected -> the two-hand model is preferred; a one-hand model
      is still evaluated for each hand individually so that a mis-detected
      extra hand does not block a valid one-hand sign.
    * One hand detected -> the one-hand model is preferred; the two-hand model
      is evaluated with the missing hand masked out.
    """
    candidates: list = []
    two_hand = models.get(2)
    one_hand = models.get(1)

    if detected_hands >= 2 and two_hand is not None:
        candidates.append((two_hand, 2, False))
    if one_hand is not None:
        candidates.append((one_hand, 1, False))
    if detected_hands == 1 and two_hand is not None:
        candidates.append((two_hand, 2, False))
    if detected_hands >= 2 and one_hand is not None and two_hand is None:
        # Only a one-hand model exists: score each detected hand separately.
        candidates.append((one_hand, 1, True))
    return candidates


def predict_from_bytes(payload: bytes, config=None) -> PredictionOutcome:
    """Decode an image payload and run the recognition pipeline."""
    from app.services.hand_detector import decode_image

    frame = decode_image(payload)
    return predict_frame(frame, config=config)
