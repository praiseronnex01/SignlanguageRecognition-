"""End-to-end verification of the machine-learning pipeline.

The script proves that the whole chain works on this machine:

1. generate labelled **synthetic** landmark samples for every validated
   gesture (several participants, one-hand and two-hand schemas),
2. export a training-ready dataset,
3. train a Random Forest (and, optionally, the SVM and KNN comparisons),
4. save the model, the label encoder and the evaluation report,
5. publish the model and run a prediction through the real API service.

.. warning::
   The samples are **synthetic**.  They exercise the code path; they are
   **not** Malawi Sign Language data and the resulting accuracy figures say
   nothing about real MSL recognition performance.  Real performance can only
   be measured on a dataset captured from real participants.

Usage::

    python scripts/verify_pipeline.py
    python scripts/verify_pipeline.py --participants 8 --per-class 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from app import create_app  # noqa: E402
from app.extensions import db  # noqa: E402
from app.models.gesture import Gesture  # noqa: E402
from app.models.participant import Participant  # noqa: E402
from app.services import dataset_service, feature_extractor  # noqa: E402

# --- Synthetic hand geometry (mirrors the test fixtures) --------------------
FINGER_CHAINS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
BASE_DIRECTIONS = {
    "thumb": (0.55, 0.62),
    "index": (0.16, 1.00),
    "middle": (0.00, 1.05),
    "ring": (-0.16, 1.00),
    "pinky": (-0.32, 0.90),
}
FINGER_LENGTHS = {
    "thumb": (0.30, 0.26, 0.21),
    "index": (0.42, 0.24, 0.18),
    "middle": (0.45, 0.26, 0.19),
    "ring": (0.42, 0.24, 0.18),
    "pinky": (0.34, 0.20, 0.16),
}
FINGER_ROOTS = {
    "index": (0.16, 0.52, 0.0),
    "middle": (0.02, 0.54, 0.0),
    "ring": (-0.12, 0.51, 0.0),
    "pinky": (-0.25, 0.46, 0.0),
}
THUMB_ROOT = (0.20, 0.12, 0.0)
CURL_INDEX = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "pinky": 4}

#: One distinct synthetic hand shape per gesture class.
def curl_for(class_index: int) -> tuple:
    """Return a deterministic, unique curl configuration for a class."""
    import random

    rng = random.Random(9001 + class_index * 17)
    return tuple(round(rng.uniform(0.05, 0.95), 4) for _ in range(5))


def synthetic_landmarks(curl, rotation=0.0, scale=1.0, offset=(0.5, 0.5, 0.0)):
    import math

    landmarks = [[0.0, 0.0, 0.0] for _ in range(21)]
    for finger, chain in FINGER_CHAINS.items():
        root = THUMB_ROOT if finger == "thumb" else FINGER_ROOTS[finger]
        direction = BASE_DIRECTIONS[finger]
        angle = math.atan2(direction[1], direction[0])
        current = root
        points = [root]
        for index, length in enumerate(FINGER_LENGTHS[finger]):
            angle += curl[CURL_INDEX[finger]] * 0.38 * (index + 1)
            current = (
                current[0] + math.cos(angle) * length,
                current[1] + math.sin(angle) * length,
                current[2],
            )
            points.append(current)
        for index, position in enumerate(points):
            landmarks[chain[index]] = list(position)

    cosine, sine = math.cos(rotation), math.sin(rotation)
    result = []
    for x, y, z in landmarks:
        result.append(
            [
                offset[0] + (x * cosine - y * sine) * scale,
                offset[1] + (x * sine + y * cosine) * scale,
                offset[2] + z * scale,
            ]
        )
    return result


def build_payload(shape_index, hand_count, jitter, rotation, scale, offset):
    from ml.feature_engineering import HandObservation

    curl = curl_for(shape_index)
    hands = []
    for hand_index in range(hand_count):
        points = synthetic_landmarks(
            curl,
            rotation=rotation + hand_index * 0.12,
            scale=scale,
            offset=offset,
        )
        jittered = [
            [value + jitter * ((index % 5) - 2) * 0.001 for value in point]
            for index, point in enumerate(points)
        ]
        hands.append(
            HandObservation(
                landmarks=jittered,
                handedness="Left" if hand_index == 0 else "Right",
                handedness_score=0.95,
            )
        )
    detection = type("Detection", (), {"hands": hands})()
    return feature_extractor.build_payload(detection)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--participants", type=int, default=6)
    parser.add_argument("--per-class", type=int, default=24)
    parser.add_argument("--algorithm", default="random_forest")
    parser.add_argument(
        "--validate-candidates",
        action="store_true",
        help=(
            "Mark the seeded candidate gestures as validated so that the "
            "pipeline can be exercised.  This simulates the administrator "
            "action that a real Malawi Sign Language signer would perform; it "
            "must NOT be used to create a real dataset."
        ),
    )
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        if args.validate_candidates:
            print(
                "[0/5] Marking the seeded candidates as validated (development "
                "shortcut - a real signer must do this in a real project)…"
            )
            for gesture in Gesture.query.filter_by(is_validated=False).all():
                gesture.is_validated = True
                gesture.validation_status = "validated"
                gesture.validation_notes = (
                    "Marked by scripts/verify_pipeline.py for pipeline verification."
                )
            db.session.commit()

        gestures = (
            Gesture.query.filter_by(is_validated=True, is_active=True)
            .order_by(Gesture.id)
            .all()
        )
        if len(gestures) < 2:
            print(
                "At least two validated gestures are required. Validate the seeded "
                "candidates from the Vocabulary page first."
            )
            return 1

        # ------------------------------------------------------- collection
        existing = dataset_service.dataset_stats()["total_samples"]
        if existing:
            print(f"[1/5] The database already holds {existing} sample(s); skipping collection.")
        else:
            print(f"[1/5] Generating synthetic samples for {len(gestures)} gesture(s)…")
            import random

            rng = random.Random(20260113)
            for participant_index in range(args.participants):
                code = f"SYN{participant_index + 1:02d}"
                participant = dataset_service.get_or_create_participant(code)
                dataset_service.set_consent(
                    participant, Participant.CONSENT_GRANTED
                )
                session = dataset_service.create_session(
                    participant, notes="Synthetic pipeline verification"
                )
                for shape_index, gesture in enumerate(gestures):
                    for _ in range(args.per_class):
                        payload = build_payload(
                            shape_index,
                            int(gesture.hand_count),
                            rng.uniform(-1, 1),
                            rng.uniform(-0.08, 0.08),
                            1.0 + rng.uniform(-0.05, 0.05),
                            (
                                0.5 + rng.uniform(-0.02, 0.02),
                                0.5 + rng.uniform(-0.02, 0.02),
                                rng.uniform(-0.01, 0.01),
                            ),
                        )
                        try:
                            dataset_service.store_sample(
                                gesture=gesture,
                                participant=participant,
                                session=session,
                                payload=payload,
                                quality_score=1.0,
                            )
                        except dataset_service.DatasetError:
                            continue
                dataset_service.close_session(session)
            print(
                f"      {dataset_service.dataset_stats()['total_samples']} sample(s) stored."
            )

        # ----------------------------------------------------------- export
        print("[2/5] Exporting the dataset…")
        try:
            metadata = dataset_service.export_dataset()
        except dataset_service.DatasetError as exc:
            print(f"      Export failed: {exc}")
            return 1
        print(
            f"      Dataset {metadata['version']}: {metadata['sample_count']} samples, "
            f"{metadata['class_count']} classes, {metadata['participant_count']} participants."
        )

        # ----------------------------------------------------------- train
        print(f"[3/5] Training the {args.algorithm} model…")
        from ml.train import run_training

        summary = run_training(
            algorithm=args.algorithm,
            dataset_version=metadata["version"],
            parameters={},
            tune=False,
        )
        print("      " + json.dumps(summary["metrics"], indent=6))

        # --------------------------------------------------------- publish
        print("[4/5] Publishing the model…")
        from app.models.model_version import ModelVersion
        from app.services import model_service

        for version in summary["models"]:
            row = ModelVersion.query.filter_by(version=version).first()
            if row is None:
                continue
            model_service.publish_model(row.id)
            print(f"      Published version {row.version}.")

        # --------------------------------------------------------- predict
        print("[5/5] Running a prediction through the API service…")
        from app.services.gesture_predictor import predict_from_detection

        for shape_index, gesture in enumerate(gestures):
            payload = build_payload(
                shape_index,
                int(gesture.hand_count),
                0.0,
                0.0,
                1.0,
                (0.5, 0.5, 0.0),
            )
            detection = type(
                "Detection",
                (),
                {
                    "hands": [
                        __import__(
                            "ml.feature_engineering", fromlist=["HandObservation"]
                        ).HandObservation(
                            landmarks=hand["landmarks"],
                            handedness=hand["handedness"],
                        )
                        for hand in payload["hands"]
                    ]
                },
            )()
            outcome = predict_from_detection(detection)
            print(
                f"      expected {gesture.label:<12} -> recognised "
                f"{str(outcome.gesture):<12} confidence {outcome.confidence:.3f} "
                f"(model {outcome.model_version})"
            )

    print("-" * 68)
    print("Pipeline verification complete. The accuracy above was measured on")
    print("SYNTHETIC landmarks and is not a Malawi Sign Language performance figure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
