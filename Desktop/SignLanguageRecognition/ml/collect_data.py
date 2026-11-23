"""Standalone webcam dataset collection tool.

The tool opens the webcam, displays a live preview with the detected hand
landmarks, and stores the extracted landmarks for the selected gesture and
participant.  **No raw video is ever written to disk** - only the normalised
landmark coordinates and the metadata required to train the classifier.

Usage
-----
Interactive (asks for the gesture and participant on the console)::

    python -m ml.collect_data

Non-interactive::

    python -m ml.collect_data --gesture HELLO --participant P001 --target 200

Once the target number of samples is reached the tool stops capturing and
offers to export a training-ready dataset.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover - optional at runtime
    pass

import cv2
import numpy as np

from ml.feature_engineering import build_feature_vector, validate_feature_vector

WINDOW_NAME = "MSL Dataset Collection"
CONNECTION_LANDMARKS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def _create_app():
    from app import create_app

    return create_app()


def _draw_overlay(frame, detection, status: str, counters: dict) -> np.ndarray:
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    for hand in detection.hands:
        points = np.asarray(
            [[int(p[0] * width), int(p[1] * height)] for p in hand.landmarks],
            dtype=np.int32,
        )
        for start, end in CONNECTION_LANDMARKS:
            cv2.line(canvas, points[start], points[end], (37, 99, 235), 2, cv2.LINE_AA)
        for index, point in enumerate(points):
            cv2.circle(canvas, tuple(point), 4, (16, 185, 129), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            hand.handedness,
            tuple(points[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (15, 23, 42),
            2,
            cv2.LINE_AA,
        )

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, 96), (16, 28, 50), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    cv2.putText(
        canvas, status, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
    )
    summary = "   ".join(f"{key}: {value}" for key, value in counters.items())
    cv2.putText(
        canvas, summary, (16, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (214, 226, 245), 1, cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        "[SPACE]/[C] capture    [S] toggle capture    [Q] quit",
        (16, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def collect(
    gesture_label: str | None = None,
    participant_code: str | None = None,
    target: int = 100,
    camera_index: int = 0,
    interactive: bool = True,
    export_after: bool = True,
    session_notes: str | None = None,
) -> dict:
    """Run the interactive collection session."""
    app = _create_app()
    with app.app_context():
        from app.models.gesture import Gesture
        from app.services import dataset_service, feature_extractor
        from app.services.hand_detector import HandDetector

        if interactive:
            if not gesture_label:
                gesture_label = input("Gesture label (upper case, e.g. HELLO): ").strip().upper()
            if not participant_code:
                participant_code = input("Participant code (anonymous, e.g. P001): ").strip().upper()
        if not gesture_label or not participant_code:
            raise SystemExit("A gesture label and a participant code are required.")

        gesture = Gesture.query.filter_by(label=gesture_label.upper()).first()
        if gesture is None:
            raise SystemExit(
                f"The gesture '{gesture_label.upper()}' is not in the vocabulary. "
                "Add it from the Vocabulary page before collecting samples."
            )
        if not gesture.is_active:
            raise SystemExit(f"The gesture '{gesture.label}' is deactivated.")

        participant = dataset_service.get_or_create_participant(participant_code.upper())
        if not participant.has_consent:
            raise SystemExit(
                f"Participant '{participant.participant_code}' has not given consent. "
                "Record consent from the Dataset Management page first."
            )

        session = dataset_service.create_session(
            participant, notes=session_notes or "Standalone collection tool"
        )
        print(f"[collect] Session {session.session_code} opened.")

        detector = HandDetector(max_hands=2)
        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            raise SystemExit(
                f"Could not open camera {camera_index}. Check that the device is "
                "connected and not in use by another application."
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        saved = 0
        duplicates = 0
        invalid = 0
        capturing = True
        frame_index = 0
        last_capture_time = 0.0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    print("[collect] The camera stream ended.")
                    break

                # Detection always runs on the RAW frame so that the handedness
                # labels produced here are identical to the ones produced by
                # the browser capture pipeline.  Only the preview is mirrored.
                detection = detector.detect(frame)
                frame_index += 1

                if capturing and detection.hands:
                    now = time.time()
                    if now - last_capture_time >= 0.25:
                        last_capture_time = now
                        expected = int(gesture.hand_count)
                        if len(detection.hands) != expected:
                            invalid += 1
                        else:
                            try:
                                payload = feature_extractor.build_payload(detection)
                                dataset_service.store_sample(
                                    gesture=gesture,
                                    participant=participant,
                                    session=session,
                                    payload=payload,
                                    quality_score=feature_extractor.quality_score(detection),
                                )
                                saved += 1
                            except Exception as exc:
                                if "duplicat" in str(exc).lower():
                                    duplicates += 1
                                else:
                                    invalid += 1
                                    print(f"[collect] Sample rejected: {exc}")

                counters = {
                    "Saved": saved,
                    "Target": target,
                    "Invalid": invalid,
                    "Duplicate": duplicates,
                    "Hands": detection.hand_count,
                }
                status = (
                    f"Collecting: {gesture.label} ({gesture.english_meaning}) "
                    f"| participant {participant.participant_code}"
                )
                preview = _draw_overlay(
                    cv2.flip(frame, 1), detection, status, counters
                )
                cv2.imshow(WINDOW_NAME, preview)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (ord(" "), ord("c")):
                    # Manual capture trigger.
                    if detection.hands:
                        expected = int(gesture.hand_count)
                        if len(detection.hands) != expected:
                            invalid += 1
                        else:
                            try:
                                payload = feature_extractor.build_payload(detection)
                                dataset_service.store_sample(
                                    gesture=gesture,
                                    participant=participant,
                                    session=session,
                                    payload=payload,
                                    quality_score=feature_extractor.quality_score(detection),
                                )
                                saved += 1
                            except Exception as exc:
                                if "duplicat" in str(exc).lower():
                                    duplicates += 1
                                else:
                                    invalid += 1
                    else:
                        invalid += 1
                if key == ord("s"):
                    capturing = not capturing
                    print(f"[collect] Automatic capture {'enabled' if capturing else 'paused'}.")
                if target and saved >= target:
                    print(f"[collect] Target of {target} samples reached.")
                    break
        except KeyboardInterrupt:
            print("\n[collect] Interrupted by the operator.")
        finally:
            capture.release()
            cv2.destroyAllWindows()
            dataset_service.close_session(session)

        print(
            f"[collect] Finished: {saved} saved, {invalid} invalid, "
            f"{duplicates} duplicates, {frame_index} frames processed."
        )

        export_info = None
        if export_after and saved > 0:
            try:
                export_info = dataset_service.export_dataset()
                print(f"[collect] Exported dataset {export_info['version']}.")
            except Exception as exc:
                print(f"[collect] Export skipped: {exc}")

        return {
            "gesture": gesture.label,
            "participant": participant.participant_code,
            "session": session.session_code,
            "saved": saved,
            "invalid": invalid,
            "duplicates": duplicates,
            "export": export_info,
        }


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect labelled MSL hand-landmark samples from a webcam."
    )
    parser.add_argument("--gesture", default=None, help="Gesture label to collect.")
    parser.add_argument(
        "--participant", default=None, help="Anonymous participant code."
    )
    parser.add_argument(
        "--target", type=int, default=100, help="Number of samples to capture."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Do not prompt for the gesture and participant on the console.",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Do not export the dataset when the session ends.",
    )
    args = parser.parse_args(argv)

    try:
        collect(
            gesture_label=args.gesture,
            participant_code=args.participant,
            target=args.target,
            camera_index=args.camera,
            interactive=not args.no_interactive,
            export_after=not args.no_export,
        )
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Collection failed: {exc}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
