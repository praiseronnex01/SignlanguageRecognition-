"""Predict a gesture from an image file using the published model.

Usage::

    python -m ml.predict --image path/to/frame.jpg
    python -m ml.predict --image frame.jpg --include-landmarks

The command prints a JSON document describing the outcome.  It is intended
for offline verification of the recognition pipeline; the web application
uses the very same service code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover - optional at runtime
    pass


def predict_image(path: str, include_landmarks: bool = False) -> dict:
    """Run the recognition pipeline against a single image file."""
    from app import create_app
    from app.services.gesture_predictor import predict_frame
    from app.services.hand_detector import HandDetector, decode_image

    if not os.path.isfile(path):
        raise FileNotFoundError(f"The image '{path}' does not exist.")

    app = create_app()
    with app.app_context():
        with open(path, "rb") as handle:
            frame = decode_image(handle.read())

        outcome = predict_frame(frame)
        result = outcome.to_dict()

        if include_landmarks:
            detection = HandDetector(max_hands=int(app.config.get("MAX_HANDS", 2))).detect(frame)
            result["landmarks"] = detection.to_dict(include_landmarks=True)
        return result


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recognise a Malawi Sign Language gesture in an image."
    )
    parser.add_argument("--image", required=True, help="Path to the image file.")
    parser.add_argument(
        "--include-landmarks",
        action="store_true",
        help="Include the detected hand landmarks in the output.",
    )
    args = parser.parse_args(argv)

    try:
        result = predict_image(args.image, include_landmarks=args.include_landmarks)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Prediction failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("recognized") else 2


if __name__ == "__main__":
    raise SystemExit(main())
