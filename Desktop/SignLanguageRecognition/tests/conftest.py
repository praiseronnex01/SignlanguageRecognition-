"""Shared pytest fixtures.

The fixtures build **synthetic** hand landmarks so that the computer-vision
and machine-learning pipelines can be exercised without a webcam and without
real participant data.

The synthetic data is a *test fixture only*.  It is never presented as
evidence of Malawi Sign Language recognition performance - real performance
figures can only come from a real, validated dataset.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app  # noqa: E402
from app.extensions import db  # noqa: E402
from app.models.user import User  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic hand geometry
# ---------------------------------------------------------------------------
#: MediaPipe hand-landmark index groups.
FINGER_CHAINS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

#: Direction of each finger relative to the palm in an open hand (x, y).
BASE_DIRECTIONS = {
    "thumb": (0.55, 0.62),
    "index": (0.16, 1.00),
    "middle": (0.00, 1.05),
    "ring": (-0.16, 1.00),
    "pinky": (-0.32, 0.90),
}

#: Segment lengths of each finger chain (three segments per chain, so the
#: chain yields the four MediaPipe landmarks of that finger).
FINGER_LENGTHS = {
    "thumb": (0.30, 0.26, 0.21),
    "index": (0.42, 0.24, 0.18),
    "middle": (0.45, 0.26, 0.19),
    "ring": (0.42, 0.24, 0.18),
    "pinky": (0.34, 0.20, 0.16),
}

#: Where each finger starts on the palm (knuckle position).
FINGER_ROOTS = {
    "index": (0.16, 0.52, 0.0),
    "middle": (0.02, 0.54, 0.0),
    "ring": (-0.12, 0.51, 0.0),
    "pinky": (-0.25, 0.46, 0.0),
}

THUMB_ROOT = (0.20, 0.12, 0.0)

#: Position of each finger inside the ``curl`` tuple.
CURL_INDEX = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "pinky": 4}


def _chain(root, direction, lengths, curl, z_drift=0.0):
    """Build one finger chain with an optional curl."""
    points = [root]
    angle = math.atan2(direction[1], direction[0])
    current = root
    for index, length in enumerate(lengths):
        angle += curl * 0.38 * (index + 1)
        current = (
            current[0] + math.cos(angle) * length,
            current[1] + math.sin(angle) * length,
            current[2] + z_drift * (index + 1),
        )
        points.append(current)
    return points


def make_hand_landmarks(
    curl=(0.0, 0.0, 0.0, 0.0, 0.0),
    rotation: float = 0.0,
    scale: float = 1.0,
    offset=(0.5, 0.5, 0.0),
    handedness: str = "Right",
    spread: float = 1.0,
) -> list:
    """Return 21 synthetic landmarks describing a hand shape.

    ``curl`` holds one value per finger (thumb, index, middle, ring, pinky);
    ``0`` is an extended finger and ``1`` a fully curled one.
    """
    landmarks = [[0.0, 0.0, 0.0] for _ in range(21)]
    landmarks[0] = [0.0, 0.0, 0.0]

    for finger, chain in FINGER_CHAINS.items():
        if finger == "thumb":
            root = THUMB_ROOT
        else:
            root = FINGER_ROOTS[finger]
        direction = BASE_DIRECTIONS[finger]
        if finger != "thumb":
            direction = (direction[0] * spread, direction[1])
        curl_value = curl[CURL_INDEX[finger]]
        points = _chain(root, direction, FINGER_LENGTHS[finger], curl_value)
        for index, position in enumerate(points):
            landmarks[chain[index]] = list(position)

    cosine = math.cos(rotation)
    sine = math.sin(rotation)
    result = []
    for x, y, z in landmarks:
        rotated_x = x * cosine - y * sine
        rotated_y = x * sine + y * cosine
        result.append(
            [
                offset[0] + rotated_x * scale,
                offset[1] + rotated_y * scale,
                offset[2] + z * scale,
            ]
        )
    return result


#: A small library of clearly distinguishable synthetic hand shapes.
HAND_SHAPES = {
    "open": (0.0, 0.0, 0.0, 0.0, 0.0),
    "fist": (0.85, 1.0, 1.0, 1.0, 1.0),
    "point": (0.85, 0.0, 1.0, 1.0, 1.0),
    "two": (0.85, 0.0, 0.0, 1.0, 1.0),
    "palm": (0.85, 1.0, 0.0, 1.0, 1.0),
    "flat": (0.2, 0.15, 0.15, 0.15, 0.15),
}


def make_hand_observation(
    shape: str = "open",
    handedness: str = "Right",
    rotation: float = 0.0,
    scale: float = 1.0,
    offset=(0.5, 0.5, 0.0),
    score: float = 0.97,
):
    """Build a :class:`HandObservation` from a named synthetic hand shape."""
    from ml.feature_engineering import HandObservation

    return HandObservation(
        landmarks=make_hand_landmarks(
            curl=HAND_SHAPES[shape],
            rotation=rotation,
            scale=scale,
            offset=offset,
            handedness=handedness,
        ),
        handedness=handedness,
        handedness_score=score,
    )


def make_detection(hands=(), frame_width=640, frame_height=480):
    """Build a :class:`DetectionResult` from a list of hand observations."""
    from app.services.hand_detector import DetectionResult

    return DetectionResult(
        hands=list(hands),
        frame_width=frame_width,
        frame_height=frame_height,
        source="test",
    )


# ---------------------------------------------------------------------------
# Application fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def app():
    """A Flask application bound to an in-memory database."""
    application = create_app("testing")
    with application.app_context():
        db.create_all()
        yield application
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def admin_user(app):
    user = User(
        full_name="Test Administrator",
        email="admin@test.local",
        role=User.ROLE_ADMIN,
        is_active=True,
    )
    user.set_password("Admin12345")
    db.session.add(user)
    db.session.commit()
    return user


@pytest.fixture()
def standard_user(app):
    user = User(
        full_name="Test User",
        email="user@test.local",
        role=User.ROLE_USER,
        is_active=True,
    )
    user.set_password("User12345")
    db.session.add(user)
    db.session.commit()
    return user


@pytest.fixture()
def login_admin(client, admin_user):
    response = client.post(
        "/auth/api/auth/login",
        json={"email": admin_user.email, "password": "Admin12345"},
    )
    assert response.status_code == 200
    return admin_user


@pytest.fixture()
def login_user(client, standard_user):
    response = client.post(
        "/auth/api/auth/login",
        json={"email": standard_user.email, "password": "User12345"},
    )
    assert response.status_code == 200
    return standard_user


@pytest.fixture()
def gesture(app):
    """A validated one-hand gesture."""
    from app.models.gesture import Gesture

    entry = Gesture(
        label="TEST_SIGN",
        english_meaning="Test sign",
        category="Testing",
        description="A synthetic gesture used by the test suite.",
        is_validated=True,
        is_active=True,
        hand_count=1,
    )
    db.session.add(entry)
    db.session.commit()
    return entry


@pytest.fixture()
def unvalidated_gesture(app):
    """A candidate (not yet verified) gesture."""
    from app.models.gesture import Gesture

    entry = Gesture(
        label="TEST_CANDIDATE",
        english_meaning="Test candidate sign",
        category="Testing",
        description="A synthetic candidate gesture used by the test suite.",
        is_validated=False,
        is_active=True,
        hand_count=1,
    )
    db.session.add(entry)
    db.session.commit()
    return entry


@pytest.fixture()
def two_hand_gesture(app):
    from app.models.gesture import Gesture

    entry = Gesture(
        label="TEST_TWO_HAND",
        english_meaning="Test two-hand sign",
        category="Testing",
        description="A synthetic two-hand gesture used by the test suite.",
        is_validated=True,
        is_active=True,
        hand_count=2,
    )
    db.session.add(entry)
    db.session.commit()
    return entry


@pytest.fixture()
def sample_jpeg(tmp_path):
    """A small, valid JPEG image that contains no hand."""
    import cv2

    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, :, 1] = 40
    path = tmp_path / "frame.jpg"
    cv2.imwrite(str(path), image)
    with open(path, "rb") as handle:
        return handle.read()


@pytest.fixture()
def cleanup_model_cache():
    from app.services import model_service

    model_service.clear_cache()
    yield
    model_service.clear_cache()


@pytest.fixture(autouse=True)
def _isolated_artifacts(tmp_path, monkeypatch):
    """Keep test runs from writing into the real artefact directories."""
    monkeypatch.setenv("TESTING", "1")
    yield
