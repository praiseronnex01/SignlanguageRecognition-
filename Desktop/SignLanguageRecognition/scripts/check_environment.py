"""Report on the state of the local environment.

Checks the Python version, the installed dependencies, the MediaPipe task
model asset, the database connection and the state of the dataset, the model
artefacts and the vocabulary.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

REQUIRED_PACKAGES = [
    "flask",
    "flask_sqlalchemy",
    "flask_migrate",
    "flask_login",
    "flask_wtf",
    "werkzeug",
    "dotenv",
    "cv2",
    "mediapipe",
    "numpy",
    "pandas",
    "sklearn",
    "joblib",
    "scipy",
]


def _check_python() -> tuple:
    version = sys.version_info
    ok = (3, 9) <= (version.major, version.minor)
    return ok, f"Python {version.major}.{version.minor}.{version.micro}"


def _check_packages() -> tuple:
    import importlib.util

    missing = []
    versions = []
    for package in REQUIRED_PACKAGES:
        module = package.replace("sklearn", "sklearn").replace("cv2", "cv2")
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        if spec is None:
            missing.append(package)
            continue
        try:
            imported = importlib.import_module(module)
            versions.append(f"{package} {getattr(imported, '__version__', '?')}")
        except Exception:  # pragma: no cover - defensive
            versions.append(f"{package} ?")

    if missing:
        return False, "Missing package(s): " + ", ".join(missing)
    return True, ", ".join(versions)


def _check_model_asset() -> tuple:
    from app.config import ASSETS_DIR

    path = os.getenv(
        "MEDIAPIPE_MODEL_PATH",
        str(ASSETS_DIR / "models" / "hand_landmarker.task"),
    )
    if not os.path.isabs(path):
        path = str(PROJECT_ROOT / path)
    if os.path.isfile(path):
        size = os.path.getsize(path) / (1024 * 1024)
        return True, f"Hand landmarker asset found at {path} ({size:.1f} MB)"
    return False, (
        f"Hand landmarker asset missing at {path}. Download it with:\n"
        '  Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/'
        'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" '
        f'-OutFile "{path}"'
    )


def _check_database(app) -> tuple:
    from sqlalchemy import text

    from app.extensions import db

    try:
        with app.app_context():
            db.session.execute(text("SELECT 1"))
            engine = db.engine.url
            return True, f"Database reachable ({engine.drivername})"
    except Exception as exc:
        return False, f"Database is not reachable: {exc}"


def _check_state(app) -> list:
    from app.extensions import db
    from app.models.gesture import Gesture
    from app.models.model_version import ModelVersion
    from app.models.participant import Participant
    from app.models.training_run import TrainingRun
    from app.models.user import User

    results = []
    with app.app_context():
        results.append(("Users", User.query.count()))
        results.append(("Gestures (total)", Gesture.query.count()))
        results.append(
            (
                "Gestures (validated)",
                Gesture.query.filter_by(is_validated=True).count(),
            )
        )
        results.append(("Participants", Participant.query.count()))
        results.append(
            (
                "Published models",
                ModelVersion.query.filter_by(
                    status=ModelVersion.STATUS_PUBLISHED
                ).count(),
            )
        )
        results.append(
            (
                "Training runs",
                TrainingRun.query.filter(
                    TrainingRun.status == TrainingRun.STATUS_COMPLETED
                ).count(),
            )
        )
    return results


def run_checks() -> int:
    from app import create_app

    print("=" * 68)
    print("Malawi Sign Language Recognition System - environment check")
    print("=" * 68)

    ok, message = _check_python()
    print(f"[{'OK' if ok else '!!'}] Python        : {message}")

    ok, message = _check_packages()
    print(f"[{'OK' if ok else '!!'}] Dependencies  : {message}")

    ok, message = _check_model_asset()
    print(f"[{'OK' if ok else '!!'}] MediaPipe asset: {message}")

    try:
        app = create_app()
    except Exception as exc:
        print(f"[!!] Application  : could not be created - {exc}")
        return 1

    ok, message = _check_database(app)
    print(f"[{'OK' if ok else '!!'}] Database      : {message}")

    print("-" * 68)
    print("Database contents")
    for label, value in _check_state(app):
        print(f"    {label:<22}: {value}")

    exported = list((PROJECT_ROOT / "dataset" / "processed").glob("dataset_v*.csv"))
    models = list((PROJECT_ROOT / "artifacts" / "models").glob("*.joblib"))
    print("-" * 68)
    print(f"Exported datasets : {len(exported)}")
    print(f"Model artefacts   : {len(models)}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(run_checks())
