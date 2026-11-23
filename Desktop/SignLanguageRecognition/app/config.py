"""Application configuration.

Every value can be overridden through environment variables (see ``.env.example``).
The configuration is intentionally free of secrets: ``SECRET_KEY`` must be
supplied in production, and the application refuses to start in production
with the development fallback key.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INSTANCE_DIR = BASE_DIR / "instance"
DATASET_DIR = BASE_DIR / "dataset"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
ENCODERS_DIR = ARTIFACTS_DIR / "encoders"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
ASSETS_DIR = BASE_DIR / "assets"
MIGRATIONS_DIR = BASE_DIR / "migrations"

for _directory in (
    INSTANCE_DIR,
    DATASET_DIR / "raw",
    DATASET_DIR / "processed",
    DATASET_DIR / "exports",
    DATASET_DIR / "metadata",
    MODELS_DIR,
    ENCODERS_DIR,
    REPORTS_DIR,
    ASSETS_DIR / "models",
):
    _directory.mkdir(parents=True, exist_ok=True)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _database_uri() -> str:
    """Resolve the database URL.

    A relative SQLite path such as ``sqlite:///instance/msl.db`` is resolved
    against the project root so that the value stays portable between
    machines.
    """
    configured = os.getenv("DATABASE_URL")
    if not configured:
        return f"sqlite:///{(INSTANCE_DIR / 'msl.db').as_posix()}"
    if configured.startswith("sqlite:///") and not configured.startswith("sqlite:////"):
        path = configured[len("sqlite:///") :]
        if path and not os.path.isabs(path):
            return f"sqlite:///{(BASE_DIR / path).as_posix()}"
    return configured


class BaseConfig:
    """Settings shared by every environment."""

    # --- Core -------------------------------------------------------------
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-insecure-secret-key-change-me")
    DEBUG = False
    TESTING = False

    # --- Database ---------------------------------------------------------
    SQLALCHEMY_DATABASE_URI = _database_uri()
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}

    # --- Sessions / security ---------------------------------------------
    PERMANENT_SESSION_LIFETIME = int(
        os.getenv("SESSION_LIFETIME_SECONDS", str(60 * 60 * 8))
    )
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = _env_bool("SESSION_COOKIE_SECURE", False)
    WTF_CSRF_TIME_LIMIT = int(os.getenv("CSRF_TIME_LIMIT", 60 * 60 * 4))
    MAX_CONTENT_LENGTH = _env_int("MAX_CONTENT_LENGTH", 12 * 1024 * 1024)  # 12 MB

    # --- Uploads ----------------------------------------------------------
    UPLOAD_FOLDER = str(INSTANCE_DIR / "uploads")
    ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

    # --- Computer vision --------------------------------------------------
    MEDIAPIPE_MODEL_PATH = os.getenv(
        "MEDIAPIPE_MODEL_PATH",
        str(ASSETS_DIR / "models" / "hand_landmarker.task"),
    )
    MAX_HANDS = _env_int("MAX_HANDS", 2)
    HAND_DETECTION_CONFIDENCE = _env_float("HAND_DETECTION_CONFIDENCE", 0.5)
    HAND_PRESENCE_CONFIDENCE = _env_float("HAND_PRESENCE_CONFIDENCE", 0.5)
    HAND_TRACKING_CONFIDENCE = _env_float("HAND_TRACKING_CONFIDENCE", 0.5)

    # --- Machine learning -------------------------------------------------
    ARTIFACTS_DIR = str(ARTIFACTS_DIR)
    MODELS_DIR = str(MODELS_DIR)
    ENCODERS_DIR = str(ENCODERS_DIR)
    REPORTS_DIR = str(REPORTS_DIR)
    DATASET_DIR = str(DATASET_DIR)
    RECOGNITION_CONFIDENCE_THRESHOLD = _env_float(
        "RECOGNITION_CONFIDENCE_THRESHOLD", 0.60
    )
    TRAIN_PARTICIPANT_RATIO = _env_float("TRAIN_PARTICIPANT_RATIO", 0.70)
    VALIDATION_PARTICIPANT_RATIO = _env_float("VALIDATION_PARTICIPANT_RATIO", 0.15)
    RANDOM_STATE = _env_int("RANDOM_STATE", 42)

    # --- Behaviour --------------------------------------------------------
    AUTO_MIGRATE_ON_START = _env_bool("AUTO_MIGRATE_ON_START", True)
    ITEMS_PER_PAGE = _env_int("ITEMS_PER_PAGE", 15)
    PROJECT_NAME = "Malawi Sign Language Recognition System"
    PROJECT_SHORT_NAME = "MSL Recognition"
    MSL_REFERENCE_URL = "https://msld.manadmw.org/"
    FEATURE_VERSION = "v1.0"


class DevelopmentConfig(BaseConfig):
    DEBUG = True
    TEMPLATES_AUTO_RELOAD = True


class TestingConfig(BaseConfig):
    TESTING = True
    DEBUG = False
    WTF_CSRF_ENABLED = False
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "TEST_DATABASE_URL", "sqlite:///:memory:"
    )
    if SQLALCHEMY_DATABASE_URI.startswith("sqlite:///") and not SQLALCHEMY_DATABASE_URI.startswith(
        "sqlite:////"
    ):
        _test_path = SQLALCHEMY_DATABASE_URI[len("sqlite:///") :]
        if _test_path and not os.path.isabs(_test_path) and _test_path != ":memory:":
            SQLALCHEMY_DATABASE_URI = f"sqlite:///{(BASE_DIR / _test_path).as_posix()}"
    SQLALCHEMY_ENGINE_OPTIONS = {}
    AUTO_MIGRATE_ON_START = False
    SECRET_KEY = "testing-secret-key"
    MEDIAPIPE_MODEL_PATH = os.getenv(
        "MEDIAPIPE_MODEL_PATH",
        str(ASSETS_DIR / "models" / "hand_landmarker.task"),
    )


class ProductionConfig(BaseConfig):
    DEBUG = False
    SESSION_COOKIE_SECURE = _env_bool("SESSION_COOKIE_SECURE", True)
    AUTO_MIGRATE_ON_START = _env_bool("AUTO_MIGRATE_ON_START", False)

    @classmethod
    def validate(cls) -> None:
        if cls.SECRET_KEY == "dev-insecure-secret-key-change-me":
            raise RuntimeError(
                "SECRET_KEY must be set to a unique value in production. "
                "Copy .env.example to .env and generate a strong key."
            )


CONFIG_MAP = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}


def get_config(name: str | None = None):
    """Resolve a configuration class from a name, the environment or a default."""
    key = (name or os.getenv("FLASK_CONFIG") or "development").strip().lower()
    if key not in CONFIG_MAP:
        raise ValueError(
            f"Unknown configuration '{key}'. Expected one of: {', '.join(CONFIG_MAP)}"
        )
    return CONFIG_MAP[key]
