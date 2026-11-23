"""Model artifact loading, caching and publication management.

Model files are always loaded from the directories owned by the application
(``artifacts/models`` and ``artifacts/encoders``).  Arbitrary user supplied
model files are never loaded - only paths recorded by the application in the
``model_versions`` table are accepted.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

import joblib

from app.extensions import db
from app.models.model_version import ModelVersion

_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


@dataclass
class LoadedModel:
    """A trained model plus everything needed to serve predictions."""

    version: str
    algorithm: str
    pipeline: object
    encoder: object
    classes: list = field(default_factory=list)
    schema: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    rejection_threshold: float | None = None
    dataset_version: str | None = None
    feature_version: str = "v1.0"
    model_path: str = ""
    encoder_path: str = ""

    @property
    def hand_count(self) -> int:
        return int(self.schema.get("hand_count", 1) or 1)

    @property
    def expected_dims(self) -> int:
        return int(self.schema.get("total_dims", 63 if self.hand_count == 1 else 128))

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "algorithm": self.algorithm,
            "classes": list(self.classes),
            "schema": self.schema,
            "metrics": self.metrics,
            "rejection_threshold": self.rejection_threshold,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "hand_count": self.hand_count,
            "model_path": self.model_path,
            "encoder_path": self.encoder_path,
        }


class ModelArtifactError(RuntimeError):
    """Raised when a model artifact cannot be loaded."""


def _is_allowed_path(path: str, allowed_root: str) -> bool:
    """Guard against loading model files from outside the artifact folders."""
    if not path:
        return False
    resolved = os.path.abspath(path)
    root = os.path.abspath(allowed_root)
    return os.path.commonpath([resolved, root]) == root


def load_artifacts(model_path: str, encoder_path: str, allowed_roots: tuple) -> LoadedModel:
    """Load a model bundle and its label encoder from disk."""
    for path, root in zip((model_path, encoder_path), allowed_roots):
        if not _is_allowed_path(path, root):
            raise ModelArtifactError(
                f"Refusing to load '{path}': model artifacts must live inside the "
                "application artifact directories."
            )
        if not os.path.isfile(path):
            raise ModelArtifactError(f"Model artifact '{path}' does not exist")

    try:
        bundle = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
    except Exception as exc:  # pragma: no cover - defensive
        raise ModelArtifactError(f"Could not load the model artifacts: {exc}") from exc

    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ModelArtifactError("The model artifact is not a valid MSL model bundle")

    classes = bundle.get("classes")
    if classes is None and hasattr(encoder, "classes_"):
        classes = [str(c) for c in encoder.classes_]

    return LoadedModel(
        version=str(bundle.get("version", "")),
        algorithm=str(bundle.get("algorithm", "")),
        pipeline=bundle["pipeline"],
        encoder=encoder,
        classes=list(classes or []),
        schema=dict(bundle.get("schema", {}) or {}),
        metrics=dict(bundle.get("metrics", {}) or {}),
        rejection_threshold=bundle.get("rejection_threshold"),
        dataset_version=bundle.get("dataset_version"),
        feature_version=str(bundle.get("feature_version", "v1.0")),
        model_path=model_path,
        encoder_path=encoder_path,
    )


def get_loaded_model(model_version: ModelVersion, config=None) -> LoadedModel:
    """Return a cached :class:`LoadedModel` for a database row."""
    from flask import current_app

    config = config or current_app.config
    key = (model_version.id, model_version.version, model_version.model_path)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    model = load_artifacts(
        model_version.model_path,
        model_version.encoder_path,
        (config["MODELS_DIR"], config["ENCODERS_DIR"]),
    )
    if not model.version:
        model.version = model_version.version
    if not model.schema:
        model.schema = model_version.schema
    if model.rejection_threshold is None:
        model.rejection_threshold = model_version.rejection_threshold
    if not model.metrics:
        model.metrics = {
            "accuracy": model_version.accuracy,
            "precision": model_version.precision,
            "recall": model_version.recall,
            "f1_score": model_version.f1_score,
        }
    with _CACHE_LOCK:
        _CACHE[key] = model
    return model


def published_model_rows():
    """Return published :class:`ModelVersion` rows, newest first."""
    return (
        ModelVersion.query.filter_by(status=ModelVersion.STATUS_PUBLISHED)
        .order_by(ModelVersion.created_at.desc(), ModelVersion.id.desc())
        .all()
    )


def published_models(config=None) -> dict:
    """Return ``{hand_count: LoadedModel}`` for every published model."""
    from flask import current_app

    config = config or current_app.config
    models: dict = {}
    for row in published_model_rows():
        try:
            model = get_loaded_model(row, config)
        except ModelArtifactError:
            current_app.logger.warning(
                "Published model %s could not be loaded and was skipped.", row.version
            )
            continue
        # The newest published model for a given hand count wins.
        models.setdefault(model.hand_count, model)
    return models


def primary_published_model(config=None):
    """Return the most recently published model, or ``None``."""
    from flask import current_app

    config = config or current_app.config
    for row in published_model_rows():
        try:
            return get_loaded_model(row, config)
        except ModelArtifactError:
            current_app.logger.warning(
                "Published model %s could not be loaded and was skipped.", row.version
            )
    return None


def publish_model(model_version_id: int, config=None) -> ModelVersion:
    """Publish a model version and archive superseded ones."""
    from flask import current_app

    config = config or current_app.config
    model_version = db.session.get(ModelVersion, model_version_id)
    if model_version is None:
        raise ModelArtifactError(f"Model version {model_version_id} was not found")
    if model_version.status == ModelVersion.STATUS_FAILED:
        raise ModelArtifactError("A failed model version cannot be published")

    hand_count = model_version.hand_count
    for other in ModelVersion.query.filter(
        ModelVersion.status == ModelVersion.STATUS_PUBLISHED
    ).all():
        if other.id != model_version.id and other.hand_count == hand_count:
            other.status = ModelVersion.STATUS_ARCHIVED

    model_version.status = ModelVersion.STATUS_PUBLISHED
    db.session.commit()
    clear_cache()
    current_app.logger.info("Published model version %s", model_version.version)
    return model_version


def archive_model(model_version_id: int) -> ModelVersion:
    model_version = db.session.get(ModelVersion, model_version_id)
    if model_version is None:
        raise ModelArtifactError(f"Model version {model_version_id} was not found")
    model_version.status = ModelVersion.STATUS_ARCHIVED
    db.session.commit()
    clear_cache()
    return model_version


def clear_cache() -> None:
    """Drop every cached model (used after publishing or in tests)."""
    with _CACHE_LOCK:
        _CACHE.clear()


def save_artifacts(
    pipeline,
    encoder,
    schema: dict,
    metrics: dict,
    algorithm: str,
    version: str,
    config=None,
    dataset_version: str | None = None,
    rejection_threshold: float | None = None,
) -> tuple:
    """Persist the model bundle and the label encoder.

    Returns ``(model_path, encoder_path)``.
    """
    from flask import current_app

    config = config or current_app.config
    models_dir = config["MODELS_DIR"]
    encoders_dir = config["ENCODERS_DIR"]
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(encoders_dir, exist_ok=True)

    model_path = os.path.join(models_dir, f"msl_{version}_{algorithm}.joblib")
    encoder_path = os.path.join(encoders_dir, f"msl_{version}_encoder.joblib")

    classes = [str(c) for c in getattr(encoder, "classes_", [])]
    bundle = {
        "version": version,
        "algorithm": algorithm,
        "pipeline": pipeline,
        "classes": classes,
        "schema": schema,
        "metrics": metrics,
        "rejection_threshold": rejection_threshold,
        "dataset_version": dataset_version,
        "feature_version": schema.get("feature_version", "v1.0"),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(bundle, model_path, compress=3)
    joblib.dump(encoder, encoder_path, compress=3)
    return model_path, encoder_path
