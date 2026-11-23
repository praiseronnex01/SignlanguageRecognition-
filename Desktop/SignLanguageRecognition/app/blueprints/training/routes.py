"""Training orchestration routes (administrator only).

Training runs are executed in a **background process** so that a long
running job never blocks an HTTP request.  The child process writes its
progress directly into the ``training_runs`` table, which the front end
polls through ``GET /api/training/status/<id>``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

from flask import Blueprint, current_app, render_template, request
from flask_login import current_user, login_required

from app.extensions import db
from app.models.dataset_sample import DatasetSample
from app.models.gesture import Gesture
from app.models.model_version import ModelVersion
from app.models.training_run import TrainingRun
from app.services import dataset_service, model_service
from app.utils import admin_required, json_error, json_success, log_audit

bp = Blueprint("training", __name__, url_prefix="/training")

ALGORITHMS = {
    "random_forest": "Random Forest",
    "svm": "Support Vector Machine",
    "knn": "K-Nearest Neighbours",
}

DEFAULT_PARAMETERS = {
    "random_forest": {
        "n_estimators": [200, 400],
        "max_depth": [None, 12, 20],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced", None],
    },
    "svm": {
        "C": [0.5, 1.0, 5.0, 10.0],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
        "class_weight": ["balanced", None],
    },
    "knn": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
}


def _training_log_path(run_id: int) -> str:
    from app.config import ARTIFACTS_DIR

    logs_dir = ARTIFACTS_DIR / "reports" / "logs"
    os.makedirs(logs_dir, exist_ok=True)
    return os.path.join(logs_dir, f"training_run_{run_id}.log")


def _spawn_training_process(run_id: int) -> subprocess.Popen:
    """Launch the training script as a detached background process."""
    project_root = current_app.config["PROJECT_ROOT"]
    log_path = _training_log_path(run_id)
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment["MSL_TRAINING_RUN_ID"] = str(run_id)

    creation_flags = 0
    if os.name == "nt":
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    handle = open(log_path, "ab")
    return subprocess.Popen(
        [sys.executable, "-m", "ml.train", "--run-id", str(run_id)],
        cwd=project_root,
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=environment,
        creationflags=creation_flags,
    )


@bp.route("")
@bp.route("/")
@login_required
@admin_required
def index():
    runs = TrainingRun.query.order_by(TrainingRun.created_at.desc()).limit(20).all()
    models = ModelVersion.query.order_by(ModelVersion.created_at.desc()).all()
    versions = dataset_service.list_dataset_versions()
    return render_template(
        "training/index.html",
        runs=runs,
        models=models,
        algorithms=ALGORITHMS,
        default_parameters=DEFAULT_PARAMETERS,
        dataset_versions=versions,
        stats=dataset_service.dataset_stats(),
    )


@bp.post("/api/training/start")
@login_required
@admin_required
def api_start():
    """Queue and launch a training run."""
    payload = request.get_json(silent=True) or request.form.to_dict()

    algorithm = (payload.get("algorithm") or "random_forest").strip().lower()
    if algorithm not in ALGORITHMS:
        return json_error(
            f"Unsupported algorithm '{algorithm}'. Expected one of: "
            f"{', '.join(sorted(ALGORITHMS))}.",
            400,
        )

    dataset_version = (payload.get("dataset_version") or "").strip()
    hand_count = payload.get("hand_count")
    try:
        hand_count = int(hand_count) if hand_count not in (None, "", "auto") else None
    except (TypeError, ValueError):
        return json_error("hand_count must be 1, 2 or 'auto'.", 400)
    if hand_count not in (None, 1, 2):
        return json_error("hand_count must be 1, 2 or 'auto'.", 400)

    parameters = payload.get("parameters") or {}
    if isinstance(parameters, str):
        try:
            parameters = json.loads(parameters)
        except ValueError:
            return json_error("The model parameters must be valid JSON.", 400)
    if not isinstance(parameters, dict):
        return json_error("The model parameters must be a JSON object.", 400)

    tune = bool(payload.get("tune", True))
    configuration = {
        "algorithm": algorithm,
        "algorithm_label": ALGORITHMS[algorithm],
        "parameters": parameters,
        "tune_hyperparameters": tune,
        "dataset_version": dataset_version or None,
        "hand_count": hand_count,
        "random_state": int(current_app.config.get("RANDOM_STATE", 42)),
        "train_ratio": float(current_app.config.get("TRAIN_PARTICIPANT_RATIO", 0.70)),
        "validation_ratio": float(
            current_app.config.get("VALIDATION_PARTICIPANT_RATIO", 0.15)
        ),
    }

    run = TrainingRun(
        status=TrainingRun.STATUS_QUEUED,
        configuration_json=json.dumps(configuration),
        progress=0.0,
        dataset_version=dataset_version or None,
    )
    db.session.add(run)
    db.session.commit()

    try:
        _spawn_training_process(run.id)
    except Exception as exc:  # pragma: no cover - environment specific
        run.status = TrainingRun.STATUS_FAILED
        run.error_message = f"Could not start the training process: {exc}"
        db.session.commit()
        return json_error(run.error_message, 500)

    run.status = TrainingRun.STATUS_RUNNING
    run.started_at = datetime.now(timezone.utc)
    db.session.commit()

    log_audit(
        "training.start",
        f"Started {ALGORITHMS[algorithm]} training run #{run.id}",
        current_user,
    )
    return json_success(run=run.to_dict(), log_path=_training_log_path(run.id)), 202


@bp.get("/api/training/status/<int:run_id>")
@login_required
@admin_required
def api_status(run_id: int):
    run = db.session.get(TrainingRun, run_id)
    if run is None:
        return json_error("Training run not found.", 404)
    return json_success(run=run.to_dict())


@bp.get("/api/training/runs")
@login_required
@admin_required
def api_runs():
    runs = TrainingRun.query.order_by(TrainingRun.created_at.desc()).limit(50).all()
    return json_success(runs=[run.to_dict() for run in runs])


@bp.get("/api/training/models")
@login_required
@admin_required
def api_models():
    models = ModelVersion.query.order_by(ModelVersion.created_at.desc()).all()
    return json_success(models=[model.to_dict() for model in models])


@bp.post("/api/training/models/<int:model_id>/publish")
@login_required
@admin_required
def api_publish(model_id: int):
    """Publish a trained model version so that live recognition can use it."""
    model = db.session.get(ModelVersion, model_id)
    if model is None:
        return json_error("Model version not found.", 404)
    if model.status == ModelVersion.STATUS_TRAINING:
        return json_error("This model version is still being trained.", 409)
    if model.status == ModelVersion.STATUS_FAILED:
        return json_error("A failed model version cannot be published.", 409)

    try:
        published = model_service.publish_model(model_id)
    except model_service.ModelArtifactError as exc:
        return json_error(str(exc), 400)

    log_audit(
        "training.publish",
        f"Published model version {published.version}",
        current_user,
    )
    return json_success(model=published.to_dict())


@bp.post("/api/training/models/<int:model_id>/archive")
@login_required
@admin_required
def api_archive(model_id: int):
    model = db.session.get(ModelVersion, model_id)
    if model is None:
        return json_error("Model version not found.", 404)
    archived = model_service.archive_model(model_id)
    log_audit("training.archive", f"Archived model version {archived.version}", current_user)
    return json_success(model=archived.to_dict())


@bp.get("/api/training/dataset-summary")
@login_required
@admin_required
def api_dataset_summary():
    """Preview the dataset that a training run would consume."""
    stats = dataset_service.dataset_stats()
    validated_classes = [
        entry["label"]
        for entry in stats["per_gesture"]
        if entry["is_validated"] and entry["sample_count"] > 0
    ]
    return json_success(
        total_samples=stats["total_samples"],
        validated_classes=validated_classes,
        validated_class_count=len(validated_classes),
        participants=stats["participant_count"],
        versions=dataset_service.list_dataset_versions(),
        ready=bool(validated_classes) and stats["participant_count"] >= 2,
    )
