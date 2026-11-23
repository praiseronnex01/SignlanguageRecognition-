"""Evaluation routes.

Every number displayed by this module is read from an evaluation report that
was produced by the training pipeline.  Nothing is hard-coded: if no model has
been trained yet the endpoints report an empty state.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from flask import Blueprint, current_app, render_template, request
from flask_login import login_required

from app.extensions import db
from app.models.evaluation_metric import EvaluationMetric
from app.models.model_version import ModelVersion
from app.models.training_run import TrainingRun
from app.utils import admin_required, json_error, json_success

bp = Blueprint("evaluation", __name__, url_prefix="/evaluation")


def _reports_dir() -> str:
    return current_app.config["REPORTS_DIR"]


def _report_path(version: str) -> str:
    safe = "".join(c for c in version if c.isalnum() or c in {".", "-", "_"})
    return os.path.join(_reports_dir(), f"evaluation_{safe}.json")


def _load_report(version: str) -> dict | None:
    path = _report_path(version)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _latest_report() -> dict | None:
    candidates = [
        model
        for model in ModelVersion.query.order_by(ModelVersion.created_at.desc()).all()
        if _load_report(model.version)
    ]
    if not candidates:
        return None
    report = _load_report(candidates[0].version)
    if report is not None:
        report.setdefault("model_version", candidates[0].version)
        report.setdefault("status", candidates[0].status)
    return report


def _comparison_report() -> dict | None:
    reports_dir = _reports_dir()
    if not os.path.isdir(reports_dir):
        return None
    files = [
        os.path.join(reports_dir, name)
        for name in os.listdir(reports_dir)
        if name.startswith("comparison_") and name.endswith(".json")
    ]
    if not files:
        return None
    newest = max(files, key=os.path.getmtime)
    try:
        with open(newest, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    data["report_path"] = newest
    return data


@bp.route("")
@bp.route("/")
@login_required
@admin_required
def index():
    models = ModelVersion.query.order_by(ModelVersion.created_at.desc()).all()
    runs = TrainingRun.query.order_by(TrainingRun.created_at.desc()).limit(20).all()
    return render_template(
        "evaluation/index.html",
        models=models,
        runs=runs,
        latest_report=_latest_report(),
        comparison=_comparison_report(),
    )


@bp.get("/api/evaluation/latest")
@login_required
@admin_required
def api_latest():
    report = _latest_report()
    if report is None:
        return json_success(
            report=None,
            message=(
                "No evaluation report is available yet. Train a model to "
                "generate real evaluation metrics."
            ),
        )
    return json_success(report=report)


@bp.get("/api/evaluation/models")
@login_required
@admin_required
def api_models():
    models = ModelVersion.query.order_by(ModelVersion.created_at.desc()).all()
    return json_success(models=[model.to_dict() for model in models])


@bp.get("/api/evaluation/confusion-matrix")
@login_required
@admin_required
def api_confusion_matrix():
    """Return the confusion matrix of a specific model version."""
    version = request.args.get("version")
    if version:
        report = _load_report(version)
        if report is None:
            return json_error("No evaluation report was found for that version.", 404)
        return json_success(
            confusion_matrix=report.get("confusion_matrix"),
            labels=report.get("confusion_matrix", {}).get("labels", []),
            version=version,
        )

    report = _latest_report()
    if report is None:
        return json_success(confusion_matrix=None, labels=[])
    return json_success(
        confusion_matrix=report.get("confusion_matrix"),
        labels=report.get("confusion_matrix", {}).get("labels", []),
        version=report.get("model_version"),
    )


@bp.get("/api/evaluation/report/<path:version>")
@login_required
@admin_required
def api_report(version: str):
    """Return the full evaluation report for a specific model version."""
    report = _load_report(version)
    if report is None:
        return json_error("No evaluation report was found for that version.", 404)
    return json_success(report=report, version=version)


@bp.get("/api/evaluation/compare")
@login_required
@admin_required
def api_compare():
    """Return the Random Forest / SVM / KNN comparison produced by the pipeline."""
    report = _comparison_report()
    if report is None:
        return json_success(
            comparison=None,
            message=(
                "No model comparison is available. Run "
                "'python -m ml.compare_models' to generate one."
            ),
        )
    return json_success(comparison=report)


@bp.get("/api/evaluation/metrics/<int:run_id>")
@login_required
@admin_required
def api_run_metrics(run_id: int):
    run = db.session.get(TrainingRun, run_id)
    if run is None:
        return json_error("Training run not found.", 404)
    metrics = (
        EvaluationMetric.query.filter_by(training_run_id=run_id)
        .order_by(EvaluationMetric.id)
        .all()
    )
    return json_success(
        run=run.to_dict(),
        metrics=[metric.to_dict() for metric in metrics],
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
