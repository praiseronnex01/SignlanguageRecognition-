"""Training pipeline for the MSL recognition models.

Usage
-----
Run through the web interface (recommended) or directly::

    python -m ml.train --algorithm random_forest --dataset-version v1
    python -m ml.train --run-id 3

The script can be executed inside an existing Flask application context, or it
will create its own application context when run from the command line.

Training is executed as a background process by the web application; progress
is written into the ``training_runs`` table so the front end can poll it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
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

import numpy as np
import pandas as pd

from ml.evaluate import (
    evaluate_predictions,
    probability_matrix,
    save_confusion_matrix_png,
    schema_for,
    select_rejection_threshold,
    write_report,
)
from ml.preprocess import (
    DatasetError,
    find_dataset_file,
    load_dataset,
    participant_split,
    validate_bundle,
)

ALGORITHMS = ("random_forest", "svm", "knn")


def _svm_uses_calibration() -> bool:
    """scikit-learn 1.9 deprecated ``SVC(probability=True)``."""
    try:
        import sklearn

        major, minor = (int(part) for part in sklearn.__version__.split(".")[:2])
        return (major, minor) >= (1, 9)
    except Exception:  # pragma: no cover - defensive
        return False


def _svm_param_prefix() -> str:
    """The grid prefix for the SVM estimator."""
    if _svm_uses_calibration():
        return "classifier__base_estimator__"
    return "classifier__"


def _build_svc(params: dict, random_state: int):
    """Build an SVM estimator that can report class probabilities."""
    from sklearn.svm import SVC

    kwargs = {
        "C": float(params.get("C", 1.0)),
        "kernel": str(params.get("kernel", "rbf")),
        "gamma": params.get("gamma", "scale"),
        "class_weight": params.get("class_weight", "balanced"),
    }
    if _svm_uses_calibration():
        from sklearn.calibration import CalibratedClassifierCV

        return CalibratedClassifierCV(SVC(**kwargs), ensemble=False)
    return SVC(probability=True, random_state=random_state, **kwargs)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_pipeline(algorithm: str, params: dict, random_state: int):
    """Build the scikit-learn pipeline for an algorithm."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    params = dict(params or {})

    if algorithm == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            min_samples_split=int(params.get("min_samples_split", 2)),
            max_features=params.get("max_features", "sqrt"),
            class_weight=params.get("class_weight", "balanced"),
            bootstrap=bool(params.get("bootstrap", True)),
            random_state=random_state,
            n_jobs=-1,
        )
    elif algorithm == "svm":
        classifier = _build_svc(params, random_state)
    elif algorithm == "knn":
        classifier = KNeighborsClassifier(
            n_neighbors=int(params.get("n_neighbors", 5)),
            weights=str(params.get("weights", "distance")),
            p=int(params.get("p", 2)),
            n_jobs=-1,
        )
    else:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. Expected one of: {ALGORITHMS}."
        )

    return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])


def parameter_grid(algorithm: str, params: dict) -> dict:
    """Build the hyper-parameter search space from the administrator input."""
    params = dict(params or {})
    grid: dict = {}

    if algorithm == "random_forest":
        n_estimators = params.get("n_estimators")
        grid["classifier__n_estimators"] = (
            [int(v) for v in n_estimators]
            if isinstance(n_estimators, (list, tuple)) and n_estimators
            else [int(n_estimators)] if n_estimators else [200, 400]
        )
        max_depth = params.get("max_depth")
        grid["classifier__max_depth"] = (
            [None if v in (None, "None", "none") else int(v) for v in max_depth]
            if isinstance(max_depth, (list, tuple)) and max_depth
            else [None]
        )
        min_samples_leaf = params.get("min_samples_leaf")
        grid["classifier__min_samples_leaf"] = (
            [int(v) for v in min_samples_leaf]
            if isinstance(min_samples_leaf, (list, tuple)) and min_samples_leaf
            else [int(min_samples_leaf)] if min_samples_leaf else [1, 2]
        )
        class_weight = params.get("class_weight")
        grid["classifier__class_weight"] = (
            list(class_weight)
            if isinstance(class_weight, (list, tuple)) and class_weight
            else [class_weight] if class_weight else ["balanced"]
        )
    elif algorithm == "svm":
        prefix = _svm_param_prefix()
        c_values = params.get("C")
        grid[f"{prefix}C"] = (
            [float(v) for v in c_values]
            if isinstance(c_values, (list, tuple)) and c_values
            else [float(c_values)] if c_values else [1.0, 10.0]
        )
        kernel = params.get("kernel")
        grid[f"{prefix}kernel"] = (
            list(kernel)
            if isinstance(kernel, (list, tuple)) and kernel
            else [str(kernel)] if kernel else ["rbf"]
        )
        gamma = params.get("gamma")
        grid[f"{prefix}gamma"] = (
            list(gamma)
            if isinstance(gamma, (list, tuple)) and gamma
            else [str(gamma)] if gamma else ["scale"]
        )
        class_weight = params.get("class_weight")
        grid[f"{prefix}class_weight"] = (
            list(class_weight)
            if isinstance(class_weight, (list, tuple)) and class_weight
            else [class_weight] if class_weight else ["balanced"]
        )
    elif algorithm == "knn":
        neighbours = params.get("n_neighbors")
        grid["classifier__n_neighbors"] = (
            [int(v) for v in neighbours]
            if isinstance(neighbours, (list, tuple)) and neighbours
            else [int(neighbours)] if neighbours else [5]
        )
        weights = params.get("weights")
        grid["classifier__weights"] = (
            list(weights)
            if isinstance(weights, (list, tuple)) and weights
            else [str(weights)] if weights else ["uniform"]
        )
    return grid


def make_cv(groups, y, random_state: int):
    """Create a group-aware cross-validation splitter."""
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    unique_groups = len(set(groups))
    n_splits = max(2, min(5, unique_groups))
    if unique_groups < 2:
        return None
    try:
        return StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
    except Exception:  # pragma: no cover - depends on the sklearn version
        return GroupKFold(n_splits=n_splits)


# ---------------------------------------------------------------------------
# Core training routine
# ---------------------------------------------------------------------------
def train_single_schema(
    dataset_path: str,
    hand_count: int,
    algorithm: str,
    parameters: dict,
    tune: bool,
    random_state: int,
    train_ratio: float,
    validation_ratio: float,
    progress=None,
) -> dict:
    """Train and evaluate one model for a single feature schema."""

    def report_progress(value: float, message: str) -> None:
        print(f"[train] {message} ({value:.0f}%)", flush=True)
        if progress is not None:
            progress(value, message)

    report_progress(10, f"Loading the dataset for the {hand_count}-hand schema")
    bundle = load_dataset(dataset_path, hand_count=hand_count)

    report_progress(20, "Validating the dataset")
    validation = validate_bundle(bundle)
    for issue in validation["issues"]:
        print(f"[train] WARNING: {issue}", flush=True)

    report_progress(30, "Splitting the dataset by participant")
    train_idx, validation_idx, test_idx, split_info = participant_split(
        bundle.groups,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        random_state=random_state,
    )

    X = bundle.X
    y = bundle.y
    groups = bundle.groups

    X_train, y_train, g_train = X[train_idx], y[train_idx], groups[train_idx]
    X_val, y_val, g_val = X[validation_idx], y[validation_idx], groups[validation_idx]
    X_test, y_test, g_test = X[test_idx], y[test_idx], groups[test_idx]

    # The label encoder is fitted on training + validation participants only.
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    encoder.fit(np.concatenate([y_train, y_val]) if len(y_val) else y_train)
    classes = [str(c) for c in encoder.classes_]

    pipeline = build_pipeline(algorithm, parameters, random_state)
    search_params = None
    best_params = dict(parameters or {})

    if tune:
        report_progress(45, "Tuning hyperparameters (grouped cross-validation)")
        from sklearn.model_selection import GridSearchCV

        grid = parameter_grid(algorithm, parameters)
        cv = make_cv(g_train, y_train, random_state)
        if grid and cv is not None:
            search = GridSearchCV(
                pipeline,
                grid,
                cv=cv,
                scoring="f1_macro",
                n_jobs=-1,
                error_score="raise",
            )
            search.fit(X_train, y_train, groups=g_train)
            pipeline = search.best_estimator_
            best_params = {
                key.replace("classifier__", ""): value
                for key, value in search.best_params_.items()
            }
            print(f"[train] Best parameters: {best_params}", flush=True)
        else:
            search_params = None

    # Refit on train + validation participants (the test participants stay untouched).
    if len(y_val):
        X_fit = np.concatenate([X_train, X_val])
        y_fit = np.concatenate([y_train, y_val])
        g_fit = np.concatenate([g_train, g_val])
    else:
        X_fit, y_fit, g_fit = X_train, y_train, g_train

    report_progress(60, "Fitting the final model")
    pipeline.fit(X_fit, y_fit)

    report_progress(75, "Evaluating on the held-out test participants")
    y_pred = pipeline.predict(X_test)
    evaluation = evaluate_predictions(
        y_test, y_pred, classes, pipeline=pipeline, X_test=X_test
    )

    # The rejection threshold is derived from the validation participants only.
    if len(y_val):
        val_probabilities = probability_matrix(pipeline, X_val, classes)
        threshold, threshold_info = select_rejection_threshold(y_val, val_probabilities, classes)
    else:
        threshold, threshold_info = 0.60, {
            "threshold": 0.60,
            "reason": "No validation participants were available.",
            "true_acceptance": 0.0,
            "false_acceptance": 0.0,
        }

    test_probabilities = probability_matrix(pipeline, X_test, classes)
    accepted = test_probabilities.max(axis=1) >= threshold if len(test_probabilities) else []
    if len(accepted):
        accuracy_at_threshold = float((y_pred[accepted] == y_test[accepted]).mean())
        acceptance_rate = float(accepted.mean())
    else:  # pragma: no cover - defensive
        accuracy_at_threshold = 0.0
        acceptance_rate = 0.0

    report_progress(85, "Saving the model artifacts")
    schema = schema_for(hand_count)
    metrics = evaluation["metrics"]
    metrics.update(
        {
            "accuracy_at_threshold": round(accuracy_at_threshold, 4),
            "acceptance_rate_at_threshold": round(acceptance_rate, 4),
        }
    )

    return {
        "pipeline": pipeline,
        "encoder": encoder,
        "classes": classes,
        "schema": schema,
        "metrics": metrics,
        "per_class": evaluation["per_class"],
        "classification_report": evaluation["classification_report"],
        "confusion_matrix": evaluation["confusion_matrix"],
        "latency": evaluation["latency"],
        "split": split_info,
        "rejection_threshold": threshold,
        "threshold_info": threshold_info,
        "best_params": best_params,
        "dataset_validation": validation,
        "hand_count": hand_count,
        "algorithm": algorithm,
    }


def available_hand_counts(dataset_path: str) -> list:
    """Return the hand configurations present in a dataset file."""
    frame = pd.read_csv(dataset_path, usecols=["hand_count"])
    return sorted({int(value) for value in frame["hand_count"].unique()})


def next_version_numbers(existing_versions, group_index: int) -> tuple:
    """Return the ``(major, minor)`` components for the next model version."""
    majors = []
    for version in existing_versions:
        try:
            majors.append(int(str(version).split(".")[0]))
        except (ValueError, IndexError):
            continue
    major = (max(majors) + 1) if majors else 1
    return major, group_index


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def run_training(
    run_id: int | None = None,
    algorithm: str | None = None,
    dataset_version: str | None = None,
    hand_count: int | None = None,
    parameters: dict | None = None,
    tune: bool = True,
) -> dict:
    """Execute a training run.  Returns a summary dictionary."""
    from app import create_app
    from app.extensions import db
    from app.models.evaluation_metric import EvaluationMetric
    from app.models.model_version import ModelVersion
    from app.models.training_run import TrainingRun
    from app.services import model_service

    app = create_app()
    with app.app_context():
        run = None
        if run_id is not None:
            run = db.session.get(TrainingRun, run_id)
            if run is None:
                raise DatasetError(f"Training run {run_id} was not found.")
            configuration = run.configuration
            algorithm = algorithm or configuration.get("algorithm", "random_forest")
            dataset_version = dataset_version or configuration.get("dataset_version")
            hand_count = hand_count or configuration.get("hand_count")
            parameters = parameters if parameters is not None else configuration.get("parameters", {})
            tune = configuration.get("tune_hyperparameters", True)
            random_state = int(configuration.get("random_state", 42))
            train_ratio = float(configuration.get("train_ratio", 0.70))
            validation_ratio = float(configuration.get("validation_ratio", 0.15))
        else:
            random_state = int(app.config.get("RANDOM_STATE", 42))
            train_ratio = float(app.config.get("TRAIN_PARTICIPANT_RATIO", 0.70))
            validation_ratio = float(app.config.get("VALIDATION_PARTICIPANT_RATIO", 0.15))

        algorithm = (algorithm or "random_forest").strip().lower()
        if algorithm not in ALGORITHMS:
            raise DatasetError(
                f"Unsupported algorithm '{algorithm}'. Expected one of: {', '.join(ALGORITHMS)}."
            )

        def set_progress(value: float, message: str) -> None:
            if run is None:
                return
            try:
                db.session.expire(run)
                run.progress = float(min(max(value, 0.0), 100.0))
                run.status = TrainingRun.STATUS_RUNNING
                run.started_at = run.started_at or datetime.now(timezone.utc)
                db.session.commit()
            except Exception:  # pragma: no cover - defensive
                db.session.rollback()

        if run is not None:
            run.status = TrainingRun.STATUS_RUNNING
            run.started_at = datetime.now(timezone.utc)
            run.progress = 2.0
            db.session.commit()

        try:
            dataset_path = find_dataset_file(dataset_version)
            print(f"[train] Dataset: {dataset_path}", flush=True)

            hand_counts = available_hand_counts(dataset_path)
            if hand_count in (1, 2):
                hand_counts = [int(hand_count)]
            if not hand_counts:
                raise DatasetError("The dataset does not contain any samples.")

            existing_versions = [row.version for row in ModelVersion.query.all()]
            major, _ = next_version_numbers(existing_versions, 0)

            created_models: list = []
            for group_index, hc in enumerate(hand_counts):
                result = train_single_schema(
                    dataset_path=dataset_path,
                    hand_count=hc,
                    algorithm=algorithm,
                    parameters=parameters or {},
                    tune=bool(tune),
                    random_state=random_state,
                    train_ratio=train_ratio,
                    validation_ratio=validation_ratio,
                    progress=set_progress,
                )

                version = f"{major}.{group_index}.0"
                model_path, encoder_path = model_service.save_artifacts(
                    pipeline=result["pipeline"],
                    encoder=result["encoder"],
                    schema=result["schema"],
                    metrics=result["metrics"],
                    algorithm=algorithm,
                    version=version,
                    dataset_version=dataset_version,
                    rejection_threshold=result["rejection_threshold"],
                )

                report = {
                    "model_version": version,
                    "algorithm": algorithm,
                    "dataset_path": dataset_path,
                    "dataset_version": dataset_version,
                    "hand_count": hc,
                    "classes": result["classes"],
                    "metrics": result["metrics"],
                    "per_class": result["per_class"],
                    "classification_report": result["classification_report"],
                    "confusion_matrix": result["confusion_matrix"],
                    "latency": result["latency"],
                    "split": result["split"],
                    "rejection_threshold": result["rejection_threshold"],
                    "threshold_analysis": result["threshold_info"],
                    "best_parameters": result["best_params"],
                    "dataset_validation": result["dataset_validation"],
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
                report_path = os.path.join(
                    app.config["REPORTS_DIR"], f"evaluation_{version}.json"
                )
                write_report(report, report_path)
                save_confusion_matrix_png(
                    result["confusion_matrix"]["matrix"],
                    result["confusion_matrix"]["labels"],
                    os.path.join(app.config["REPORTS_DIR"], f"confusion_{version}.png"),
                )

                model_version = ModelVersion(
                    version=version,
                    algorithm=algorithm,
                    model_path=model_path,
                    encoder_path=encoder_path,
                    feature_schema=json.dumps(result["schema"]),
                    accuracy=result["metrics"].get("accuracy"),
                    precision=result["metrics"].get("precision_macro"),
                    recall=result["metrics"].get("recall_macro"),
                    f1_score=result["metrics"].get("f1_macro"),
                    status=ModelVersion.STATUS_CANDIDATE,
                    rejection_threshold=result["rejection_threshold"],
                    dataset_version=dataset_version,
                    feature_version=result["schema"].get("feature_version", "v1.0"),
                    training_run_id=run.id if run else None,
                    notes=json.dumps(result["best_params"]),
                )
                db.session.add(model_version)
                db.session.commit()
                created_models.append(model_version)

                if run is not None:
                    for name, value in result["metrics"].items():
                        if isinstance(value, (int, float)):
                            db.session.add(
                                EvaluationMetric(
                                    training_run_id=run.id,
                                    metric_name=name,
                                    metric_value=float(value),
                                    metric_group="test",
                                )
                            )
                    db.session.add(
                        EvaluationMetric(
                            training_run_id=run.id,
                            metric_name="rejection_threshold",
                            metric_value=float(result["rejection_threshold"]),
                            metric_group="threshold",
                        )
                    )
                    db.session.commit()

            if run is not None:
                run.status = TrainingRun.STATUS_COMPLETED
                run.progress = 100.0
                run.completed_at = datetime.now(timezone.utc)
                run.duration_seconds = (
                    (run.completed_at - run.started_at).total_seconds()
                    if run.started_at
                    else None
                )
                if created_models:
                    run.model_version_id = created_models[0].id
                run.dataset_version = dataset_version
                db.session.commit()

            return {
                "status": "completed",
                "models": [model.version for model in created_models],
                "metrics": {
                    model.version: {
                        "accuracy": model.accuracy,
                        "precision": model.precision,
                        "recall": model.recall,
                        "f1_score": model.f1_score,
                    }
                    for model in created_models
                },
            }

        except Exception as exc:
            app.logger.exception("Training run failed")
            if run is not None:
                db.session.rollback()
                run = db.session.get(TrainingRun, run.id)
                if run is not None:
                    run.status = TrainingRun.STATUS_FAILED
                    run.error_message = f"{exc}\n{traceback.format_exc(limit=5)}"
                    run.completed_at = datetime.now(timezone.utc)
                    db.session.commit()
            raise


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a Malawi Sign Language gesture classifier."
    )
    parser.add_argument("--run-id", type=int, default=None, help="Existing training run id.")
    parser.add_argument(
        "--algorithm",
        default="random_forest",
        choices=list(ALGORITHMS),
        help="Classifier to train.",
    )
    parser.add_argument(
        "--dataset-version", default=None, help="Dataset version, e.g. 'v1'."
    )
    parser.add_argument(
        "--hand-count",
        default="auto",
        help="Feature schema to train: 1, 2 or 'auto'.",
    )
    parser.add_argument(
        "--parameters", default="{}", help="JSON object with hyper-parameters."
    )
    parser.add_argument(
        "--no-tune", action="store_true", help="Skip hyper-parameter search."
    )
    args = parser.parse_args(argv)

    try:
        parameters = json.loads(args.parameters)
    except ValueError:
        print("--parameters must be a JSON object.", file=sys.stderr)
        return 2

    hand_count = None
    if str(args.hand_count).lower() not in {"auto", "", "none"}:
        hand_count = int(args.hand_count)

    try:
        summary = run_training(
            run_id=args.run_id,
            algorithm=args.algorithm,
            dataset_version=args.dataset_version,
            hand_count=hand_count,
            parameters=parameters,
            tune=not args.no_tune,
        )
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
