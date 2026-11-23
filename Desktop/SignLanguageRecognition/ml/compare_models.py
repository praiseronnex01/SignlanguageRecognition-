"""Compare Random Forest, SVM and KNN on the same participant-independent split.

Usage::

    python -m ml.compare_models --dataset-version v1

The resulting JSON report is what the Evaluation page renders when no
algorithm-specific report is available.
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

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # pragma: no cover - optional at runtime
    pass

from ml.evaluate import save_confusion_matrix_png, write_report
from ml.preprocess import DatasetError, find_dataset_file, validate_bundle
from ml.train import ALGORITHMS, available_hand_counts, train_single_schema

ALGORITHM_LABELS = {
    "random_forest": "Random Forest",
    "svm": "Support Vector Machine",
    "knn": "K-Nearest Neighbours",
}


def compare(
    dataset_version: str | None = None,
    hand_count: int | None = None,
    tune: bool = False,
    random_state: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> dict:
    """Train every algorithm on the same split and collect the metrics."""
    from app import create_app

    app = create_app()
    with app.app_context():
        dataset_path = find_dataset_file(dataset_version)
        hand_counts = available_hand_counts(dataset_path)
        if hand_count in (1, 2):
            hand_counts = [int(hand_count)]

        results: dict = {}
        for hc in hand_counts:
            for algorithm in ALGORITHMS:
                print(f"[compare] Training {algorithm} ({hc}-hand schema)...", flush=True)
                result = train_single_schema(
                    dataset_path=dataset_path,
                    hand_count=hc,
                    algorithm=algorithm,
                    parameters={},
                    tune=tune,
                    random_state=random_state,
                    train_ratio=train_ratio,
                    validation_ratio=validation_ratio,
                    progress=None,
                )
                results.setdefault(f"hand_{hc}", {})[algorithm] = {
                    "label": ALGORITHM_LABELS[algorithm],
                    "metrics": result["metrics"],
                    "per_class": result["per_class"],
                    "latency": result["latency"],
                    "rejection_threshold": result["rejection_threshold"],
                    "confusion_matrix": result["confusion_matrix"],
                    "split": result["split"],
                    "best_parameters": result["best_params"],
                }
                save_confusion_matrix_png(
                    result["confusion_matrix"]["matrix"],
                    result["confusion_matrix"]["labels"],
                    os.path.join(
                        app.config["REPORTS_DIR"], f"comparison_{algorithm}_h{hc}.png"
                    ),
                )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_path": dataset_path,
            "dataset_version": dataset_version,
            "tuned": bool(tune),
            "results": results,
            "best_by_schema": {
                schema: max(
                    entries.items(),
                    key=lambda item: item[1]["metrics"].get("f1_macro", 0.0),
                )[0]
                for schema, entries in results.items()
            },
        }
        report_path = os.path.join(app.config["REPORTS_DIR"], f"comparison_{timestamp}.json")
        write_report(report, report_path)
        report["report_path"] = report_path
        return report


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare Random Forest, SVM and KNN classifiers."
    )
    parser.add_argument("--dataset-version", default=None)
    parser.add_argument("--hand-count", default="auto")
    parser.add_argument("--tune", action="store_true", help="Tune every algorithm.")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    hand_count = None
    if str(args.hand_count).lower() not in {"auto", "", "none"}:
        hand_count = int(args.hand_count)

    try:
        report = compare(
            dataset_version=args.dataset_version,
            hand_count=hand_count,
            tune=args.tune,
            random_state=args.random_state,
        )
    except DatasetError as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Comparison failed: {exc}", file=sys.stderr)
        raise

    for schema, entries in report["results"].items():
        print(f"\n{schema}")
        print(f"{'algorithm':<28}{'accuracy':>12}{'macro F1':>12}{'p95 ms':>10}")
        for algorithm, data in entries.items():
            metrics = data["metrics"]
            latency = (data.get("latency") or {}).get("p95_ms", 0.0)
            print(
                f"{ALGORITHM_LABELS[algorithm]:<28}"
                f"{metrics.get('accuracy', 0):>12.4f}"
                f"{metrics.get('f1_macro', 0):>12.4f}"
                f"{latency:>10.3f}"
            )
    print(f"\nReport written to {report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
