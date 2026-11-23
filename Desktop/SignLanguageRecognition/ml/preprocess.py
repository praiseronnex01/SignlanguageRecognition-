"""Dataset loading, validation and participant-independent splitting.

The most important responsibility of this module is preventing **data
leakage**: frames captured from the same participant must never appear in
both the training and the test set.  All splits are therefore performed on
*participant groups*, never on individual samples.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from ml.feature_engineering import feature_dimensions

FEATURE_PREFIX = "f"
REQUIRED_COLUMNS = [
    "sample_id",
    "gesture_label",
    "participant_code",
    "hand_count",
    "feature_version",
]


class DatasetError(RuntimeError):
    """Raised when a dataset cannot be loaded or is not usable."""


@dataclass
class DatasetBundle:
    """A validated dataset ready for model training."""

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    hand_counts: np.ndarray
    feature_columns: list = field(default_factory=list)
    classes: list = field(default_factory=list)
    participants: list = field(default_factory=list)
    source_path: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        return int(self.X.shape[0])

    @property
    def feature_count(self) -> int:
        return int(self.X.shape[1])

    def summary(self) -> dict:
        return {
            "samples": self.sample_count,
            "features": self.feature_count,
            "classes": len(self.classes),
            "participants": len(self.participants),
            "source_path": self.source_path,
            "hand_count": sorted({int(h) for h in self.hand_counts}),
        }


def feature_columns_for(hand_count: int) -> list:
    """Return the CSV column names that hold the features for a hand count."""
    return [
        f"{FEATURE_PREFIX}{index}" for index in range(feature_dimensions(hand_count))
    ]


def find_dataset_file(dataset_version: str | None = None, dataset_dir: str | None = None) -> str:
    """Resolve the dataset CSV to train on.

    ``dataset_version`` accepts ``"v1"``, ``"1"`` or ``"dataset_v1.csv"``.
    When omitted the most recently modified exported dataset is used.
    """
    if dataset_dir is None:
        from app.config import DATASET_DIR

        dataset_dir = str(DATASET_DIR)
    processed = os.path.join(dataset_dir, "processed")

    if dataset_version:
        name = str(dataset_version).strip().lower()
        if name.endswith(".csv"):
            candidate = os.path.join(processed, name)
            if os.path.isfile(candidate):
                return candidate
            raise DatasetError(f"Dataset file '{name}' was not found in {processed}.")
        number = name.lstrip("v")
        if not number.isdigit():
            raise DatasetError(f"Invalid dataset version '{dataset_version}'.")
        candidate = os.path.join(processed, f"dataset_v{number}.csv")
        if os.path.isfile(candidate):
            return candidate
        raise DatasetError(
            f"Dataset version '{dataset_version}' has not been exported yet. "
            "Export the dataset from the Dataset Management page first."
        )

    if not os.path.isdir(processed):
        raise DatasetError(f"Dataset directory '{processed}' does not exist.")
    available = [
        os.path.join(processed, name)
        for name in os.listdir(processed)
        if name.startswith("dataset_v") and name.endswith(".csv")
    ]
    if not available:
        raise DatasetError(
            "No exported dataset was found. Collect samples and export the "
            "dataset before training a model."
        )
    return max(available, key=os.path.getmtime)


def load_dataset(
    path: str, hand_count: int | None = None, drop_invalid: bool = True
) -> DatasetBundle:
    """Load and validate a dataset CSV.

    ``hand_count`` restricts the dataset to a single feature schema (1 or 2).
    When it is ``None`` every hand configuration present in the file is kept,
    but the resulting feature matrix can only represent one schema at a time -
    callers therefore normally train one model per hand configuration.
    """
    if not os.path.isfile(path):
        raise DatasetError(f"Dataset file '{path}' does not exist.")

    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetError(
            f"The dataset is missing required column(s): {', '.join(missing)}"
        )

    if hand_count is not None:
        frame = frame[frame["hand_count"].astype(int) == int(hand_count)]
        if frame.empty:
            raise DatasetError(
                f"The dataset does not contain any samples for the "
                f"{hand_count}-hand feature schema."
            )

    hand_counts = frame["hand_count"].astype(int).unique().tolist()
    if len(hand_counts) > 1:
        raise DatasetError(
            "The dataset mixes 1-hand and 2-hand samples. Train one model per "
            "hand configuration (use --hand-count to select one)."
        )

    resolved_hand_count = int(hand_counts[0])
    columns = feature_columns_for(resolved_hand_count)
    available = [column for column in columns if column in frame.columns]
    if len(available) != len(columns):
        raise DatasetError(
            f"The dataset does not contain all {len(columns)} feature columns "
            f"required by the {resolved_hand_count}-hand schema."
        )

    features = frame[columns].apply(pd.to_numeric, errors="coerce")
    invalid_mask = features.isna().any(axis=1)
    infinite_mask = ~np.isfinite(features.to_numpy(dtype=np.float64)).all(axis=1)
    invalid_mask = invalid_mask | infinite_mask

    if drop_invalid and invalid_mask.any():
        removed = int(invalid_mask.sum())
        frame = frame[~invalid_mask]
        features = features[~invalid_mask]
        print(f"[preprocess] Removed {removed} invalid sample(s).")
    elif invalid_mask.any():
        raise DatasetError(
            f"The dataset contains {int(invalid_mask.sum())} invalid sample(s)."
        )

    if frame.empty:
        raise DatasetError("The dataset does not contain any usable samples.")

    labels = frame["gesture_label"].astype(str).to_numpy()
    groups = frame["participant_code"].astype(str).to_numpy()
    X = features.to_numpy(dtype=np.float64)
    y = labels
    hand_count_array = frame["hand_count"].astype(int).to_numpy()

    classes = sorted({str(label) for label in labels})
    participants = sorted({str(group) for group in groups})

    if len(classes) < 2:
        raise DatasetError(
            "At least two gesture classes are required to train a classifier "
            f"(found {len(classes)})."
        )

    return DatasetBundle(
        X=X,
        y=y,
        groups=groups,
        hand_counts=hand_count_array,
        feature_columns=columns,
        classes=classes,
        participants=participants,
        source_path=os.path.abspath(path),
        metadata={
            "hand_count": resolved_hand_count,
            "feature_dims": len(columns),
            "created_from": os.path.basename(path),
        },
    )


def class_distribution(labels) -> dict:
    """Return the sample count per class."""
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts)}


def participant_distribution(groups) -> dict:
    """Return the sample count per participant."""
    values, counts = np.unique(np.asarray(groups), return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts)}


def validate_bundle(bundle: DatasetBundle) -> dict:
    """Produce a validation report for a loaded dataset."""
    distribution = class_distribution(bundle.y)
    participants = participant_distribution(bundle.groups)
    counts = list(distribution.values())

    issues: list = []
    if len(bundle.participants) < 2:
        issues.append(
            "Only one participant is present. Participant-independent validation "
            "is impossible - collect samples from at least three participants."
        )
    for label, count in distribution.items():
        if count < 5:
            issues.append(f"Class '{label}' has only {count} sample(s).")
    if counts and min(counts) / max(counts) < 0.25:
        issues.append(
            "The dataset is strongly imbalanced; consider balancing it before training."
        )
    for participant, count in participants.items():
        if count < 20:
            issues.append(
                f"Participant '{participant}' contributed only {count} sample(s)."
            )

    return {
        "samples": bundle.sample_count,
        "features": bundle.feature_count,
        "classes": len(bundle.classes),
        "participants": len(bundle.participants),
        "class_distribution": distribution,
        "participant_distribution": participants,
        "issues": issues,
        "ready": len(issues) == 0,
    }


def participant_split(
    groups,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    random_state: int = 42,
):
    """Split sample indices into participant-disjoint train/validation/test sets.

    Returns ``(train_idx, validation_idx, test_idx, info)``.
    """
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)

    if n_groups < 2:
        raise DatasetError(
            "At least two participants are required for a participant-independent split."
        )

    test_ratio = max(0.0, 1.0 - train_ratio - validation_ratio)

    if n_groups == 2:
        # Degenerate case: hold one participant out for testing.
        splitter = GroupShuffleSplit(n_splits=1, test_size=1, random_state=random_state)
        train_val_idx, test_idx = next(
            splitter.split(np.zeros(len(groups)), groups=groups)
        )
        train_idx = np.array([], dtype=int)
        validation_idx = train_val_idx
    elif n_groups == 3:
        splitter = GroupShuffleSplit(n_splits=1, test_size=1, random_state=random_state)
        train_val_idx, test_idx = next(
            splitter.split(np.zeros(len(groups)), groups=groups)
        )
        inner = GroupShuffleSplit(
            n_splits=1, test_size=validation_ratio / (train_ratio + validation_ratio),
            random_state=random_state + 1,
        )
        train_idx, validation_idx = next(
            inner.split(
                np.zeros(len(train_val_idx)), groups=groups[train_val_idx]
            )
        )
        validation_idx = train_val_idx[validation_idx]
        train_idx = train_val_idx[train_idx]
    else:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_ratio, random_state=random_state
        )
        train_val_idx, test_idx = next(
            splitter.split(np.zeros(len(groups)), groups=groups)
        )
        inner_ratio = validation_ratio / max(train_ratio + validation_ratio, 1e-9)
        inner = GroupShuffleSplit(
            n_splits=1, test_size=inner_ratio, random_state=random_state + 1
        )
        train_idx, validation_idx = next(
            inner.split(
                np.zeros(len(train_val_idx)), groups=groups[train_val_idx]
            )
        )
        validation_idx = train_val_idx[validation_idx]
        train_idx = train_val_idx[train_idx]

    train_groups = sorted({str(g) for g in groups[train_idx]})
    validation_groups = sorted({str(g) for g in groups[validation_idx]})
    test_groups = sorted({str(g) for g in groups[test_idx]})

    overlap = (
        set(train_groups) & set(validation_groups)
        | set(train_groups) & set(test_groups)
        | set(validation_groups) & set(test_groups)
    )
    if overlap:
        raise DatasetError(
            "Participant leakage detected between the train, validation and test "
            f"splits: {sorted(overlap)}"
        )

    info = {
        "train_participants": train_groups,
        "validation_participants": validation_groups,
        "test_participants": test_groups,
        "train_samples": int(len(train_idx)),
        "validation_samples": int(len(validation_idx)),
        "test_samples": int(len(test_idx)),
        "train_ratio": round(len(train_idx) / max(len(groups), 1), 4),
        "validation_ratio": round(len(validation_idx) / max(len(groups), 1), 4),
        "test_ratio": round(len(test_idx) / max(len(groups), 1), 4),
    }
    return train_idx, validation_idx, test_idx, info
