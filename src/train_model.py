"""Train a RandomForest gesture classifier from collected hand landmarks.

Command:
    python src/train_model.py
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Allow direct execution with: python src/train_model.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    ENCODER_PATH,
    LABELS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    REQUIRED_SIGNS,
)
from src.preprocess import NO_DATA_MESSAGE, load_keypoint_dataset, normalize_batch  # noqa: E402
from src.utils import ensure_directories  # noqa: E402


MIN_SAMPLES_PER_SIGN = 30


@dataclass
class TrainingResult:
    """Small container used by both CLI and Streamlit UI."""

    samples: int
    classes: list[str]
    accuracy: Optional[float]
    report: str
    confusion_matrix_text: str
    model_path: str
    encoder_path: str
    warnings: list[str]


def load_expected_labels(labels_path: Path = LABELS_PATH) -> list[str]:
    """Load configured labels from labels.json, with REQUIRED_SIGNS fallback."""
    if not labels_path.exists():
        return REQUIRED_SIGNS.copy()

    try:
        with labels_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        return REQUIRED_SIGNS.copy()

    if isinstance(data, list):
        return [str(label) for label in data]
    if isinstance(data, dict):
        labels = data.get("labels") or data.get("required_signs") or data.get("signs")
        if labels:
            return [str(label) for label in labels]

    return REQUIRED_SIGNS.copy()


def sample_count_warnings(
    labels: pd.Series,
    expected_labels: Optional[list[str]] = None,
    minimum_samples: int = MIN_SAMPLES_PER_SIGN,
) -> list[str]:
    """Warn when any configured sign has fewer than the recommended samples."""
    counts = labels.value_counts()
    signs_to_check = expected_labels or sorted(counts.index.tolist())
    low_sample_signs = [
        f"{sign} ({int(counts.get(sign, 0))})"
        for sign in signs_to_check
        if int(counts.get(sign, 0)) < minimum_samples
    ]

    if not low_sample_signs:
        return []

    return [
        "Warning: These signs have fewer than "
        f"{minimum_samples} samples: {', '.join(low_sample_signs)}"
    ]


def _can_split(encoded_labels: np.ndarray, test_size: float) -> bool:
    """Check if train_test_split can produce a useful validation split."""
    _, class_counts = np.unique(encoded_labels, return_counts=True)
    class_count = len(class_counts)
    test_count = math.ceil(len(encoded_labels) * test_size)
    train_count = len(encoded_labels) - test_count
    return (
        len(encoded_labels) >= 4
        and class_count >= 2
        and class_counts.min() >= 2
        and test_count >= class_count
        and train_count >= class_count
    )


def _format_confusion_matrix(matrix: np.ndarray, class_names: list[str]) -> str:
    """Create a readable confusion matrix table for terminal and Streamlit."""
    matrix_frame = pd.DataFrame(matrix, index=class_names, columns=class_names)
    return matrix_frame.to_string()


def train_model(test_size: float = 0.2, n_estimators: int = 200) -> TrainingResult:
    """Train and save the gesture model and label encoder."""
    ensure_directories()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_keypoint_dataset()
    if y.empty:
        raise FileNotFoundError(NO_DATA_MESSAGE)

    warnings = sample_count_warnings(y, load_expected_labels())

    X_normalized = normalize_batch(X)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = list(label_encoder.classes_)

    if len(class_names) < 2:
        raise ValueError("Please collect samples for at least two signs before training.")

    if not _can_split(y_encoded, test_size):
        raise ValueError(
            "Not enough samples to create a train/test split. "
            "Collect more samples for each sign first."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized,
        y_encoded,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
    matrix_text = _format_confusion_matrix(matrix, class_names)

    # Save a final model trained on all collected samples after validation.
    model.fit(X_normalized, y_encoded)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    return TrainingResult(
        samples=len(y),
        classes=class_names,
        accuracy=accuracy,
        report=report,
        confusion_matrix_text=matrix_text,
        model_path=str(MODEL_PATH),
        encoder_path=str(ENCODER_PATH),
        warnings=warnings,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a RandomForest gesture classifier from collected CSV data."
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        result = train_model(
            test_size=args.test_size,
            n_estimators=args.n_estimators,
        )
    except FileNotFoundError as exc:
        print(str(exc))
        return
    except ValueError as exc:
        print(f"Training failed: {exc}")
        return

    for warning in result.warnings:
        print(warning)

    print(f"Loaded {result.samples} samples.")
    print(f"Classes: {', '.join(result.classes)}")
    print(f"Accuracy: {result.accuracy:.2%}")
    print("\nClassification Report:")
    print(result.report)
    print("Confusion Matrix:")
    print(result.confusion_matrix_text)
    print(f"\nSaved model to: {result.model_path}")
    print(f"Saved label encoder to: {result.encoder_path}")


if __name__ == "__main__":
    main()
