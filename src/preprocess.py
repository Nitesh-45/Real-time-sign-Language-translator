"""Dataset loading and preprocessing helpers for gesture training."""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config import FEATURE_VECTOR_SIZE, KEYPOINTS_DIR, NUM_LANDMARKS


NO_DATA_MESSAGE = "No dataset found. Please collect samples first."


def landmark_columns() -> list[str]:
    """Return the expected feature columns: f1, f2, ... f63."""
    return [f"f{index}" for index in range(1, FEATURE_VECTOR_SIZE + 1)]


def legacy_landmark_columns() -> list[str]:
    """Return older x0, y0, z0... column names for backwards compatibility."""
    columns: list[str] = []
    for index in range(NUM_LANDMARKS):
        columns.extend([f"x{index}", f"y{index}", f"z{index}"])
    return columns


def validate_feature_vector(keypoints: Iterable[float]) -> np.ndarray:
    """Validate that one sample has exactly 63 landmark values."""
    vector = np.asarray(list(keypoints), dtype=np.float32)
    if vector.shape[0] != FEATURE_VECTOR_SIZE:
        raise ValueError(
            f"Expected {FEATURE_VECTOR_SIZE} landmark values, got {vector.shape[0]}."
        )
    return vector


def normalize_keypoints(keypoints: Iterable[float]) -> np.ndarray:
    """Normalize one 63-value landmark vector relative to the wrist.

    The dataset collector already stores normalized landmarks when using the
    current `HandTracker`. This function is kept here so old raw CSV files and
    future utility calls are processed consistently before training.
    """
    vector = validate_feature_vector(keypoints)
    landmarks = vector.reshape(NUM_LANDMARKS, 3)

    wrist = landmarks[0].copy()
    normalized = landmarks - wrist

    scale = np.linalg.norm(normalized, axis=1).max()
    if scale < 1e-6:
        scale = 1.0

    return (normalized / scale).flatten().astype(np.float32)


def normalize_batch(features: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Normalize every row in a dataframe or array of landmark features."""
    values = np.asarray(features, dtype=np.float32)
    if values.size == 0:
        raise ValueError(NO_DATA_MESSAGE)
    return np.vstack([normalize_keypoints(row) for row in values])


def load_keypoint_csv_files(keypoints_dir: Path = KEYPOINTS_DIR) -> list[Path]:
    """Find all CSV files saved by the dataset collector."""
    if not keypoints_dir.exists():
        raise FileNotFoundError(NO_DATA_MESSAGE)

    csv_files = sorted(keypoints_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(NO_DATA_MESSAGE)
    return csv_files


def combine_keypoint_csvs(keypoints_dir: Path = KEYPOINTS_DIR) -> pd.DataFrame:
    """Load and combine all CSV files from data/keypoints/ into one dataframe."""
    csv_files = load_keypoint_csv_files(keypoints_dir)
    frames: list[pd.DataFrame] = []

    for csv_file in csv_files:
        try:
            frame = pd.read_csv(csv_file)
        except pd.errors.EmptyDataError:
            continue

        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise FileNotFoundError(NO_DATA_MESSAGE)

    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise FileNotFoundError(NO_DATA_MESSAGE)
    return combined


def get_available_feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Return feature columns supported by the combined dataset."""
    current_columns = landmark_columns()
    if all(column in dataset.columns for column in current_columns):
        return current_columns

    old_columns = legacy_landmark_columns()
    if all(column in dataset.columns for column in old_columns):
        return old_columns

    missing = [column for column in current_columns if column not in dataset.columns]
    raise ValueError(
        "Dataset is missing landmark feature columns. "
        f"Expected label,f1...f63. Missing examples: {', '.join(missing[:5])}"
    )


def split_features_and_labels(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate a combined dataset into feature matrix X and label vector y."""
    if "label" not in dataset.columns:
        raise ValueError("Dataset CSV files must contain a 'label' column.")

    feature_columns = get_available_feature_columns(dataset)
    cleaned = dataset.dropna(subset=["label", *feature_columns]).copy()
    if cleaned.empty:
        raise FileNotFoundError(NO_DATA_MESSAGE)

    cleaned["label"] = cleaned["label"].astype(str).str.strip()
    cleaned = cleaned[cleaned["label"] != ""]
    if cleaned.empty:
        raise FileNotFoundError(NO_DATA_MESSAGE)

    X = cleaned[feature_columns].astype(np.float32)
    y = cleaned["label"]
    return X, y


def load_keypoint_dataset(keypoints_dir: Path = KEYPOINTS_DIR) -> tuple[pd.DataFrame, pd.Series]:
    """Load all collected keypoint CSVs and return X features and y labels."""
    dataset = combine_keypoint_csvs(keypoints_dir)
    return split_features_and_labels(dataset)
