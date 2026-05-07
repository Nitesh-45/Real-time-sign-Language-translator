"""Webcam dataset collector for custom hand-sign samples.

Command-line usage:
    python src/dataset_collector.py --label water

Controls:
    s - save the current hand landmark sample
    q - quit the collector window
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message="SymbolDatabase.GetPrototype\\(\\) is deprecated.*",
    category=UserWarning,
)

# Allow direct execution with: python src/dataset_collector.py --label water
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    CAMERA_INDEX,
    FEATURE_VECTOR_SIZE,
    KEYPOINTS_DIR,
    LABELS_PATH,
    RAW_IMAGES_DIR,
    REQUIRED_SIGNS,
)
from src.hand_tracker import HandTracker  # noqa: E402
from src.preprocess import validate_feature_vector  # noqa: E402
from src.utils import ensure_directories  # noqa: E402


def feature_columns() -> list[str]:
    """Return CSV feature column names f1, f2, ... f63."""
    return [f"f{index}" for index in range(1, FEATURE_VECTOR_SIZE + 1)]


class DatasetCollector:
    """Collect and store MediaPipe hand landmarks for one or more signs."""

    def __init__(
        self,
        keypoints_dir: Path = KEYPOINTS_DIR,
        raw_images_dir: Path = RAW_IMAGES_DIR,
        labels_path: Path = LABELS_PATH,
    ) -> None:
        self.keypoints_dir = keypoints_dir
        self.raw_images_dir = raw_images_dir
        self.labels_path = labels_path
        ensure_directories()
        self.keypoints_dir.mkdir(parents=True, exist_ok=True)
        self.raw_images_dir.mkdir(parents=True, exist_ok=True)
        self.allowed_labels = self._load_or_create_labels()

    def _load_or_create_labels(self) -> list[str]:
        """Load labels.json, creating it with default labels if missing."""
        if not self.labels_path.exists():
            self.labels_path.parent.mkdir(parents=True, exist_ok=True)
            with self.labels_path.open("w", encoding="utf-8") as file:
                json.dump({"labels": REQUIRED_SIGNS}, file, indent=2)
            return REQUIRED_SIGNS.copy()

        try:
            with self.labels_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (json.JSONDecodeError, OSError):
            data = {"labels": REQUIRED_SIGNS}
            with self.labels_path.open("w", encoding="utf-8") as file:
                json.dump(data, file, indent=2)

        if isinstance(data, list):
            labels = data
        elif isinstance(data, dict):
            labels = data.get("labels") or data.get("required_signs") or data.get("signs")
        else:
            labels = None

        if not labels:
            raise ValueError(
                "labels.json must contain a list of labels or a key named "
                "'labels', 'required_signs', or 'signs'."
            )

        return [str(label) for label in labels]

    def validate_label(self, label: str) -> str:
        """Return a normalized label if it exists in labels.json."""
        normalized_label = label.strip().lower().replace(" ", "_")
        if normalized_label not in self.allowed_labels:
            allowed = ", ".join(self.allowed_labels)
            raise ValueError(
                f"Invalid label '{label}'. Choose one of these labels: {allowed}"
            )
        return normalized_label

    def save_sample(
        self,
        label: str,
        keypoints,
        frame=None,
        save_image: bool = False,
    ) -> Path:
        """Append one landmark sample to data/keypoints/<label>.csv.

        The CSV schema is exactly:
            label, f1, f2, f3, ... f63
        """
        label = self.validate_label(label)
        vector = validate_feature_vector(keypoints)
        csv_path = self.keypoints_dir / f"{label}.csv"

        row = {"label": label}
        row.update(
            {column: float(value) for column, value in zip(feature_columns(), vector)}
        )

        pd.DataFrame([row], columns=["label", *feature_columns()]).to_csv(
            csv_path,
            mode="a",
            index=False,
            header=not csv_path.exists(),
        )

        sample_count = self.count_samples(label)
        self._update_label_count(label, sample_count)

        if save_image and frame is not None:
            image_dir = self.raw_images_dir / label
            image_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_dir / f"{label}_{sample_count:04d}.jpg"), frame)

        return csv_path

    def _update_label_count(self, label: str, sample_count: int) -> None:
        """Update sample_counts in this collector's labels.json file."""
        if self.labels_path.exists():
            try:
                with self.labels_path.open("r", encoding="utf-8") as file:
                    data = json.load(file)
            except (json.JSONDecodeError, OSError):
                data = {"labels": self.allowed_labels}
        else:
            data = {"labels": self.allowed_labels}

        if isinstance(data, list):
            data = {"labels": data}

        data.setdefault("sample_counts", {})
        data["sample_counts"][label] = sample_count

        with self.labels_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    def count_samples(self, label: str) -> int:
        """Count rows already collected for a sign label."""
        label = self.validate_label(label)
        csv_path = self.keypoints_dir / f"{label}.csv"
        if not csv_path.exists():
            return 0
        try:
            return len(pd.read_csv(csv_path))
        except pd.errors.EmptyDataError:
            return 0

    def dataset_summary(self) -> pd.DataFrame:
        """Return a small table showing sample counts for every allowed label."""
        rows = [
            {"sign": label, "samples": self.count_samples(label)}
            for label in self.allowed_labels
        ]
        return pd.DataFrame(rows)


def _draw_collector_text(
    frame,
    label: str,
    sample_count: int,
    hand_detected: bool,
    auto_save: bool = False,
) -> None:
    """Draw label, count, and keyboard controls on the camera preview."""
    if hand_detected and auto_save:
        status = "Hand detected - auto saving"
    elif hand_detected:
        status = "Hand detected - press 's' to save"
    else:
        status = "No hand detected"

    controls = "Controls: q = quit" if auto_save else "Controls: s = save | q = quit"
    lines = [
        f"Label: {label}",
        f"Samples saved: {sample_count}",
        status,
        controls,
    ]

    cv2.rectangle(frame, (10, 10), (560, 135), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        color = (0, 255, 0) if hand_detected or index != 2 else (0, 200, 255)
        cv2.putText(
            frame,
            line,
            (25, 42 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )


def collect_dataset_cli(
    label: str,
    camera_index: int = CAMERA_INDEX,
    save_images: bool = False,
    max_samples: Optional[int] = None,
    auto_save: bool = False,
    auto_save_delay: float = 0.12,
    start_delay: int = 3,
) -> None:
    """Open webcam and let the user save samples with the 's' key."""
    collector = DatasetCollector()
    label = collector.validate_label(label)
    tracker = HandTracker(draw_landmarks=True)
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        tracker.close()
        raise RuntimeError("Could not open webcam. Check camera permissions or camera index.")

    print(f"Collecting samples for label: {label}")
    if auto_save:
        print("Auto-save is ON. Keep the hand visible. Press 'q' to quit.")
    else:
        print("Press 's' to save the current sample. Press 'q' to quit.")

    if start_delay > 0:
        print(f"Starting in {start_delay} seconds. Get the '{label}' gesture ready.")
        time.sleep(start_delay)

    last_auto_save_time = 0.0
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Could not read a frame from the webcam.")
                break

            result = tracker.process_frame(frame)
            keypoints = result.keypoints
            sample_count = collector.count_samples(label)

            preview = result.annotated_frame
            _draw_collector_text(
                preview,
                label,
                sample_count,
                keypoints is not None,
                auto_save=auto_save,
            )
            cv2.imshow("Dataset Collector", preview)

            if auto_save and keypoints is not None:
                now = time.monotonic()
                if now - last_auto_save_time >= auto_save_delay:
                    collector.save_sample(label, keypoints, frame, save_images)
                    sample_count = collector.count_samples(label)
                    last_auto_save_time = now
                    print(f"Saved sample {sample_count} for '{label}'.")

                    if max_samples is not None and sample_count >= max_samples:
                        print(f"Reached max samples ({max_samples}).")
                        break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                if keypoints is None:
                    print("No hand detected. Sample not saved.")
                    continue

                collector.save_sample(label, keypoints, frame, save_images)
                sample_count = collector.count_samples(label)
                print(f"Saved sample {sample_count} for '{label}'.")

                if max_samples is not None and sample_count >= max_samples:
                    print(f"Reached max samples ({max_samples}).")
                    break

            if key == ord("q"):
                print("Collector closed.")
                break
    finally:
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect webcam hand-landmark samples for a custom sign."
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Sign label to collect, for example: water",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help="OpenCV camera index. Default: 0",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit. Without this, press q when finished.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Also save raw camera frames in data/raw_images/<label>/.",
    )
    parser.add_argument(
        "--auto-save",
        action="store_true",
        help="Automatically save samples whenever a hand is detected.",
    )
    parser.add_argument(
        "--auto-save-delay",
        type=float,
        default=0.12,
        help="Seconds between auto-saved samples. Default: 0.12",
    )
    parser.add_argument(
        "--start-delay",
        type=int,
        default=3,
        help="Seconds to wait before collecting starts. Default: 3",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_dataset_cli(
        label=args.label,
        camera_index=args.camera,
        save_images=args.save_images,
        max_samples=args.max_samples,
        auto_save=args.auto_save,
        auto_save_delay=args.auto_save_delay,
        start_delay=args.start_delay,
    )
