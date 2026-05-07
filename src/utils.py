import json
from pathlib import Path
from typing import Any, Dict

import cv2

from config import (
    EMOJIS_PATH,
    KEYPOINTS_DIR,
    LABELS_PATH,
    MODELS_DIR,
    RAW_IMAGES_DIR,
    REQUIRED_SIGNS,
)


DEFAULT_EMOJI = "🔤"


def ensure_directories() -> None:
    """Create project runtime folders if they do not already exist."""
    for directory in [KEYPOINTS_DIR, RAW_IMAGES_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return default


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def format_sign_name(sign: str | None) -> str:
    if not sign:
        return "-"
    return sign.replace("_", " ").title()


def load_emoji_mapping() -> Dict[str, str]:
    """Load sign-to-emoji mappings from assets/emojis.json."""
    return load_json(EMOJIS_PATH, default={}) or {}


def load_emoji_map() -> Dict[str, str]:
    """Backward-compatible alias for older code."""
    return load_emoji_mapping()


def get_emoji(sign: str | None) -> str:
    """Return the mapped emoji for a sign, or a default symbol if missing."""
    if not sign:
        return DEFAULT_EMOJI
    return load_emoji_mapping().get(sign, DEFAULT_EMOJI)


def load_supported_labels() -> list[str]:
    """Load supported labels from labels.json, falling back to config labels."""
    data = load_json(LABELS_PATH, default=None)
    if isinstance(data, list):
        labels = data
    elif isinstance(data, dict):
        labels = data.get("labels") or data.get("required_signs") or data.get("signs")
    else:
        labels = None

    if not labels:
        return REQUIRED_SIGNS.copy()

    cleaned = [
        str(label).strip().lower().replace(" ", "_")
        for label in labels
        if str(label).strip()
    ]
    return cleaned or REQUIRED_SIGNS.copy()


def update_label_count(label: str, count: int) -> None:
    data = load_json(LABELS_PATH, default={"required_signs": [], "sample_counts": {}})
    data.setdefault("sample_counts", {})
    data["sample_counts"][label] = count
    save_json(LABELS_PATH, data)


def draw_status_box(frame, text: str, color=(20, 120, 20)):
    """Draw a compact readable status label on a BGR frame."""
    if frame is None:
        return frame
    cv2.rectangle(frame, (10, 10), (10 + max(280, len(text) * 13), 50), (0, 0, 0), -1)
    cv2.putText(frame, text, (22, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return frame
