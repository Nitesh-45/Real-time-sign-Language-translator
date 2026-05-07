"""Real-time webcam gesture prediction.

Command:
    python src/predictor.py
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import joblib
import numpy as np

# Allow direct execution with: python src/predictor.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CAMERA_INDEX, CONFIDENCE_THRESHOLD, ENCODER_PATH, MODEL_PATH  # noqa: E402
from src.hand_tracker import HandTracker  # noqa: E402
from src.preprocess import validate_feature_vector  # noqa: E402


PredictionDict = dict[str, Optional[str] | float]


class SignPredictor:
    """Load the trained model and predict hand signs from webcam frames."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        encoder_path: Path = ENCODER_PATH,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        hand_tracker: Optional[HandTracker] = None,
    ) -> None:
        """Initialize model paths, threshold, and a MediaPipe hand tracker."""
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.confidence_threshold = confidence_threshold
        self.hand_tracker = hand_tracker
        self.model = None
        self.label_encoder = None
        self.model_loaded = False
        self.last_error: Optional[str] = None
        self.load_model()

    def load_model(self) -> bool:
        """Load model and label encoder from disk.

        Returns:
            True when both files are loaded successfully. False when files are
            missing or unreadable. Prediction will then return `model_missing`.
        """
        if not self.model_path.exists() or not self.encoder_path.exists():
            self.last_error = (
                "Model files are missing. Collect samples first, then train with: "
                "python src/train_model.py"
            )
            self.model_loaded = False
            return False

        try:
            self.model = joblib.load(self.model_path)
            self.label_encoder = joblib.load(self.encoder_path)
            self.model_loaded = True
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = f"Could not load model files: {exc}"
            self.model_loaded = False
            return False

    def predict(self, frame) -> PredictionDict:
        """Predict a sign from one OpenCV webcam frame.

        Returns:
            {"label": predicted_label, "confidence": confidence_score, "status": "success"}
            {"label": None, "confidence": 0, "status": "no_hand"}
            {"label": None, "confidence": confidence_score, "status": "low_confidence"}
            {"label": None, "confidence": 0, "status": "model_missing"}
        """
        if not self.model_loaded:
            return {"label": None, "confidence": 0.0, "status": "model_missing"}

        try:
            landmarks = self._get_hand_tracker().get_landmarks(frame)
        except Exception as exc:
            self.last_error = f"Hand landmark extraction failed: {exc}"
            return {"label": None, "confidence": 0.0, "status": "no_hand"}

        if landmarks is None:
            return {"label": None, "confidence": 0.0, "status": "no_hand"}

        return self.predict_from_landmarks(landmarks)

    def predict_from_landmarks(self, landmarks) -> PredictionDict:
        """Predict a sign from an already extracted 63-value landmark list."""
        if not self.model_loaded:
            return {"label": None, "confidence": 0.0, "status": "model_missing"}

        try:
            features = validate_feature_vector(landmarks).reshape(1, -1)
            label, confidence = self._predict_label_and_confidence(features)
        except Exception as exc:
            self.last_error = f"Prediction failed: {exc}"
            return {"label": None, "confidence": 0.0, "status": "model_missing"}

        if confidence < self.confidence_threshold:
            return {
                "label": None,
                "confidence": confidence,
                "status": "low_confidence",
            }

        return {
            "label": label,
            "confidence": confidence,
            "status": "success",
        }

    def _predict_label_and_confidence(self, features: np.ndarray) -> tuple[str, float]:
        """Run the scikit-learn model and decode the predicted label."""
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[0]
            best_index = int(np.argmax(probabilities))
            encoded_label = self.model.classes_[best_index]
            confidence = float(probabilities[best_index])
        else:
            encoded_label = self.model.predict(features)[0]
            confidence = 1.0

        label = str(self.label_encoder.inverse_transform([encoded_label])[0])
        return label, confidence

    def _get_hand_tracker(self) -> HandTracker:
        """Create HandTracker lazily so loading a model does not initialize MediaPipe."""
        if self.hand_tracker is None:
            self.hand_tracker = HandTracker(draw_landmarks=True)
        return self.hand_tracker

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self.hand_tracker is not None:
            self.hand_tracker.close()


def status_text(prediction: PredictionDict) -> str:
    """Convert a prediction dictionary into readable webcam overlay text."""
    status = prediction["status"]
    if status == "success":
        return f"{prediction['label']} ({prediction['confidence']:.0%})"
    if status == "low_confidence":
        return f"Gesture not clear ({prediction['confidence']:.0%})"
    if status == "model_missing":
        return "Model missing - run python src/train_model.py"
    return "No hand detected"


def run_webcam_prediction(camera_index: int = CAMERA_INDEX) -> None:
    """Open webcam and print/draw real-time predictions until q is pressed."""
    predictor = SignPredictor()
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        predictor.close()
        print("Could not open webcam. Check camera permissions or camera index.")
        return

    if not predictor.model_loaded:
        print(predictor.last_error or "Model files are missing.")

    print("Running webcam prediction. Press 'q' to quit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("Could not read a frame from the webcam.")
                break

            prediction = predictor.predict(frame)
            if predictor.hand_tracker is not None:
                preview = predictor.hand_tracker.draw_landmarks(frame)
            else:
                preview = frame.copy()
            text = status_text(prediction)

            cv2.rectangle(preview, (10, 10), (560, 58), (0, 0, 0), -1)
            cv2.putText(
                preview,
                text,
                (24, 43),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if prediction["status"] == "success" else (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Real-Time Sign Prediction", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        predictor.close()
        cv2.destroyAllWindows()


def test_webcam_prediction(camera_index: int = CAMERA_INDEX) -> None:
    """Simple OpenCV webcam test function for this module."""
    run_webcam_prediction(camera_index=camera_index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time webcam sign prediction.")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test_webcam_prediction(camera_index=args.camera)
