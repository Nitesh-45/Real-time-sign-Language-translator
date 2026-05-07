from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from config import FEATURE_VECTOR_SIZE, NUM_LANDMARKS

try:
    import mediapipe as mp
except ImportError as exc:
    mp = None
    MEDIAPIPE_IMPORT_ERROR = (
        "MediaPipe is not installed. Run: pip install -r requirements.txt"
    )
else:
    MEDIAPIPE_IMPORT_ERROR = None


@dataclass
class HandDetectionResult:
    annotated_frame: np.ndarray
    keypoints: Optional[list[float]]
    handedness: Optional[str]


class HandTracker:
    """Detect one hand and return stable, normalized MediaPipe landmarks.

    The public `get_landmarks` method returns one flattened feature list:

    x1, y1, z1, x2, y2, z2 ... x21, y21, z21

    The values are normalized relative to the wrist landmark, which makes the
    model less sensitive to where the hand appears inside the webcam frame.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        draw_landmarks: bool = True,
    ) -> None:
        """Create a MediaPipe Hands detector.

        Args:
            static_image_mode: Use True for independent images, False for webcam video.
            max_num_hands: Number of hands to detect. The MVP uses one hand.
            min_detection_confidence: Minimum confidence for the first detection.
            min_tracking_confidence: Minimum confidence for tracking across frames.
            draw_landmarks: Whether `process_frame` should return an annotated frame.
        """
        self.should_draw_landmarks = draw_landmarks
        self.initialization_error: Optional[str] = None
        self.mp_hands = None
        self.mp_drawing = None
        self.mp_styles = None
        self.hands = None

        if mp is None:
            self.initialization_error = MEDIAPIPE_IMPORT_ERROR
            return

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        try:
            self.hands = self.mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except Exception as exc:
            self.initialization_error = f"Could not initialize MediaPipe Hands: {exc}"
            self.hands = None

    @property
    def available(self) -> bool:
        """Return True when MediaPipe Hands is ready to process frames."""
        return self.hands is not None

    def process_frame(self, frame: np.ndarray) -> HandDetectionResult:
        """Return an annotated frame plus normalized landmarks.

        This helper keeps the older app code simple. For new code, prefer
        `get_landmarks(frame)` when you only need features, and
        `draw_landmarks(frame)` when you only need visualization.
        """
        if self._is_empty_frame(frame):
            return HandDetectionResult(frame, None, None)

        if not self.available:
            return HandDetectionResult(frame.copy(), None, None)

        result = self._detect_hands(frame)
        annotated = frame.copy()
        if result is None or not result.multi_hand_landmarks:
            return HandDetectionResult(annotated, None, None)

        hand_landmarks = result.multi_hand_landmarks[0]
        keypoints = self.extract_keypoints(hand_landmarks)
        handedness = None

        if result.multi_handedness:
            handedness = result.multi_handedness[0].classification[0].label

        if self.should_draw_landmarks:
            self._draw_detected_landmarks(annotated, hand_landmarks)

        return HandDetectionResult(annotated, keypoints, handedness)

    def get_landmarks(self, frame: np.ndarray) -> Optional[list[float]]:
        """Extract normalized 63-value hand landmarks from a BGR webcam frame.

        Args:
            frame: One OpenCV frame in BGR format.

        Returns:
            A flattened list of 63 normalized values if a hand is detected.
            Returns None when no hand is detected or when the frame is empty.
        """
        if self._is_empty_frame(frame):
            return None

        if not self.available:
            return None

        result = self._detect_hands(frame)
        if result is None or not result.multi_hand_landmarks:
            return None

        return self.extract_keypoints(result.multi_hand_landmarks[0])

    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """Draw detected hand landmarks and connections on a frame.

        Args:
            frame: One OpenCV frame in BGR format.

        Returns:
            A copy of the frame with hand landmarks drawn. If no hand is
            detected, the original frame content is returned unchanged.
        """
        if self._is_empty_frame(frame):
            return frame

        annotated = frame.copy()
        if not self.available:
            return annotated

        result = self._detect_hands(frame)
        if result is None or not result.multi_hand_landmarks:
            return annotated

        self._draw_detected_landmarks(annotated, result.multi_hand_landmarks[0])
        return annotated

    def _detect_hands(self, frame: np.ndarray) -> Optional[Any]:
        """Run MediaPipe Hands safely on a BGR frame."""
        if not self.available:
            return None

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except cv2.error:
            return None

        try:
            rgb_frame.flags.writeable = False
            result = self.hands.process(rgb_frame)
            rgb_frame.flags.writeable = True
            return result
        except Exception as exc:
            self.initialization_error = f"MediaPipe processing failed: {exc}"
            return None

    def _draw_detected_landmarks(self, frame: np.ndarray, hand_landmarks: Any) -> None:
        """Draw one already-detected MediaPipe hand on the provided frame."""
        if not self.available:
            return

        self.mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_styles.get_default_hand_landmarks_style(),
            self.mp_styles.get_default_hand_connections_style(),
        )

    @staticmethod
    def extract_keypoints(hand_landmarks: Any) -> list[float]:
        """Convert 21 MediaPipe landmarks into a normalized 63-value list."""
        raw_values: list[float] = []
        for landmark in hand_landmarks.landmark:
            raw_values.extend([landmark.x, landmark.y, landmark.z])
        return HandTracker.normalize_landmarks(raw_values)

    @staticmethod
    def normalize_landmarks(landmarks: list[float]) -> list[float]:
        """Normalize landmarks relative to the wrist landmark.

        The wrist is landmark 0. Subtracting it makes the hand location
        relative, and scaling by the largest wrist-relative distance makes the
        features less sensitive to hand size and distance from the camera.
        """
        vector = np.asarray(landmarks, dtype=np.float32)
        if vector.shape[0] != FEATURE_VECTOR_SIZE:
            raise ValueError("MediaPipe returned an unexpected number of landmarks.")

        points = vector.reshape(NUM_LANDMARKS, 3)
        wrist = points[0].copy()
        normalized = points - wrist

        scale = np.linalg.norm(normalized, axis=1).max()
        if scale < 1e-6:
            scale = 1.0

        return (normalized / scale).flatten().astype(float).tolist()

    @staticmethod
    def _is_empty_frame(frame: Optional[np.ndarray]) -> bool:
        """Return True when OpenCV cannot safely process the frame."""
        return frame is None or not hasattr(frame, "size") or frame.size == 0

    def close(self) -> None:
        """Release the MediaPipe resources owned by this tracker."""
        if self.hands is not None:
            self.hands.close()
