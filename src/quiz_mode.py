"""Quiz mode for beginner sign-language practice."""

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Allow direct execution with: python src/quiz_mode.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIDENCE_THRESHOLD, LABELS_PATH, REQUIRED_SIGNS  # noqa: E402


@dataclass
class QuizAnswer:
    """Result returned after checking one predicted answer."""

    correct: bool
    message: str
    target_sign: str
    predicted_sign: Optional[str]
    confidence: float
    next_question: Optional[str] = None


class QuizManager:
    """Manage quiz questions, answer checking, score, and accuracy."""

    def __init__(
        self,
        labels_path: Path = LABELS_PATH,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self.labels_path = labels_path
        self.confidence_threshold = confidence_threshold
        self.signs = self._load_labels()
        self.current_question: Optional[str] = None
        self.total_questions = 0
        self.correct_answers = 0
        self.wrong_attempts = 0
        self.next_question()

    def _load_labels(self) -> list[str]:
        """Load available quiz signs from labels.json.

        Supported formats:
            ["hello", "water"]
            {"labels": [...]}
            {"required_signs": [...]}
            {"signs": [...]}
        """
        if not self.labels_path.exists():
            self.labels_path.parent.mkdir(parents=True, exist_ok=True)
            with self.labels_path.open("w", encoding="utf-8") as file:
                json.dump({"required_signs": REQUIRED_SIGNS}, file, indent=2)
            return REQUIRED_SIGNS.copy()

        try:
            with self.labels_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (json.JSONDecodeError, OSError):
            return REQUIRED_SIGNS.copy()

        if isinstance(data, list):
            labels = data
        elif isinstance(data, dict):
            labels = data.get("labels") or data.get("required_signs") or data.get("signs")
        else:
            labels = None

        if not labels:
            return REQUIRED_SIGNS.copy()

        return [str(label).strip().lower().replace(" ", "_") for label in labels if str(label).strip()]

    def get_current_question(self) -> str:
        """Return the current target sign, creating one if needed."""
        if self.current_question is None:
            return self.next_question()
        return self.current_question

    def check_answer(self, predicted_label: Optional[str], confidence: float) -> QuizAnswer:
        """Check a predicted sign against the current target sign.

        A correct answer requires both:
            1. predicted label matches the target sign
            2. confidence is at or above the configured threshold
        """
        target = self.get_current_question()
        normalized_prediction = (
            predicted_label.strip().lower().replace(" ", "_")
            if predicted_label
            else None
        )

        if not normalized_prediction or confidence < self.confidence_threshold:
            self.wrong_attempts += 1
            return QuizAnswer(
                correct=False,
                message="Try again",
                target_sign=target,
                predicted_sign=normalized_prediction,
                confidence=confidence,
            )

        if normalized_prediction == target:
            self.correct_answers += 1
            next_target = self.next_question()
            return QuizAnswer(
                correct=True,
                message="Correct!",
                target_sign=target,
                predicted_sign=normalized_prediction,
                confidence=confidence,
                next_question=next_target,
            )

        self.wrong_attempts += 1
        return QuizAnswer(
            correct=False,
            message="Try again",
            target_sign=target,
            predicted_sign=normalized_prediction,
            confidence=confidence,
        )

    def check_live_frame(self, predictor, frame) -> QuizAnswer:
        """Use a Predictor instance to detect a live frame and check the answer.

        The predictor is expected to return dictionaries from
        `SignPredictor.predict(frame)`, including `label` and `confidence`.
        """
        prediction = predictor.predict(frame)
        return self.check_answer(
            predicted_label=prediction.get("label"),
            confidence=float(prediction.get("confidence", 0.0)),
        )

    def next_question(self) -> str:
        """Randomly select the next target sign from labels.json."""
        if not self.signs:
            self.signs = REQUIRED_SIGNS.copy()

        previous = self.current_question
        if len(self.signs) == 1:
            self.current_question = self.signs[0]
        else:
            choices = [sign for sign in self.signs if sign != previous]
            self.current_question = random.choice(choices)

        self.total_questions += 1
        return self.current_question

    def get_score(self) -> dict[str, int | float]:
        """Return total questions, correct answers, wrong attempts, and accuracy."""
        total_attempts = self.correct_answers + self.wrong_attempts
        accuracy = (
            (self.correct_answers / total_attempts) * 100
            if total_attempts > 0
            else 0.0
        )
        return {
            "total_questions": self.total_questions,
            "correct_answers": self.correct_answers,
            "wrong_attempts": self.wrong_attempts,
            "accuracy": accuracy,
        }

    def reset_quiz(self) -> None:
        """Clear score and start again with a new random question."""
        self.current_question = None
        self.total_questions = 0
        self.correct_answers = 0
        self.wrong_attempts = 0
        self.next_question()

    # Compatibility helpers for older app code.
    def new_question(self) -> str:
        return self.next_question()

    def check_prediction(self, predicted_sign: Optional[str], confidence: float) -> QuizAnswer:
        return self.check_answer(predicted_sign, confidence)

    def accuracy(self) -> float:
        return float(self.get_score()["accuracy"])

    def reset(self) -> None:
        self.reset_quiz()


# Backward-compatible name for earlier versions of this project.
QuizMode = QuizManager


def _run_simple_tests() -> None:
    """Simple test mode for direct execution."""
    manager = QuizManager()
    manager.signs = ["water", "food"]
    manager.current_question = "water"
    manager.total_questions = 1
    manager.correct_answers = 0
    manager.wrong_attempts = 0

    wrong = manager.check_answer("food", 0.95)
    assert wrong.correct is False
    assert wrong.message == "Try again"
    assert manager.get_score()["wrong_attempts"] == 1

    low_confidence = manager.check_answer("water", 0.10)
    assert low_confidence.correct is False
    assert low_confidence.message == "Try again"
    assert manager.get_score()["wrong_attempts"] == 2

    correct = manager.check_answer("water", 0.95)
    assert correct.correct is True
    assert correct.next_question in manager.signs
    score = manager.get_score()
    assert score["correct_answers"] == 1
    assert score["total_questions"] == 2
    assert round(float(score["accuracy"]), 2) == 33.33

    manager.reset_quiz()
    reset_score = manager.get_score()
    assert reset_score["correct_answers"] == 0
    assert reset_score["wrong_attempts"] == 0
    assert reset_score["total_questions"] == 1

    print("QuizManager simple tests passed.")


if __name__ == "__main__":
    _run_simple_tests()
