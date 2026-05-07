"""Build a readable sentence from predicted sign labels."""

import time
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Optional

# Allow direct execution with: python src/sentence_builder.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import MAX_SENTENCE_SIGNS


GRAMMAR_MAPPING = {
    "water": "I need water",
    "food": "I need food",
    "medicine": "I need medicine",
    "help": "Please help me",
    "thank_you": "Thank you so much",
    "hello": "Hello",
    "stop": "Please stop",
    "yes": "Yes",
    "no": "No",
    "good": "I am good",
    "bad": "I am not feeling good",
    "please": "Please",
    "doctor": "I need a doctor",
    "pain": "I am in pain",
    "toilet": "I need to use the toilet",
    "home": "I want to go home",
    "phone": "Please call someone",
    "sorry": "I am sorry",
    "more": "I need more",
    "emergency": "This is an emergency",
}


class SentenceBuilder:
    """Maintain detected words and convert them into useful sentence output.

    The predictor can produce the same label for many consecutive frames. This
    class avoids flooding the sentence with duplicates by:

    1. ignoring continuous repeated words
    2. applying a cooldown before the same word can be added again
    """

    def __init__(
        self,
        cooldown_seconds: float = 2.0,
        max_words: int = MAX_SENTENCE_SIGNS,
    ) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.words: Deque[str] = deque(maxlen=max_words)
        self.last_word: Optional[str] = None
        self.last_added_at: dict[str, float] = {}

    def add_word(self, word: Optional[str]) -> bool:
        """Add a detected sign word if it is not a duplicate.

        Args:
            word: Predicted sign label, for example `water` or `thank_you`.

        Returns:
            True if the word was added, False if it was ignored.
        """
        if not word:
            return False

        normalized_word = word.strip().lower().replace(" ", "_")
        if not normalized_word:
            return False

        # Avoid adding the same sign repeatedly while it remains in view.
        if normalized_word == self.last_word:
            return False

        now = time.monotonic()
        last_time = self.last_added_at.get(normalized_word)
        if last_time is not None and now - last_time < self.cooldown_seconds:
            self.last_word = normalized_word
            return False

        self.words.append(normalized_word)
        self.last_word = normalized_word
        self.last_added_at[normalized_word] = now
        return True

    def get_words(self) -> list[str]:
        """Return the detected sign words in insertion order."""
        return list(self.words)

    def get_sentence(self) -> str:
        """Return a live sentence built from detected sign words."""
        phrases = [
            GRAMMAR_MAPPING.get(word, word.replace("_", " ").capitalize())
            for word in self.words
        ]
        return ". ".join(phrases)

    def clear_sentence(self) -> None:
        """Clear all detected words and duplicate/cooldown state."""
        self.words.clear()
        self.last_word = None
        self.last_added_at.clear()

    # Backwards-compatible aliases used by older app code or notebooks.
    def update(self, predicted_sign: Optional[str]) -> bool:
        return self.add_word(predicted_sign)

    def get_signs(self) -> list[str]:
        return self.get_words()

    def clear(self) -> None:
        self.clear_sentence()


def _run_simple_tests() -> None:
    """Small unit-like test for direct execution."""
    builder = SentenceBuilder(cooldown_seconds=2.0)

    assert builder.add_word("water") is True
    assert builder.get_words() == ["water"]
    assert builder.get_sentence() == "I need water"

    # Continuous duplicate should not be added.
    assert builder.add_word("water") is False
    assert builder.get_words() == ["water"]

    assert builder.add_word("food") is True
    assert builder.get_sentence() == "I need water. I need food"

    builder.clear_sentence()
    assert builder.add_word("Thank_you") is True
    assert builder.get_sentence() == "Thank you so much"

    # Same word inside cooldown should be ignored even after another word.
    builder.clear_sentence()
    assert builder.add_word("water") is True
    assert builder.add_word("food") is True
    assert builder.add_word("water") is False
    assert builder.get_words() == ["water", "food"]

    # Simulate cooldown expiry.
    builder.last_word = "food"
    builder.last_added_at["water"] -= 2.1
    assert builder.add_word("water") is True
    assert builder.get_words() == ["water", "food", "water"]

    builder.clear_sentence()
    assert builder.get_words() == []
    assert builder.get_sentence() == ""

    print("SentenceBuilder simple tests passed.")


if __name__ == "__main__":
    _run_simple_tests()
