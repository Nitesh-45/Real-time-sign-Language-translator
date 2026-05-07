"""Text-to-speech helper for translated sign-language sentences."""

import threading
from typing import Optional


class TextToSpeech:
    """Speak text locally using pyttsx3 with safe error handling.

    pyttsx3 depends on the operating system's speech/audio drivers. On some
    machines it may fail to initialize, so this wrapper keeps the app from
    crashing and returns readable status messages instead.
    """

    def __init__(self, rate: int = 155, volume: float = 1.0) -> None:
        """Initialize the pyttsx3 speech engine.

        Args:
            rate: Speech speed in words per minute. 155 is clear for demos.
            volume: Speech volume from 0.0 to 1.0.
        """
        self.engine = None
        self.error: Optional[str] = None
        self._speech_thread: Optional[threading.Thread] = None
        self._pyttsx3 = None
        self.rate = rate
        self.volume = volume

        try:
            import pyttsx3

            self._pyttsx3 = pyttsx3
        except Exception as exc:
            self.error = str(exc)
            self._pyttsx3 = None

    @property
    def available(self) -> bool:
        """Return True when pyttsx3 is available."""
        return self._pyttsx3 is not None

    def speak(self, text: str) -> str:
        """Speak the provided text.

        Args:
            text: Sentence to speak.

        Returns:
            A short status message suitable for Streamlit display.
        """
        clean_text = (text or "").strip()
        if not clean_text:
            return "Nothing to speak."

        if not self.available:
            return f"Text-to-speech unavailable: {self.error}"

        # Stop any current speech before starting the new sentence.
        if self._speech_thread and self._speech_thread.is_alive():
            self.stop()

        self._speech_thread = threading.Thread(
            target=self._speak_blocking,
            args=(clean_text,),
            daemon=True,
        )
        self._speech_thread.start()
        return "Speaking sentence."

    def stop(self) -> str:
        """Stop current speech if the engine is active."""
        if not self.available:
            return "Text-to-speech unavailable."

        try:
            if self.engine is not None:
                self.engine.stop()
            return "Speech stopped."
        except Exception as exc:
            self.error = str(exc)
            return f"Could not stop speech: {exc}"

    def _speak_blocking(self, text: str) -> None:
        """Run pyttsx3 speech on a background thread with a fresh engine."""
        try:
            self.engine = self._pyttsx3.init()
            self.engine.setProperty("rate", self.rate)
            self.engine.setProperty("volume", self.volume)
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as exc:
            self.error = str(exc)
        finally:
            try:
                if self.engine is not None:
                    self.engine.stop()
            except Exception:
                pass
            self.engine = None
