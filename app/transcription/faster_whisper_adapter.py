from __future__ import annotations

from threading import Lock
from typing import Any, Iterable, Protocol

from app.core.config import AppSettings
from app.core.exceptions import TranscriptionModelLoadError


class SpeechToTextAdapter(Protocol):
    def transcribe(self, audio_path: str) -> tuple[Iterable[Any], Any]: ...


class FasterWhisperAdapter:
    """Lazy, reusable wrapper around faster-whisper.

    Import and model construction happen only when transcription is actually needed,
    so normal API startup and unit tests never download model weights.
    """

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._model: Any | None = None
        self._load_lock = Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel

                kwargs: dict[str, Any] = {
                    "device": self._settings.whisper_device,
                    "compute_type": self._settings.whisper_compute_type,
                }
                if self._settings.whisper_model_cache_dir is not None:
                    kwargs["download_root"] = str(self._settings.whisper_model_cache_dir)
                self._model = WhisperModel(self._settings.whisper_model_size, **kwargs)
            except Exception as exc:
                raise TranscriptionModelLoadError("Local transcription model could not be loaded.") from exc
        return self._model

    def transcribe(self, audio_path: str):
        model = self._get_model()
        language = None if self._settings.whisper_language.strip().lower() == "auto" else self._settings.whisper_language
        try:
            return model.transcribe(
                audio_path,
                language=language,
                beam_size=self._settings.whisper_beam_size,
                vad_filter=self._settings.whisper_vad_filter,
                word_timestamps=self._settings.whisper_word_timestamps,
                temperature=self._settings.whisper_temperature,
                condition_on_previous_text=self._settings.whisper_condition_on_previous_text,
            )
        except Exception as exc:
            raise TranscriptionModelLoadError("Local transcription engine could not start transcription.") from exc
