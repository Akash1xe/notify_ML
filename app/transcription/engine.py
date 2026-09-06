from __future__ import annotations

import math
import os
import shutil
import time
import wave
from pathlib import Path
from typing import Callable

from app.core.config import AppSettings
from app.core.exceptions import AudioChunkTranscriptionError, JobCancelledError, RawTranscriptionError
from app.storage.workspace import WorkspaceManager
from app.transcription.faster_whisper_adapter import SpeechToTextAdapter
from app.transcription.models import (
    RawChunkStatus,
    RawChunkTranscript,
    RawTranscriptManifest,
    RawTranscriptSegment,
    RawTranscriptStats,
    RawWord,
    TranscriptionChunk,
    TranscriptionPreparationManifest,
)
from app.transcription.repository import TranscriptionRepository
from app.video_analysis.fingerprints import stable_hash

TRANSCRIPTION_ENGINE_ALGORITHM_VERSION = "1"
CP_RAW_TRANSCRIPTION = "RAW_TRANSCRIPTION_COMPLETE"

ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]


class TranscriptionEngine:
    def __init__(
        self,
        settings: AppSettings,
        workspace: WorkspaceManager,
        repository: TranscriptionRepository,
        adapter: SpeechToTextAdapter,
    ) -> None:
        self._settings = settings
        self._workspace = workspace
        self._repository = repository
        self._adapter = adapter

    def config_fingerprint(self) -> str:
        s = self._settings
        return stable_hash(
            {
                "algorithm_version": TRANSCRIPTION_ENGINE_ALGORITHM_VERSION,
                "model": s.whisper_model_size,
                "device": s.whisper_device,
                "compute_type": s.whisper_compute_type,
                "language": s.whisper_language,
                "beam_size": s.whisper_beam_size,
                "vad_filter": s.whisper_vad_filter,
                "word_timestamps": s.whisper_word_timestamps,
                "temperature": s.whisper_temperature,
                "condition_on_previous_text": s.whisper_condition_on_previous_text,
            }
        )

    def expected_chunk_fingerprint(self, preparation: TranscriptionPreparationManifest, chunk: TranscriptionChunk) -> str:
        return stable_hash(
            {
                "audio_fingerprint": preparation.audio_fingerprint,
                "preparation_fingerprint": preparation.artifact_fingerprint,
                "chunk": chunk.model_dump(mode="json"),
                "config_fingerprint": self.config_fingerprint(),
                "algorithm_version": TRANSCRIPTION_ENGINE_ALGORITHM_VERSION,
            }
        )

    def validate_cached_chunk(
        self, job_id: str, preparation: TranscriptionPreparationManifest, chunk: TranscriptionChunk
    ) -> RawChunkTranscript | None:
        path = self._workspace.raw_transcript_chunk_path(job_id, chunk.chunk_id)
        if not path.exists():
            return None
        try:
            value = self._repository.load_raw_chunk(job_id, chunk.chunk_id)
        except Exception:
            return None
        expected = self.expected_chunk_fingerprint(preparation, chunk)
        if (
            value.status is not RawChunkStatus.COMPLETED
            or value.chunk_fingerprint != expected
            or value.config_fingerprint != self.config_fingerprint()
            or value.audio_fingerprint != preparation.audio_fingerprint
            or value.chunk != chunk
        ):
            return None
        if any(s.absolute_end_seconds > preparation.audio.duration_seconds + 0.1 for s in value.segments):
            return None
        return value

    def _safe_source(self, job_id: str, preparation: TranscriptionPreparationManifest) -> Path:
        root = self._workspace.workspace(job_id)
        source = (root / preparation.audio_path).resolve()
        try:
            source.relative_to(root)
        except ValueError as exc:
            raise RawTranscriptionError("Transcription audio path escaped the job workspace.") from exc
        if not source.exists():
            raise RawTranscriptionError("Transcription audio is missing.")
        return source

    def _extract_chunk_wav(
        self, source: Path, target: Path, chunk: TranscriptionChunk, cancel_check: CancelCheck
    ) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)
        try:
            with wave.open(str(source), "rb") as src:
                if src.getsampwidth() != 2 or src.getnchannels() != 1 or src.getframerate() != 16000:
                    raise RawTranscriptionError("Prepared transcription audio no longer matches the 16 kHz mono PCM contract.")
                rate = src.getframerate()
                start_frame = int(round(chunk.decode_start_seconds * rate))
                end_frame = min(src.getnframes(), int(round(chunk.decode_end_seconds * rate)))
                src.setpos(min(start_frame, src.getnframes()))
                with wave.open(str(target), "wb") as dst:
                    dst.setnchannels(src.getnchannels())
                    dst.setsampwidth(src.getsampwidth())
                    dst.setframerate(rate)
                    remaining = max(0, end_frame - start_frame)
                    block = rate * 10
                    while remaining > 0:
                        if cancel_check():
                            raise JobCancelledError("Transcription was cancelled.")
                        data = src.readframes(min(block, remaining))
                        if not data:
                            break
                        dst.writeframes(data)
                        remaining -= len(data) // (src.getsampwidth() * src.getnchannels())
        except JobCancelledError:
            target.unlink(missing_ok=True)
            raise
        except wave.Error as exc:
            target.unlink(missing_ok=True)
            raise AudioChunkTranscriptionError(f"Unable to prepare chunk {chunk.chunk_id} for transcription.") from exc

    @staticmethod
    def _info_value(info, name: str, default=None):
        return getattr(info, name, default) if info is not None else default

    def _convert_segment(
        self,
        *,
        raw,
        chunk: TranscriptionChunk,
        ordinal: int,
        audio_duration: float,
    ) -> RawTranscriptSegment | None:
        text = str(getattr(raw, "text", "")).strip()
        if not text:
            return None
        local_start = max(0.0, float(getattr(raw, "start", 0.0)))
        local_end = max(local_start, float(getattr(raw, "end", local_start)))
        decode_duration = chunk.decode_end_seconds - chunk.decode_start_seconds
        if local_end > decode_duration + 0.25:
            raise AudioChunkTranscriptionError(f"Chunk {chunk.chunk_id} produced an invalid timestamp.")
        local_end = min(local_end, decode_duration)
        absolute_start = chunk.decode_start_seconds + local_start
        absolute_end = chunk.decode_start_seconds + local_end
        if absolute_end > audio_duration + 0.1:
            if absolute_end <= audio_duration + 0.25:
                absolute_end = audio_duration
            else:
                raise AudioChunkTranscriptionError(f"Chunk {chunk.chunk_id} produced a timestamp outside the lecture.")
        absolute_start = min(absolute_start, audio_duration)
        words: list[RawWord] = []
        for word in list(getattr(raw, "words", None) or []):
            w_start = chunk.decode_start_seconds + max(0.0, float(getattr(word, "start", 0.0)))
            w_end = chunk.decode_start_seconds + max(0.0, float(getattr(word, "end", 0.0)))
            w_start = max(absolute_start, min(absolute_end, w_start))
            w_end = max(w_start, min(absolute_end, w_end))
            probability = getattr(word, "probability", None)
            words.append(
                RawWord(
                    word=str(getattr(word, "word", "")),
                    start_seconds=w_start,
                    end_seconds=w_end,
                    probability=float(probability) if probability is not None else None,
                )
            )
        midpoint = (absolute_start + absolute_end) / 2
        return RawTranscriptSegment(
            segment_id=f"{chunk.chunk_id}:{ordinal}",
            chunk_id=chunk.chunk_id,
            source_ordinal=ordinal,
            local_start_seconds=local_start,
            local_end_seconds=local_end,
            absolute_start_seconds=absolute_start,
            absolute_end_seconds=absolute_end,
            text=text,
            avg_logprob=(float(getattr(raw, "avg_logprob")) if getattr(raw, "avg_logprob", None) is not None else None),
            no_speech_prob=(float(getattr(raw, "no_speech_prob")) if getattr(raw, "no_speech_prob", None) is not None else None),
            compression_ratio=(float(getattr(raw, "compression_ratio")) if getattr(raw, "compression_ratio", None) is not None else None),
            is_within_logical_range=(chunk.logical_start_seconds <= midpoint <= chunk.logical_end_seconds),
            words=words,
        )

    def transcribe_chunk(
        self,
        *,
        job_id: str,
        preparation: TranscriptionPreparationManifest,
        chunk: TranscriptionChunk,
        cancel_check: CancelCheck,
    ) -> RawChunkTranscript:
        cached = self.validate_cached_chunk(job_id, preparation, chunk)
        if cached is not None:
            return cached
        source = self._safe_source(job_id, preparation)
        temp_dir = self._workspace.transcript_temp_audio_dir(job_id)
        temp = temp_dir / f"chunk_{chunk.chunk_id:04d}.wav"
        expected_chunk_fp = self.expected_chunk_fingerprint(preparation, chunk)
        last_error: Exception | None = None
        for attempt in range(self._settings.whisper_chunk_max_retries + 1):
            if cancel_check():
                raise JobCancelledError("Transcription was cancelled.")
            started = time.monotonic()
            try:
                self._extract_chunk_wav(source, temp, chunk, cancel_check)
                raw_segments, info = self._adapter.transcribe(str(temp))
                segments: list[RawTranscriptSegment] = []
                empty = 0
                for ordinal, raw in enumerate(raw_segments):
                    if cancel_check():
                        raise JobCancelledError("Transcription was cancelled.")
                    item = self._convert_segment(
                        raw=raw,
                        chunk=chunk,
                        ordinal=ordinal,
                        audio_duration=preparation.audio.duration_seconds,
                    )
                    if item is None:
                        empty += 1
                    else:
                        segments.append(item)
                language = self._info_value(info, "language")
                language_probability = self._info_value(info, "language_probability")
                core = {
                    "chunk_fingerprint": expected_chunk_fp,
                    "config_fingerprint": self.config_fingerprint(),
                    "segments": [s.model_dump(mode="json") for s in segments],
                    "language": language,
                }
                result = RawChunkTranscript(
                    algorithm_version=TRANSCRIPTION_ENGINE_ALGORITHM_VERSION,
                    chunk=chunk,
                    audio_fingerprint=preparation.audio_fingerprint,
                    config_fingerprint=self.config_fingerprint(),
                    chunk_fingerprint=expected_chunk_fp,
                    artifact_fingerprint=stable_hash(core),
                    status=RawChunkStatus.COMPLETED,
                    language=str(language) if language else None,
                    language_probability=(float(language_probability) if language_probability is not None else None),
                    segments=segments,
                    empty_segment_count=empty,
                    transcription_seconds=time.monotonic() - started,
                )
                self._repository.save_raw_chunk(job_id, chunk.chunk_id, result)
                return result
            except JobCancelledError:
                raise
            except Exception as exc:
                last_error = exc
                if attempt >= self._settings.whisper_chunk_max_retries:
                    if isinstance(exc, AudioChunkTranscriptionError):
                        raise
                    raise AudioChunkTranscriptionError(f"Unable to transcribe chunk {chunk.chunk_id}.") from exc
            finally:
                temp.unlink(missing_ok=True)
        raise AudioChunkTranscriptionError(f"Unable to transcribe chunk {chunk.chunk_id}.") from last_error

    def assemble(
        self,
        *,
        preparation: TranscriptionPreparationManifest,
        chunks: list[RawChunkTranscript],
        wall_clock_seconds: float,
    ) -> RawTranscriptManifest:
        ordered = sorted(chunks, key=lambda c: c.chunk.chunk_id)
        segments = sorted(
            [s for c in ordered for s in c.segments],
            key=lambda s: (s.absolute_start_seconds, s.absolute_end_seconds, s.chunk_id, s.source_ordinal),
        )
        speech_duration = sum(max(0.0, s.absolute_end_seconds - s.absolute_start_seconds) for s in segments)
        word_count = sum(len(s.words) if s.words else len(s.text.split()) for s in segments)
        empty_count = sum(c.empty_segment_count for c in ordered)
        languages = [c.language for c in ordered if c.language]
        primary_language = max(set(languages), key=languages.count) if languages else None
        probs = [c.language_probability for c in ordered if c.language == primary_language and c.language_probability is not None]
        warnings: list[str] = []
        if not segments:
            warnings.append("NO_SPEECH_DETECTED")
        elif preparation.audio.duration_seconds > 300 and speech_duration / preparation.audio.duration_seconds < 0.01:
            warnings.append("LOW_TRANSCRIPT_COVERAGE")
        chunk_fps = [c.artifact_fingerprint for c in ordered]
        stats = RawTranscriptStats(
            audio_duration_seconds=preparation.audio.duration_seconds,
            chunk_count=len(preparation.chunks),
            completed_chunk_count=len(ordered),
            raw_segment_count=len(segments),
            word_count=word_count,
            empty_segment_count=empty_count,
            transcribed_speech_duration_seconds=speech_duration,
            wall_clock_transcription_seconds=max(0.0, wall_clock_seconds),
            real_time_factor=(wall_clock_seconds / preparation.audio.duration_seconds if preparation.audio.duration_seconds > 0 else 0.0),
        )
        core = {
            "config": self.config_fingerprint(),
            "preparation": preparation.artifact_fingerprint,
            "chunks": chunk_fps,
            "segments": [s.model_dump(mode="json") for s in segments],
        }
        return RawTranscriptManifest(
            algorithm_version=TRANSCRIPTION_ENGINE_ALGORITHM_VERSION,
            model_name=self._settings.whisper_model_size,
            device=self._settings.whisper_device,
            compute_type=self._settings.whisper_compute_type,
            language=primary_language,
            language_probability=(sum(probs) / len(probs) if probs else None),
            config_fingerprint=self.config_fingerprint(),
            preparation_fingerprint=preparation.artifact_fingerprint,
            audio_fingerprint=preparation.audio_fingerprint,
            chunk_fingerprints=chunk_fps,
            artifact_fingerprint=stable_hash(core),
            stats=stats,
            warnings=warnings,
            segments=segments,
        )

    def process(
        self,
        job_id: str,
        *,
        preparation: TranscriptionPreparationManifest,
        progress_callback: ProgressCallback = lambda _: None,
        cancel_check: CancelCheck = lambda: False,
    ) -> RawTranscriptManifest:
        overall_started = time.monotonic()
        results: list[RawChunkTranscript] = []
        total = sum(c.duration_seconds for c in preparation.chunks) or 1.0
        completed_duration = 0.0
        for chunk in preparation.chunks:
            if cancel_check():
                raise JobCancelledError("Transcription was cancelled.")
            result = self.transcribe_chunk(job_id=job_id, preparation=preparation, chunk=chunk, cancel_check=cancel_check)
            results.append(result)
            completed_duration += chunk.duration_seconds
            progress_callback(min(100.0, completed_duration * 100.0 / total))
        manifest = self.assemble(
            preparation=preparation,
            chunks=results,
            wall_clock_seconds=sum(item.transcription_seconds for item in results),
        )
        self._repository.save_raw_transcript(job_id, manifest)
        shutil.rmtree(self._workspace.transcript_temp_audio_dir(job_id), ignore_errors=True)
        progress_callback(100.0)
        return manifest
