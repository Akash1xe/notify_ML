from __future__ import annotations

import math
import statistics
import time
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from app.candidate_analysis.cache import CP_CANDIDATES_READY
from app.candidate_analysis.repository import CandidateAnalysisRepository
from app.core.config import AppSettings
from app.core.exceptions import AudioPreparationError, JobCancelledError
from app.jobs.checkpoints import CheckpointStore
from app.media.models import AudioResult, MediaInspection
from app.storage.workspace import WorkspaceManager
from app.transcription.models import (
    AudioDiagnostics,
    AudioFormatInfo,
    AudioPreparationRejection,
    AudioPreparationStats,
    AudioPreparationWarning,
    TranscriptionChunk,
    TranscriptionPreparationManifest,
)
from app.transcription.repository import TranscriptionRepository
from app.video_analysis.fingerprints import stable_hash

AUDIO_PREPARATION_ALGORITHM_VERSION = "1"
CP_AUDIO_READY = "AUDIO_READY_FOR_TRANSCRIPTION"

ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]


class AudioPreparationService:
    def __init__(
        self,
        settings: AppSettings,
        workspace: WorkspaceManager,
        checkpoints: CheckpointStore,
        repository: TranscriptionRepository,
        candidates: CandidateAnalysisRepository,
    ) -> None:
        self._settings = settings
        self._workspace = workspace
        self._checkpoints = checkpoints
        self._repository = repository
        self._candidates = candidates

    def config_fingerprint(self) -> str:
        s = self._settings
        return stable_hash(
            {
                "algorithm_version": AUDIO_PREPARATION_ALGORITHM_VERSION,
                "duration_tolerance_seconds": s.audio_video_duration_tolerance_seconds,
                "duration_tolerance_ratio": s.audio_video_duration_tolerance_ratio,
                "min_duration": s.audio_min_duration_seconds,
                "diagnostic_window": s.audio_diagnostic_window_seconds,
                "silence_rms": s.audio_silence_rms_threshold,
                "silent_ratio": s.audio_effectively_silent_ratio,
                "high_clipping_ratio": s.audio_high_clipping_ratio,
                "chunk_seconds": s.transcription_chunk_seconds,
                "chunk_overlap": s.transcription_chunk_overlap_seconds,
                "expected": {"sample_rate": 16000, "channels": 1, "bits": 16, "codec": "pcm_s16le"},
            }
        )

    def _load_upstream(self, job_id: str) -> tuple[AudioResult, MediaInspection, str | None]:
        import json
        from pydantic import ValidationError

        if not self._checkpoints.is_completed(job_id, CP_CANDIDATES_READY):
            raise AudioPreparationError("CANDIDATES_READY is required before transcription preparation.")
        audio_manifest_path = self._workspace.audio_manifest_path(job_id)
        if not audio_manifest_path.exists():
            raise AudioPreparationError(AudioPreparationRejection.AUDIO_MANIFEST_MISSING.value)
        try:
            audio = AudioResult.model_validate(json.loads(audio_manifest_path.read_text(encoding="utf-8")))
            media = MediaInspection.model_validate(
                json.loads(self._workspace.media_inspection_path(job_id).read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, ValidationError) as exc:
            raise AudioPreparationError("Phase-2 audio/media metadata is missing or invalid.") from exc
        try:
            candidate_summary_fingerprint = stable_hash(self._candidates.load_summary(job_id).model_dump(mode="json"))
        except Exception:
            candidate_summary_fingerprint = None
        return audio, media, candidate_summary_fingerprint

    def _safe_audio_path(self, job_id: str, relative: str) -> Path:
        root = self._workspace.workspace(job_id)
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise AudioPreparationError(AudioPreparationRejection.UNSAFE_AUDIO_PATH.value) from exc
        return path

    def _diagnose_wav(
        self,
        path: Path,
        cancel_check: CancelCheck,
        progress_callback: ProgressCallback,
    ) -> tuple[AudioFormatInfo, AudioDiagnostics]:
        if not path.exists():
            raise AudioPreparationError(AudioPreparationRejection.AUDIO_FILE_MISSING.value)
        if path.stat().st_size <= 44:
            raise AudioPreparationError(AudioPreparationRejection.EMPTY_AUDIO.value)
        rms_values: list[float] = []
        total_samples = 0
        sum_sq = 0.0
        sum_samples = 0.0
        peak = 0.0
        clipping = 0
        silent_windows = 0
        window_count = 0
        try:
            with wave.open(str(path), "rb") as wav:
                channels = wav.getnchannels()
                sample_rate = wav.getframerate()
                sample_width = wav.getsampwidth()
                frame_count = wav.getnframes()
                duration = frame_count / sample_rate if sample_rate else 0.0
                frames_per_window = max(1, int(sample_rate * self._settings.audio_diagnostic_window_seconds))
                while True:
                    if cancel_check():
                        raise JobCancelledError("Audio preparation was cancelled.")
                    raw = wav.readframes(frames_per_window)
                    if not raw:
                        break
                    if sample_width != 2:
                        # The format will be rejected below, but do not decode unsupported width incorrectly.
                        break
                    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
                    if samples.size == 0:
                        continue
                    sq = float(np.dot(samples, samples))
                    rms = math.sqrt(sq / samples.size)
                    rms_values.append(rms)
                    sum_sq += sq
                    sum_samples += float(samples.sum())
                    total_samples += int(samples.size)
                    peak = max(peak, float(np.max(np.abs(samples))))
                    clipping += int(np.count_nonzero(np.abs(samples) >= (32760 / 32768.0)))
                    silent_windows += int(rms <= self._settings.audio_silence_rms_threshold)
                    window_count += 1
                    if frame_count > 0:
                        progress_callback(min(100.0, wav.tell() * 100.0 / frame_count))
        except JobCancelledError:
            raise
        except (wave.Error, OSError, EOFError) as exc:
            raise AudioPreparationError(AudioPreparationRejection.AUDIO_DECODE_FAILED.value) from exc

        bits = sample_width * 8
        info = AudioFormatInfo(
            sample_rate_hz=sample_rate,
            channels=channels,
            bits_per_sample=bits,
            duration_seconds=duration,
            file_size_bytes=path.stat().st_size,
            frame_count=frame_count,
        )
        mean_rms = math.sqrt(sum_sq / total_samples) if total_samples else 0.0
        diagnostics = AudioDiagnostics(
            mean_rms=min(1.0, mean_rms),
            median_rms=min(1.0, statistics.median(rms_values) if rms_values else 0.0),
            peak_amplitude=min(1.0, peak),
            silent_window_ratio=(silent_windows / window_count if window_count else 1.0),
            clipping_ratio=(clipping / total_samples if total_samples else 0.0),
            dc_offset=max(-1.0, min(1.0, sum_samples / total_samples if total_samples else 0.0)),
            effectively_silent=(window_count == 0 or silent_windows / max(window_count, 1) >= self._settings.audio_effectively_silent_ratio),
        )
        return info, diagnostics

    def _chunks(self, duration: float, candidate_times: list[float]) -> list[TranscriptionChunk]:
        size = self._settings.transcription_chunk_seconds
        overlap = self._settings.transcription_chunk_overlap_seconds
        chunks: list[TranscriptionChunk] = []
        start = 0.0
        chunk_id = 1
        while start < duration - 1e-9:
            end = min(duration, start + size)
            count = sum(1 for t in candidate_times if start <= t < end or (end == duration and t <= end))
            chunks.append(
                TranscriptionChunk(
                    chunk_id=chunk_id,
                    logical_start_seconds=round(start, 6),
                    logical_end_seconds=round(end, 6),
                    decode_start_seconds=round(max(0.0, start - overlap), 6),
                    decode_end_seconds=round(min(duration, end + overlap), 6),
                    duration_seconds=round(end - start, 6),
                    contains_candidate_timestamps=count > 0,
                    candidate_count=count,
                )
            )
            start = end
            chunk_id += 1
        return chunks

    def process(
        self,
        job_id: str,
        *,
        progress_callback: ProgressCallback = lambda _: None,
        cancel_check: CancelCheck = lambda: False,
    ) -> TranscriptionPreparationManifest:
        started = time.monotonic()
        audio_manifest, media, candidate_summary_fingerprint = self._load_upstream(job_id)
        audio_path = self._safe_audio_path(job_id, audio_manifest.path)
        info, diagnostics = self._diagnose_wav(audio_path, cancel_check, progress_callback)
        warnings: list[str] = []
        rejections: list[str] = []
        if info.sample_rate_hz != 16000:
            rejections.append(AudioPreparationRejection.INVALID_SAMPLE_RATE.value)
        if info.channels != 1:
            rejections.append(AudioPreparationRejection.INVALID_CHANNEL_COUNT.value)
        if info.bits_per_sample != 16:
            rejections.append(AudioPreparationRejection.INVALID_SAMPLE_WIDTH.value)
        if info.duration_seconds < self._settings.audio_min_duration_seconds:
            rejections.append(AudioPreparationRejection.EMPTY_AUDIO.value)
        expected_pcm_bytes = info.frame_count * info.channels * max(1, info.bits_per_sample // 8)
        if expected_pcm_bytes > 0 and info.file_size_bytes < max(44, int(expected_pcm_bytes * 0.85)):
            rejections.append(AudioPreparationRejection.AUDIO_FILE_SIZE_IMPLAUSIBLE.value)
        video_duration = float(media.duration_seconds or audio_manifest.duration_seconds)
        delta = abs(info.duration_seconds - video_duration)
        delta_ratio = delta / video_duration if video_duration > 0 else 0.0
        tolerance = max(
            self._settings.audio_video_duration_tolerance_seconds,
            video_duration * self._settings.audio_video_duration_tolerance_ratio,
        )
        if delta > tolerance:
            rejections.append(AudioPreparationRejection.AUDIO_VIDEO_DURATION_MISMATCH.value)
        elif delta > min(0.25, tolerance):
            warnings.append(AudioPreparationWarning.AUDIO_VIDEO_MINOR_DURATION_MISMATCH.value)
        if diagnostics.effectively_silent:
            rejections.append(AudioPreparationRejection.AUDIO_EFFECTIVELY_SILENT.value)
        elif diagnostics.silent_window_ratio >= 0.75:
            warnings.append(AudioPreparationWarning.HIGH_SILENCE_RATIO.value)
        if diagnostics.clipping_ratio >= self._settings.audio_high_clipping_ratio:
            warnings.append(AudioPreparationWarning.HIGH_CLIPPING_RATIO.value)
        if diagnostics.mean_rms <= self._settings.audio_silence_rms_threshold * 2 and not diagnostics.effectively_silent:
            warnings.append(AudioPreparationWarning.LOW_OVERALL_ENERGY.value)
        try:
            candidate_times = [item.timestamp_seconds for item in self._candidates.load_handoff(job_id, include_alternates=True)]
        except Exception:
            candidate_times = []
        chunks = self._chunks(info.duration_seconds, candidate_times)
        audio_fingerprint = stable_hash(
            {
                "manifest": audio_manifest.audio_fingerprint.model_dump(mode="json"),
                "size": info.file_size_bytes,
                "duration": round(info.duration_seconds, 6),
                "sample_rate": info.sample_rate_hz,
                "channels": info.channels,
                "bits": info.bits_per_sample,
            }
        )
        media_fingerprint = stable_hash(
            {"source": media.source_fingerprint.model_dump(mode="json"), "duration": media.duration_seconds}
        )
        config_fp = self.config_fingerprint()
        core = {
            "algorithm_version": AUDIO_PREPARATION_ALGORITHM_VERSION,
            "audio_fingerprint": audio_fingerprint,
            "media_fingerprint": media_fingerprint,
            "config_fingerprint": config_fp,
            "audio": info.model_dump(mode="json"),
            "diagnostics": diagnostics.model_dump(mode="json"),
            "chunks": [c.model_dump(mode="json") for c in chunks],
            "warnings": warnings,
            "rejections": rejections,
        }
        manifest = TranscriptionPreparationManifest(
            algorithm_version=AUDIO_PREPARATION_ALGORITHM_VERSION,
            audio_fingerprint=audio_fingerprint,
            media_fingerprint=media_fingerprint,
            candidate_summary_fingerprint=candidate_summary_fingerprint,
            config_fingerprint=config_fp,
            artifact_fingerprint=stable_hash(core),
            audio_path=audio_manifest.path,
            audio=info,
            diagnostics=diagnostics,
            video_duration_seconds=video_duration,
            stats=AudioPreparationStats(
                chunk_count=len(chunks),
                candidate_annotated_chunk_count=sum(c.contains_candidate_timestamps for c in chunks),
                duration_delta_seconds=delta,
                duration_delta_ratio=delta_ratio,
            ),
            warnings=warnings,
            rejection_reasons=rejections,
            is_valid=not rejections,
            chunks=chunks,
        )
        self._repository.save_preparation(job_id, manifest)
        if rejections:
            raise AudioPreparationError("Audio is not suitable for transcription: " + ", ".join(rejections))
        progress_callback(100.0)
        _ = started  # timing is tracked by the Phase-5 orchestrator.
        return manifest
