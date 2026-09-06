from __future__ import annotations

import asyncio
import math
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.candidate_analysis.repository import CandidateAnalysisRepository
from app.core.config import AppSettings
from app.core.logging import JobEventLogger
from app.media.models import AudioResult, AudioStreamInfo, MediaInspection, VideoStreamInfo
from app.media.probe import fingerprint
from app.storage.workspace import atomic_write_json
from app.transcription.alignment import CandidateTranscriptAlignmentService
from app.transcription.cache import TranscriptCacheCoordinator
from app.transcription.context import CandidateTranscriptContextService
from app.transcription.engine import TranscriptionEngine
from app.transcription.normalization import TranscriptNormalizationService
from app.transcription.pipeline import TranscriptionPipeline
from app.transcription.preparation import AudioPreparationService
from app.transcription.repository import TranscriptionRepository
from tests.phase4_helpers import build_phase4_context
from tests.test_phase4_cache_pipeline import build_pipeline as build_phase4_pipeline


class FakeTranscriptionAdapter:
    def __init__(self, outputs=None):
        self.calls = 0
        self.outputs = outputs or []

    def transcribe(self, audio_path: str):
        call = self.calls
        self.calls += 1
        with wave.open(str(audio_path), "rb") as wav:
            duration = wav.getnframes() / wav.getframerate()
        if self.outputs:
            spec = self.outputs[min(call, len(self.outputs) - 1)]
        else:
            spec = [
                (1.0, 3.0, "The producer sends the event to the broker."),
                (4.0, 5.5, "Now the broker receives the event."),
            ]
        segments = []
        for idx, item in enumerate(spec):
            start, end, text = item[:3]
            if start >= duration:
                continue
            end = min(end, duration)
            words = []
            tokens = text.split()
            if tokens:
                step = max((end - start) / len(tokens), 0.01)
                for w_idx, token in enumerate(tokens):
                    words.append(
                        SimpleNamespace(
                            word=(" " if w_idx else "") + token,
                            start=start + w_idx * step,
                            end=min(end, start + (w_idx + 1) * step),
                            probability=0.9,
                        )
                    )
            segments.append(
                SimpleNamespace(
                    start=start,
                    end=end,
                    text=text,
                    avg_logprob=-0.1 - idx * 0.01,
                    no_speech_prob=0.02,
                    compression_ratio=1.0,
                    words=words,
                )
            )
        return iter(segments), SimpleNamespace(language="en", language_probability=0.95)


def write_wav(path: Path, duration: float, *, sample_rate: int = 16000, channels: int = 1, silent: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = int(round(duration * sample_rate))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        block = sample_rate
        written = 0
        while written < total_frames:
            count = min(block, total_frames - written)
            if silent:
                mono = np.zeros(count, dtype="<i2")
            else:
                t = (np.arange(count) + written) / sample_rate
                mono = (np.sin(2 * math.pi * 220 * t) * 3000).astype("<i2")
            if channels == 1:
                data = mono.tobytes()
            else:
                data = np.repeat(mono[:, None], channels, axis=1).astype("<i2").tobytes()
            wav.writeframes(data)
            written += count


def build_phase5_context(tmp_path: Path, *, settings_overrides=None, adapter=None, silent=False):
    overrides = {
        "processor_mode": "transcription",
        "transcription_chunk_seconds": 6.0,
        "transcription_chunk_overlap_seconds": 1.0,
    }
    overrides.update(settings_overrides or {})
    ctx = build_phase4_context(tmp_path, settings_overrides=overrides)
    phase4_pipeline, *_ = build_phase4_pipeline(ctx)
    asyncio.run(phase4_pipeline.process(ctx[4].id, finalize_job=False))
    settings, workspace, jobs, checkpoints, job, *_ = ctx
    duration = ctx[-1].stats.total_duration_seconds
    audio_path = workspace.audio_path(job.id)
    write_wav(audio_path, duration, silent=silent)
    source = workspace.source_dir(job.id) / "video.mp4"
    source_fp = fingerprint(source)
    audio_fp = fingerprint(audio_path)
    audio_result = AudioResult(
        path="audio/audio.wav",
        duration_seconds=duration,
        file_size_bytes=audio_fp.file_size_bytes,
        source_video_id="dQw4w9WgXcQ",
        source_fingerprint=source_fp,
        audio_fingerprint=audio_fp,
    )
    atomic_write_json(workspace.audio_manifest_path(job.id), audio_result.model_dump(mode="json"))
    media = MediaInspection(
        file_path="source/video.mp4",
        container="mp4",
        duration_seconds=duration,
        file_size_bytes=source.stat().st_size,
        video=VideoStreamInfo(codec="h264", width=640, height=360, fps=30, duration_seconds=duration),
        audio=AudioStreamInfo(codec="aac", sample_rate=48000, channels=2, duration_seconds=duration),
        source_fingerprint=source_fp,
    )
    atomic_write_json(workspace.media_inspection_path(job.id), media.model_dump(mode="json"))
    fake = adapter or FakeTranscriptionAdapter()
    return ctx, fake


def build_phase5_pipeline(ctx, adapter):
    settings, workspace, jobs, checkpoints, *_ = ctx
    candidate_repo = CandidateAnalysisRepository(workspace)
    repository = TranscriptionRepository(workspace)
    preparation = AudioPreparationService(settings, workspace, checkpoints, repository, candidate_repo)
    engine = TranscriptionEngine(settings, workspace, repository, adapter)
    normalization = TranscriptNormalizationService(settings, repository)
    alignment = CandidateTranscriptAlignmentService(settings, repository, candidate_repo)
    context = CandidateTranscriptContextService(settings, repository, candidate_repo)
    cache = TranscriptCacheCoordinator(
        workspace,
        checkpoints,
        repository,
        candidate_repo,
        preparation,
        engine,
        normalization,
        alignment,
        context,
    )
    pipeline = TranscriptionPipeline(
        jobs=jobs,
        checkpoints=checkpoints,
        events=JobEventLogger(workspace),
        preparation=preparation,
        engine=engine,
        normalization=normalization,
        alignment=alignment,
        context=context,
        cache=cache,
        repository=repository,
    )
    return pipeline, cache, repository, preparation, engine, normalization, alignment, context
