from __future__ import annotations

import bisect
import statistics
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from app.candidate_analysis.repository import CandidateAnalysisRepository
from app.core.config import AppSettings
from app.core.exceptions import JobCancelledError, TemporalContextError
from app.jobs.checkpoints import CheckpointStore
from app.semantic_analysis.models import (
    TEMPORAL_CONTEXT_ALGORITHM_VERSION,
    TemporalContextStats,
    TemporalContextsManifest,
    TemporalContextType,
    TemporalFrame,
    TemporalFrameRole,
    TemporalVisualContext,
)
from app.semantic_analysis.preparation import CP_SEMANTIC_INPUT_READY
from app.semantic_analysis.repository import SemanticRepository
from app.storage.workspace import WorkspaceManager
from app.video_analysis.fingerprints import stable_hash
from app.video_analysis.models import FrameQuality, TimelineState
from app.video_analysis.repository import FrameAnalysisRepository

CP_TEMPORAL_VISUAL_CONTEXT_READY = "TEMPORAL_VISUAL_CONTEXT_READY"
CancelCheck = Callable[[], bool]
ProgressCallback = Callable[[int], None]


class FrameTimelineIndex:
    def __init__(self, frames: list[FrameQuality]) -> None:
        self.frames = sorted(frames, key=lambda x: (x.timestamp_seconds, x.index))
        self.timestamps = [x.timestamp_seconds for x in self.frames]
        self.by_index = {x.index: x for x in self.frames}

    def between(self, start: float, end: float) -> list[FrameQuality]:
        left = bisect.bisect_left(self.timestamps, start)
        right = bisect.bisect_right(self.timestamps, end)
        return self.frames[left:right]


class TemporalVisualContextService:
    def __init__(
        self,
        settings: AppSettings,
        workspace: WorkspaceManager,
        checkpoints: CheckpointStore,
        semantic_repository: SemanticRepository,
        frame_repository: FrameAnalysisRepository,
        candidate_repository: CandidateAnalysisRepository,
    ) -> None:
        self._settings = settings
        self._workspace = workspace
        self._checkpoints = checkpoints
        self._semantic = semantic_repository
        self._frames = frame_repository
        self._candidates = candidate_repository

    def config_fingerprint(self) -> str:
        return stable_hash(
            {
                "algorithm_version": TEMPORAL_CONTEXT_ALGORITHM_VERSION,
                "previous_max_seconds": self._settings.temporal_context_previous_max_seconds,
                "next_max_seconds": self._settings.temporal_context_next_max_seconds,
                "min_gap_seconds": self._settings.temporal_context_min_frame_gap_seconds,
                "preferred_gap_seconds": self._settings.temporal_context_preferred_frame_gap_seconds,
                "min_visual_difference": self._settings.temporal_context_min_visual_difference,
                "duplicate_difference_threshold": self._settings.temporal_context_duplicate_difference_threshold,
                "quality_policy": "timeline-quality-distance-v1",
            }
        )

    def _safe_path(self, job_id: str, relative_path: str) -> Path:
        root = self._workspace.workspace(job_id)
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise TemporalContextError("Temporal frame path is outside the job workspace.") from exc
        if not path.is_file():
            raise TemporalContextError("Temporal frame file is missing.")
        return path

    @staticmethod
    def _timeline_state(timestamp: float, segments) -> TimelineState | None:
        for segment in segments:
            if segment.start_timestamp_seconds - 1e-6 <= timestamp <= segment.end_timestamp_seconds + 1e-6:
                return segment.state
        return None

    def _difference(self, current_path: Path, other_path: Path) -> float | None:
        current = cv2.imread(str(current_path), cv2.IMREAD_GRAYSCALE)
        other = cv2.imread(str(other_path), cv2.IMREAD_GRAYSCALE)
        if current is None or other is None:
            return None
        if current.shape != other.shape:
            other = cv2.resize(other, (current.shape[1], current.shape[0]), interpolation=cv2.INTER_AREA)
        return float(np.mean(cv2.absdiff(current, other)) / 255.0)

    def _eligible(self, frame: FrameQuality, current_index: int, candidate_ts: float) -> bool:
        if frame.index == current_index or not frame.is_valid or frame.is_black or not frame.processed_path:
            return False
        return abs(frame.timestamp_seconds - candidate_ts) >= self._settings.temporal_context_min_frame_gap_seconds - 1e-6

    def _choose_neighbor(
        self,
        job_id: str,
        *,
        candidates: list[FrameQuality],
        current: FrameQuality,
        current_path: Path,
        candidate_ts: float,
        boundary_ts: float | None,
        window_start: float,
        window_end: float,
        timeline_segments,
        direction: str,
    ) -> tuple[FrameQuality | None, float | None]:
        scored: list[tuple[tuple, FrameQuality, float | None]] = []
        for frame in candidates:
            if not self._eligible(frame, current.index, candidate_ts):
                continue
            if direction == "previous" and frame.timestamp_seconds >= candidate_ts:
                continue
            if direction == "next" and frame.timestamp_seconds <= candidate_ts:
                continue
            state = self._timeline_state(frame.timestamp_seconds, timeline_segments)
            if state in {TimelineState.INVALID, TimelineState.BLACK_TRANSITION}:
                continue
            frame_path = self._safe_path(job_id, frame.processed_path)
            diff = self._difference(current_path, frame_path)
            if diff is not None and diff <= self._settings.temporal_context_duplicate_difference_threshold:
                continue
            gap = abs(candidate_ts - frame.timestamp_seconds)
            quality = frame.quality_score or 0.0
            preferred = abs(gap - self._settings.temporal_context_preferred_frame_gap_seconds)
            if direction == "previous":
                prior_changing = state is TimelineState.CHANGING
                pre_boundary = boundary_ts is not None and frame.timestamp_seconds < boundary_ts
                priority = (0 if prior_changing else 1, 0 if pre_boundary else 1, preferred, -quality, -frame.timestamp_seconds, frame.index)
            else:
                same_window = window_start <= frame.timestamp_seconds <= window_end
                later_changing = state is TimelineState.CHANGING
                priority = (0 if same_window else 1, 0 if later_changing else 1, preferred, -quality, frame.timestamp_seconds, frame.index)
            scored.append((priority, frame, diff))
        if not scored:
            return None, None
        scored.sort(key=lambda x: x[0])
        return scored[0][1], scored[0][2]

    def _to_temporal_frame(
        self,
        frame: FrameQuality,
        role: TemporalFrameRole,
        *,
        window_start: float,
        window_end: float,
        boundary_ts: float | None,
        timeline_segments,
    ) -> TemporalFrame:
        duration = max(0.0, window_end - window_start)
        rel = None
        if duration > 0 and window_start <= frame.timestamp_seconds <= window_end:
            rel = min(1.0, max(0.0, (frame.timestamp_seconds - window_start) / duration))
        return TemporalFrame(
            role=role,
            frame_index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            relative_path=frame.processed_path or frame.source_path,
            timeline_state=self._timeline_state(frame.timestamp_seconds, timeline_segments),
            quality_score=frame.quality_score,
            sharpness_score=frame.sharpness_score,
            relative_position_in_window=rel,
            seconds_relative_to_boundary=(frame.timestamp_seconds - boundary_ts) if boundary_ts is not None else None,
        )

    def process(self, job_id: str, *, progress_callback: ProgressCallback | None = None, cancel_check: CancelCheck | None = None) -> TemporalContextsManifest:
        if not self._checkpoints.is_completed(job_id, CP_SEMANTIC_INPUT_READY):
            raise TemporalContextError("SEMANTIC_INPUT_READY checkpoint is required.")
        inputs = self._semantic.load_input_manifest(job_id)
        preprocessing = self._frames.load_preprocessing(job_id)
        sampling = self._frames.load_sampling(job_id)
        timeline = self._frames.load_timeline(job_id)
        differences = self._frames.load_differences(job_id)
        stability = self._candidates.load_stability_windows(job_id)
        boundaries = self._candidates.load_boundaries(job_id)
        index = FrameTimelineIndex(preprocessing.frames)
        contexts: list[TemporalVisualContext] = []
        prev_gaps: list[float] = []
        next_gaps: list[float] = []

        for ordinal, item in enumerate(inputs.inputs):
            if cancel_check and cancel_check():
                raise JobCancelledError(f"Job {job_id} was cancelled")
            current_quality = index.by_index.get(item.semantic_frame.frame_index)
            if current_quality is None or not current_quality.processed_path:
                raise TemporalContextError("CURRENT semantic frame is missing from preprocessing manifest.")
            current_path = self._safe_path(job_id, current_quality.processed_path)
            previous_candidates = index.between(
                max(0.0, item.candidate_timestamp_seconds - self._settings.temporal_context_previous_max_seconds),
                item.candidate_timestamp_seconds,
            )
            next_candidates = index.between(
                item.candidate_timestamp_seconds,
                item.candidate_timestamp_seconds + self._settings.temporal_context_next_max_seconds,
            )
            previous, prev_diff = self._choose_neighbor(
                job_id,
                candidates=previous_candidates,
                current=current_quality,
                current_path=current_path,
                candidate_ts=item.candidate_timestamp_seconds,
                boundary_ts=item.visual_boundary_timestamp_seconds,
                window_start=item.stable_window_start_seconds,
                window_end=item.stable_window_end_seconds,
                timeline_segments=timeline.segments,
                direction="previous",
            )
            next_frame, next_diff = self._choose_neighbor(
                job_id,
                candidates=next_candidates,
                current=current_quality,
                current_path=current_path,
                candidate_ts=item.candidate_timestamp_seconds,
                boundary_ts=item.visual_boundary_timestamp_seconds,
                window_start=item.stable_window_start_seconds,
                window_end=item.stable_window_end_seconds,
                timeline_segments=timeline.segments,
                direction="next",
            )
            current_tf = self._to_temporal_frame(
                current_quality,
                TemporalFrameRole.CURRENT,
                window_start=item.stable_window_start_seconds,
                window_end=item.stable_window_end_seconds,
                boundary_ts=item.visual_boundary_timestamp_seconds,
                timeline_segments=timeline.segments,
            )
            previous_tf = self._to_temporal_frame(previous, TemporalFrameRole.PREVIOUS, window_start=item.stable_window_start_seconds, window_end=item.stable_window_end_seconds, boundary_ts=item.visual_boundary_timestamp_seconds, timeline_segments=timeline.segments) if previous else None
            next_tf = self._to_temporal_frame(next_frame, TemporalFrameRole.NEXT, window_start=item.stable_window_start_seconds, window_end=item.stable_window_end_seconds, boundary_ts=item.visual_boundary_timestamp_seconds, timeline_segments=timeline.segments) if next_frame else None
            if previous_tf and next_tf:
                ctype = TemporalContextType.TRIPLET
            elif previous_tf:
                ctype = TemporalContextType.PREVIOUS_CURRENT
            elif next_tf:
                ctype = TemporalContextType.CURRENT_NEXT
            else:
                ctype = TemporalContextType.CURRENT_ONLY
            prev_gap = item.candidate_timestamp_seconds - previous.timestamp_seconds if previous else None
            next_gap = next_frame.timestamp_seconds - item.candidate_timestamp_seconds if next_frame else None
            if prev_gap is not None:
                prev_gaps.append(prev_gap)
            if next_gap is not None:
                next_gaps.append(next_gap)
            fingerprint_payload = {
                "candidate_input_fingerprint": item.semantic_input_fingerprint,
                "current": current_tf.model_dump(mode="json"),
                "previous": previous_tf.model_dump(mode="json") if previous_tf else None,
                "next": next_tf.model_dump(mode="json") if next_tf else None,
                "config": self.config_fingerprint(),
            }
            contexts.append(
                TemporalVisualContext(
                    candidate_id=item.candidate_id,
                    stable_window_id=item.stable_window_id,
                    current=current_tf,
                    previous=previous_tf,
                    next=next_tf,
                    context_type=ctype,
                    previous_gap_seconds=prev_gap,
                    next_gap_seconds=next_gap,
                    previous_to_current_change_score=prev_diff,
                    current_to_next_change_score=next_diff,
                    temporal_context_fingerprint=stable_hash(fingerprint_payload),
                )
            )
            if progress_callback:
                progress_callback(int(((ordinal + 1) / max(1, len(inputs.inputs))) * 100))

        contexts.sort(key=lambda x: (x.current.timestamp_seconds, x.candidate_id))
        type_counts = {t: sum(x.context_type is t for x in contexts) for t in TemporalContextType}
        warnings: list[str] = []
        if contexts and type_counts[TemporalContextType.CURRENT_ONLY] / len(contexts) >= 0.6:
            warnings.append("HIGH_CURRENT_ONLY_RATIO")
        stats = TemporalContextStats(
            context_count=len(contexts),
            triplet_count=type_counts[TemporalContextType.TRIPLET],
            previous_current_count=type_counts[TemporalContextType.PREVIOUS_CURRENT],
            current_next_count=type_counts[TemporalContextType.CURRENT_NEXT],
            current_only_count=type_counts[TemporalContextType.CURRENT_ONLY],
            mean_previous_gap_seconds=statistics.fmean(prev_gaps) if prev_gaps else 0.0,
            mean_next_gap_seconds=statistics.fmean(next_gaps) if next_gaps else 0.0,
            median_previous_gap_seconds=statistics.median(prev_gaps) if prev_gaps else 0.0,
            median_next_gap_seconds=statistics.median(next_gaps) if next_gaps else 0.0,
            warnings=warnings,
        )
        config_fp = self.config_fingerprint()
        payload = {
            "semantic_input_fingerprint": inputs.artifact_fingerprint,
            "frame_manifest_fingerprint": sampling.artifact_fingerprint,
            "preprocessing_fingerprint": preprocessing.artifact_fingerprint,
            "timeline_fingerprint": timeline.artifact_fingerprint,
            "differences_fingerprint": differences.artifact_fingerprint,
            "stability_windows_fingerprint": stability.artifact_fingerprint,
            "boundaries_fingerprint": boundaries.artifact_fingerprint,
            "config_fingerprint": config_fp,
            "contexts": [x.model_dump(mode="json") for x in contexts],
        }
        manifest = TemporalContextsManifest(
            semantic_input_fingerprint=inputs.artifact_fingerprint,
            frame_manifest_fingerprint=sampling.artifact_fingerprint,
            preprocessing_fingerprint=preprocessing.artifact_fingerprint,
            timeline_fingerprint=timeline.artifact_fingerprint,
            differences_fingerprint=differences.artifact_fingerprint,
            stability_windows_fingerprint=stability.artifact_fingerprint,
            boundaries_fingerprint=boundaries.artifact_fingerprint,
            config_fingerprint=config_fp,
            artifact_fingerprint=stable_hash(payload),
            stats=stats,
            contexts=contexts,
        )
        self._semantic.save_temporal_contexts(job_id, manifest)
        return manifest
