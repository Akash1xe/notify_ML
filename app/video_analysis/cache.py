from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.core.config import AppSettings
from app.jobs.checkpoints import CheckpointStore
from app.media.probe import fingerprint
from app.storage.workspace import WorkspaceManager
from app.video_analysis.changes import (
    major_change_artifact_fingerprint,
    major_change_config_fingerprint,
)
from app.video_analysis.differences import (
    difference_artifact_fingerprint,
    difference_config_fingerprint,
)
from app.video_analysis.models import (
    AnalysisArtifactCheck,
    AnalysisArtifactState,
    AnalysisCacheSnapshot,
    DifferenceManifest,
    FrameAnalysisSummary,
    MajorChangesManifest,
    Phase3ResumeStage,
    PreprocessingManifest,
    SamplingManifest,
    TimelineManifest,
)
from app.video_analysis.preprocessing import (
    preprocessing_artifact_fingerprint,
    preprocessing_config_fingerprint,
)
from app.video_analysis.sampling import sampling_artifact_fingerprint, sampling_config_fingerprint
from app.video_analysis.timeline import timeline_artifact_fingerprint, timeline_config_fingerprint


CP_FRAMES_SAMPLED = "FRAMES_SAMPLED"
CP_FRAMES_PREPROCESSED = "FRAMES_PREPROCESSED"
CP_FRAME_DIFFERENCES = "FRAME_DIFFERENCES_CALCULATED"
CP_MAJOR_CHANGES = "MAJOR_CHANGES_DETECTED"
CP_TIMELINE = "TEMPORAL_TIMELINE_BUILT"
CP_FRAME_ANALYSIS = "FRAME_ANALYSIS_COMPLETE"

T = TypeVar("T", bound=BaseModel)


def _check(state: AnalysisArtifactState, reason: str | None = None) -> AnalysisArtifactCheck:
    return AnalysisArtifactCheck(state=state, reason=reason)


class FrameAnalysisCacheManager:
    def __init__(
        self,
        settings: AppSettings,
        workspace: WorkspaceManager,
        checkpoints: CheckpointStore,
    ) -> None:
        self._settings = settings
        self._workspace = workspace
        self._checkpoints = checkpoints

    def _load_model(self, path: Path, model: type[T]) -> tuple[T | None, AnalysisArtifactCheck]:
        if not path.exists():
            return None, _check(AnalysisArtifactState.MISSING, "manifest_missing")
        try:
            value = model.model_validate(json.loads(path.read_text(encoding="utf-8")))
            return value, _check(AnalysisArtifactState.VALID)
        except (OSError, json.JSONDecodeError, ValidationError, ValueError, TypeError):
            return None, _check(AnalysisArtifactState.CORRUPT, "invalid_json_or_schema")

    def _safe_relative(self, job_id: str, relative: str) -> Path | None:
        root = self._workspace.workspace(job_id)
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate

    def validate_sampling(
        self, job_id: str, source_path: Path
    ) -> tuple[AnalysisArtifactCheck, SamplingManifest | None]:
        if self._workspace.sampled_frames_temp_dir(job_id).exists():
            return _check(AnalysisArtifactState.PARTIAL, "partial_output_present"), None
        if not self._checkpoints.is_completed(job_id, CP_FRAMES_SAMPLED):
            return _check(AnalysisArtifactState.MISSING, "checkpoint_missing"), None
        manifest, result = self._load_model(self._workspace.frame_manifest_path(job_id), SamplingManifest)
        if manifest is None:
            return result, None
        if not source_path.exists():
            return _check(AnalysisArtifactState.MISSING, "source_video_missing"), None
        if fingerprint(source_path) != manifest.source_fingerprint:
            return _check(AnalysisArtifactState.STALE, "source_fingerprint_mismatch"), None
        if manifest.config_fingerprint != sampling_config_fingerprint(self._settings):
            return _check(AnalysisArtifactState.STALE, "configuration_mismatch"), None
        if sampling_artifact_fingerprint(manifest) != manifest.artifact_fingerprint:
            return _check(AnalysisArtifactState.CORRUPT, "artifact_fingerprint_mismatch"), None
        if manifest.actual_frame_count != len(manifest.frames):
            return _check(AnalysisArtifactState.CORRUPT, "count_mismatch"), None
        for frame in manifest.frames:
            path = self._safe_relative(job_id, frame.relative_path)
            if path is None:
                return _check(AnalysisArtifactState.CORRUPT, "unsafe_artifact_path"), None
            if not path.exists() or not path.is_file() or path.stat().st_size <= 0:
                return _check(AnalysisArtifactState.MISSING, "referenced_frame_missing"), None
            if path.stat().st_size != frame.file_size_bytes:
                return _check(AnalysisArtifactState.STALE, "sampled_frame_size_changed"), None
        return _check(AnalysisArtifactState.VALID), manifest

    def validate_preprocessing(
        self, job_id: str, sampling: SamplingManifest | None
    ) -> tuple[AnalysisArtifactCheck, PreprocessingManifest | None]:
        if sampling is None:
            return _check(AnalysisArtifactState.STALE, "upstream_invalid"), None
        if self._workspace.processed_frames_temp_dir(job_id).exists():
            return _check(AnalysisArtifactState.PARTIAL, "partial_output_present"), None
        if not self._checkpoints.is_completed(job_id, CP_FRAMES_PREPROCESSED):
            return _check(AnalysisArtifactState.MISSING, "checkpoint_missing"), None
        manifest, result = self._load_model(
            self._workspace.preprocessing_manifest_path(job_id), PreprocessingManifest
        )
        if manifest is None:
            return result, None
        if manifest.sampling_fingerprint != sampling.artifact_fingerprint:
            return _check(AnalysisArtifactState.STALE, "sampling_dependency_mismatch"), None
        if manifest.config_fingerprint != preprocessing_config_fingerprint(self._settings):
            return _check(AnalysisArtifactState.STALE, "configuration_mismatch"), None
        if preprocessing_artifact_fingerprint(manifest) != manifest.artifact_fingerprint:
            return _check(AnalysisArtifactState.CORRUPT, "artifact_fingerprint_mismatch"), None
        if manifest.stats.total_frames != len(manifest.frames):
            return _check(AnalysisArtifactState.CORRUPT, "count_mismatch"), None
        for frame in manifest.frames:
            if not frame.is_valid:
                continue
            if not frame.processed_path:
                return _check(AnalysisArtifactState.CORRUPT, "processed_path_missing"), None
            path = self._safe_relative(job_id, frame.processed_path)
            if path is None:
                return _check(AnalysisArtifactState.CORRUPT, "unsafe_artifact_path"), None
            if not path.exists() or path.stat().st_size <= 0:
                return _check(AnalysisArtifactState.MISSING, "processed_frame_missing"), None
        return _check(AnalysisArtifactState.VALID), manifest

    def validate_differences(
        self, job_id: str, preprocessing: PreprocessingManifest | None
    ) -> tuple[AnalysisArtifactCheck, DifferenceManifest | None]:
        if preprocessing is None:
            return _check(AnalysisArtifactState.STALE, "upstream_invalid"), None
        if not self._checkpoints.is_completed(job_id, CP_FRAME_DIFFERENCES):
            return _check(AnalysisArtifactState.MISSING, "checkpoint_missing"), None
        manifest, result = self._load_model(self._workspace.differences_path(job_id), DifferenceManifest)
        if manifest is None:
            return result, None
        if manifest.preprocessing_fingerprint != preprocessing.artifact_fingerprint:
            return _check(AnalysisArtifactState.STALE, "preprocessing_dependency_mismatch"), None
        if manifest.config_fingerprint != difference_config_fingerprint(self._settings):
            return _check(AnalysisArtifactState.STALE, "configuration_mismatch"), None
        if difference_artifact_fingerprint(manifest) != manifest.artifact_fingerprint:
            return _check(AnalysisArtifactState.CORRUPT, "artifact_fingerprint_mismatch"), None
        expected = max(0, len(preprocessing.frames) - 1)
        if manifest.stats.total_comparisons != len(manifest.comparisons) or len(manifest.comparisons) != expected:
            return _check(AnalysisArtifactState.CORRUPT, "count_mismatch"), None
        return _check(AnalysisArtifactState.VALID), manifest

    def validate_major_changes(
        self, job_id: str, differences: DifferenceManifest | None
    ) -> tuple[AnalysisArtifactCheck, MajorChangesManifest | None]:
        if differences is None:
            return _check(AnalysisArtifactState.STALE, "upstream_invalid"), None
        if not self._checkpoints.is_completed(job_id, CP_MAJOR_CHANGES):
            return _check(AnalysisArtifactState.MISSING, "checkpoint_missing"), None
        manifest, result = self._load_model(self._workspace.major_changes_path(job_id), MajorChangesManifest)
        if manifest is None:
            return result, None
        if manifest.differences_fingerprint != differences.artifact_fingerprint:
            return _check(AnalysisArtifactState.STALE, "differences_dependency_mismatch"), None
        if manifest.config_fingerprint != major_change_config_fingerprint(self._settings):
            return _check(AnalysisArtifactState.STALE, "configuration_mismatch"), None
        if major_change_artifact_fingerprint(manifest) != manifest.artifact_fingerprint:
            return _check(AnalysisArtifactState.CORRUPT, "artifact_fingerprint_mismatch"), None
        valid_pairs = {
            (item.previous_frame_index, item.current_frame_index)
            for item in differences.comparisons
        }
        for event in manifest.events:
            if event.previous_frame_index is None or event.current_frame_index is None:
                continue
            if (event.previous_frame_index, event.current_frame_index) not in valid_pairs:
                return _check(AnalysisArtifactState.CORRUPT, "event_reference_missing"), None
        return _check(AnalysisArtifactState.VALID), manifest

    def validate_timeline(
        self,
        job_id: str,
        differences: DifferenceManifest | None,
        major_changes: MajorChangesManifest | None,
    ) -> tuple[AnalysisArtifactCheck, TimelineManifest | None]:
        if differences is None or major_changes is None:
            return _check(AnalysisArtifactState.STALE, "upstream_invalid"), None
        if not self._checkpoints.is_completed(job_id, CP_TIMELINE):
            return _check(AnalysisArtifactState.MISSING, "checkpoint_missing"), None
        manifest, result = self._load_model(self._workspace.timeline_path(job_id), TimelineManifest)
        if manifest is None:
            return result, None
        if manifest.differences_fingerprint != differences.artifact_fingerprint:
            return _check(AnalysisArtifactState.STALE, "differences_dependency_mismatch"), None
        if manifest.major_changes_fingerprint != major_changes.artifact_fingerprint:
            return _check(AnalysisArtifactState.STALE, "major_changes_dependency_mismatch"), None
        if manifest.config_fingerprint != timeline_config_fingerprint(self._settings):
            return _check(AnalysisArtifactState.STALE, "configuration_mismatch"), None
        if timeline_artifact_fingerprint(manifest) != manifest.artifact_fingerprint:
            return _check(AnalysisArtifactState.CORRUPT, "artifact_fingerprint_mismatch"), None
        return _check(AnalysisArtifactState.VALID), manifest

    def _completion_check(self, job_id: str, timeline: TimelineManifest | None) -> AnalysisArtifactCheck:
        if timeline is None or not self._checkpoints.is_completed(job_id, CP_FRAME_ANALYSIS):
            return _check(AnalysisArtifactState.MISSING, "checkpoint_missing")
        summary, result = self._load_model(
            self._workspace.analysis_summary_path(job_id), FrameAnalysisSummary
        )
        if summary is None:
            return result
        if summary.timeline_fingerprint != timeline.artifact_fingerprint:
            return _check(AnalysisArtifactState.STALE, "timeline_dependency_mismatch")
        return _check(AnalysisArtifactState.VALID)

    def inspect(self, job_id: str, source_path: Path) -> AnalysisCacheSnapshot:
        sampling_check, sampling = self.validate_sampling(job_id, source_path)
        if not sampling_check.valid:
            stale = _check(AnalysisArtifactState.STALE, "upstream_invalid")
            return AnalysisCacheSnapshot(
                sampling=sampling_check,
                preprocessing=stale,
                differences=stale,
                major_changes=stale,
                timeline=stale,
                frame_analysis_complete=_check(AnalysisArtifactState.MISSING, "artifact_or_checkpoint_missing"),
                resume_stage=Phase3ResumeStage.SAMPLING_FRAMES,
            )
        preprocessing_check, preprocessing = self.validate_preprocessing(job_id, sampling)
        if not preprocessing_check.valid:
            stale = _check(AnalysisArtifactState.STALE, "upstream_invalid")
            return AnalysisCacheSnapshot(
                sampling=sampling_check,
                preprocessing=preprocessing_check,
                differences=stale,
                major_changes=stale,
                timeline=stale,
                frame_analysis_complete=_check(AnalysisArtifactState.MISSING, "artifact_or_checkpoint_missing"),
                resume_stage=Phase3ResumeStage.PREPROCESSING_FRAMES,
            )
        differences_check, differences = self.validate_differences(job_id, preprocessing)
        if not differences_check.valid:
            stale = _check(AnalysisArtifactState.STALE, "upstream_invalid")
            return AnalysisCacheSnapshot(
                sampling=sampling_check,
                preprocessing=preprocessing_check,
                differences=differences_check,
                major_changes=stale,
                timeline=stale,
                frame_analysis_complete=_check(AnalysisArtifactState.MISSING, "artifact_or_checkpoint_missing"),
                resume_stage=Phase3ResumeStage.DETECTING_CHANGES,
            )
        major_check, major = self.validate_major_changes(job_id, differences)
        if not major_check.valid:
            return AnalysisCacheSnapshot(
                sampling=sampling_check,
                preprocessing=preprocessing_check,
                differences=differences_check,
                major_changes=major_check,
                timeline=_check(AnalysisArtifactState.STALE, "upstream_invalid"),
                frame_analysis_complete=_check(AnalysisArtifactState.MISSING, "artifact_or_checkpoint_missing"),
                resume_stage=Phase3ResumeStage.DETECTING_MAJOR_CHANGES,
            )
        timeline_check, timeline = self.validate_timeline(job_id, differences, major)
        if not timeline_check.valid:
            return AnalysisCacheSnapshot(
                sampling=sampling_check,
                preprocessing=preprocessing_check,
                differences=differences_check,
                major_changes=major_check,
                timeline=timeline_check,
                frame_analysis_complete=_check(AnalysisArtifactState.MISSING, "artifact_or_checkpoint_missing"),
                resume_stage=Phase3ResumeStage.DETECTING_STABILITY,
            )
        return AnalysisCacheSnapshot(
            sampling=sampling_check,
            preprocessing=preprocessing_check,
            differences=differences_check,
            major_changes=major_check,
            timeline=timeline_check,
            frame_analysis_complete=self._completion_check(job_id, timeline),
            resume_stage=Phase3ResumeStage.PHASE3_READY,
        )

    def reconcile(self, job_id: str, source_path: Path) -> AnalysisCacheSnapshot:
        """Validate persisted Phase-3 state and clear stale completion markers/artifacts."""
        snapshot = self.inspect(job_id, source_path)
        if snapshot.resume_stage is not Phase3ResumeStage.PHASE3_READY:
            self.invalidate_from(job_id, snapshot.resume_stage)
        elif not snapshot.frame_analysis_complete.valid:
            # Stage artifacts are valid; only the umbrella completion/summary may
            # need to be rebuilt by Phase 3.7.
            self._checkpoints.invalidate(job_id, CP_FRAME_ANALYSIS)
            self._workspace.analysis_summary_path(job_id).unlink(missing_ok=True)
        return snapshot

    def invalidate_from(self, job_id: str, stage: Phase3ResumeStage) -> None:
        order = [
            Phase3ResumeStage.SAMPLING_FRAMES,
            Phase3ResumeStage.PREPROCESSING_FRAMES,
            Phase3ResumeStage.DETECTING_CHANGES,
            Phase3ResumeStage.DETECTING_MAJOR_CHANGES,
            Phase3ResumeStage.DETECTING_STABILITY,
        ]
        checkpoints = [
            CP_FRAMES_SAMPLED,
            CP_FRAMES_PREPROCESSED,
            CP_FRAME_DIFFERENCES,
            CP_MAJOR_CHANGES,
            CP_TIMELINE,
        ]
        if stage is Phase3ResumeStage.PHASE3_READY:
            self._checkpoints.invalidate(job_id, CP_FRAME_ANALYSIS)
            self._workspace.analysis_summary_path(job_id).unlink(missing_ok=True)
            return
        start = order.index(stage)
        self._checkpoints.invalidate(job_id, CP_FRAME_ANALYSIS)
        for checkpoint in checkpoints[start:]:
            self._checkpoints.invalidate(job_id, checkpoint)
        self._workspace.analysis_summary_path(job_id).unlink(missing_ok=True)

        if start <= 0:
            shutil.rmtree(self._workspace.sampled_frames_dir(job_id), ignore_errors=True)
            self._workspace.frame_manifest_path(job_id).unlink(missing_ok=True)
        if start <= 1:
            shutil.rmtree(self._workspace.processed_frames_dir(job_id), ignore_errors=True)
            self._workspace.preprocessing_manifest_path(job_id).unlink(missing_ok=True)
        if start <= 2:
            self._workspace.differences_path(job_id).unlink(missing_ok=True)
        if start <= 3:
            self._workspace.major_changes_path(job_id).unlink(missing_ok=True)
        if start <= 4:
            self._workspace.timeline_path(job_id).unlink(missing_ok=True)

    def cleanup_partial_artifacts(self, job_id: str) -> list[str]:
        removed: list[str] = []
        root = self._workspace.workspace(job_id)
        for directory in (
            self._workspace.sampled_frames_temp_dir(job_id),
            self._workspace.processed_frames_temp_dir(job_id),
        ):
            if directory.exists():
                removed.append(self._workspace.relative_to_workspace(job_id, directory))
                shutil.rmtree(directory, ignore_errors=True)
        analysis = self._workspace.analysis_dir(job_id)
        if analysis.exists():
            for path in analysis.glob("*.tmp"):
                if path.is_file():
                    removed.append(self._workspace.relative_to_workspace(job_id, path))
                    path.unlink(missing_ok=True)
            for path in analysis.glob("*.json.tmp"):
                if path.is_file():
                    removed.append(self._workspace.relative_to_workspace(job_id, path))
                    path.unlink(missing_ok=True)
        # Protect against malformed external paths: cleanup only touched known
        # locations derived from WorkspaceManager.
        _ = root
        return removed
