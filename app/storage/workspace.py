from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

from app.core.exceptions import StorageError


WORKSPACE_DIRS = (
    "source",
    "audio",
    "frames",
    "analysis",
    "candidates",
    "screenshots",
    "transcript",
    "semantic",
    "decisions",
    "output",
    "logs",
)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, path)
    except OSError as exc:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise StorageError(f"Unable to persist {path.name}") from exc


class WorkspaceManager:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def ensure_root(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def _normalized_job_id(self, job_id: str) -> str:
        try:
            return str(UUID(str(job_id)))
        except (ValueError, TypeError, AttributeError) as exc:
            raise StorageError("Invalid job id for workspace path") from exc

    def workspace(self, job_id: str) -> Path:
        normalized = self._normalized_job_id(job_id)
        candidate = (self.root / normalized).resolve()
        if candidate.parent != self.root:
            raise StorageError("Unsafe workspace path")
        return candidate

    def create_workspace(self, job_id: str) -> Path:
        workspace = self.workspace(job_id)
        workspace.mkdir(parents=True, exist_ok=True)
        for name in WORKSPACE_DIRS:
            (workspace / name).mkdir(exist_ok=True)
        return workspace

    def ensure_workspace(self, job_id: str) -> Path:
        return self.create_workspace(job_id)

    def delete_workspace(self, job_id: str) -> None:
        workspace = self.workspace(job_id)
        if workspace.exists():
            shutil.rmtree(workspace)

    def job_metadata_path(self, job_id: str) -> Path:
        return self.workspace(job_id) / "job.json"

    def checkpoints_path(self, job_id: str) -> Path:
        return self.workspace(job_id) / "checkpoints.json"

    def ingestion_path(self, job_id: str) -> Path:
        return self.workspace(job_id) / "ingestion.json"

    def source_dir(self, job_id: str) -> Path:
        return self.workspace(job_id) / "source"

    def youtube_metadata_path(self, job_id: str) -> Path:
        return self.source_dir(job_id) / "metadata.json"

    def download_manifest_path(self, job_id: str) -> Path:
        return self.source_dir(job_id) / "download.json"

    def media_inspection_path(self, job_id: str) -> Path:
        return self.source_dir(job_id) / "media.json"

    def audio_dir(self, job_id: str) -> Path:
        return self.workspace(job_id) / "audio"

    def audio_path(self, job_id: str) -> Path:
        return self.audio_dir(job_id) / "audio.wav"

    def audio_temp_path(self, job_id: str) -> Path:
        return self.audio_dir(job_id) / "audio.tmp.wav"

    def audio_manifest_path(self, job_id: str) -> Path:
        return self.audio_dir(job_id) / "audio.json"

    def frames_dir(self, job_id: str) -> Path:
        return self.workspace(job_id) / "frames"

    def sampled_frames_dir(self, job_id: str) -> Path:
        return self.frames_dir(job_id) / "sampled"

    def sampled_frames_temp_dir(self, job_id: str) -> Path:
        return self.frames_dir(job_id) / "sampled.tmp"

    def frame_manifest_path(self, job_id: str) -> Path:
        return self.frames_dir(job_id) / "manifest.json"

    def processed_frames_dir(self, job_id: str) -> Path:
        return self.frames_dir(job_id) / "processed"

    def processed_frames_temp_dir(self, job_id: str) -> Path:
        return self.frames_dir(job_id) / "processed.tmp"

    def preprocessing_manifest_path(self, job_id: str) -> Path:
        return self.frames_dir(job_id) / "preprocessing.json"

    def analysis_dir(self, job_id: str) -> Path:
        return self.workspace(job_id) / "analysis"

    def differences_path(self, job_id: str) -> Path:
        return self.analysis_dir(job_id) / "differences.json"

    def major_changes_path(self, job_id: str) -> Path:
        return self.analysis_dir(job_id) / "major_changes.json"

    def timeline_path(self, job_id: str) -> Path:
        return self.analysis_dir(job_id) / "timeline.json"

    def analysis_summary_path(self, job_id: str) -> Path:
        return self.analysis_dir(job_id) / "summary.json"

    def evaluation_path(self, job_id: str) -> Path:
        return self.analysis_dir(job_id) / "evaluation.json"

    def candidates_dir(self, job_id: str) -> Path:
        return self.workspace(job_id) / "candidates"

    def stability_windows_path(self, job_id: str) -> Path:
        return self.candidates_dir(job_id) / "stability_windows.json"

    def boundaries_path(self, job_id: str) -> Path:
        return self.candidates_dir(job_id) / "boundaries.json"

    def generated_candidates_path(self, job_id: str) -> Path:
        return self.candidates_dir(job_id) / "generated_candidates.json"

    def scored_candidates_path(self, job_id: str) -> Path:
        return self.candidates_dir(job_id) / "scored_candidates.json"

    def ranked_candidates_path(self, job_id: str) -> Path:
        return self.candidates_dir(job_id) / "ranked_candidates.json"

    def candidate_selections_path(self, job_id: str) -> Path:
        return self.candidates_dir(job_id) / "selections.json"

    def candidate_summary_path(self, job_id: str) -> Path:
        return self.candidates_dir(job_id) / "summary.json"

    def candidate_evaluation_path(self, job_id: str) -> Path:
        return self.candidates_dir(job_id) / "evaluation.json"

    def screenshots_dir(self, job_id: str) -> Path:
        return self.workspace(job_id) / "screenshots"

    def transcript_dir(self, job_id: str) -> Path:
        return self.workspace(job_id) / "transcript"


    def transcript_preparation_path(self, job_id: str) -> Path:
        return self.transcript_dir(job_id) / "preparation.json"

    def raw_transcript_chunks_dir(self, job_id: str) -> Path:
        return self.transcript_dir(job_id) / "raw_chunks"

    def raw_transcript_chunk_path(self, job_id: str, chunk_id: int) -> Path:
        return self.raw_transcript_chunks_dir(job_id) / f"chunk_{chunk_id:04d}.json"

    def raw_transcript_path(self, job_id: str) -> Path:
        return self.transcript_dir(job_id) / "raw_transcript.json"

    def normalized_transcript_path(self, job_id: str) -> Path:
        return self.transcript_dir(job_id) / "transcript.json"

    def candidate_alignment_path(self, job_id: str) -> Path:
        return self.transcript_dir(job_id) / "candidate_alignment.json"

    def transcript_contexts_path(self, job_id: str) -> Path:
        return self.transcript_dir(job_id) / "contexts.json"

    def transcript_summary_path(self, job_id: str) -> Path:
        return self.transcript_dir(job_id) / "summary.json"

    def transcript_evaluation_path(self, job_id: str) -> Path:
        return self.transcript_dir(job_id) / "evaluation.json"

    def transcript_temp_audio_dir(self, job_id: str) -> Path:
        return self.transcript_dir(job_id) / "chunks.tmp"

    def semantic_dir(self, job_id: str) -> Path:
        return self.workspace(job_id) / "semantic"

    def semantic_input_manifest_path(self, job_id: str) -> Path:
        return self.semantic_dir(job_id) / "input_manifest.json"

    def semantic_temporal_contexts_path(self, job_id: str) -> Path:
        return self.semantic_dir(job_id) / "temporal_contexts.json"

    def semantic_raw_results_dir(self, job_id: str) -> Path:
        return self.semantic_dir(job_id) / "raw_model_results"

    def semantic_candidate_result_path(self, job_id: str, candidate_id: int) -> Path:
        if candidate_id < 1:
            raise StorageError("Invalid semantic candidate id")
        return self.semantic_raw_results_dir(job_id) / f"candidate_{candidate_id:06d}.json"

    def semantic_results_path(self, job_id: str) -> Path:
        return self.semantic_dir(job_id) / "semantic_results.json"

    def semantic_selections_path(self, job_id: str) -> Path:
        return self.semantic_dir(job_id) / "selections.json"

    def semantic_summary_path(self, job_id: str) -> Path:
        return self.semantic_dir(job_id) / "summary.json"

    def semantic_evaluation_path(self, job_id: str) -> Path:
        return self.semantic_dir(job_id) / "evaluation.json"

    def decisions_dir(self, job_id: str) -> Path:
        return self.workspace(job_id) / "decisions"

    def output_dir(self, job_id: str) -> Path:
        return self.workspace(job_id) / "output"

    def logs_dir(self, job_id: str) -> Path:
        return self.workspace(job_id) / "logs"

    def relative_to_workspace(self, job_id: str, path: Path) -> str:
        resolved = path.resolve()
        workspace = self.workspace(job_id)
        try:
            return resolved.relative_to(workspace).as_posix()
        except ValueError as exc:
            raise StorageError("Artifact path is outside job workspace") from exc

    def workspace_size(self, job_id: str) -> int:
        total = 0
        root = self.workspace(job_id)
        if not root.exists():
            return 0
        for path in root.rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
        return total

    def cleanup_expired(
        self,
        retention_hours: int,
        *,
        protected_job_ids: set[str] | None = None,
        dry_run: bool = False,
    ) -> list[str]:
        self.ensure_root()
        protected = {str(UUID(job_id)) for job_id in (protected_job_ids or set())}
        cutoff = datetime.now(UTC) - timedelta(hours=retention_hours)
        deleted: list[str] = []

        for child in self.root.iterdir():
            if not child.is_dir() or child.name in protected:
                continue
            try:
                UUID(child.name)
            except ValueError:
                continue
            modified = datetime.fromtimestamp(child.stat().st_mtime, tz=UTC)
            if modified < cutoff:
                if not dry_run:
                    shutil.rmtree(child)
                deleted.append(child.name)
        return deleted
