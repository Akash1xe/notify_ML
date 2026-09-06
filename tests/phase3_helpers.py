from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from app.core.config import AppSettings
from app.jobs.checkpoints import CheckpointStore
from app.media.probe import fingerprint
from app.storage.workspace import WorkspaceManager, atomic_write_json
from app.video_analysis.cache import CP_FRAMES_SAMPLED
from app.video_analysis.models import SampledFrame, SamplingManifest
from app.video_analysis.sampling import (
    SAMPLING_ALGORITHM_VERSION,
    sampling_artifact_fingerprint,
    sampling_config_fingerprint,
)


def build_phase3_workspace(tmp_path: Path, **settings_overrides):
    values = {
        "storage_root": tmp_path / "jobs",
        "processor_mode": "analysis",
        "frame_disk_safety_margin_mb": 0,
        "audio_disk_safety_margin_mb": 0,
        "download_disk_safety_margin_mb": 0,
    }
    values.update(settings_overrides)
    settings = AppSettings(**values)
    workspace = WorkspaceManager(settings.storage_root)
    workspace.ensure_root()
    job_id = str(uuid4())
    workspace.create_workspace(job_id)
    return settings, workspace, job_id


def synthetic_frame(kind: str = "blank", width: int = 640, height: int = 360) -> np.ndarray:
    if kind == "black":
        return np.zeros((height, width, 3), dtype=np.uint8)
    if kind == "gray":
        return np.full((height, width, 3), 128, dtype=np.uint8)
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    if kind == "text":
        cv2.putText(image, "Kafka", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 4)
        cv2.line(image, (50, 180), (500, 180), (0, 0, 0), 5)
    elif kind == "dense":
        for y in range(30, height, 30):
            cv2.line(image, (20, y), (width - 20, y), (0, 0, 0), 2)
        for x in range(20, width, 50):
            cv2.line(image, (x, 20), (x, height - 20), (0, 0, 0), 1)
    elif kind == "cursor":
        cv2.rectangle(image, (300, 170), (310, 180), (0, 0, 0), -1)
    elif kind == "checker":
        block = 40
        for y in range(0, height, block):
            for x in range(0, width, block):
                if (x // block + y // block) % 2 == 0:
                    image[y:y + block, x:x + block] = 0
    return image


def create_sample_manifest(
    settings: AppSettings,
    workspace: WorkspaceManager,
    job_id: str,
    kinds: list[str],
    *,
    fps: float | None = None,
) -> tuple[Path, SamplingManifest]:
    fps = fps or settings.frame_sample_fps
    source = workspace.source_dir(job_id) / "video.mp4"
    source.write_bytes(b"phase3-source-video")
    sampled_dir = workspace.sampled_frames_dir(job_id)
    sampled_dir.mkdir(parents=True, exist_ok=True)
    records: list[SampledFrame] = []
    for index, kind in enumerate(kinds, start=1):
        path = sampled_dir / f"frame_{index:08d}.jpg"
        assert cv2.imwrite(str(path), synthetic_frame(kind), [cv2.IMWRITE_JPEG_QUALITY, 95])
        records.append(
            SampledFrame(
                index=index,
                timestamp_seconds=(index - 1) / fps,
                relative_path=f"frames/sampled/{path.name}",
                file_size_bytes=path.stat().st_size,
            )
        )
    manifest = SamplingManifest(
        algorithm_version=SAMPLING_ALGORITHM_VERSION,
        source_video="source/video.mp4",
        source_fingerprint=fingerprint(source),
        sample_fps=fps,
        interval_seconds=1.0 / fps,
        video_duration_seconds=max(1.0, len(kinds) / fps),
        expected_frame_count=len(kinds),
        actual_frame_count=len(kinds),
        jpeg_quality=settings.frame_jpeg_quality,
        config_fingerprint=sampling_config_fingerprint(settings),
        artifact_fingerprint="pending",
        frames=records,
    )
    manifest.artifact_fingerprint = sampling_artifact_fingerprint(manifest)
    atomic_write_json(workspace.frame_manifest_path(job_id), manifest.model_dump(mode="json"))
    checkpoints = CheckpointStore(workspace)
    checkpoints.mark_completed(job_id, CP_FRAMES_SAMPLED)
    return source, manifest
