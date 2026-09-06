from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobStage(str, Enum):
    QUEUED = "QUEUED"
    PREPARING = "PREPARING"
    DOWNLOADING = "DOWNLOADING"
    INSPECTING_MEDIA = "INSPECTING_MEDIA"
    EXTRACTING_AUDIO = "EXTRACTING_AUDIO"
    INGESTION_COMPLETE = "INGESTION_COMPLETE"
    TRANSCRIBING = "TRANSCRIBING"
    VALIDATING_AUDIO = "VALIDATING_AUDIO"
    TRANSCRIBING_AUDIO = "TRANSCRIBING_AUDIO"
    NORMALIZING_TRANSCRIPT = "NORMALIZING_TRANSCRIPT"
    ALIGNING_TRANSCRIPT = "ALIGNING_TRANSCRIPT"
    BUILDING_TRANSCRIPT_CONTEXT = "BUILDING_TRANSCRIPT_CONTEXT"
    SAMPLING_FRAMES = "SAMPLING_FRAMES"
    PREPROCESSING_FRAMES = "PREPROCESSING_FRAMES"
    DETECTING_CHANGES = "DETECTING_CHANGES"
    DETECTING_MAJOR_CHANGES = "DETECTING_MAJOR_CHANGES"
    DETECTING_STABILITY = "DETECTING_STABILITY"
    FRAME_ANALYSIS_COMPLETE = "FRAME_ANALYSIS_COMPLETE"
    GENERATING_CANDIDATES = "GENERATING_CANDIDATES"
    BUILDING_TEMPORAL_CONTEXT = "BUILDING_TEMPORAL_CONTEXT"
    AI_ANALYSIS = "AI_ANALYSIS"
    SELECTING_SCREENSHOTS = "SELECTING_SCREENSHOTS"
    DEDUPLICATING = "DEDUPLICATING"
    GENERATING_PDF = "GENERATING_PDF"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobErrorInfo(BaseModel):
    code: str
    message: str
    category: str | None = None
    failed_stage: str | None = None


class Job(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    id: str
    source_url: str
    status: JobStatus = JobStatus.QUEUED
    stage: JobStage = JobStage.QUEUED
    progress: int = Field(default=0, ge=0, le=100)
    message: str = "Job queued"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: JobErrorInfo | None = None

    @model_validator(mode="after")
    def validate_terminal_state(self) -> "Job":
        if self.status is JobStatus.COMPLETED and self.progress != 100:
            raise ValueError("Completed jobs must have progress 100")
        return self


class JobCreateRequest(BaseModel):
    source_url: HttpUrl


class JobListResponse(BaseModel):
    items: list[Job]
    total: int
