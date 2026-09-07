from __future__ import annotations

import asyncio
import math
import os
import statistics
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from app.core.config import AppSettings
from app.core.exceptions import JobCancelledError, ScreenshotQualityError
from app.jobs.checkpoints import CheckpointStore
from app.screenshots.extraction import FrameExtractionBackend, FFmpegFrameExtractionBackend, CP_SOURCE_SCREENSHOTS_EXTRACTED
from app.screenshots.models import (
    SCREENSHOT_QUALITY_ALGORITHM_VERSION,
    ExposureState,
    QualityAttemptRecord,
    QualityManifest,
    QualitySelectionRecord,
    QualityStats,
    ScreenshotQualityMetrics,
    ScreenshotQualityState,
    SourceScreenshotRecord,
)
from app.screenshots.repository import ScreenshotRepository
from app.screenshots.utils import canonical_fingerprint, decode_rgb, file_sha256, resolve_job_path
from app.semantic_analysis.repository import SemanticRepository
from app.storage.workspace import WorkspaceManager

CP_QUALITY_SCREENSHOTS_READY = 'QUALITY_SCREENSHOTS_READY'


class ScreenshotQualityValidator:
    def __init__(self, settings: AppSettings, workspace: WorkspaceManager, checkpoints: CheckpointStore, repository: ScreenshotRepository, semantic_repository: SemanticRepository, extraction_backend: FrameExtractionBackend | None = None) -> None:
        self._settings = settings
        self._workspace = workspace
        self._checkpoints = checkpoints
        self._repository = repository
        self._semantic = semantic_repository
        self._backend = extraction_backend or FFmpegFrameExtractionBackend(str(settings.ffmpeg_path) if settings.ffmpeg_path else "ffmpeg")
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_screenshot_extractions)

    def config_fingerprint(self) -> str:
        s=self._settings
        return canonical_fingerprint({
            'algorithm':SCREENSHOT_QUALITY_ALGORITHM_VERSION,
            'sharpness_sat':s.screenshot_sharpness_saturation,
            'blur':s.screenshot_blur_score_threshold,
            'min':s.screenshot_min_quality_score,'good':s.screenshot_good_quality_score,
            'radius':s.screenshot_quality_search_radius_seconds,'step':s.screenshot_quality_search_step_seconds,
            'attempts':s.max_screenshot_quality_attempts,'search_acceptable':s.screenshot_search_on_acceptable,
            'distance_weight':s.screenshot_timestamp_distance_weight,'min_improve':s.screenshot_min_quality_improvement_to_shift,
            'weights':[s.screenshot_quality_sharpness_weight,s.screenshot_quality_contrast_weight,s.screenshot_quality_exposure_weight,s.screenshot_quality_content_weight,s.screenshot_quality_blankness_weight],
        })

    def expected_fingerprint(self, extraction_fingerprint: str, selected_sha: str | None = None) -> str:
        return canonical_fingerprint({'extraction':extraction_fingerprint,'config':self.config_fingerprint(),'selected_sha':selected_sha or ''})

    def measure(self, path: Path) -> ScreenshotQualityMetrics:
        image = decode_rgb(path)
        rgb=np.asarray(image,dtype=np.uint8)
        gray=cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        sharp_raw=float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharp=min(1.0, sharp_raw/self._settings.screenshot_sharpness_saturation)
        mean=float(gray.mean()/255.0); std=float(gray.std()/255.0)
        contrast=min(1.0, std/0.22)
        black=float(np.mean(gray<=12)); white=float(np.mean(gray>=243))
        edges=cv2.Canny(gray,80,180)
        edge_density=float(np.mean(edges>0))
        content=min(1.0, edge_density/0.10 + min(std/0.20,1.0)*0.25)
        # White/dark backgrounds are valid when meaningful edge structure is present.
        blank = ((black>0.92 or white>0.94) and edge_density<0.01 and std<0.08)
        under=mean<0.055 and edge_density<0.012 and std<0.10
        over=mean>0.96 and edge_density<0.012 and std<0.10
        exposure_state=ExposureState.UNDEREXPOSED if under else ExposureState.OVEREXPOSED if over else ExposureState.NORMAL
        exposure_score=0.0 if (under or over) else max(0.35, 1.0 - max(0.0, abs(mean-0.5)-0.32)*2.0)
        blankness=0.0 if blank else min(1.0, 0.45 + edge_density*7.0 + std*2.0)
        score=(
            sharp*self._settings.screenshot_quality_sharpness_weight +
            contrast*self._settings.screenshot_quality_contrast_weight +
            exposure_score*self._settings.screenshot_quality_exposure_weight +
            content*self._settings.screenshot_quality_content_weight +
            blankness*self._settings.screenshot_quality_blankness_weight
        )
        score=float(max(0.0,min(1.0,score)))
        blurry=sharp<self._settings.screenshot_blur_score_threshold
        if blank or under or over:
            state=ScreenshotQualityState.POOR
        elif score>=self._settings.screenshot_good_quality_score and not blurry:
            state=ScreenshotQualityState.GOOD
        elif score>=self._settings.screenshot_min_quality_score:
            state=ScreenshotQualityState.ACCEPTABLE
        else:
            state=ScreenshotQualityState.POOR
        return ScreenshotQualityMetrics(
            sharpness_raw=sharp_raw,sharpness_score=sharp,brightness_mean=mean,brightness_std=std,
            contrast_score=contrast,black_ratio=black,white_ratio=white,edge_density=edge_density,content_density=content,
            exposure_state=exposure_state,is_black_or_blank=blank,is_overexposed=over,is_underexposed=under,is_blurry=blurry,
            quality_score=score,quality_state=state,
        )

    def _search_timestamps(self, t: float, duration: float, start: float, end: float) -> list[float]:
        radius=self._settings.screenshot_quality_search_radius_seconds; step=self._settings.screenshot_quality_search_step_seconds
        values=[t]
        k=1
        while len(values)<self._settings.max_screenshot_quality_attempts and k*step<=radius+1e-9:
            for sign in (-1,1):
                x=t+sign*k*step
                if 0<=x<=duration and start-1e-6<=x<=end+1e-6:
                    values.append(round(x,6))
                    if len(values)>=self._settings.max_screenshot_quality_attempts: break
            k+=1
        return values

    async def validate_candidate(self, job_id: str, extraction: SourceScreenshotRecord, *, cancel_check: Callable[[], bool] | None = None) -> QualitySelectionRecord:
        exact_path=resolve_job_path(self._workspace,job_id,extraction.screenshot_relative_path)
        if file_sha256(exact_path)!=extraction.file_sha256:
            raise ScreenshotQualityError('Exact screenshot integrity check failed.')
        exact=self.measure(exact_path)
        semantic_input=self._semantic.get_semantic_input(job_id,extraction.candidate_id)
        should_search=exact.quality_state in {ScreenshotQualityState.POOR, ScreenshotQualityState.INVALID} or (exact.quality_state is ScreenshotQualityState.ACCEPTABLE and self._settings.screenshot_search_on_acceptable)
        attempts=[QualityAttemptRecord(timestamp_seconds=extraction.requested_timestamp_seconds,distance_seconds=0,quality_score=exact.quality_score,quality_state=exact.quality_state,selected=True)]
        selected_path=exact_path; selected_ts=extraction.requested_timestamp_seconds; selected=exact
        if should_search and self._settings.screenshot_quality_search_radius_seconds>0:
            source=resolve_job_path(self._workspace,job_id,extraction.source_video_relative_path)
            # Duration from source bounds: semantic window is the strongest drift guard; source duration can be derived from max window / requested.
            duration=max(semantic_input.stable_window_end_seconds, extraction.requested_timestamp_seconds+self._settings.screenshot_quality_search_radius_seconds)
            timestamps=self._search_timestamps(extraction.requested_timestamp_seconds,duration,semantic_input.stable_window_start_seconds,semantic_input.stable_window_end_seconds)
            best_adjusted=exact.quality_score
            for ts in timestamps[1:]:
                if cancel_check and cancel_check(): raise JobCancelledError('Screenshot quality search cancelled.')
                temp_dir=self._workspace.screenshot_quality_candidates_dir(job_id); temp_dir.mkdir(parents=True,exist_ok=True)
                temp=temp_dir/f'candidate_{extraction.candidate_id:06d}_{int(round(ts*1000)):012d}.tmp.png'
                try:
                    async with self._semaphore:
                        await asyncio.to_thread(self._backend.extract,source=source,timestamp_seconds=ts,output=temp,timeout_seconds=self._settings.source_screenshot_timeout_seconds,cancel_check=cancel_check)
                    metrics=self.measure(temp)
                    dist=abs(ts-extraction.requested_timestamp_seconds)
                    normalized_dist=dist/max(self._settings.screenshot_quality_search_radius_seconds,1e-9)
                    adjusted=metrics.quality_score-self._settings.screenshot_timestamp_distance_weight*normalized_dist
                    attempts.append(QualityAttemptRecord(timestamp_seconds=ts,distance_seconds=dist,quality_score=metrics.quality_score,quality_state=metrics.quality_state,selected=False))
                    improvement=metrics.quality_score-exact.quality_score
                    if improvement>=self._settings.screenshot_min_quality_improvement_to_shift and adjusted>best_adjusted+1e-9 and not metrics.is_black_or_blank:
                        final=self._workspace.screenshot_quality_selected_path(job_id,extraction.candidate_id)
                        final.parent.mkdir(parents=True,exist_ok=True)
                        if selected_path != exact_path and selected_path != final and selected_path.exists():
                            selected_path.unlink(missing_ok=True)
                        os.replace(temp,final)
                        selected_path=final; selected_ts=ts; selected=metrics; best_adjusted=adjusted
                    else:
                        temp.unlink(missing_ok=True)
                finally:
                    temp.unlink(missing_ok=True)
        # mark attempts selected
        for a in attempts: a.selected=abs(a.timestamp_seconds-selected_ts)<1e-7
        selected_sha=file_sha256(selected_path)
        record=QualitySelectionRecord(
            candidate_id=extraction.candidate_id,stable_window_id=extraction.stable_window_id,
            semantic_timestamp_seconds=extraction.requested_timestamp_seconds,selected_timestamp_seconds=selected_ts,
            timestamp_shift_seconds=selected_ts-extraction.requested_timestamp_seconds,
            exact_quality_score=exact.quality_score,selected_quality_score=selected.quality_score,
            selected_relative_path=self._workspace.relative_to_workspace(job_id,selected_path),selected_file_sha256=selected_sha,
            width=decode_rgb(selected_path).width,height=decode_rgb(selected_path).height,quality_state=selected.quality_state,
            metrics=selected,exact_metrics=exact,used_timestamp_fallback=abs(selected_ts-extraction.requested_timestamp_seconds)>1e-7,
            extraction_fingerprint=extraction.extraction_fingerprint,quality_config_fingerprint=self.config_fingerprint(),
            quality_fingerprint=self.expected_fingerprint(extraction.extraction_fingerprint,selected_sha),attempts=attempts,
        )
        self._repository.save_quality_record(job_id,record)
        return record

    def build_manifest(self, job_id: str, extraction_manifest, records: list[QualitySelectionRecord], *, cached_count: int=0) -> QualityManifest:
        records=sorted(records,key=lambda r:(r.selected_timestamp_seconds,r.candidate_id)); shifts=[abs(r.timestamp_shift_seconds) for r in records]
        exact=[r.exact_quality_score for r in records]; selected=[r.selected_quality_score for r in records]
        improvements=[r.selected_quality_score-r.exact_quality_score for r in records]
        fallback_search=sum(1 for r in records if len(r.attempts)>1); fallback_sel=sum(1 for r in records if r.used_timestamp_fallback)
        warnings=[]
        if records and fallback_sel/len(records)>self._settings.phase7_high_fallback_ratio: warnings.append('HIGH_SCREENSHOT_FALLBACK_RATE')
        if any(r.quality_state is ScreenshotQualityState.POOR for r in records): warnings.append('POOR_FINAL_SCREENSHOT_QUALITY')
        stats=QualityStats(candidate_count=len(records),exact_accepted_count=sum(1 for r in records if len(r.attempts)==1),fallback_search_count=fallback_search,fallback_selected_count=fallback_sel,poor_remaining_count=sum(r.quality_state is ScreenshotQualityState.POOR for r in records),invalid_count=sum(r.quality_state is ScreenshotQualityState.INVALID for r in records),cached_quality_count=cached_count,mean_absolute_timestamp_shift=statistics.mean(shifts) if shifts else 0,median_absolute_timestamp_shift=statistics.median(shifts) if shifts else 0,max_absolute_timestamp_shift=max(shifts) if shifts else 0,mean_exact_quality_score=statistics.mean(exact) if exact else 0,mean_selected_quality_score=statistics.mean(selected) if selected else 0,median_selected_quality_score=statistics.median(selected) if selected else 0,mean_quality_improvement=statistics.mean(improvements) if improvements else 0,max_quality_improvement=max(improvements) if improvements else 0,warnings=warnings)
        artifact=canonical_fingerprint({'extraction':extraction_manifest.artifact_fingerprint,'config':self.config_fingerprint(),'records':[r.quality_fingerprint for r in records]})
        manifest=QualityManifest(extraction_manifest_fingerprint=extraction_manifest.artifact_fingerprint,config_fingerprint=self.config_fingerprint(),artifact_fingerprint=artifact,stats=stats,screenshots=records)
        self._repository.save_quality_manifest(job_id,manifest); self._checkpoints.mark_completed(job_id,CP_QUALITY_SCREENSHOTS_READY); return manifest
