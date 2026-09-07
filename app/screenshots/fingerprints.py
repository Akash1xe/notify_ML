from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from app.core.config import AppSettings
from app.core.exceptions import JobCancelledError, VisualFingerprintError
from app.jobs.checkpoints import CheckpointStore
from app.screenshots.models import (
    VISUAL_FINGERPRINT_ALGORITHM_VERSION,
    FingerprintManifest,
    FingerprintStats,
    QualityManifest,
    QualitySelectionRecord,
    VisualFingerprintRecord,
)
from app.screenshots.repository import ScreenshotRepository
from app.screenshots.utils import atomic_save_png, canonical_fingerprint, decode_rgb, file_sha256, resolve_job_path
from app.storage.workspace import WorkspaceManager

CP_VISUAL_FINGERPRINTS_READY='VISUAL_FINGERPRINTS_READY'


def _bits_to_hex(bits: np.ndarray) -> str:
    flat=np.asarray(bits,dtype=np.uint8).reshape(-1)
    pad=(-len(flat))%4
    if pad: flat=np.pad(flat,(0,pad))
    return ''.join(f'{int("".join(str(int(v)) for v in flat[i:i+4]),2):x}' for i in range(0,len(flat),4))


def compute_phash(image: Image.Image, hash_size: int=8) -> str:
    gray=np.asarray(image.convert('L').resize((hash_size*4,hash_size*4),Image.Resampling.LANCZOS),dtype=np.float32)
    dct=cv2.dct(gray)
    low=dct[:hash_size,:hash_size].copy(); vals=low.flatten()[1:]
    median=float(np.median(vals)) if vals.size else 0.0
    return _bits_to_hex(low>median)


def compute_dhash(image: Image.Image, hash_size: int=8) -> str:
    arr=np.asarray(image.convert('L').resize((hash_size+1,hash_size),Image.Resampling.LANCZOS),dtype=np.uint8)
    return _bits_to_hex(arr[:,1:]>arr[:,:-1])


def hamming_distance_hex(a: str,b: str) -> int:
    if len(a)!=len(b): raise ValueError('hash lengths differ')
    return (int(a,16)^int(b,16)).bit_count()


def normalized_hamming_distance(a: str,b: str) -> float:
    return hamming_distance_hex(a,b)/max(1,len(a)*4)


class VisualFingerprintGenerator:
    def __init__(self,settings:AppSettings,workspace:WorkspaceManager,checkpoints:CheckpointStore,repository:ScreenshotRepository)->None:
        self._settings=settings; self._workspace=workspace; self._checkpoints=checkpoints; self._repository=repository

    def config_fingerprint(self)->str:
        s=self._settings
        return canonical_fingerprint({'algorithm':VISUAL_FINGERPRINT_ALGORITHM_VERSION,'phash':s.visual_phash_size,'dhash':s.visual_dhash_size,'long_edge':s.visual_comparison_long_edge,'edge_low':s.visual_edge_low_threshold,'edge_high':s.visual_edge_high_threshold})

    def expected_fingerprint(self,record:QualitySelectionRecord)->str:
        return canonical_fingerprint({'sha':record.selected_file_sha256,'config':self.config_fingerprint()})

    def generate_candidate(self,job_id:str,quality:QualitySelectionRecord,*,cancel_check:Callable[[],bool]|None=None)->VisualFingerprintRecord:
        if cancel_check and cancel_check(): raise JobCancelledError('Fingerprint generation cancelled.')
        source=resolve_job_path(self._workspace,job_id,quality.selected_relative_path)
        if file_sha256(source)!=quality.selected_file_sha256: raise VisualFingerprintError('Quality-selected screenshot integrity mismatch.')
        image=decode_rgb(source); phash=compute_phash(image,self._settings.visual_phash_size); dhash=compute_dhash(image,self._settings.visual_dhash_size)
        scale=min(1.0,self._settings.visual_comparison_long_edge/max(image.width,image.height)); size=(max(1,round(image.width*scale)),max(1,round(image.height*scale)))
        gray=image.convert('L').resize(size,Image.Resampling.LANCZOS)
        gray_path=self._workspace.screenshot_gray_thumbnail_path(job_id,quality.candidate_id); atomic_save_png(gray.convert('RGB'),gray_path)
        arr=np.asarray(gray,dtype=np.uint8); edges=cv2.Canny(arr,self._settings.visual_edge_low_threshold,self._settings.visual_edge_high_threshold)
        edge_img=Image.fromarray(edges,mode='L').convert('RGB'); edge_path=self._workspace.screenshot_edge_map_path(job_id,quality.candidate_id); atomic_save_png(edge_img,edge_path)
        edge_hash=compute_dhash(edge_img,self._settings.visual_dhash_size)
        rec=VisualFingerprintRecord(candidate_id=quality.candidate_id,stable_window_id=quality.stable_window_id,selected_timestamp_seconds=quality.selected_timestamp_seconds,screenshot_relative_path=quality.selected_relative_path,source_file_sha256=quality.selected_file_sha256,phash=phash,dhash=dhash,edge_hash=edge_hash,thumbnail_relative_path=self._workspace.relative_to_workspace(job_id,gray_path),thumbnail_file_sha256=file_sha256(gray_path),edge_map_relative_path=self._workspace.relative_to_workspace(job_id,edge_path),edge_map_file_sha256=file_sha256(edge_path),width=quality.width,height=quality.height,fingerprint_config_fingerprint=self.config_fingerprint(),visual_fingerprint=self.expected_fingerprint(quality))
        self._repository.save_fingerprint_record(job_id,rec); return rec

    def build_manifest(self,job_id:str,quality_manifest:QualityManifest,records:list[VisualFingerprintRecord],*,cached_count:int=0)->FingerprintManifest:
        records=sorted(records,key=lambda r:(r.selected_timestamp_seconds,r.candidate_id)); ph=[r.phash for r in records]; dh=[r.dhash for r in records]
        stats=FingerprintStats(candidate_count=len(quality_manifest.screenshots),fingerprint_count=len(records),cached_fingerprint_count=cached_count,failed_fingerprint_count=max(0,len(quality_manifest.screenshots)-len(records)),identical_phash_count=len(ph)-len(set(ph)),identical_dhash_count=len(dh)-len(set(dh)),total_thumbnail_bytes=sum(resolve_job_path(self._workspace,job_id,r.thumbnail_relative_path).stat().st_size for r in records),warnings=['HIGH_IDENTICAL_PERCEPTUAL_HASH_RATE'] if records and (len(ph)-len(set(ph)))/len(records)>0.5 else [])
        artifact=canonical_fingerprint({'quality':quality_manifest.artifact_fingerprint,'config':self.config_fingerprint(),'records':[r.visual_fingerprint for r in records]})
        manifest=FingerprintManifest(quality_manifest_fingerprint=quality_manifest.artifact_fingerprint,config_fingerprint=self.config_fingerprint(),artifact_fingerprint=artifact,stats=stats,fingerprints=records)
        self._repository.save_fingerprint_manifest(job_id,manifest); self._checkpoints.mark_completed(job_id,CP_VISUAL_FINGERPRINTS_READY); return manifest
