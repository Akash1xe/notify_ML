from __future__ import annotations

import os
import platform
import shutil
import subprocess
from enum import Enum

import psutil
from pydantic import BaseModel

from app.core.config import AppSettings
from app.media.models import MediaToolsCapabilities
from app.media.tools import MediaToolsService


class ProcessingProfile(str, Enum):
    LOW = "LOW"
    BALANCED = "BALANCED"
    QUALITY = "QUALITY"


class SystemCapabilities(BaseModel):
    operating_system: str
    cpu_architecture: str
    logical_cpu_count: int
    memory_gb: float
    gpu_available: bool
    gpu_name: str | None
    cuda_available: bool
    recommended_profile: ProcessingProfile
    media_tools: MediaToolsCapabilities | None = None


def _detect_nvidia_gpu() -> tuple[bool, str | None]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return False, None
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return (bool(names), names[0] if names else None)
    except (OSError, subprocess.SubprocessError):
        return False, None


def get_system_capabilities(settings: AppSettings | None = None) -> SystemCapabilities:
    gpu_available, gpu_name = _detect_nvidia_gpu()
    memory_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    cpu_count = os.cpu_count() or 1

    if gpu_available and memory_gb >= 16:
        profile = ProcessingProfile.QUALITY
    elif memory_gb >= 8 and cpu_count >= 4:
        profile = ProcessingProfile.BALANCED
    else:
        profile = ProcessingProfile.LOW

    media_tools = None
    if settings is not None:
        media_tools = MediaToolsService(settings).capabilities()

    return SystemCapabilities(
        operating_system=platform.system(),
        cpu_architecture=platform.machine(),
        logical_cpu_count=cpu_count,
        memory_gb=memory_gb,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        cuda_available=gpu_available,
        recommended_profile=profile,
        media_tools=media_tools,
    )
