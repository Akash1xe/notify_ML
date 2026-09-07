from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from app.core.exceptions import ScreenshotDecodeError, ScreenshotPathError
from app.storage.workspace import WorkspaceManager
from app.video_analysis.fingerprints import stable_hash


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_job_path(workspace: WorkspaceManager, job_id: str, relative_path: str, *, must_exist: bool = True) -> Path:
    root = workspace.workspace(job_id)
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ScreenshotPathError('Screenshot artifact path escaped the job workspace.') from exc
    if must_exist and (not path.exists() or not path.is_file() or path.stat().st_size <= 0):
        raise ScreenshotPathError(f'Missing screenshot artifact: {relative_path}')
    return path


def decode_rgb(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image.load()
            return image.convert('RGB')
    except Exception as exc:
        raise ScreenshotDecodeError(f'Unable to decode screenshot: {path.name}') from exc


def image_dimensions(path: Path) -> tuple[int, int]:
    image = decode_rgb(path)
    return image.width, image.height


def atomic_save_png(image: Image.Image, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_path.with_name(final_path.stem + '.tmp' + final_path.suffix)
    tmp.unlink(missing_ok=True)
    try:
        image.save(tmp, format='PNG', optimize=False)
        # Decode before promotion.
        decode_rgb(tmp)
        tmp.replace(final_path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def canonical_fingerprint(value: Any) -> str:
    return stable_hash(value)


def grayscale_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert('L'), dtype=np.uint8)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), p))
