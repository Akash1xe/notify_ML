from __future__ import annotations

import re
import statistics
import unicodedata
from pathlib import Path
from typing import Iterable

from PIL import Image

from app.core.exceptions import DocumentInputIntegrityError
from app.screenshots.utils import canonical_fingerprint, file_sha256
from app.storage.workspace import WorkspaceManager


def resolve_document_artifact_path(
    workspace: WorkspaceManager,
    job_id: str,
    relative_path: str,
    *,
    must_exist: bool = True,
) -> Path:
    if not relative_path or Path(relative_path).is_absolute():
        raise DocumentInputIntegrityError("Document artifact path must be a safe relative path.")
    root = workspace.workspace(job_id)
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise DocumentInputIntegrityError("Document artifact path escaped the job workspace.") from exc
    if must_exist and (not path.exists() or not path.is_file() or path.stat().st_size <= 0):
        raise DocumentInputIntegrityError(f"Document dependency is missing: {relative_path}")
    return path


def image_info(path: Path) -> tuple[int, int, str]:
    try:
        with Image.open(path) as image:
            image.load()
            return image.width, image.height, (image.format or "UNKNOWN").upper()
    except Exception as exc:
        raise DocumentInputIntegrityError(f"Unable to decode document image: {path.name}") from exc


def format_timestamp(seconds: float) -> str:
    whole = max(0, int(seconds))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def mm_to_points(value: float) -> float:
    return value * 72.0 / 25.4


def content_type_display_name(value: str) -> str:
    mapping = {
        "SLIDE": "Slide",
        "WHITEBOARD": "Whiteboard",
        "BLACKBOARD": "Blackboard",
        "CODE": "Code",
        "DIAGRAM": "Diagram",
        "EQUATION": "Equation",
        "DOCUMENT": "Document",
        "UI_DEMO": "UI Demo",
        "MIXED": "Mixed",
        "OTHER": "Other",
        "EMPTY_OR_LOW_INFORMATION": "Low information",
        "UNKNOWN": "Unknown",
    }
    return mapping.get(str(value).upper(), "Unknown")


def normalize_plain_text(value: str) -> str:
    text = unicodedata.normalize("NFC", value or "")
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    return re.sub(r"\s+", " ", text).strip()


def truncate_words(text: str, *, max_characters: int, max_words: int) -> tuple[str, bool]:
    text = normalize_plain_text(text)
    words = text.split()
    truncated = False
    if len(words) > max_words:
        words = words[:max_words]
        truncated = True
    result = " ".join(words)
    if len(result) > max_characters:
        clipped = result[:max_characters].rstrip()
        if " " in clipped:
            clipped = clipped.rsplit(" ", 1)[0]
        result = clipped
        truncated = True
    if truncated and result:
        result = result.rstrip(" .,…") + "…"
    return result, truncated


def wrap_approx(text: str, *, width_points: float, font_size: float, max_lines: int) -> tuple[str, bool]:
    """Backend-independent conservative wrapping used only for render planning."""
    text = normalize_plain_text(text)
    if not text:
        return "", False
    max_chars = max(1, int(width_points / max(font_size * 0.58, 1.0)))
    tokens: list[str] = []
    for word in text.split():
        if len(word) <= max_chars:
            tokens.append(word)
        else:
            tokens.extend(word[i : i + max_chars] for i in range(0, len(word), max_chars))
    lines: list[str] = []
    current = ""
    truncated = False
    for token in tokens:
        candidate = token if not current else f"{current} {token}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = token
        if len(lines) >= max_lines:
            truncated = True
            current = ""
            break
    if current:
        if len(lines) < max_lines:
            lines.append(current)
        else:
            truncated = True
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    if truncated and lines:
        room = max(1, max_chars - 1)
        lines[-1] = lines[-1][:room].rstrip(" .,…") + "…"
    return "\n".join(lines), truncated


def safe_download_filename(title: str, *, max_length: int = 120) -> str:
    text = normalize_plain_text(title or "Lecture Notes")
    text = text.replace("/", " ").replace("\\", " ")
    text = re.sub(r"[^\w\- .()]+", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "-", text).strip("-._ ")
    if not text:
        text = "lecture-notes"
    if text.lower().endswith(".pdf"):
        text = text[:-4]
    text = text[:max_length].rstrip("-._ ") or "lecture-notes"
    return f"{text}.pdf"


def mean_or_none(values: Iterable[float]) -> float | None:
    data = list(values)
    return float(statistics.mean(data)) if data else None


def median_or_none(values: Iterable[float]) -> float | None:
    data = list(values)
    return float(statistics.median(data)) if data else None


__all__ = [
    "canonical_fingerprint",
    "file_sha256",
    "resolve_document_artifact_path",
    "image_info",
    "format_timestamp",
    "mm_to_points",
    "content_type_display_name",
    "normalize_plain_text",
    "truncate_words",
    "wrap_approx",
    "safe_download_filename",
    "mean_or_none",
    "median_or_none",
]
