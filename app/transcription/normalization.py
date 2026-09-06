from __future__ import annotations

import math
import re
import statistics
import time
from difflib import SequenceMatcher
from typing import Callable

from app.core.config import AppSettings
from app.core.exceptions import JobCancelledError, TranscriptNormalizationError
from app.transcription.models import (
    NormalizationFlag,
    NormalizedTranscriptManifest,
    NormalizedTranscriptSegment,
    RawTranscriptManifest,
    RawTranscriptSegment,
    TranscriptStats,
)
from app.transcription.repository import TranscriptionRepository
from app.video_analysis.fingerprints import stable_hash

TRANSCRIPT_NORMALIZATION_ALGORITHM_VERSION = "1"
CP_TRANSCRIPT_NORMALIZED = "TRANSCRIPT_NORMALIZED"

ProgressCallback = Callable[[float], None]
CancelCheck = Callable[[], bool]


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _comparison_text(text: str) -> str:
    cleaned = _clean_text(text).lower()
    return re.sub(r"[^\w\s]", "", cleaned, flags=re.UNICODE)


def _similarity(a: str, b: str) -> float:
    aa, bb = _comparison_text(a), _comparison_text(b)
    if not aa or not bb:
        return 0.0
    if aa in bb or bb in aa:
        return 1.0
    return SequenceMatcher(None, aa, bb, autojunk=False).ratio()


class TranscriptNormalizationService:
    def __init__(self, settings: AppSettings, repository: TranscriptionRepository) -> None:
        self._settings = settings
        self._repository = repository

    def config_fingerprint(self) -> str:
        s = self._settings
        return stable_hash(
            {
                "algorithm_version": TRANSCRIPT_NORMALIZATION_ALGORITHM_VERSION,
                "overlap_time_tolerance": s.transcript_overlap_time_tolerance_seconds,
                "duplicate_similarity": s.transcript_duplicate_similarity_threshold,
                "merge_gap": s.transcript_merge_max_gap_seconds,
                "max_duration": s.transcript_max_segment_duration_seconds,
                "max_characters": s.transcript_max_segment_characters,
                "short_max_duration": s.transcript_short_segment_max_duration_seconds,
            }
        )

    @staticmethod
    def _winner_key(segment: RawTranscriptSegment) -> tuple:
        return (
            1 if segment.is_within_logical_range else 0,
            -(segment.no_speech_prob if segment.no_speech_prob is not None else 1.0),
            segment.avg_logprob if segment.avg_logprob is not None else -999.0,
            len(_clean_text(segment.text)),
            # deterministic fallback prefers earlier source id lexically by negating elsewhere
        )

    def _is_duplicate(self, a: RawTranscriptSegment, b: RawTranscriptSegment) -> bool:
        tolerance = self._settings.transcript_overlap_time_tolerance_seconds
        time_close = not (
            a.absolute_end_seconds < b.absolute_start_seconds - tolerance
            or b.absolute_end_seconds < a.absolute_start_seconds - tolerance
        )
        if not time_close:
            return False
        overlap = max(
            0.0,
            min(a.absolute_end_seconds, b.absolute_end_seconds)
            - max(a.absolute_start_seconds, b.absolute_start_seconds),
        )
        shorter = min(
            max(0.001, a.absolute_end_seconds - a.absolute_start_seconds),
            max(0.001, b.absolute_end_seconds - b.absolute_start_seconds),
        )
        temporal_support = overlap / shorter
        return (
            temporal_support >= 0.25
            and _similarity(a.text, b.text) >= self._settings.transcript_duplicate_similarity_threshold
        )

    def _split_long(self, item: dict) -> list[dict]:
        duration = item["end"] - item["start"]
        if duration <= self._settings.transcript_max_segment_duration_seconds:
            return [item]
        words = item["words"]
        if not words:
            item["flags"].add(NormalizationFlag.LONG_SEGMENT_PRESERVED.value)
            return [item]
        groups: list[list] = []
        current: list = []
        group_start = None
        chars = 0
        for word in words:
            if group_start is None:
                group_start = word.start_seconds
            projected_duration = word.end_seconds - group_start
            projected_chars = chars + len(word.word)
            if current and (
                projected_duration > self._settings.transcript_max_segment_duration_seconds
                or projected_chars > self._settings.transcript_max_segment_characters
            ):
                groups.append(current)
                current = []
                group_start = word.start_seconds
                chars = 0
            current.append(word)
            chars += len(word.word)
        if current:
            groups.append(current)
        if len(groups) <= 1:
            item["flags"].add(NormalizationFlag.LONG_SEGMENT_PRESERVED.value)
            return [item]
        split: list[dict] = []
        for group in groups:
            text = "".join(w.word for w in group).strip()
            if not text:
                continue
            split.append(
                {
                    **item,
                    "start": group[0].start_seconds,
                    "end": group[-1].end_seconds,
                    "text": _clean_text(text),
                    "words": group,
                    "flags": set(item["flags"]),
                }
            )
        return split or [item]

    def process(
        self,
        job_id: str,
        *,
        raw: RawTranscriptManifest,
        progress_callback: ProgressCallback = lambda _: None,
        cancel_check: CancelCheck = lambda: False,
    ) -> NormalizedTranscriptManifest:
        started = time.monotonic()
        ordered = sorted(
            raw.segments,
            key=lambda s: (s.absolute_start_seconds, s.absolute_end_seconds, s.chunk_id, s.source_ordinal),
        )
        accepted: list[dict] = []
        duplicate_count = 0
        rejected_count = 0
        for idx, segment in enumerate(ordered):
            if cancel_check():
                raise JobCancelledError("Transcript normalization was cancelled.")
            text = _clean_text(segment.text)
            if not text:
                rejected_count += 1
                continue
            item = {
                "raw": segment,
                "start": segment.absolute_start_seconds,
                "end": segment.absolute_end_seconds,
                "text": text,
                "source_ids": [segment.segment_id],
                "chunk_ids": [segment.chunk_id],
                "words": list(segment.words),
                "avg_logprob": segment.avg_logprob,
                "no_speech_prob": segment.no_speech_prob,
                "flags": ({NormalizationFlag.TRIMMED_WHITESPACE.value} if text != segment.text else set()),
            }
            duplicate_index = None
            for j in range(len(accepted) - 1, -1, -1):
                prior_raw = accepted[j]["raw"]
                if segment.absolute_start_seconds - prior_raw.absolute_end_seconds > self._settings.transcript_overlap_time_tolerance_seconds:
                    break
                if self._is_duplicate(prior_raw, segment):
                    duplicate_index = j
                    break
            if duplicate_index is None:
                accepted.append(item)
            else:
                duplicate_count += 1
                prior = accepted[duplicate_index]
                prior_raw = prior["raw"]
                choose_new = self._winner_key(segment) > self._winner_key(prior_raw)
                winner = item if choose_new else prior
                loser = prior if choose_new else item
                winner["source_ids"] = list(dict.fromkeys(winner["source_ids"] + loser["source_ids"]))
                winner["chunk_ids"] = list(dict.fromkeys(winner["chunk_ids"] + loser["chunk_ids"]))
                winner["flags"].add(NormalizationFlag.DEDUPLICATED_OVERLAP.value)
                accepted[duplicate_index] = winner
            if ordered:
                progress_callback((idx + 1) * 50.0 / len(ordered))

        split_items: list[dict] = []
        for item in accepted:
            split_items.extend(self._split_long(item))
        split_items.sort(key=lambda x: (x["start"], x["end"], x["source_ids"]))

        merged: list[dict] = []
        merge_count = 0
        for item in split_items:
            if cancel_check():
                raise JobCancelledError("Transcript normalization was cancelled.")
            if not merged:
                merged.append(item)
                continue
            prev = merged[-1]
            gap = item["start"] - prev["end"]
            combined_duration = max(prev["end"], item["end"]) - prev["start"]
            combined_text = _clean_text(prev["text"] + " " + item["text"])
            should_merge = (
                gap >= -0.05
                and gap <= self._settings.transcript_merge_max_gap_seconds
                and combined_duration <= self._settings.transcript_max_segment_duration_seconds
                and len(combined_text) <= self._settings.transcript_max_segment_characters
            )
            if should_merge:
                merge_count += 1
                prev["end"] = max(prev["end"], item["end"])
                prev["text"] = combined_text
                prev["source_ids"] = list(dict.fromkeys(prev["source_ids"] + item["source_ids"]))
                prev["chunk_ids"] = list(dict.fromkeys(prev["chunk_ids"] + item["chunk_ids"]))
                prev["words"] = sorted(prev["words"] + item["words"], key=lambda w: (w.start_seconds, w.end_seconds))
                vals = [x for x in (prev["avg_logprob"], item["avg_logprob"]) if x is not None]
                prev["avg_logprob"] = sum(vals) / len(vals) if vals else None
                ns = [x for x in (prev["no_speech_prob"], item["no_speech_prob"]) if x is not None]
                prev["no_speech_prob"] = sum(ns) / len(ns) if ns else None
                prev["flags"].update(item["flags"])
                prev["flags"].add(NormalizationFlag.MERGED_ADJACENT.value)
            else:
                merged.append(item)

        normalized: list[NormalizedTranscriptSegment] = []
        for index, item in enumerate(merged, 1):
            normalized.append(
                NormalizedTranscriptSegment(
                    segment_id=index,
                    start_seconds=max(0.0, item["start"]),
                    end_seconds=max(item["start"], item["end"]),
                    text=item["text"],
                    source_segment_ids=item["source_ids"],
                    source_chunk_ids=item["chunk_ids"],
                    word_count=len(item["text"].split()),
                    language=raw.language,
                    avg_logprob=item["avg_logprob"],
                    no_speech_prob=item["no_speech_prob"],
                    normalization_flags=sorted(item["flags"]),
                    words=item["words"],
                )
            )

        if any(a.end_seconds > raw.stats.audio_duration_seconds + 0.25 for a in normalized):
            raise TranscriptNormalizationError("Normalized transcript contains timestamps outside the lecture.")
        intervals = [(s.start_seconds, min(s.end_seconds, raw.stats.audio_duration_seconds)) for s in normalized]
        union = 0.0
        if intervals:
            cur_start, cur_end = intervals[0]
            for start_i, end_i in intervals[1:]:
                if start_i <= cur_end:
                    cur_end = max(cur_end, end_i)
                else:
                    union += max(0.0, cur_end - cur_start)
                    cur_start, cur_end = start_i, end_i
            union += max(0.0, cur_end - cur_start)
        gaps = [max(0.0, b.start_seconds - a.end_seconds) for a, b in zip(normalized, normalized[1:])]
        stats = TranscriptStats(
            raw_segment_count=len(raw.segments),
            normalized_segment_count=len(normalized),
            deduplicated_segment_count=duplicate_count,
            merged_segment_count=merge_count,
            rejected_segment_count=rejected_count,
            word_count=sum(s.word_count for s in normalized),
            speech_start_seconds=(normalized[0].start_seconds if normalized else None),
            speech_end_seconds=(normalized[-1].end_seconds if normalized else None),
            total_transcribed_span_seconds=union,
            speech_coverage_ratio=(min(1.0, union / raw.stats.audio_duration_seconds) if raw.stats.audio_duration_seconds > 0 else 0.0),
            mean_gap_seconds=(statistics.mean(gaps) if gaps else 0.0),
            median_gap_seconds=(statistics.median(gaps) if gaps else 0.0),
            max_gap_seconds=(max(gaps) if gaps else 0.0),
        )
        languages = sorted({s.language for s in normalized if s.language})
        warnings: list[str] = []
        if raw.segments and duplicate_count / len(raw.segments) > 0.5:
            warnings.append("HIGH_OVERLAP_DUPLICATION")
        if raw.segments and rejected_count / len(raw.segments) > 0.5:
            warnings.append("HIGH_REJECTED_SEGMENT_RATIO")
        core = {
            "raw": raw.artifact_fingerprint,
            "config": self.config_fingerprint(),
            "segments": [s.model_dump(mode="json") for s in normalized],
            "stats": stats.model_dump(mode="json"),
        }
        manifest = NormalizedTranscriptManifest(
            algorithm_version=TRANSCRIPT_NORMALIZATION_ALGORITHM_VERSION,
            raw_transcript_fingerprint=raw.artifact_fingerprint,
            config_fingerprint=self.config_fingerprint(),
            artifact_fingerprint=stable_hash(core),
            primary_language=raw.language,
            observed_languages=languages,
            warnings=warnings,
            stats=stats,
            segments=normalized,
        )
        self._repository.save_transcript(job_id, manifest)
        progress_callback(100.0)
        _ = started
        return manifest
