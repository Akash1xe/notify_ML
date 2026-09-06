from __future__ import annotations


class NotifyError(Exception):
    """Base class for expected application failures."""

    code = "notify_error"
    category = "INTERNAL"


class JobNotFoundError(NotifyError):
    code = "job_not_found"


class InvalidJobTransitionError(NotifyError):
    code = "invalid_job_transition"


class JobCancelledError(NotifyError):
    code = "job_cancelled"
    category = "CANCELLED"


class StorageError(NotifyError):
    code = "storage_error"
    category = "STORAGE"


class ProcessingError(NotifyError):
    code = "processing_error"


class InvalidYouTubeURLError(NotifyError):
    code = "invalid_youtube_url"
    category = "INPUT"


class VideoUnavailableError(NotifyError):
    code = "video_unavailable"
    category = "YOUTUBE"


class PrivateVideoError(VideoUnavailableError):
    code = "private_video"


class LiveVideoNotSupportedError(NotifyError):
    code = "live_video_not_supported"
    category = "YOUTUBE"


class MetadataExtractionError(NotifyError):
    code = "metadata_extraction_failed"
    category = "YOUTUBE"


class VideoDownloadError(NotifyError):
    code = "video_download_failed"
    category = "DOWNLOAD"


class DownloadCancelledError(JobCancelledError):
    code = "download_cancelled"


class InsufficientDiskSpaceError(NotifyError):
    code = "insufficient_disk_space"
    category = "STORAGE"


class DownloadOutputMissingError(VideoDownloadError):
    code = "download_output_missing"


class FFmpegNotFoundError(NotifyError):
    code = "ffmpeg_not_found"
    category = "MEDIA"


class FFprobeNotFoundError(NotifyError):
    code = "ffprobe_not_found"
    category = "MEDIA"


class MediaInspectionError(NotifyError):
    code = "media_inspection_failed"
    category = "MEDIA"


class MediaProbeTimeoutError(MediaInspectionError):
    code = "media_probe_timeout"


class CorruptMediaError(MediaInspectionError):
    code = "corrupt_media"


class VideoStreamMissingError(MediaInspectionError):
    code = "video_stream_missing"


class AudioStreamMissingError(NotifyError):
    code = "audio_stream_missing"
    category = "AUDIO"


class AudioExtractionError(NotifyError):
    code = "audio_extraction_failed"
    category = "AUDIO"


class AudioExtractionTimeoutError(AudioExtractionError):
    code = "audio_extraction_timeout"


class AudioValidationError(AudioExtractionError):
    code = "audio_validation_failed"


class AudioOutputMissingError(AudioExtractionError):
    code = "audio_output_missing"


class CacheValidationError(NotifyError):
    code = "cache_validation_failed"
    category = "STORAGE"


class CleanupError(NotifyError):
    code = "cleanup_failed"
    category = "STORAGE"


class FrameSamplingError(NotifyError):
    code = "frame_sampling_failed"
    category = "FRAME_ANALYSIS"


class FrameSamplingTimeoutError(FrameSamplingError):
    code = "frame_sampling_timeout"


class FrameOutputMissingError(FrameSamplingError):
    code = "frame_output_missing"


class FramePreprocessingError(NotifyError):
    code = "frame_preprocessing_failed"
    category = "FRAME_ANALYSIS"


class TooManyInvalidFramesError(FramePreprocessingError):
    code = "too_many_invalid_frames"


class DifferenceAnalysisError(NotifyError):
    code = "difference_analysis_failed"
    category = "FRAME_ANALYSIS"


class TooManyInvalidComparisonsError(DifferenceAnalysisError):
    code = "too_many_invalid_comparisons"


class MajorChangeAnalysisError(NotifyError):
    code = "major_change_analysis_failed"
    category = "FRAME_ANALYSIS"


class TimelineGenerationError(NotifyError):
    code = "timeline_generation_failed"
    category = "FRAME_ANALYSIS"


class FrameAnalysisValidationError(NotifyError):
    code = "frame_analysis_validation_failed"
    category = "FRAME_ANALYSIS"
