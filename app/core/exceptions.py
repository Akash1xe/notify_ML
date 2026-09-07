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


class CandidateAnalysisError(NotifyError):
    code = "candidate_analysis_failed"
    category = "CANDIDATE_ANALYSIS"


class StabilityWindowDetectionError(CandidateAnalysisError):
    code = "stability_window_detection_failed"


class BoundaryDetectionError(CandidateAnalysisError):
    code = "boundary_detection_failed"


class CandidateGenerationError(CandidateAnalysisError):
    code = "candidate_generation_failed"


class CandidateHeuristicsError(CandidateAnalysisError):
    code = "candidate_heuristics_failed"


class CandidateRankingError(CandidateAnalysisError):
    code = "candidate_ranking_failed"


class CandidateAnalysisValidationError(CandidateAnalysisError):
    code = "candidate_analysis_validation_failed"


class TranscriptionError(NotifyError):
    code = "transcription_failed"
    category = "TRANSCRIPTION"


class AudioPreparationError(TranscriptionError):
    code = "audio_preparation_failed"


class TranscriptionModelLoadError(TranscriptionError):
    code = "transcription_model_load_failed"


class AudioChunkTranscriptionError(TranscriptionError):
    code = "audio_chunk_transcription_failed"


class RawTranscriptionError(TranscriptionError):
    code = "raw_transcription_failed"


class TranscriptNormalizationError(TranscriptionError):
    code = "transcript_normalization_failed"


class TranscriptAlignmentError(TranscriptionError):
    code = "transcript_alignment_failed"


class TranscriptContextError(TranscriptionError):
    code = "transcript_context_failed"


class Phase5ValidationError(TranscriptionError):
    code = "phase5_validation_failed"


class SemanticAnalysisError(NotifyError):
    code = "semantic_analysis_failed"
    category = "SEMANTIC_ANALYSIS"


class SemanticInputError(SemanticAnalysisError):
    code = "semantic_input_failed"


class TemporalContextError(SemanticAnalysisError):
    code = "temporal_context_failed"


class VLMRuntimeError(SemanticAnalysisError):
    code = "vlm_runtime_failed"


class VLMDependencyMissingError(VLMRuntimeError):
    code = "vlm_dependency_missing"


class VLMModelLoadError(VLMRuntimeError):
    code = "vlm_model_load_failed"


class VLMModelNotFoundError(VLMModelLoadError):
    code = "vlm_model_not_found"


class VLMUnsupportedConfigurationError(VLMRuntimeError):
    code = "vlm_unsupported_configuration"


class VLMOutOfMemoryError(VLMRuntimeError):
    code = "vlm_out_of_memory"


class VLMInferenceError(VLMRuntimeError):
    code = "vlm_inference_failed"


class VLMImageInputError(VLMRuntimeError):
    code = "vlm_image_input_error"


class VLMCancelledError(JobCancelledError):
    code = "vlm_cancelled"


class SemanticDecisionError(SemanticAnalysisError):
    code = "semantic_decision_failed"


class Phase6ValidationError(SemanticAnalysisError):
    code = "phase6_validation_failed"
