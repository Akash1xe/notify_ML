from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from app.core.config import AppSettings
from app.core.exceptions import JobCancelledError, Phase6ValidationError, SemanticAnalysisError
from app.jobs.checkpoints import CheckpointStore
from app.semantic_analysis.models import (
    CandidateSemanticArtifact,
    CompletionState,
    SEMANTIC_ANALYSIS_ALGORITHM_VERSION,
    SEMANTIC_PROMPT_VERSION,
    SemanticAnalysisStats,
    SemanticParseStatus,
    SemanticResultsManifest,
    VLGenerationConfig,
)
from app.semantic_analysis.parsing import parse_semantic_result
from app.semantic_analysis.prompts import SemanticPromptBuilder
from app.semantic_analysis.repository import SemanticRepository
from app.semantic_analysis.runtime import QwenRuntimeManager
from app.storage.workspace import WorkspaceManager
from app.video_analysis.fingerprints import stable_hash

CP_SEMANTIC_ANALYSIS_COMPLETE = "SEMANTIC_ANALYSIS_COMPLETE"
CancelCheck = Callable[[], bool]
ProgressCallback = Callable[[int], None]


class SemanticCandidateAnalyzer:
    def __init__(
        self,
        settings: AppSettings,
        workspace: WorkspaceManager,
        checkpoints: CheckpointStore,
        repository: SemanticRepository,
        runtime: QwenRuntimeManager,
        prompt_builder: SemanticPromptBuilder,
    ) -> None:
        self._settings = settings
        self._workspace = workspace
        self._checkpoints = checkpoints
        self._repository = repository
        self._runtime = runtime
        self._prompts = prompt_builder

    def config_fingerprint(self) -> str:
        return stable_hash(
            {
                "algorithm_version": SEMANTIC_ANALYSIS_ALGORITHM_VERSION,
                "prompt_version": SEMANTIC_PROMPT_VERSION,
                "max_prompt_chars": self._settings.semantic_prompt_max_characters,
                "invalid_output_retries": self._settings.semantic_invalid_output_retries,
                "generation": self.generation_config().model_dump(mode="json"),
            }
        )

    def generation_config(self) -> VLGenerationConfig:
        return VLGenerationConfig(
            max_new_tokens=self._settings.qwen_vl_max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            repetition_penalty=1.0,
        )

    def inference_config_fingerprint(self) -> str:
        return stable_hash(
            {
                "model_tier": self._settings.qwen_vl_model_tier,
                "model_name": self._settings.qwen_vl_model_name,
                "fallback_tier": self._settings.qwen_vl_fallback_model_tier,
                "fallback_enabled": self._settings.qwen_vl_enable_fallback,
                "revision": self._settings.qwen_vl_model_revision,
                "device": self._settings.qwen_vl_device,
                "dtype": self._settings.qwen_vl_dtype,
                "quantization": self._settings.qwen_vl_quantization,
                "max_edge": self._settings.qwen_vl_max_image_long_edge,
                "prompt_version": SEMANTIC_PROMPT_VERSION,
                "analysis_version": SEMANTIC_ANALYSIS_ALGORITHM_VERSION,
                "generation": self.generation_config().model_dump(mode="json"),
            }
        )

    def expected_candidate_fingerprint(self, semantic_input, context) -> str:
        return stable_hash(
            {
                "semantic_input_fingerprint": semantic_input.semantic_input_fingerprint,
                "temporal_context_fingerprint": context.temporal_context_fingerprint,
                "prompt_fingerprint": self._prompts.fingerprint(semantic_input, context),
                "inference_config_fingerprint": self.inference_config_fingerprint(),
                "algorithm_version": SEMANTIC_ANALYSIS_ALGORITHM_VERSION,
            }
        )

    def cached_candidate_valid(self, job_id: str, semantic_input, context) -> CandidateSemanticArtifact | None:
        try:
            artifact = self._repository.load_candidate_artifact(job_id, semantic_input.candidate_id)
        except Exception:
            return None
        expected = self.expected_candidate_fingerprint(semantic_input, context)
        if artifact.artifact_fingerprint != expected:
            return None
        if artifact.semantic_input_fingerprint != semantic_input.semantic_input_fingerprint:
            return None
        if artifact.temporal_context_fingerprint != context.temporal_context_fingerprint:
            return None
        if artifact.inference_fingerprint != self.inference_config_fingerprint():
            return None
        if artifact.prompt_version != SEMANTIC_PROMPT_VERSION:
            return None
        if artifact.parse_status is not SemanticParseStatus.SUCCESS or artifact.semantic_result is None:
            return None
        return artifact

    def _safe_image_path(self, job_id: str, relative_path: str) -> Path:
        root = self._workspace.workspace(job_id)
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise SemanticAnalysisError("Semantic image path is unsafe.") from exc
        if not path.is_file():
            raise SemanticAnalysisError("Semantic image is missing.")
        return path

    def _load_images(self, job_id: str, context) -> list[Image.Image]:
        frames = []
        if context.previous:
            frames.append(context.previous)
        frames.append(context.current)
        if context.next:
            frames.append(context.next)
        images: list[Image.Image] = []
        try:
            for frame in frames:
                path = self._safe_image_path(job_id, frame.relative_path)
                with Image.open(path) as source:
                    images.append(source.copy())
            return images
        except Exception:
            for image in images:
                image.close()
            raise

    def analyze_candidate(self, job_id: str, semantic_input, context, *, cancel_check: CancelCheck | None = None) -> CandidateSemanticArtifact:
        cached = self.cached_candidate_valid(job_id, semantic_input, context)
        if cached is not None:
            return cached
        if cancel_check and cancel_check():
            raise JobCancelledError(f"Job {job_id} was cancelled")
        images = self._load_images(job_id, context)
        last_text = ""
        last_status = SemanticParseStatus.EMPTY_RESPONSE
        runtime_result = None
        try:
            for attempt in range(1, self._settings.semantic_invalid_output_retries + 2):
                prompt = self._prompts.build(semantic_input, context, repair=attempt > 1)
                runtime_result = self._runtime.generate(
                    images,
                    prompt,
                    self.generation_config(),
                    cancel_check=cancel_check,
                )
                last_text = runtime_result.text
                last_status, parsed = parse_semantic_result(last_text, semantic_input.candidate_id)
                if parsed is not None:
                    expected_fp = self.expected_candidate_fingerprint(semantic_input, context)
                    artifact = CandidateSemanticArtifact(
                        candidate_id=semantic_input.candidate_id,
                        semantic_input_fingerprint=semantic_input.semantic_input_fingerprint,
                        temporal_context_fingerprint=context.temporal_context_fingerprint,
                        inference_fingerprint=self.inference_config_fingerprint(),
                        prompt_version=SEMANTIC_PROMPT_VERSION,
                        analysis_algorithm_version=SEMANTIC_ANALYSIS_ALGORITHM_VERSION,
                        prompt_fingerprint=self._prompts.fingerprint(semantic_input, context),
                        raw_text=last_text,
                        parse_status=SemanticParseStatus.SUCCESS,
                        semantic_result=parsed,
                        resolved_model=runtime_result.model_name,
                        fallback_used=runtime_result.fallback_used,
                        fallback_reason=runtime_result.fallback_reason,
                        runtime_fingerprint=runtime_result.runtime_fingerprint,
                        inference_seconds=runtime_result.generation_seconds,
                        attempt_count=attempt,
                        artifact_fingerprint=expected_fp,
                    )
                    self._repository.save_candidate_artifact(job_id, semantic_input.candidate_id, artifact)
                    return artifact
            if runtime_result is not None:
                failed = CandidateSemanticArtifact(
                    candidate_id=semantic_input.candidate_id,
                    semantic_input_fingerprint=semantic_input.semantic_input_fingerprint,
                    temporal_context_fingerprint=context.temporal_context_fingerprint,
                    inference_fingerprint=self.inference_config_fingerprint(),
                    prompt_version=SEMANTIC_PROMPT_VERSION,
                    analysis_algorithm_version=SEMANTIC_ANALYSIS_ALGORITHM_VERSION,
                    prompt_fingerprint=self._prompts.fingerprint(semantic_input, context),
                    raw_text=last_text,
                    parse_status=last_status,
                    semantic_result=None,
                    resolved_model=runtime_result.model_name,
                    fallback_used=runtime_result.fallback_used,
                    fallback_reason=runtime_result.fallback_reason,
                    runtime_fingerprint=runtime_result.runtime_fingerprint,
                    inference_seconds=runtime_result.generation_seconds,
                    attempt_count=self._settings.semantic_invalid_output_retries + 1,
                    artifact_fingerprint=self.expected_candidate_fingerprint(semantic_input, context),
                )
                self._repository.save_candidate_artifact(job_id, semantic_input.candidate_id, failed)
            raise SemanticAnalysisError(
                f"Semantic output for candidate {semantic_input.candidate_id} failed schema validation after bounded retries ({last_status.value})."
            )
        finally:
            for image in images:
                image.close()

    def rebuild_aggregate(self, job_id: str, *, cached_count: int = 0) -> SemanticResultsManifest:
        inputs = self._repository.load_input_manifest(job_id)
        contexts = self._repository.load_temporal_contexts(job_id)
        ctx_by_id = {x.candidate_id: x for x in contexts.contexts}
        artifacts: list[CandidateSemanticArtifact] = []
        for item in inputs.inputs:
            ctx = ctx_by_id.get(item.candidate_id)
            if ctx is None:
                raise Phase6ValidationError("Semantic aggregate cannot resolve temporal context.")
            artifact = self.cached_candidate_valid(job_id, item, ctx)
            if artifact is None:
                raise Phase6ValidationError("Semantic aggregate cannot be rebuilt from invalid candidate artifacts.")
            artifacts.append(artifact)
        results = [x.semantic_result for x in artifacts if x.semantic_result is not None]
        results.sort(key=lambda x: x.candidate_id)
        confidences = [x.confidence for x in results]
        timings = [x.inference_seconds for x in artifacts]
        completion = Counter(x.completion_state for x in results)
        model_counts = Counter(x.resolved_model for x in artifacts)
        warnings: list[str] = []
        if len(model_counts) > 1:
            warnings.append("MIXED_VLM_MODELS")
        if results and completion[CompletionState.TRANSITION] / len(results) >= 0.8:
            warnings.append("POSSIBLE_TRANSITION_BIAS")
        if results and completion[CompletionState.COMPLETE] / len(results) >= 0.95:
            warnings.append("POSSIBLE_COMPLETION_BIAS")
        retry_count = sum(max(0, x.attempt_count - 1) for x in artifacts)
        stats = SemanticAnalysisStats(
            candidate_count=len(inputs.inputs),
            successful_analysis_count=len(results),
            failed_analysis_count=len(inputs.inputs) - len(results),
            cached_analysis_count=cached_count,
            complete_count=completion[CompletionState.COMPLETE],
            mostly_complete_count=completion[CompletionState.MOSTLY_COMPLETE],
            incomplete_count=completion[CompletionState.INCOMPLETE],
            transition_count=completion[CompletionState.TRANSITION],
            uncertain_count=completion[CompletionState.UNCERTAIN],
            mean_confidence=statistics.fmean(confidences) if confidences else 0.0,
            median_confidence=statistics.median(confidences) if confidences else 0.0,
            mean_inference_seconds=statistics.fmean(timings) if timings else 0.0,
            median_inference_seconds=statistics.median(timings) if timings else 0.0,
            p90_inference_seconds=_percentile(timings, 90),
            total_inference_seconds=sum(timings),
            retry_count=retry_count,
            retry_success_count=sum(x.attempt_count > 1 and x.parse_status is SemanticParseStatus.SUCCESS for x in artifacts),
            fallback_candidate_count=sum(x.fallback_used for x in artifacts),
            model_usage_counts=dict(model_counts),
            warnings=warnings,
        )
        config_fp = self.config_fingerprint()
        fp_map = {str(x.candidate_id): x.artifact_fingerprint for x in artifacts}
        payload = {
            "semantic_input_fingerprint": inputs.artifact_fingerprint,
            "temporal_contexts_fingerprint": contexts.artifact_fingerprint,
            "inference_fingerprint": self.inference_config_fingerprint(),
            "config_fingerprint": config_fp,
            "candidate_artifact_fingerprints": fp_map,
            "results": [x.model_dump(mode="json") for x in results],
        }
        manifest = SemanticResultsManifest(
            semantic_input_fingerprint=inputs.artifact_fingerprint,
            temporal_contexts_fingerprint=contexts.artifact_fingerprint,
            inference_fingerprint=self.inference_config_fingerprint(),
            config_fingerprint=config_fp,
            artifact_fingerprint=stable_hash(payload),
            stats=stats,
            results=results,
            candidate_artifact_fingerprints=fp_map,
        )
        self._repository.save_results(job_id, manifest)
        return manifest

    def process(
        self,
        job_id: str,
        *,
        candidate_ids: list[int] | None = None,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> SemanticResultsManifest:
        inputs = self._repository.load_input_manifest(job_id)
        contexts = self._repository.load_temporal_contexts(job_id)
        ctx_by_id = {x.candidate_id: x for x in contexts.contexts}
        selected = set(candidate_ids) if candidate_ids is not None else {x.candidate_id for x in inputs.inputs}
        cached_count = 0
        work = [x for x in inputs.inputs if x.candidate_id in selected]
        for ordinal, semantic_input in enumerate(work):
            if cancel_check and cancel_check():
                raise JobCancelledError(f"Job {job_id} was cancelled")
            context = ctx_by_id.get(semantic_input.candidate_id)
            if context is None:
                raise Phase6ValidationError("Semantic input has no temporal context.")
            if self.cached_candidate_valid(job_id, semantic_input, context) is not None:
                cached_count += 1
            else:
                self.analyze_candidate(job_id, semantic_input, context, cancel_check=cancel_check)
            if progress_callback:
                progress_callback(int(((ordinal + 1) / max(1, len(work))) * 100))
        # Candidates outside candidate_ids still have to be valid for a complete aggregate.
        return self.rebuild_aggregate(job_id, cached_count=cached_count)


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * (percentile / 100)
    lo, hi = int(idx), min(int(idx) + 1, len(ordered) - 1)
    weight = idx - lo
    return float(ordered[lo] * (1 - weight) + ordered[hi] * weight)
