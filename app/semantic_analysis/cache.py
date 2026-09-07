from __future__ import annotations

from pathlib import Path

from app.candidate_analysis.repository import CandidateAnalysisRepository
from app.jobs.checkpoints import CheckpointStore
from app.semantic_analysis.analysis import CP_SEMANTIC_ANALYSIS_COMPLETE, SemanticCandidateAnalyzer
from app.semantic_analysis.decision import CP_SEMANTIC_CANDIDATES_READY, SemanticDecisionEngine
from app.semantic_analysis.models import (
    CandidateCacheCheck,
    CandidateCacheState,
    Phase6ResumePlan,
    Phase6ResumeStage,
    SemanticCacheSnapshot,
    SemanticParseStatus,
)
from app.semantic_analysis.preparation import CP_SEMANTIC_INPUT_READY, SemanticInputPreparationService
from app.semantic_analysis.repository import SemanticRepository
from app.semantic_analysis.temporal_context import CP_TEMPORAL_VISUAL_CONTEXT_READY, TemporalVisualContextService
from app.storage.workspace import WorkspaceManager
from app.transcription.repository import TranscriptionRepository
from app.video_analysis.fingerprints import stable_hash
from app.video_analysis.models import AnalysisArtifactCheck, AnalysisArtifactState
from app.video_analysis.repository import FrameAnalysisRepository


class SemanticCacheCoordinator:
    def __init__(
        self,
        workspace: WorkspaceManager,
        checkpoints: CheckpointStore,
        repository: SemanticRepository,
        candidate_repository: CandidateAnalysisRepository,
        transcription_repository: TranscriptionRepository,
        frame_repository: FrameAnalysisRepository,
        preparation: SemanticInputPreparationService,
        temporal_context: TemporalVisualContextService,
        analyzer: SemanticCandidateAnalyzer,
        decision: SemanticDecisionEngine,
    ) -> None:
        self._workspace = workspace
        self._checkpoints = checkpoints
        self._repository = repository
        self._candidates = candidate_repository
        self._transcripts = transcription_repository
        self._frames = frame_repository
        self._preparation = preparation
        self._temporal = temporal_context
        self._analyzer = analyzer
        self._decision = decision

    @staticmethod
    def _check(state: AnalysisArtifactState, reason: str | None = None) -> AnalysisArtifactCheck:
        return AnalysisArtifactCheck(state=state, reason=reason)

    def validate_semantic_input(self, job_id: str):
        path = self._workspace.semantic_input_manifest_path(job_id)
        if not path.exists():
            return self._check(AnalysisArtifactState.MISSING, "artifact_missing"), None
        try:
            manifest = self._repository.load_input_manifest(job_id)
            selections = self._candidates.load_selections(job_id)
            ranked = self._candidates.load_ranked_candidates(job_id)
            contexts = self._transcripts.load_contexts(job_id)
            alignment = self._transcripts.load_alignment(job_id)
            sampling = self._frames.load_sampling(job_id)
            preprocessing = self._frames.load_preprocessing(job_id)
            if manifest.candidate_selections_fingerprint != selections.artifact_fingerprint:
                return self._check(AnalysisArtifactState.STALE, "candidate_selection_mismatch"), manifest
            if manifest.ranked_candidates_fingerprint != ranked.artifact_fingerprint:
                return self._check(AnalysisArtifactState.STALE, "ranking_mismatch"), manifest
            if manifest.transcript_contexts_fingerprint != contexts.artifact_fingerprint or manifest.transcript_alignment_fingerprint != alignment.artifact_fingerprint:
                return self._check(AnalysisArtifactState.STALE, "transcript_context_mismatch"), manifest
            if manifest.frame_manifest_fingerprint != sampling.artifact_fingerprint or manifest.preprocessing_fingerprint != preprocessing.artifact_fingerprint:
                return self._check(AnalysisArtifactState.STALE, "frame_dependency_mismatch"), manifest
            if manifest.config_fingerprint != self._preparation.config_fingerprint():
                return self._check(AnalysisArtifactState.STALE, "config_mismatch"), manifest
            ids = [x.candidate_id for x in manifest.inputs]
            if len(ids) != len(set(ids)):
                return self._check(AnalysisArtifactState.CORRUPT, "duplicate_candidate"), manifest
            expected_payload = {
                "candidate_selections_fingerprint": manifest.candidate_selections_fingerprint,
                "ranked_candidates_fingerprint": manifest.ranked_candidates_fingerprint,
                "transcript_contexts_fingerprint": manifest.transcript_contexts_fingerprint,
                "transcript_alignment_fingerprint": manifest.transcript_alignment_fingerprint,
                "frame_manifest_fingerprint": manifest.frame_manifest_fingerprint,
                "preprocessing_fingerprint": manifest.preprocessing_fingerprint,
                "config_fingerprint": manifest.config_fingerprint,
                "records": [x.model_dump(mode="json") for x in manifest.inputs],
            }
            if stable_hash(expected_payload) != manifest.artifact_fingerprint:
                return self._check(AnalysisArtifactState.CORRUPT, "artifact_fingerprint_mismatch"), manifest
            root = self._workspace.workspace(job_id)
            for item in manifest.inputs:
                path = (root / item.semantic_frame.relative_path).resolve()
                try:
                    path.relative_to(root)
                except ValueError:
                    return self._check(AnalysisArtifactState.CORRUPT, "unsafe_frame_path"), manifest
                if not path.is_file():
                    return self._check(AnalysisArtifactState.CORRUPT, "frame_reference_missing"), manifest
            if not self._checkpoints.is_completed(job_id, CP_SEMANTIC_INPUT_READY):
                return self._check(AnalysisArtifactState.PARTIAL, "checkpoint_missing"), manifest
            return self._check(AnalysisArtifactState.VALID), manifest
        except Exception:
            return self._check(AnalysisArtifactState.CORRUPT, "invalid_json_or_schema"), None

    def validate_temporal_context(self, job_id: str, input_manifest=None):
        if input_manifest is None:
            input_check, input_manifest = self.validate_semantic_input(job_id)
            if not input_check.valid or input_manifest is None:
                return self._check(AnalysisArtifactState.STALE, "semantic_input_invalid"), None
        path = self._workspace.semantic_temporal_contexts_path(job_id)
        if not path.exists():
            return self._check(AnalysisArtifactState.MISSING, "artifact_missing"), None
        try:
            manifest = self._repository.load_temporal_contexts(job_id)
            sampling = self._frames.load_sampling(job_id)
            preprocessing = self._frames.load_preprocessing(job_id)
            timeline = self._frames.load_timeline(job_id)
            differences = self._frames.load_differences(job_id)
            stability = self._candidates.load_stability_windows(job_id)
            boundaries = self._candidates.load_boundaries(job_id)
            if manifest.semantic_input_fingerprint != input_manifest.artifact_fingerprint:
                return self._check(AnalysisArtifactState.STALE, "semantic_input_mismatch"), manifest
            deps = (
                manifest.frame_manifest_fingerprint == sampling.artifact_fingerprint,
                manifest.preprocessing_fingerprint == preprocessing.artifact_fingerprint,
                manifest.timeline_fingerprint == timeline.artifact_fingerprint,
                manifest.differences_fingerprint == differences.artifact_fingerprint,
                manifest.stability_windows_fingerprint == stability.artifact_fingerprint,
                manifest.boundaries_fingerprint == boundaries.artifact_fingerprint,
            )
            if not all(deps):
                return self._check(AnalysisArtifactState.STALE, "visual_dependency_mismatch"), manifest
            if manifest.config_fingerprint != self._temporal.config_fingerprint():
                return self._check(AnalysisArtifactState.STALE, "config_mismatch"), manifest
            expected_payload = {
                "semantic_input_fingerprint": manifest.semantic_input_fingerprint,
                "frame_manifest_fingerprint": manifest.frame_manifest_fingerprint,
                "preprocessing_fingerprint": manifest.preprocessing_fingerprint,
                "timeline_fingerprint": manifest.timeline_fingerprint,
                "differences_fingerprint": manifest.differences_fingerprint,
                "stability_windows_fingerprint": manifest.stability_windows_fingerprint,
                "boundaries_fingerprint": manifest.boundaries_fingerprint,
                "config_fingerprint": manifest.config_fingerprint,
                "contexts": [x.model_dump(mode="json") for x in manifest.contexts],
            }
            if stable_hash(expected_payload) != manifest.artifact_fingerprint:
                return self._check(AnalysisArtifactState.CORRUPT, "artifact_fingerprint_mismatch"), manifest
            input_ids = {x.candidate_id for x in input_manifest.inputs}
            context_ids = [x.candidate_id for x in manifest.contexts]
            if len(context_ids) != len(set(context_ids)) or set(context_ids) != input_ids:
                return self._check(AnalysisArtifactState.CORRUPT, "candidate_reference_mismatch"), manifest
            root = self._workspace.workspace(job_id)
            for context in manifest.contexts:
                frames = [context.current] + ([context.previous] if context.previous else []) + ([context.next] if context.next else [])
                for frame in frames:
                    path = (root / frame.relative_path).resolve()
                    try:
                        path.relative_to(root)
                    except ValueError:
                        return self._check(AnalysisArtifactState.CORRUPT, "unsafe_frame_path"), manifest
                    if not path.is_file():
                        return self._check(AnalysisArtifactState.CORRUPT, "frame_reference_missing"), manifest
            if not self._checkpoints.is_completed(job_id, CP_TEMPORAL_VISUAL_CONTEXT_READY):
                return self._check(AnalysisArtifactState.PARTIAL, "checkpoint_missing"), manifest
            return self._check(AnalysisArtifactState.VALID), manifest
        except Exception:
            return self._check(AnalysisArtifactState.CORRUPT, "invalid_json_or_schema"), None

    def candidate_checks(self, job_id: str, input_manifest, temporal_manifest) -> list[CandidateCacheCheck]:
        contexts = {x.candidate_id: x for x in temporal_manifest.contexts}
        checks: list[CandidateCacheCheck] = []
        for item in input_manifest.inputs:
            context = contexts.get(item.candidate_id)
            if context is None:
                checks.append(CandidateCacheCheck(candidate_id=item.candidate_id, state=CandidateCacheState.STALE, reason="temporal_context_missing"))
                continue
            path = self._workspace.semantic_candidate_result_path(job_id, item.candidate_id)
            if not path.exists():
                checks.append(CandidateCacheCheck(candidate_id=item.candidate_id, state=CandidateCacheState.MISSING, reason="artifact_missing"))
                continue
            try:
                artifact = self._repository.load_candidate_artifact(job_id, item.candidate_id)
                expected = self._analyzer.expected_candidate_fingerprint(item, context)
                if artifact.artifact_fingerprint != expected or artifact.inference_fingerprint != self._analyzer.inference_config_fingerprint():
                    checks.append(CandidateCacheCheck(candidate_id=item.candidate_id, state=CandidateCacheState.STALE, reason="dependency_mismatch"))
                elif artifact.parse_status is not SemanticParseStatus.SUCCESS or artifact.semantic_result is None:
                    checks.append(CandidateCacheCheck(candidate_id=item.candidate_id, state=CandidateCacheState.FAILED, reason=artifact.parse_status.value))
                else:
                    checks.append(CandidateCacheCheck(candidate_id=item.candidate_id, state=CandidateCacheState.VALID))
            except Exception:
                checks.append(CandidateCacheCheck(candidate_id=item.candidate_id, state=CandidateCacheState.CORRUPT, reason="invalid_json_or_schema"))
        return checks

    def validate_semantic_analysis(self, job_id: str, input_manifest=None, temporal_manifest=None):
        if input_manifest is None:
            check, input_manifest = self.validate_semantic_input(job_id)
            if not check.valid or input_manifest is None:
                return self._check(AnalysisArtifactState.STALE, "semantic_input_invalid"), None, []
        if temporal_manifest is None:
            check, temporal_manifest = self.validate_temporal_context(job_id, input_manifest)
            if not check.valid or temporal_manifest is None:
                return self._check(AnalysisArtifactState.STALE, "temporal_context_invalid"), None, []
        candidate_checks = self.candidate_checks(job_id, input_manifest, temporal_manifest)
        invalid = [x for x in candidate_checks if x.state is not CandidateCacheState.VALID]
        if invalid:
            state = AnalysisArtifactState.PARTIAL if any(x.state in {CandidateCacheState.VALID} for x in candidate_checks) else AnalysisArtifactState.STALE
            return self._check(state, "candidate_results_incomplete"), None, candidate_checks
        path = self._workspace.semantic_results_path(job_id)
        if not path.exists():
            return self._check(AnalysisArtifactState.PARTIAL, "aggregate_missing"), None, candidate_checks
        try:
            manifest = self._repository.load_results(job_id)
            if manifest.semantic_input_fingerprint != input_manifest.artifact_fingerprint or manifest.temporal_contexts_fingerprint != temporal_manifest.artifact_fingerprint:
                return self._check(AnalysisArtifactState.STALE, "aggregate_dependency_mismatch"), manifest, candidate_checks
            if manifest.inference_fingerprint != self._analyzer.inference_config_fingerprint() or manifest.config_fingerprint != self._analyzer.config_fingerprint():
                return self._check(AnalysisArtifactState.STALE, "inference_config_mismatch"), manifest, candidate_checks
            expected_map = {str(x.candidate_id): self._repository.load_candidate_artifact(job_id, x.candidate_id).artifact_fingerprint for x in input_manifest.inputs}
            if manifest.candidate_artifact_fingerprints != expected_map:
                return self._check(AnalysisArtifactState.CORRUPT, "candidate_aggregate_mismatch"), manifest, candidate_checks
            expected_payload = {
                "semantic_input_fingerprint": manifest.semantic_input_fingerprint,
                "temporal_contexts_fingerprint": manifest.temporal_contexts_fingerprint,
                "inference_fingerprint": manifest.inference_fingerprint,
                "config_fingerprint": manifest.config_fingerprint,
                "candidate_artifact_fingerprints": manifest.candidate_artifact_fingerprints,
                "results": [x.model_dump(mode="json") for x in manifest.results],
            }
            if stable_hash(expected_payload) != manifest.artifact_fingerprint:
                return self._check(AnalysisArtifactState.CORRUPT, "artifact_fingerprint_mismatch"), manifest, candidate_checks
            if not self._checkpoints.is_completed(job_id, CP_SEMANTIC_ANALYSIS_COMPLETE):
                return self._check(AnalysisArtifactState.PARTIAL, "checkpoint_missing"), manifest, candidate_checks
            return self._check(AnalysisArtifactState.VALID), manifest, candidate_checks
        except Exception:
            return self._check(AnalysisArtifactState.CORRUPT, "aggregate_invalid"), None, candidate_checks

    def validate_decision(self, job_id: str, results_manifest=None):
        if results_manifest is None:
            check, results_manifest, _ = self.validate_semantic_analysis(job_id)
            if not check.valid or results_manifest is None:
                return self._check(AnalysisArtifactState.STALE, "semantic_analysis_invalid"), None
        path = self._workspace.semantic_selections_path(job_id)
        if not path.exists():
            return self._check(AnalysisArtifactState.MISSING, "artifact_missing"), None
        try:
            manifest = self._repository.load_selections(job_id)
            ranked = self._candidates.load_ranked_candidates(job_id)
            selections = self._candidates.load_selections(job_id)
            if manifest.semantic_results_fingerprint != results_manifest.artifact_fingerprint:
                return self._check(AnalysisArtifactState.STALE, "semantic_results_mismatch"), manifest
            if manifest.ranked_candidates_fingerprint != ranked.artifact_fingerprint or manifest.candidate_selections_fingerprint != selections.artifact_fingerprint:
                return self._check(AnalysisArtifactState.STALE, "phase4_dependency_mismatch"), manifest
            if manifest.config_fingerprint != self._decision.config_fingerprint():
                return self._check(AnalysisArtifactState.STALE, "decision_config_mismatch"), manifest
            result_ids = {x.candidate_id for x in results_manifest.results}
            if any(x not in result_ids for x in manifest.selected_candidate_ids):
                return self._check(AnalysisArtifactState.CORRUPT, "selected_candidate_reference_broken"), manifest
            expected_payload = {
                "semantic_results_fingerprint": manifest.semantic_results_fingerprint,
                "ranked_candidates_fingerprint": manifest.ranked_candidates_fingerprint,
                "candidate_selections_fingerprint": manifest.candidate_selections_fingerprint,
                "config_fingerprint": manifest.config_fingerprint,
                "candidate_decisions": [x.model_dump(mode="json") for x in manifest.candidate_decisions],
                "window_decisions": [x.model_dump(mode="json") for x in manifest.window_decisions],
                "selected_candidate_ids": manifest.selected_candidate_ids,
            }
            if stable_hash(expected_payload) != manifest.artifact_fingerprint:
                return self._check(AnalysisArtifactState.CORRUPT, "artifact_fingerprint_mismatch"), manifest
            if not self._checkpoints.is_completed(job_id, CP_SEMANTIC_CANDIDATES_READY):
                return self._check(AnalysisArtifactState.PARTIAL, "checkpoint_missing"), manifest
            return self._check(AnalysisArtifactState.VALID), manifest
        except Exception:
            return self._check(AnalysisArtifactState.CORRUPT, "invalid_json_or_schema"), None

    def inspect(self, job_id: str) -> SemanticCacheSnapshot:
        input_check, input_manifest = self.validate_semantic_input(job_id)
        if not input_check.valid or input_manifest is None:
            plan = Phase6ResumePlan(resume_stage=Phase6ResumeStage.SEMANTIC_INPUT_PREPARATION)
            invalid = self._check(AnalysisArtifactState.STALE, "upstream_invalid")
            return SemanticCacheSnapshot(semantic_input=input_check, temporal_context=invalid, semantic_analysis=invalid, semantic_decision=invalid, semantic_candidates_ready=invalid, resume_plan=plan)
        temporal_check, temporal_manifest = self.validate_temporal_context(job_id, input_manifest)
        if not temporal_check.valid or temporal_manifest is None:
            plan = Phase6ResumePlan(resume_stage=Phase6ResumeStage.TEMPORAL_CONTEXT_CONSTRUCTION, reuse_semantic_input=True)
            invalid = self._check(AnalysisArtifactState.STALE, "upstream_invalid")
            return SemanticCacheSnapshot(semantic_input=input_check, temporal_context=temporal_check, semantic_analysis=invalid, semantic_decision=invalid, semantic_candidates_ready=invalid, resume_plan=plan)
        analysis_check, results, candidate_checks = self.validate_semantic_analysis(job_id, input_manifest, temporal_manifest)
        ids_to_analyze = [x.candidate_id for x in candidate_checks if x.state is not CandidateCacheState.VALID]
        if not analysis_check.valid or results is None:
            rebuild = not ids_to_analyze and bool(candidate_checks)
            if not input_manifest.inputs:
                rebuild = True
            plan = Phase6ResumePlan(
                resume_stage=Phase6ResumeStage.SEMANTIC_ANALYSIS,
                reuse_semantic_input=True,
                reuse_temporal_context=True,
                candidate_ids_to_analyze=ids_to_analyze,
                rebuild_semantic_aggregate=rebuild,
                rerun_decision_engine=True,
            )
            invalid = self._check(AnalysisArtifactState.STALE, "upstream_invalid")
            return SemanticCacheSnapshot(semantic_input=input_check, temporal_context=temporal_check, semantic_analysis=analysis_check, semantic_decision=invalid, semantic_candidates_ready=invalid, candidate_checks=candidate_checks, resume_plan=plan)
        decision_check, _ = self.validate_decision(job_id, results)
        if not decision_check.valid:
            plan = Phase6ResumePlan(resume_stage=Phase6ResumeStage.SEMANTIC_DECISION, reuse_semantic_input=True, reuse_temporal_context=True, rerun_decision_engine=True)
            return SemanticCacheSnapshot(semantic_input=input_check, temporal_context=temporal_check, semantic_analysis=analysis_check, semantic_decision=decision_check, semantic_candidates_ready=decision_check, candidate_checks=candidate_checks, resume_plan=plan)
        ready = self._check(AnalysisArtifactState.VALID)
        plan = Phase6ResumePlan(resume_stage=Phase6ResumeStage.PHASE6_READY, reuse_semantic_input=True, reuse_temporal_context=True)
        return SemanticCacheSnapshot(semantic_input=input_check, temporal_context=temporal_check, semantic_analysis=analysis_check, semantic_decision=decision_check, semantic_candidates_ready=ready, candidate_checks=candidate_checks, resume_plan=plan)

    def cleanup_partial_artifacts(self, job_id: str) -> list[str]:
        semantic = self._workspace.semantic_dir(job_id)
        if not semantic.exists():
            return []
        removed: list[str] = []
        for path in semantic.rglob("*.tmp"):
            if path.is_file():
                path.unlink(missing_ok=True)
                removed.append(path.relative_to(semantic).as_posix())
        return removed

    def invalidate_from(self, job_id: str, stage: Phase6ResumeStage) -> None:
        paths: list[Path] = []
        checkpoints: list[str] = []
        if stage is Phase6ResumeStage.SEMANTIC_INPUT_PREPARATION:
            paths += [self._workspace.semantic_input_manifest_path(job_id), self._workspace.semantic_temporal_contexts_path(job_id), self._workspace.semantic_results_path(job_id), self._workspace.semantic_selections_path(job_id), self._workspace.semantic_summary_path(job_id)]
            checkpoints += [CP_SEMANTIC_INPUT_READY, CP_TEMPORAL_VISUAL_CONTEXT_READY, CP_SEMANTIC_ANALYSIS_COMPLETE, CP_SEMANTIC_CANDIDATES_READY]
        elif stage is Phase6ResumeStage.TEMPORAL_CONTEXT_CONSTRUCTION:
            paths += [self._workspace.semantic_temporal_contexts_path(job_id), self._workspace.semantic_results_path(job_id), self._workspace.semantic_selections_path(job_id), self._workspace.semantic_summary_path(job_id)]
            checkpoints += [CP_TEMPORAL_VISUAL_CONTEXT_READY, CP_SEMANTIC_ANALYSIS_COMPLETE, CP_SEMANTIC_CANDIDATES_READY]
        elif stage is Phase6ResumeStage.SEMANTIC_ANALYSIS:
            paths += [self._workspace.semantic_results_path(job_id), self._workspace.semantic_selections_path(job_id), self._workspace.semantic_summary_path(job_id)]
            checkpoints += [CP_SEMANTIC_ANALYSIS_COMPLETE, CP_SEMANTIC_CANDIDATES_READY]
        elif stage is Phase6ResumeStage.SEMANTIC_DECISION:
            paths += [self._workspace.semantic_selections_path(job_id), self._workspace.semantic_summary_path(job_id)]
            checkpoints += [CP_SEMANTIC_CANDIDATES_READY]
        for path in paths:
            path.unlink(missing_ok=True)
        for cp in checkpoints:
            self._checkpoints.invalidate(job_id, cp)

    def reconcile(self, job_id: str) -> SemanticCacheSnapshot:
        snapshot = self.inspect(job_id)
        # Repair stale completion checkpoints conservatively.
        if not snapshot.semantic_input.valid:
            self._checkpoints.invalidate(job_id, CP_SEMANTIC_INPUT_READY)
        if not snapshot.temporal_context.valid:
            self._checkpoints.invalidate(job_id, CP_TEMPORAL_VISUAL_CONTEXT_READY)
        if not snapshot.semantic_analysis.valid:
            self._checkpoints.invalidate(job_id, CP_SEMANTIC_ANALYSIS_COMPLETE)
        if not snapshot.semantic_decision.valid:
            self._checkpoints.invalidate(job_id, CP_SEMANTIC_CANDIDATES_READY)
        return snapshot
