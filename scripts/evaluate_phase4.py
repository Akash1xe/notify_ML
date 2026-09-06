from __future__ import annotations

import argparse
from pathlib import Path

from app.candidate_analysis.evaluation import Phase4Evaluator
from app.candidate_analysis.repository import CandidateAnalysisRepository
from app.core.config import AppSettings
from app.storage.workspace import WorkspaceManager
from app.video_analysis.repository import FrameAnalysisRepository


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a completed Notify Phase-4 job workspace")
    parser.add_argument("job_id", help="UUID job id under STORAGE_ROOT")
    parser.add_argument("--storage-root", default="storage/jobs")
    parser.add_argument("--persist", action="store_true", help="Write candidates/evaluation.json")
    args = parser.parse_args()

    settings = AppSettings(storage_root=Path(args.storage_root))
    workspace = WorkspaceManager(settings.storage_root)
    candidate_repository = CandidateAnalysisRepository(workspace)
    frame_repository = FrameAnalysisRepository(workspace)
    report = Phase4Evaluator(
        settings, workspace, candidate_repository, frame_repository
    ).evaluate(args.job_id, persist=args.persist)

    print(f"Lecture duration: {report.lecture_duration_seconds:.1f}s")
    print(f"Stable windows: {report.stability_windows}")
    print(f"Valid boundaries: {report.valid_boundaries}")
    print(f"Generated candidates: {report.generated_candidates}")
    print(f"Valid candidates: {report.valid_candidates}")
    print(f"Primary / alternates: {report.primary_candidates} / {report.alternate_candidates}")
    print(f"Ambiguous / clear winners: {report.ambiguous_windows} / {report.clear_winners}")
    print(f"Primary density: {report.primary_density_per_minute:.2f}/min")
    print(f"Mean boundary score: {report.mean_boundary_score:.3f}")
    print(f"Mean completeness: {report.mean_completeness:.3f}")
    print(f"Mean transition risk: {report.mean_transition_risk:.3f}")
    print(f"Mean primary rank score: {report.mean_primary_ranking_score:.3f}")
    if report.warnings:
        print("Warnings:")
        for warning in report.warnings:
            print(f"- {warning}")


if __name__ == "__main__":
    main()
