from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.video_analysis.evaluation import evaluate_phase3
from app.video_analysis.models import (
    DifferenceManifest,
    FrameAnalysisSummary,
    MajorChangesManifest,
    PreprocessingManifest,
    TimelineManifest,
)


def load(path: Path, model):
    return model.model_validate(json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate an existing Notify Phase-3 job workspace")
    parser.add_argument("workspace", type=Path, help="Path to storage/jobs/<job_id>")
    args = parser.parse_args()
    root = args.workspace.resolve()
    summary = load(root / "analysis" / "summary.json", FrameAnalysisSummary)
    preprocessing = load(root / "frames" / "preprocessing.json", PreprocessingManifest)
    differences = load(root / "analysis" / "differences.json", DifferenceManifest)
    major = load(root / "analysis" / "major_changes.json", MajorChangesManifest)
    timeline = load(root / "analysis" / "timeline.json", TimelineManifest)
    report = evaluate_phase3(
        summary=summary,
        preprocessing=preprocessing,
        differences=differences,
        major_changes=major,
        timeline=timeline,
    )
    output = root / "analysis" / "evaluation.json"
    temp = output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
    os.replace(temp, output)

    minutes = report.lecture_duration_seconds / 60 if report.lecture_duration_seconds else 0
    print(f"Lecture duration: {minutes:.1f} min")
    print(f"Sampled frames: {report.sampled_frames}")
    print(f"Invalid / black / blurry: {report.invalid_frames} / {report.black_frames} / {report.blurry_frames}")
    print(f"Difference p95 / p99: {report.score_distribution['p95']:.3f} / {report.score_distribution['p99']:.3f}")
    print(f"Major events: {report.major_events} ({report.events_per_minute:.2f}/min)")
    print(f"Stable / changing ratio: {report.stable_ratio:.1%} / {report.changing_ratio:.1%}")
    print(f"Segments per minute: {report.segments_per_minute:.2f}")
    if report.example_event_timestamps:
        print("Example event timestamps:", ", ".join(f"{value:.1f}s" for value in report.example_event_timestamps))
    for warning in report.warnings:
        print(f"WARNING {warning.code}: {warning.message}")
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
