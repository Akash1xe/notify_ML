from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.performance.models import ArtifactMetrics, PerformanceMode
from app.performance.profiler import PipelinePerformanceProfiler, calculate_workspace_size


def main() -> int:
    parser = argparse.ArgumentParser(description='Profile Notify pipeline stages or CI-safe synthetic workload')
    parser.add_argument('--mode', default='cold', choices=['cold', 'warm', 'cache-hit'])
    parser.add_argument('--sample', default='ci_synthetic')
    parser.add_argument('--video-duration', type=float, default=60.0)
    parser.add_argument('--workspace')
    parser.add_argument('--output')
    args = parser.parse_args()
    mode = {'cold': PerformanceMode.COLD_RUN, 'warm': PerformanceMode.WARM_RUN, 'cache-hit': PerformanceMode.CACHE_HIT_RUN}[args.mode]
    profiler = PipelinePerformanceProfiler()
    # Deterministic CI-safe workload; real integrations can wrap actual stage calls with the same profiler.
    profiler.start_stage('CI_SYNTHETIC_WORKLOAD')
    total = 0
    for i in range(25000 if mode is not PerformanceMode.CACHE_HIT_RUN else 1000):
        total += (i * 17) % 101
    profiler.stop_stage('CI_SYNTHETIC_WORKLOAD', input_count=25000, output_count=1, cache_hit=mode is PerformanceMode.CACHE_HIT_RUN)
    workspace_bytes = calculate_workspace_size(Path(args.workspace)) if args.workspace else 0
    run = profiler.build_run(run_id='ci-profile', sample_id=args.sample, mode=mode, pipeline_fingerprint='phase9-ci', video_duration_seconds=args.video_duration, counts={'synthetic_checksum': total}, artifact_metrics=ArtifactMetrics(workspace_bytes=workspace_bytes))
    payload = run.model_dump(mode='json') | {'total_runtime_seconds': run.total_runtime_seconds, 'real_time_factor': run.real_time_factor, 'processing_seconds_per_video_minute': run.processing_seconds_per_video_minute, 'fingerprint': run.fingerprint}
    if args.output:
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == '__main__': raise SystemExit(main())
