# Performance Profiling

Notify profiles wall-clock time with monotonic timers and lightweight CPU/RSS sampling. GPU metrics are optional and CPU-only operation remains supported.

Modes:
- `COLD_RUN`: full path with no reusable stage artifacts.
- `WARM_RUN`: runtime/model may already be initialized.
- `CACHE_HIT_RUN`: valid artifacts should make the run lightweight.

Useful normalized metrics are real-time factor (`processing_seconds / video_seconds`) and processing seconds per video minute. Performance changes are accepted only when the quality guardrail remains green and memory/disk tradeoffs stay bounded. CI uses a synthetic workload; real model measurements must record hardware context.
