# Notify v1.0.0

Notify v1 turns a YouTube lecture into focused visual notes using local-first processing: visual timeline analysis, candidate ranking, local transcription, Qwen3-VL semantic selection, source-quality screenshot extraction, visual deduplication and deterministic PDF generation. The React frontend supports job submission, processing progress, result preview and streamed PDF download.

v1 also includes dependency-aware resume/cache behavior, offline quality evaluation and calibration tooling, performance/resource profiling, bounded stress scenarios, controlled failure/recovery testing, and local readiness/diagnostics.

## Known limitations

Real semantic/transcription throughput is hardware dependent. Very low-quality video, heavy teacher occlusion, rapid handwriting, private/unavailable YouTube sources and extremely long/high-resolution media may reduce quality or hit configured resource limits. Real-model quality/performance measurements require local media/model availability and are not faked in CI.
