# Notify v1 Architecture

```text
YouTube URL
  -> ingestion (yt-dlp / FFmpeg)
  -> OpenCV visual analysis
  -> stability + candidate ranking
  -> local transcription + context
  -> Qwen3-VL semantic analysis
  -> source-quality screenshot extraction
  -> quality refinement + deduplication
  -> final screenshot selection
  -> document input/layout/render plan
  -> PDF generation
  -> result API
  -> React frontend
```

Supporting systems are dependency-aware cache/resume, offline evaluation/calibration, performance/stress profiling, controlled failure recovery and local diagnostics/readiness.

Main job workspace areas are `source`, visual-analysis artifacts, `candidates`, `transcript`, `semantic`, `screenshots`, `document` and optional `diagnostics`. Phase-specific manifests and fingerprints determine the earliest stage that needs to rerun after a configuration or integrity change.
