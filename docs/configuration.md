# Notify Configuration

Notify uses `AppSettings`/environment variables as the canonical configuration. Explicit constructor overrides (tests/calibration) take precedence over environment values; environment values take precedence over defaults.

Key groups are core/API/storage, ingestion/FFmpeg, frame analysis, candidates, transcription, semantic analysis, screenshot selection, document/PDF, resource limits and local diagnostics. Use `.env.example` as the authoritative list of supported settings and `python scripts/check_environment.py` before processing a real lecture.

Never place secrets in `VITE_*` variables: frontend environment variables are public browser configuration. Fake processors/adapters are for tests and must not be used as release defaults.
