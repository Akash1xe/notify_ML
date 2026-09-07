# Notify Diagnostics

- `GET /health`: process liveness only.
- `GET /ready`: validates the configured local environment without eagerly loading large models.
- `python scripts/check_environment.py [--json] [--deep]`: environment doctor; deep mode also checks optional model runtime packages.

Structured Phase-9 diagnostics use stable event names and local JSON/JSONL data. Sensitive keys (tokens, cookies, passwords, API keys, authorization values) are redacted. Full transcripts, image bytes and full Qwen prompts are never emitted by default.

Typical failures and retryability:

| Code | Meaning | Typical action |
|---|---|---|
| `FFMPEG_NOT_AVAILABLE` | FFmpeg cannot be executed | install/fix dependency, retry |
| `TRANSCRIPTION_MODEL_UNAVAILABLE` | local transcription runtime/model unavailable | fix dependency, retry |
| `SEMANTIC_MODEL_UNAVAILABLE` | local VLM runtime/model unavailable | fix dependency, retry |
| `STORAGE_FULL` | artifact write failed due to storage | free/change storage, retry |
| `RESOURCE_LIMIT_EXCEEDED` | configured hard resource bound exceeded | adjust config/input, retry |
| `DOCUMENT_CORRUPT` | final PDF failed integrity validation | regenerate PDF stage |
