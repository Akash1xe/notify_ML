# Notify ML

Notify is a **local-first lecture processing application**. The final product will accept a YouTube lecture, identify completed teaching states such as slides, boards, diagrams, and code, select useful screenshots with a local vision-language model, remove duplicates, and generate a PDF.

This repository is the ML-oriented implementation track. **Phase 2 is complete:** Notify can now turn a supported recorded YouTube URL into a validated local source video plus normalized transcription-ready audio, with job persistence, cancellation, caching, restart recovery, and cleanup.

## Current architecture

```text
Client
  |
  v
FastAPI
  |
  v
Job Service ----------------------> isolated job workspace
  |
  v
Bounded Job Runner
  |
  v
IngestionPipeline
  |
  +--> YouTube URL validation
  |      |
  |      +--> yt-dlp metadata (no download)
  |
  +--> yt-dlp video download (<= 1080p by default)
  |
  +--> FFmpeg / ffprobe capability check
  |
  +--> ffprobe media inspection
  |
  +--> FFmpeg normalized audio extraction
  |
  +--> dependency-aware cache + checkpoints
  |
  v
INGESTION_COMPLETE
```

Phase 3 can consume the local video and media metadata without knowing anything about YouTube or `yt-dlp`.

## Phase 2 output contract

A successful job produces this structure:

```text
storage/jobs/<job_id>/
├── source/
│   ├── metadata.json
│   ├── video.*
│   ├── download.json
│   └── media.json
├── audio/
│   ├── audio.wav
│   └── audio.json
├── logs/
│   └── events.ndjson
├── checkpoints.json
├── ingestion.json
└── job.json
```

`audio/audio.wav` is normalized to:

```text
WAV
PCM signed 16-bit
16 kHz
mono
```

It is ready for a future local Whisper/faster-whisper stage. **Phase 2 does not perform transcription.**

## Phase 2 sub-phases implemented

### 2.1 YouTube URL validation + metadata

Supported common forms include:

```text
https://www.youtube.com/watch?v=VIDEO_ID
https://youtube.com/watch?v=VIDEO_ID
https://youtu.be/VIDEO_ID
https://www.youtube.com/shorts/VIDEO_ID
https://youtube.com/live/VIDEO_ID
```

Query parameters do not affect canonical video identification. URLs are normalized to:

```text
https://www.youtube.com/watch?v=VIDEO_ID
```

Metadata is extracted through the `yt-dlp` Python API with download disabled. Notify stores a typed, bounded representation instead of persisting the entire raw `yt-dlp` payload.

Current live/upcoming streams are rejected. Finished live replays may be processed like normal recorded videos when `yt-dlp` exposes a usable replay.

Private, deleted, unavailable, malformed, or unsupported sources become domain-level job failures rather than raw stack traces.

No YouTube API key and no browser cookies are required or accessed automatically.

### 2.2 Video download

The downloader uses `yt-dlp` behind a dedicated adapter and stores media only inside the job workspace.

Default strategy:

```text
maximum height: 1080p
preferred merge container: mp4
bounded download retries
bounded fragment retries
```

The actual filename is controlled internally (`video.*`); untrusted YouTube titles never become paths.

Download progress is mapped into global job progress. Cancellation is checked from the `yt-dlp` progress hook. `.part`, `.ytdl`, and other temporary files never count as completed media.

A configured maximum download size and best-effort disk-space check protect local storage.

### 2.3 FFmpeg detection + media inspection

Notify detects `ffmpeg` and `ffprobe` from PATH, or accepts explicitly configured binary paths.

`ffprobe` JSON is normalized into typed media information including:

- container
- duration
- file size / bitrate
- video codec/profile
- width / height
- FPS (including fractional values such as `30000/1001`)
- pixel format
- rotation
- audio codec
- audio sample rate
- channels / layout

A missing video stream, corrupt/truncated media, ffprobe timeout, or missing required tool fails cleanly.

### 2.4 Audio extraction

FFmpeg extracts only audio and normalizes it using the equivalent settings:

```text
-vn
-ac 1
-ar 16000
-c:a pcm_s16le
```

Extraction writes to `audio.tmp.wav` first. Only a successful, validated result is atomically promoted to `audio.wav`.

Cancellation terminates FFmpeg and removes the temporary output. Extraction uses a duration-aware timeout and estimates WAV disk usage before starting.

### 2.5 Caching, resume, and cleanup

Cache validity requires both:

```text
completion checkpoint
+
valid underlying artifact
```

The dependency graph is:

```text
source URL
    |
 metadata
    |
  video
    |
inspection
    |
  audio
```

An upstream change invalidates dependent state. Examples:

```text
video missing/change
  -> redownload video
  -> re-run inspection
  -> re-extract audio

audio missing
  -> keep metadata/video/inspection
  -> re-extract audio only
```

Known partial files are removed safely inside the job workspace. Cleanup can run in dry-run mode and deletes only expired terminal (`COMPLETED`, `FAILED`, `CANCELLED`) jobs; active/queued jobs are protected.

### 2.6 Final integration + failure handling

`IngestionPipeline` is the single Phase-2 orchestrator. It performs cache inspection, selects the resume point, runs only required stages, validates the final artifact chain, writes `ingestion.json`, and records `INGESTION_COMPLETE`.

Failures include structured codes/categories such as input, YouTube, download, media, audio, storage, cancellation, and internal processing failures. Detailed diagnostics remain in per-job logs while the public job response gets a safe error object.

Failed or cancelled jobs can be explicitly retried. Completed upstream artifacts are reused when valid.

## Job lifecycle

```text
QUEUED -> RUNNING -> COMPLETED
   |         |
   |         +--> FAILED -> retry -> QUEUED
   |         |
   +---------+--> CANCELLED -> retry -> QUEUED
```

A job that was `RUNNING` when the application stopped is recovered as `QUEUED` at startup. Cache inspection then selects the first incomplete ingestion stage.

## Processing stages currently used

```text
PREPARING
DOWNLOADING
INSPECTING_MEDIA
EXTRACTING_AUDIO
INGESTION_COMPLETE
```

Future stages are still modeled for Phase 3+.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Application / implementation phase |
| `GET` | `/health` | Health check |
| `POST` | `/api/jobs` | Create and queue a lecture job |
| `GET` | `/api/jobs` | List jobs |
| `GET` | `/api/jobs/{job_id}` | Live status/progress/error |
| `POST` | `/api/jobs/{job_id}/cancel` | Cancel queued/running work |
| `POST` | `/api/jobs/{job_id}/retry` | Retry failed/cancelled job |
| `GET` | `/api/jobs/{job_id}/metadata` | Normalized YouTube metadata |
| `GET` | `/api/jobs/{job_id}/ingestion` | Final Phase-2 ingestion result |
| `GET` | `/api/system/capabilities` | CPU/RAM/GPU and FFmpeg tool availability |
| `POST` | `/api/system/cleanup` | Dry-run or execute retention cleanup |

Create a job:

```json
{
  "source_url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

At this stage `COMPLETED` means **all currently implemented Phase-2 ingestion work has finished**. The job message explicitly says the lecture is ready for frame analysis; it does not imply a PDF exists.

## Configuration

Copy:

```bash
cp .env.example .env
```

Important Phase-2 values:

```text
PROCESSOR_MODE=ingestion
VIDEO_MAX_HEIGHT=1080
VIDEO_PREFERRED_CONTAINER=mp4
MAX_VIDEO_DOWNLOAD_GB=4
VIDEO_DOWNLOAD_RETRIES=3
VIDEO_FRAGMENT_RETRIES=3
DOWNLOAD_DISK_SAFETY_MARGIN_MB=512
FFMPEG_PATH=
FFPROBE_PATH=
MEDIA_PROBE_TIMEOUT_SECONDS=30
AUDIO_EXTRACTION_TIMEOUT_MULTIPLIER=3
AUDIO_DISK_SAFETY_MARGIN_MB=256
AUDIO_DURATION_TOLERANCE_RATIO=0.01
```

Blank FFmpeg paths mean auto-detect from PATH.

`PROCESSOR_MODE=fake` is retained only for deterministic foundation tests/development. Normal usage should use `ingestion`.

## Install

Python 3.11+ is required.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

### FFmpeg

Windows (one option):

```powershell
winget install Gyan.FFmpeg
```

macOS:

```bash
brew install ffmpeg
```

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install ffmpeg
```

Verify:

```bash
ffmpeg -version
ffprobe -version
```

## Run

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open API docs at:

```text
http://127.0.0.1:8000/docs
```

## Tests

```bash
pytest
```

Automated tests deliberately mock YouTube and FFmpeg process boundaries. CI therefore does **not** need live YouTube access and does not depend on a particular system FFmpeg installation.

Tests cover Phase-1 regression behavior plus URL parsing, metadata normalization, format selection, downloader progress/cancellation, media parsing, audio command/progress behavior, cache dependency validation, partial cleanup, retry, failure classification, the complete Phase-2 orchestrator, and the real FastAPI/background-runner flow with mocked external boundaries.

## Optional real-world smoke test

Not part of CI. Requires internet and FFmpeg:

```bash
python scripts/smoke_ingestion.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

It creates a real Notify job and prints stage/progress until ingestion finishes.

## Security / local-first guarantees

Phase 2 uses:

- no paid API
- no cloud storage
- no Redis
- no Kafka
- no Celery
- no database server
- no browser-cookie scraping
- no YouTube API key
- no ML model yet

Subprocess commands use argument arrays rather than shell strings. Artifact paths are constrained to UUID job workspaces, and external titles are not used as filenames.

## Phase 3 handoff

Phase 3 can assume a successful ingestion provides:

```text
source video path
duration
width
height
fps
video codec
normalized audio path
```

Phase 3 will implement **frame sampling + visual change detection**. It should not re-run YouTube validation/download logic.

Not implemented yet:

- OpenCV frame sampling
- SSIM / scene-change detection
- Whisper/faster-whisper transcription
- Qwen3-VL
- screenshot selection
- PDF generation
