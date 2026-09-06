# Notify ML

Notify is a **local-first AI lecture screenshot extractor**. The final product will accept a YouTube lecture URL, detect meaningful completed teaching states (slides, boards, diagrams, code), use a local vision-language model to judge which moments are worth keeping, remove duplicate screenshots, and generate a clean PDF.

This repository is the ML-oriented implementation track. **Phase 1 is complete here as the application foundation and job-processing shell.** No video, Whisper, Qwen, OpenCV, or PDF processing has been added yet.

## Target pipeline

```text
YouTube URL
   -> video ingestion
   -> frame sampling
   -> change + stability detection
   -> timestamped transcription
   -> temporal context
   -> local Qwen3-VL analysis
   -> best screenshot selection
   -> deduplication
   -> PDF
```

## Phase 1 architecture

```text
Client
  |
  v
FastAPI
  |
  v
Job Service ---------> Workspace Manager
  |
  v
Bounded Job Queue
  |
  v
Background Runner
  |
  v
Processor contract (FakeProcessor in Phase 1)
  |
  +----> Progress + state machine
  +----> Checkpoints
  +----> Per-job logs
```

The Phase-1 processor is intentionally fake. It proves that long-running work, progress, cancellation, concurrency limits, failure isolation, persistence, and restart recovery work before expensive video/ML stages are introduced.

## Job lifecycle

```text
QUEUED -> RUNNING -> COMPLETED
   |         |
   |         +------> FAILED
   |         |
   +---------+------> CANCELLED
```

A job that was `RUNNING` when the application stopped is recovered as `QUEUED` on the next startup and safely retried.

## Future processing stages already modeled

- `PREPARING`
- `DOWNLOADING`
- `EXTRACTING_AUDIO`
- `TRANSCRIBING`
- `SAMPLING_FRAMES`
- `DETECTING_CHANGES`
- `DETECTING_STABILITY`
- `GENERATING_CANDIDATES`
- `BUILDING_TEMPORAL_CONTEXT`
- `AI_ANALYSIS`
- `SELECTING_SCREENSHOTS`
- `DEDUPLICATING`
- `GENERATING_PDF`

These enum values make future phases plug into the same job engine without redesigning the API.

## Local workspace layout

Each job receives an isolated UUID workspace:

```text
storage/jobs/<job_id>/
├── source/
├── audio/
├── frames/
├── candidates/
├── screenshots/
├── transcript/
├── decisions/
├── output/
├── logs/
│   └── events.ndjson
├── checkpoints.json
└── job.json
```

Critical JSON metadata is written atomically to reduce corruption risk on crashes.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Application information |
| `GET` | `/health` | Health check |
| `POST` | `/api/jobs` | Create and queue a processing job |
| `GET` | `/api/jobs` | List jobs |
| `GET` | `/api/jobs/{job_id}` | Read live status/progress |
| `POST` | `/api/jobs/{job_id}/cancel` | Cancel a queued/running job |
| `GET` | `/api/system/capabilities` | Inspect local CPU/RAM/GPU capability |

Example job request:

```json
{
  "source_url": "https://www.youtube.com/watch?v=example"
}
```

## Configuration

Copy the sample file:

```bash
cp .env.example .env
```

Supported variables:

```text
APP_NAME
APP_ENV
HOST
PORT
STORAGE_ROOT
JOB_RETENTION_HOURS
MAX_CONCURRENT_JOBS
LOG_LEVEL
FAKE_PROCESSOR_STEP_DELAY
SIMULATED_FAILURE_AT_PROGRESS
```

`SIMULATED_FAILURE_AT_PROGRESS` is for Phase-1 development/failure testing only.

## Setup

Python 3.11+ is required.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -e ".[dev]"
```

Run:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open API documentation at `http://127.0.0.1:8000/docs`.

## Tests

```bash
pytest
```

The test suite covers configuration/startup, job persistence and state transitions, workspace safety, the REST API, background execution, cancellation, concurrency limits, injected failures, recovery primitives, and hardware capability detection.

## Phase 2 integration point

Phase 2 will replace/extend the `FakeProcessor` contract with **YouTube ingestion** using `yt-dlp` and FFmpeg. The job API, queue, progress model, workspaces, recovery logic, and checkpoint infrastructure remain unchanged.

## Constraints intentionally preserved

Phase 1 has:

- no paid API
- no cloud dependency
- no Redis
- no Kafka
- no microservices
- no database server
- no ML model
- no video processing

This keeps Notify cheap, local, and easy to iterate while the ML pipeline is still experimental.
