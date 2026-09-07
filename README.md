# Notify ML

Notify is a **local-first lecture processing application**. The long-term product accepts a YouTube lecture, identifies completed teaching states such as slides, boards, diagrams, and code, selects useful screenshots with a local vision-language model, removes duplicates, and generates a PDF.

**Phase 6 is complete.** Notify can ingest a lecture, build visual candidates, transcribe locally with faster-whisper, align speech to candidate timestamps, construct compact temporal visual context, analyze candidates through a local Qwen3-VL runtime, and deterministically select semantically useful completed visual states with candidate-level cache/recovery.

No paid API or cloud vision/speech service is required. Qwen model weights are downloaded only on explicit uncached semantic inference (unless local-files-only mode is enabled); automated tests use fake STT/VLM adapters and never download model weights.

## Current pipeline

```text
YouTube URL
    |
    v
Phase 2 - IngestionPipeline
    |
    +--> validate/normalize YouTube URL
    +--> yt-dlp metadata
    +--> yt-dlp video download
    +--> ffprobe media inspection
    +--> FFmpeg 16 kHz mono PCM audio
    +--> ingestion cache/recovery
    |
    v
INGESTION_COMPLETE
    |
    v
Phase 3 - FrameAnalysisPipeline
    |
    +--> 3.1 FFmpeg frame sampling
    +--> 3.2 OpenCV quality/preprocessing
    +--> 3.3 visual difference scoring
    +--> 3.4 major transition detection
    +--> 3.5 temporal activity timeline
    +--> 3.6 dependency-aware cache/recovery
    +--> 3.7 final validation + summary/evaluation
    |
    v
FRAME_ANALYSIS_COMPLETE
    |
    v
Phase 4 - CandidateAnalysisPipeline
    |
    +--> 4.1 stability windows
    +--> 4.2 CHANGING/transition -> STABLE boundaries
    +--> 4.3 strategic candidate timestamps
    +--> 4.4 quality/completeness heuristics
    +--> 4.5 per-window PRIMARY/ALTERNATE ranking
    +--> 4.6 dependency-aware cache/recovery
    +--> 4.7 final validation + summary/evaluation
    |
    v
CANDIDATES_READY
```

`PROCESSOR_MODE=semantic` runs Phases 2–6. `PROCESSOR_MODE=transcription` stops after Phase 5, `candidates` stops after Phase 4, `analysis` after Phase 3, `ingestion` after Phase 2, and `fake` remains available for deterministic foundation tests. The `.env.example` keeps the existing transcription default unless you explicitly select semantic mode.

## Workspace contract

A successful Phase-4 job produces conceptually:

```text
storage/jobs/<job_id>/
├── source/
│   ├── metadata.json
│   ├── video.*
│   ├── download.json
│   └── media.json
│
├── audio/
│   ├── audio.wav
│   └── audio.json
│
├── frames/
│   ├── sampled/
│   │   ├── frame_00000001.jpg
│   │   └── ...
│   ├── processed/
│   │   ├── frame_00000001.jpg
│   │   └── ...
│   ├── manifest.json
│   └── preprocessing.json
│
├── analysis/
│   ├── differences.json
│   ├── major_changes.json
│   ├── timeline.json
│   ├── summary.json
│   └── evaluation.json       # only when Phase-3 evaluator is run
│
├── candidates/
│   ├── stability_windows.json
│   ├── boundaries.json
│   ├── generated_candidates.json
│   ├── scored_candidates.json
│   ├── ranked_candidates.json
│   ├── selections.json
│   ├── summary.json
│   └── evaluation.json       # only when Phase-4 evaluator is run
│
├── logs/
│   └── events.ndjson
├── checkpoints.json
├── ingestion.json
└── job.json
```

`frames/sampled/*` and `frames/processed/*` are **analysis artifacts**, not final PDF screenshots. Later screenshot selection must return to `source/video.*` for the cleanest/high-quality exact frame.

# Phase 2 — ingestion

## YouTube ingestion

Notify accepts common recorded-video forms such as:

```text
https://www.youtube.com/watch?v=VIDEO_ID
https://youtu.be/VIDEO_ID
https://www.youtube.com/shorts/VIDEO_ID
https://youtube.com/live/VIDEO_ID
```

URLs are normalized to a canonical video ID/URL. Metadata is extracted through the `yt-dlp` Python API with download disabled first. Current live/upcoming streams, private videos, deleted/unavailable videos, malformed URLs, and unsupported sources fail through typed domain errors.

Video downloads are capped to 1080p by default with bounded retries, controlled internal filenames, disk checks, progress hooks, and cancellation. FFmpeg/ffprobe are detected locally; no browser cookies or YouTube API key are read automatically.

## Normalized audio

The Phase-2 audio artifact is:

```text
WAV
pcm_s16le
16 kHz
mono
```

Extraction writes to a temporary file, validates the result, then atomically promotes it to `audio/audio.wav`.

## Phase-2 caching

```text
source URL
   ↓
metadata
   ↓
video
   ↓
media inspection
   ↓
audio
```

A cache entry requires a completion checkpoint **and** a valid artifact. Upstream changes invalidate downstream artifacts. Restart recovery uses files/fingerprints rather than trusting `job.stage` alone.

# Phase 3 — visual frame analysis

## 3.1 Frame sampling

Phase 3 consumes the local Phase-2 video; it does not re-run YouTube logic.

Default sampling:

```text
FRAME_SAMPLE_FPS=1.0
```

Examples:

```text
1 FPS   -> one frame each second
2 FPS   -> one frame each 0.5 seconds
0.5 FPS -> one frame each 2 seconds
```

FFmpeg samples into:

```text
frames/sampled.tmp/
```

and only after successful validation promotes it to:

```text
frames/sampled/
```

`frames/manifest.json` stores explicit numeric timestamps, source fingerprint, sampling configuration, frame count, relative paths, sizes, and an artifact fingerprint. Analysis never relies on filenames as the source of timestamp truth.

Safety controls include:

- maximum sampling FPS
- maximum sampled-frame count
- estimated JPEG disk usage
- disk safety margin
- duration-aware sampling timeout
- cancellation that terminates FFmpeg
- partial-directory cleanup

## 3.2 Frame quality and preprocessing

Phase 3 uses `opencv-python-headless`. Original sampled frames are never overwritten.

Each valid sampled frame produces a derived analysis frame under:

```text
frames/processed/
```

The default analysis width is 640 px with aspect ratio preserved. Smaller frames are not upscaled.

Per-frame classical CV measurements include:

```text
brightness          0..1
contrast            0..1
Laplacian sharpness
edge density        0..1
near-black ratio    0..1
black flag
very-dark flag
blurry flag
low-information flag
quality score       0..1
```

Corrupt images are recorded rather than silently dropped. A small invalid ratio is tolerated; exceeding `MAX_INVALID_FRAME_RATIO` fails the stage.

## 3.3 Visual difference scoring

Consecutive processed frames are compared using four independent signals:

```text
1. mean absolute grayscale pixel difference
2. SSIM difference
3. perceptual hash distance (OpenCV DCT pHash implementation)
4. Canny edge-map difference
```

All difference metrics are normalized so:

```text
0.0 = almost identical
1.0 = extremely different
```

The default combined score is:

```text
difference =
    pixel * 0.25
  + SSIM  * 0.35
  + pHash * 0.20
  + edge  * 0.20
```

Weights do not need to sum to 1; Notify normalizes them internally. Individual metrics are preserved in `analysis/differences.json` together with mean/median/max and p50/p75/p90/p95/p99 score statistics.

The comparison implementation keeps only neighboring images in memory rather than loading an entire lecture into RAM.

## 3.4 Major visual changes

Phase 3 does not use one brittle threshold such as `score > 0.5`.

It derives lecture-specific thresholds using robust statistics:

```text
median
MAD (median absolute deviation)
p90
p95
p99
absolute minimum floors
```

A simplified view is:

```text
major threshold = max(
    configured absolute floor,
    p95,
    median + robust_multiplier * scaled_MAD
)
```

Large nearby spikes are clustered into one transition and the strongest comparison becomes the representative timestamp. Black/fade sequences are collapsed into `BLACK_TRANSITION` events rather than emitted as repeated content changes. Invalid comparison gaps are preserved explicitly.

Current conservative event types are:

```text
MAJOR_VISUAL_CHANGE
VERY_MAJOR_VISUAL_CHANGE
BLACK_TRANSITION
INVALID_GAP
```

These events intentionally do **not** claim semantic meanings such as `NEW_SLIDE` or `BOARD_CLEARED` yet.

## 3.5 Temporal timeline

Point-level difference scores are converted into ordered visual-activity segments:

```text
STABLE
CHANGING
MAJOR_TRANSITION
BLACK_TRANSITION
INVALID
```

The timeline uses adaptive low-change thresholds plus an absolute stable ceiling, hysteresis, minimum stable/changing durations, and controlled weak-gap bridging.

This avoids noisy state flicker such as:

```text
STABLE -> CHANGING -> STABLE -> CHANGING
```

when scores merely hover around one threshold.

Major/black transitions and invalid regions remain hard boundaries. Actual timestamps drive all duration logic; Notify never assumes that one sampled frame equals one second.

The most important Phase-4 handoff pattern is:

```text
CHANGING
    ↓
STABLE
```

which may later represent completed writing, code, a diagram, or a settled slide. Phase 3 does not decide that semantic meaning.

## 3.6 Phase-3 cache and restart recovery

The dependency graph is explicit:

```text
Phase-2 source video
        ↓
frame sampling
        ↓
preprocessing
        ↓
difference scoring
        ↓
major changes
        ↓
timeline
```

Every stage stores:

```text
algorithm/schema version
configuration fingerprint
upstream artifact fingerprint
its own artifact fingerprint
completion checkpoint
```

Cache states are:

```text
VALID
MISSING
STALE
CORRUPT
PARTIAL
```

Examples of precise invalidation:

```text
FRAME_SAMPLE_FPS changes
  -> sampling + every downstream Phase-3 stage reruns

ANALYSIS_FRAME_WIDTH changes
  -> keep sampled frames
  -> preprocessing + downstream rerun

DIFF_SSIM_WEIGHT changes
  -> keep all image artifacts
  -> differences + downstream rerun

MAJOR_CHANGE_MERGE_WINDOW_SECONDS changes
  -> major changes + timeline rerun

MIN_STABLE_DURATION_SECONDS changes
  -> timeline only reruns
```

Known temporary artifacts such as `sampled.tmp`, `processed.tmp`, and atomic JSON temp files are cleaned before recovery. Paths are always resolved through the UUID job workspace.

A recovered RUNNING job becomes QUEUED as established in Phase 1; Phase-3 artifact validation then selects the real resume stage. `job.stage` alone is not treated as truth.

## 3.7 Integration and evaluation

`FrameAnalysisPipeline` is the single Phase-3 orchestrator. `NotifyPipeline` runs:

```text
IngestionPipeline(finalize_job=False)
        ↓
FrameAnalysisPipeline(finalize_job=True)
```

so Phase 2 can finish without prematurely terminalizing the job.

A validated `analysis/summary.json` contains counts, artifact fingerprints, per-stage timing, and disk usage. Only after final validation is `FRAME_ANALYSIS_COMPLETE` written.

`COMPLETED` currently means **all implemented work through Phase 3 has finished**. It does not mean screenshots or a PDF exist. The success message is:

```text
Frame analysis completed; ready for stability candidate generation
```

### Evaluation tool

Evaluation is downstream diagnostics only; it never changes production thresholds automatically.

Run:

```bash
python scripts/evaluate_phase3.py storage/jobs/<job_id>
```

The report includes:

- sampled/valid/invalid/black/blurry frame counts
- p50/p75/p90/p95/p99/max difference scores
- major events per minute
- stable/changing ratios
- segment density
- mean/median stable and changing durations
- example event timestamps
- warnings for suspicious fragmentation, invalid-frame ratios, extreme event density, or almost-all-stable/changing timelines

It writes:

```text
analysis/evaluation.json
```

Changing the evaluation script does not invalidate Phase-3 production artifacts.

# API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Application / implementation phase |
| `GET` | `/health` | Health check |
| `POST` | `/api/jobs` | Create and queue a lecture job |
| `GET` | `/api/jobs` | List jobs |
| `GET` | `/api/jobs/{job_id}` | Live status/progress/error |
| `POST` | `/api/jobs/{job_id}/cancel` | Cancel queued/running work |
| `POST` | `/api/jobs/{job_id}/retry` | Retry failed/cancelled work |
| `GET` | `/api/jobs/{job_id}/metadata` | Normalized YouTube metadata |
| `GET` | `/api/jobs/{job_id}/ingestion` | Phase-2 ingestion result |
| `GET` | `/api/jobs/{job_id}/analysis` | Compact Phase-3 summary |
| `GET` | `/api/jobs/{job_id}/analysis/cache` | Phase-3 cache states + resume stage |
| `GET` | `/api/system/capabilities` | CPU/RAM/GPU + FFmpeg availability |
| `POST` | `/api/system/cleanup` | Dry-run/execute retention cleanup |

The analysis summary endpoint intentionally does not return thousands of frame/comparison records.

# Progress allocation

Current processing allocation:

```text
Phase 2 ingestion               0-50
3.1 frame sampling             50-60
3.2 frame preprocessing        60-67
3.3 difference scoring         67-74
3.4 major transitions          74-79
3.5 temporal timeline          79-84
3.7 final validation           84-86
terminal development state       100
```

The unused range is intentionally reserved for later candidate generation, transcription, VLM judging, screenshot selection, deduplication, and PDF generation.

Global job progress is monotonic, including after restart.

# Configuration

Copy:

```bash
cp .env.example .env
```

Important Phase-2 values:

```text
PROCESSOR_MODE=analysis
VIDEO_MAX_HEIGHT=1080
VIDEO_PREFERRED_CONTAINER=mp4
MAX_VIDEO_DOWNLOAD_GB=4
FFMPEG_PATH=
FFPROBE_PATH=
AUDIO_EXTRACTION_TIMEOUT_MULTIPLIER=3
```

Important Phase-3 values:

```text
FRAME_SAMPLE_FPS=1.0
FRAME_SAMPLE_MAX_FPS=5.0
MAX_SAMPLED_FRAMES=20000
FRAME_ESTIMATED_SIZE_KB=150
FRAME_DISK_SAFETY_MARGIN_MB=512
FRAME_JPEG_QUALITY=90

ANALYSIS_FRAME_WIDTH=640
MAX_INVALID_FRAME_RATIO=0.02
DARK_FRAME_THRESHOLD=0.05
BLACK_PIXEL_VALUE_THRESHOLD=10
BLACK_PIXEL_RATIO_THRESHOLD=0.95
BLUR_VARIANCE_THRESHOLD=50

DIFF_PIXEL_WEIGHT=0.25
DIFF_SSIM_WEIGHT=0.35
DIFF_PHASH_WEIGHT=0.20
DIFF_EDGE_WEIGHT=0.20
DIFF_PHASH_SIZE=8
DIFF_CANNY_LOW=100
DIFF_CANNY_HIGH=200
MAX_INVALID_COMPARISON_RATIO=0.02

MAJOR_CHANGE_MIN_SCORE=0.30
VERY_MAJOR_CHANGE_MIN_SCORE=0.60
MAJOR_CHANGE_ROBUST_MULTIPLIER=6
MAJOR_CHANGE_MERGE_WINDOW_SECONDS=2

TIMELINE_STABLE_MIN_SCORE=0.03
TIMELINE_STABLE_MAX_SCORE=0.10
TIMELINE_STABLE_ROBUST_MULTIPLIER=1.5
TIMELINE_EXIT_STABLE_FACTOR=1.6
MIN_STABLE_DURATION_SECONDS=2
MIN_CHANGING_DURATION_SECONDS=1
MAX_STABLE_GAP_SECONDS=0.5
```

Thresholds are deliberately versioned/configurable because lecture styles differ. Evaluation should guide tuning; it must not silently rewrite `.env`.

# Install

Python 3.11+ is required.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
winget install Gyan.FFmpeg
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

Ubuntu/Debian FFmpeg:

```bash
sudo apt update
sudo apt install ffmpeg
```

Verify:

```bash
ffmpeg -version
ffprobe -version
```

# Run

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

# Tests

```bash
pytest
```

Tests do not require live YouTube access. External YouTube/FFmpeg boundaries are mocked where appropriate while the real application orchestration, OpenCV preprocessing, difference metrics, robust thresholding, temporal segmentation, cache logic, and FastAPI/background runner are exercised.

Phase-3 coverage includes:

- sampling interval/count/config validation
- FFmpeg command construction
- max-frame/disk safeguards
- sampling success/cancellation/timeout cleanup
- aspect-ratio resize without upscaling
- brightness/contrast/sharpness/edge/black-frame metrics
- corrupt-frame tolerance
- pixel/SSIM/pHash/edge comparison behavior
- metric weight normalization
- invalid comparison handling
- adaptive major thresholds
- event clustering and black transitions
- stable/changing timeline smoothing and hysteresis
- timestamp-aware duration
- precise cache invalidation and restart selection
- duplicate-execution serialization
- full Phase-2 -> Phase-3 FastAPI flow with mocked external media boundaries
- Phase-3 evaluation diagnostics

# Manual tools

Phase-2 real ingestion smoke test:

```bash
python scripts/smoke_ingestion.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Standalone FFmpeg frame-sampling smoke test:

```bash
python scripts/sample_frames.py path/to/video.mp4 --fps 1
```

Evaluate an already completed Phase-3 workspace:

```bash
python scripts/evaluate_phase3.py storage/jobs/<job_id>
```

# Security and local-first guarantees

Current implementation uses:

- no paid API
- no cloud storage
- no Redis
- no Kafka
- no Celery
- no database server
- no browser-cookie scraping
- no YouTube API key
- no OCR
- no Whisper yet
- no VLM/Qwen yet
- no GPU requirement for Phase 3

Subprocess commands use argument arrays instead of `shell=True`. User-controlled video titles never become filesystem paths. Persisted relative artifact paths are validated to remain inside the UUID job workspace before use/deletion.

# Phase 4 — visual candidate analysis

Phase 4 deliberately reduces thousands of sampled frames to a compact set of visual opportunities before transcript/VLM work. It is metadata-driven except for candidate-only heuristic inspection.

## 4.1 Stability windows

Phase-3 `STABLE` segments are validated by duration, difference statistics and existing frame-quality metadata. Black/invalid-heavy windows can be rejected while dark but informative content remains eligible. Each window preserves previous/next timeline state and major-transition proximity.

## 4.2 Stable boundaries

Notify detects conservative temporal patterns such as:

```text
CHANGING -> STABLE
MAJOR_TRANSITION -> STABLE
BLACK_TRANSITION -> STABLE
```

It measures pre/post activity drop and records boundary strength without claiming semantic labels such as “diagram complete.”

## 4.3 Candidate generation

A valid opportunity produces only a few strategic timestamps (`SETTLED_START`, `MID_STABLE`, `PRE_EXIT`, and longer-window variants). Candidates are mapped to the nearest valid non-black analysis frame using real timestamps, spacing limits and bounded search radius. `MAX_CANDIDATES_PER_WINDOW` and `MAX_TOTAL_CANDIDATES` prevent candidate explosion.

## 4.4 Explainable heuristics

Each candidate keeps separate features including:

```text
visual quality
content density
local stability
transition risk / safety
preceding activity
content accumulation
positional completeness
completeness heuristic
```

`completeness_heuristic_score` is not semantic truth. It only estimates whether a visual state looks like a useful settled state using deterministic classical signals.

## 4.5 Ranking and selection

Candidates compete only within their own stability window. Ranking combines quality, heuristic completeness, safety, local stability, boundary strength, density and accumulation. The result preserves all ranked candidates while selecting a `PRIMARY` and, when useful, an `ALTERNATE`. Close top scores mark the window as ambiguous for later VLM comparison.

## 4.6 Cache and resume

```text
Phase 3 timeline
  -> 4.1 windows
  -> 4.2 boundaries
  -> 4.3 generated candidates
  -> 4.4 heuristic scores
  -> 4.5 rankings/selections
```

Every stage validates its checkpoint, artifact schema, deterministic config fingerprint, upstream fingerprint and internal references. A ranking-weight change reruns only 4.5; a candidate-generation setting change reruns 4.3–4.5; a Phase-3 timeline change invalidates all Phase 4. Known `.json.tmp` files are cleaned on recovery and never count as valid cache.

## Phase-4 APIs

```text
GET /api/jobs/{job_id}/candidates
GET /api/jobs/{job_id}/candidates/selections
GET /api/jobs/{job_id}/candidates/cache
```

The summary endpoints do not embed image bytes or thousands of raw frame records.

## Evaluation

Evaluate an already completed Phase-4 workspace:

```bash
python scripts/evaluate_phase4.py <job_id> --storage-root storage/jobs --persist
```

The evaluator reports window/boundary/candidate density, heuristic distributions, ambiguity, winner-type skew and diagnostic warnings. It never auto-tunes production thresholds.

# Phase 5/6 handoff

Later transcription/VLM stages can consume a compact handoff containing PRIMARY/ALTERNATE timestamp, frame references, ranking score, selection confidence, ambiguity, boundary evidence, visual quality, heuristic completeness and transition safety. Sampled/processed frames remain proxies; final screenshots must later be extracted from `source/video.*`.

Not implemented yet:

- exact high-quality screenshot extraction from source video
- final cross-window visual deduplication
- PDF generation


## Phase 5 — Local transcription and candidate context

Phase 5 turns the Phase-2 normalized WAV and Phase-4 retained visual candidates into timestamped transcript context for later semantic visual reasoning. The production path remains local-first: `faster-whisper` is loaded lazily, CPU `int8` is the default, CUDA is optional, and CI/unit tests inject a fake speech-to-text adapter so model downloads are never required.

```text
CANDIDATES_READY + audio/audio.wav
        ↓
Audio validation + streaming signal diagnostics
        ↓
Deterministic logical chunk plan
        ↓
faster-whisper raw chunk transcripts
        ↓
Overlap-aware transcript normalization
        ↓
Indexed candidate ↔ speech alignment
        ↓
Before / current / after candidate context
        ↓
TRANSCRIPT_CONTEXT_READY
```

### Phase-5 workspace contract

```text
transcript/
├── preparation.json
├── raw_chunks/
│   ├── chunk_0001.json
│   └── ...
├── raw_transcript.json
├── transcript.json
├── candidate_alignment.json
├── contexts.json
├── summary.json
└── evaluation.json        # optional diagnostics
```

`preparation.json` verifies the WAV can actually be decoded as 16 kHz mono 16-bit PCM, checks audio/video duration drift, measures RMS/silence/clipping in streaming windows, and defines full-lecture logical chunks with bounded decode overlap. Silence and modest clipping are diagnostics; structurally unusable or effectively silent audio is rejected.

Raw transcription is persisted per chunk before final assembly. Completed chunk artifacts are fingerprinted by audio identity, chunk definition, model configuration and algorithm version. A restart can therefore reuse valid chunks, recompute only a missing/corrupt chunk, or rebuild `raw_transcript.json` entirely from chunk artifacts without loading Whisper. Overlap duplicates are deliberately kept raw and resolved in Phase 5.3 with conservative time + text similarity while retaining raw provenance.

The normalized transcript remains fine-grained for temporal queries. Phase 5.4 builds one transcript timeline index and aligns only retained PRIMARY/ALTERNATE visual candidates using overlap/previous/next speech, optional word timing, and a bounded speech-proximity heuristic. Speech proximity never reranks or rejects visual candidates. Phase 5.5 then builds bounded `before_text`, `current_text`, and `after_text` packages; CURRENT speech has highest retention priority, oldest BEFORE text and farthest AFTER text are trimmed first, and transcript wording/Unicode are preserved exactly from normalization.

The Phase-5 cache dependency chain is:

```text
Phase-2 audio
  ↓
5.1 preparation
  ↓
5.2 raw chunks + raw transcript
  ↓
5.3 normalized transcript
  ↓            Phase-4 selections
5.4 alignment ←────────────┘
  ↓
5.5 contexts
```

Changes are invalidated precisely: chunk-plan changes restart 5.1+, Whisper configuration restarts 5.2+, normalization settings restart 5.3+, alignment settings or candidate-selection changes restart 5.4+, and context-window changes restart only 5.5. Routine cache inspection reads metadata/fingerprints and does not decode audio or initialize Whisper.

Use the local evaluator after a completed job:

```bash
python scripts/evaluate_phase5.py storage/jobs/<job-id>
```

It reports audio diagnostics, chunk/transcription structure, overlap normalization, speech coverage, candidate-speech alignment and context size/truncation warnings. It never auto-tunes model or production thresholds.

Useful Phase-5 API endpoints include:

- `GET /api/jobs/{job_id}/transcript/preparation` — audio/chunk preparation metadata.
- `GET /api/jobs/{job_id}/transcript/summary` — compact Phase-5 summary.
- `GET /api/jobs/{job_id}/transcript/alignment` — alignment metadata without duplicating transcript text.
- `GET /api/jobs/{job_id}/transcript/contexts` — compact context statistics.
- `GET /api/jobs/{job_id}/transcript/contexts/{candidate_id}` — one structured candidate context.
- `GET /api/jobs/{job_id}/transcript/cache` — compact cache/resume state.

`TRANSCRIPT_CONTEXT_READY` means visual candidates now have deterministic transcript context suitable for Phase 6. It does **not** mean the transcript is semantically perfect, the visual is approved, or a final PDF screenshot has been selected. Phase 6 remains responsible for Qwen3-VL semantic reasoning.


## Phase 6 — Local semantic visual analysis

Phase 6 consumes `TRANSCRIPT_CONTEXT_READY` and reduces the retained PRIMARY/ALTERNATE candidate set to semantically preferred completed teaching states. It deliberately keeps model inference candidate-scoped rather than sending the lecture or all frames to a VLM.

```text
TRANSCRIPT_CONTEXT_READY
        ↓
6.1 compact semantic input
        ↓
6.3 PREVIOUS / CURRENT / NEXT visual context
        ↓
6.4 Qwen3-VL structured semantic analysis
        ↓
6.5 deterministic completion/usefulness decision engine
        ↓
6.6 candidate-level cache + restart recovery
        ↓
6.7 final validation + evaluation
        ↓
SEMANTIC_CANDIDATES_READY
```

### Qwen runtime

The verified model tier mapping is:

```text
2b → Qwen/Qwen3-VL-2B-Instruct
4b → Qwen/Qwen3-VL-4B-Instruct   # default
8b → Qwen/Qwen3-VL-8B-Instruct
```

The runtime is lazy: application startup, Phase 1–5 processing, semantic cache inspection, and summary APIs do not load model weights. `QWEN_VL_DEVICE=auto` uses CUDA only when hardware/memory policy allows it and otherwise resolves to CPU. Dtype, revision, local-files-only behavior, image count/size limits and optional CUDA quantization are explicit configuration. The 2B model can be used as a bounded resource fallback when the primary 4B runtime hits an out-of-memory condition. Per-candidate artifacts retain the actual resolved model/fallback metadata.

Install heavy VLM dependencies only when real local Qwen inference is required:

```bash
pip install -e ".[vlm]"
```

Inspect runtime/hardware configuration without loading the model:

```bash
python scripts/smoke_qwen_runtime.py
```

Explicit model loading is opt-in:

```bash
python scripts/smoke_qwen_runtime.py --load-model
```

### Semantic workspace contract

```text
semantic/
├── input_manifest.json
├── temporal_contexts.json
├── raw_model_results/
│   ├── candidate_000042.json
│   └── ...
├── semantic_results.json
├── selections.json
├── summary.json
└── evaluation.json        # optional diagnostics
```

`input_manifest.json` joins Phase-4 retained candidate identity/frame metadata with Phase-5 transcript context without copying all prior diagnostics. `temporal_contexts.json` keeps CURRENT immutable while selecting at most one bounded meaningful PREVIOUS and NEXT processed frame, excluding invalid/black/near-duplicate frames. The VLM prompt explicitly labels image roles, includes only BEFORE/CURRENT/AFTER transcript context, requests JSON-only structured judgments, and never asks for chain-of-thought.

Semantic inference is persisted per candidate. A crash after candidate 20 can reuse candidates 1–20; a missing `semantic_results.json` is rebuilt from valid candidate artifacts without Qwen; a decision-weight change reruns only the deterministic decision engine. Prompt/model/runtime changes invalidate 6.4+, temporal-context changes invalidate affected candidates, while Phase-4 ranking-only or decision-weight changes preserve VLM inference.

### Phase-6 APIs

```text
GET /api/system/vlm
GET /api/jobs/{job_id}/semantic/input
GET /api/jobs/{job_id}/semantic/input/{candidate_id}
GET /api/jobs/{job_id}/semantic/context
GET /api/jobs/{job_id}/semantic/context/{candidate_id}
GET /api/jobs/{job_id}/semantic/selections
GET /api/jobs/{job_id}/semantic/summary
GET /api/jobs/{job_id}/semantic/cache
```

`GET /api/system/vlm` reports runtime/hardware configuration without downloading or loading Qwen. Job summary/cache endpoints do not return raw prompts or model responses.

Evaluate an existing Phase-6 workspace without running Qwen:

```bash
python scripts/evaluate_phase6.py storage/jobs/<job-id>
```

The evaluator reports temporal-context shape, content/completion/usefulness distributions, PRIMARY-vs-ALTERNATE outcomes, fallback/model use, inference latency, semantic selectivity and diagnostic warnings. It never auto-tunes the prompt, model, temporal windows or decision thresholds.

### Phase-7 handoff

`SEMANTIC_CANDIDATES_READY` exposes a compact selected-candidate handoff containing exact candidate timestamp, analysis frame reference, stable-window provenance, content type, completion/usefulness labels and deterministic semantic decision score. Phase 7 must use the exact timestamp to extract a clean high-resolution frame from `source/video.*`, then perform cross-window visual deduplication. Phase 6 does **not** extract final screenshots, deduplicate across windows, or generate a PDF.
