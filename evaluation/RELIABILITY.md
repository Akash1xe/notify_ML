# Reliability

Failure injection uses `FailureInjector` with a no-op production default. Test scenarios trigger controlled failures at stage boundaries, persistence boundaries and iteration points without scattering test flags through production code.

The recovery planner composes existing phase cache validators, chooses the earliest invalid phase, preserves valid upstream artifacts and removes only job-local temporary files. Reliability scenarios include download/FFmpeg failures, interrupted sampling/transcription/semantic work, screenshot/hash corruption, PDF crash/corruption, restart, permission/disk failures, cancellation and concurrent retry.

A scenario passes only when the typed error, no-false-ready invariant, cleanup, cache preservation, resume stage and retry/isolation expectations all pass.
