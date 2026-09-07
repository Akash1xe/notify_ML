# Calibration

Calibration always starts from a locked Phase-9.2 baseline. The parameter registry is an explicit allow-list of behavior-affecting Phase 3/4/6/7 settings. Search is bounded (`LOCAL_SWEEP`, small grid or explicit manual profiles), deterministic, and never rewrites `.env` during experiments.

Guardrails reject new critical failures, new required-state misses, excessive precision loss and candidate explosion. A profile is recommended only after measured benchmark improvement. If no tested profile safely beats the baseline, the correct outcome is **no production threshold change**.

Use calibration samples for search and validation/regression samples only for final confirmation when the corpus is large enough. Small/no-split datasets must be treated as overfitting risk.
