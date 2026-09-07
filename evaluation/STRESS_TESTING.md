# Stress Testing

Stress tiers are `CI`, `LOCAL_STANDARD`, `LOCAL_HEAVY` and `MANUAL_EXTREME`. CI is offline and uses logical stage-heavy fixtures instead of multi-hour videos/models.

Scenarios cover long slides/whiteboard/code, candidate and duplicate pressure, huge transcript alignment, many semantic candidates, 500-page PDF behavior and a 500-item result gallery.

The stress runner records duration, RSS/VRAM where available, workspace/source/temp sizes, frame/candidate/screenshot counts and PDF pages/bytes. Resource limits are typed as hard, soft or warning-only. Hard-limit failure must preserve valid upstream cache and must never produce a false ready checkpoint.
