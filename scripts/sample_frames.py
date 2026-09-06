from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual FFmpeg frame-sampling smoke tool")
    parser.add_argument("video", type=Path)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("sampled_frames"))
    args = parser.parse_args()
    if args.fps <= 0 or args.fps > 5:
        raise SystemExit("--fps must be > 0 and <= 5")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("FFmpeg was not found on PATH")
    source = args.video.resolve()
    if not source.is_file():
        raise SystemExit(f"Video not found: {source}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pattern = output / "frame_%08d.jpg"
    command = [
        ffmpeg, "-y", "-i", str(source), "-vf", f"fps=fps={args.fps:g}:start_time=0:round=near",
        "-q:v", "5", "-start_number", "1", str(pattern),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit((result.stderr or "FFmpeg sampling failed")[-1000:])
    files = sorted(output.glob("frame_*.jpg"))
    manifest = {
        "sample_fps": args.fps,
        "interval_seconds": 1.0 / args.fps,
        "frame_count": len(files),
        "frames": [
            {"index": i, "timestamp_seconds": (i - 1) / args.fps, "path": path.name}
            for i, path in enumerate(files, start=1)
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Sampled {len(files)} frames into {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
