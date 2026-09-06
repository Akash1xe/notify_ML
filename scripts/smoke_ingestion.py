"""Manual end-to-end Phase-2 smoke test.

Requires internet access, yt-dlp, and FFmpeg/ffprobe. It is intentionally not
part of automated CI.
"""
from __future__ import annotations

import argparse
import time

from fastapi.testclient import TestClient

from app.core.config import AppSettings
from app.main import create_app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Public recorded YouTube lecture URL")
    args = parser.parse_args()

    settings = AppSettings(processor_mode="ingestion")
    app = create_app(settings)
    with TestClient(app) as client:
        created = client.post("/api/jobs", json={"source_url": args.url})
        created.raise_for_status()
        job_id = created.json()["id"]
        print(f"job={job_id}")
        while True:
            job = client.get(f"/api/jobs/{job_id}").json()
            print(f"{job['status']:10} {job['stage']:20} {job['progress']:3}% {job['message']}")
            if job["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                break
            time.sleep(1)
        if job["status"] == "COMPLETED":
            result = client.get(f"/api/jobs/{job_id}/ingestion").json()
            print("ingestion:", result)
            return 0
        print("error:", job.get("error"))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
