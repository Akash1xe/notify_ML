from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from app.core.config import AppSettings
from app.semantic_analysis.runtime import QwenRuntimeManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Notify Qwen3-VL runtime without downloading weights by default")
    parser.add_argument("--load-model", action="store_true", help="Explicitly load the configured model and run one tiny smoke inference")
    args = parser.parse_args()
    runtime = QwenRuntimeManager(AppSettings())
    print(json.dumps(runtime.inspect_runtime().model_dump(mode="json"), indent=2))
    if args.load_model:
        image = Image.new("RGB", (512, 256), "white")
        result = runtime.generate([image], "Describe whether this blank image contains meaningful lecture content. Reply briefly.")
        print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
