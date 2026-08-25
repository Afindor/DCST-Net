from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcstnet.pipeline import run_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete DCST-Net example pipeline")
    parser.add_argument("--config", default="configs/demo.toml")
    arguments = parser.parse_args()
    metrics = run_all(arguments.config)
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=True))


if __name__ == "__main__":
    main()
