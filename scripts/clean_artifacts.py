from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    for relative in ("artifacts", "runs"):
        target = ROOT / relative
        if target.is_dir():
            shutil.rmtree(target)
    for directory in ROOT.rglob("__pycache__"):
        shutil.rmtree(directory)


if __name__ == "__main__":
    main()
