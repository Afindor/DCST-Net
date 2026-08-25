from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SOURCE_MANIFEST.sha256"
EXCLUDED_PARTS = {
    ".git",
    ".ruff_cache",
    ".venv",
    "artifacts",
    "build",
    "dcst_net.egg-info",
    "dist",
    "runs",
    "__pycache__",
}


def source_files() -> list[Path]:
    return [
        path
        for path in sorted(ROOT.rglob("*"))
        if path.is_file()
        and path != MANIFEST
        and not any(part in EXCLUDED_PARTS for part in path.parts)
        and path.name != ".DS_Store"
    ]


def lines() -> list[str]:
    values = []
    for path in source_files():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        values.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    return values


def verify() -> None:
    if not MANIFEST.is_file():
        raise FileNotFoundError("SOURCE_MANIFEST.sha256 is missing")
    expected = MANIFEST.read_text(encoding="utf-8").splitlines()
    actual = lines()
    if expected != actual:
        expected_set = set(expected)
        actual_set = set(actual)
        changed = sorted(expected_set ^ actual_set)
        details = "\n".join(changed[:20])
        raise ValueError(f"Source manifest mismatch:\n{details}")
    print(f"Source manifest verified ({len(actual)} files)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write or verify the source SHA-256 manifest")
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    if arguments.write:
        MANIFEST.write_text("\n".join(lines()) + "\n", encoding="utf-8")
        print(f"Source manifest written: {MANIFEST.name}")
    else:
        verify()


if __name__ == "__main__":
    main()
