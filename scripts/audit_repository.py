from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".toml", ".txt", ".csv"}
MARKERS = ("to" + "do", "t" + "bd", "place" + "holder", "fix" + "me")
PROHIBITED = re.compile(r"\b(?:" + "|".join(MARKERS) + r")\b", re.IGNORECASE)
IDENTITY_PATTERNS = (
    re.compile(r"/" + "Users/" + r"[^/\s]+"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
)
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


def main() -> None:
    failures = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".py" and not text.isascii():
            failures.append(f"non-ASCII Python source: {path.relative_to(ROOT)}")
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=str(path))
            except SyntaxError as error:
                failures.append(f"Python syntax error: {path.relative_to(ROOT)}:{error.lineno}")
        if PROHIBITED.search(text):
            failures.append(f"prohibited marker: {path.relative_to(ROOT)}")
        for pattern in IDENTITY_PATTERNS:
            if pattern.search(text):
                failures.append(f"identity-like string: {path.relative_to(ROOT)}")
    if failures:
        print("Repository audit failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("Repository audit passed")


if __name__ == "__main__":
    main()
