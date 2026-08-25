from __future__ import annotations

import json
import unittest
from pathlib import Path

from dcstnet.pipeline import run_all

ROOT = Path(__file__).resolve().parents[1]


class PipelineTest(unittest.TestCase):
    def test_complete_demo_pipeline(self) -> None:
        metrics = run_all(ROOT / "configs/demo.toml")
        output = ROOT / "artifacts/demo"
        self.assertGreater(metrics["trajectory_points"], 0)
        self.assertTrue((output / "checkpoints/phase_a.pt").is_file())
        self.assertTrue((output / "cache/phase_a_trends.npz").is_file())
        self.assertTrue((output / "checkpoints/phase_b.pt").is_file())
        self.assertTrue((output / "evaluation/predictions.csv").is_file())
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["config_sha256"], load_digest())


def load_digest() -> str:
    from dcstnet.config import load_config

    return load_config(ROOT / "configs/demo.toml").digest()


if __name__ == "__main__":
    unittest.main()
