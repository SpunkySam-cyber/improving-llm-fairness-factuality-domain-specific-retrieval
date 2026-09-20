from __future__ import annotations

import sys
import unittest
from pathlib import Path


VERSION_DIR = Path(__file__).resolve().parents[1]
if str(VERSION_DIR) not in sys.path:
    sys.path.insert(0, str(VERSION_DIR))

from run_fresh_benchmark import request_cache_key, select_video  # noqa: E402


class FreshBenchmarkTests(unittest.TestCase):
    def test_default_selection_is_shortest(self) -> None:
        manifest = {
            "videos": [
                {"video_id": "long", "duration_seconds": 1000},
                {"video_id": "short", "duration_seconds": 700},
            ]
        }
        self.assertEqual(select_video(manifest, None)["video_id"], "short")

    def test_explicit_selection_must_be_in_fixed_pilot(self) -> None:
        manifest = {"videos": [{"video_id": "a", "duration_seconds": 700}]}
        with self.assertRaises(ValueError):
            select_video(manifest, "missing")

    def test_cache_key_changes_with_model_or_prompt(self) -> None:
        base = {
            "model": "model-a",
            "system_prompt": "system",
            "user_message": "text",
            "max_tokens": 100,
            "stage": "detect",
        }
        changed = dict(base, model="model-b")
        self.assertNotEqual(request_cache_key(base), request_cache_key(changed))
        self.assertEqual(request_cache_key(base), request_cache_key(dict(base)))


if __name__ == "__main__":
    unittest.main()
