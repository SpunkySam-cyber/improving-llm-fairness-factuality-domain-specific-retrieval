from __future__ import annotations

# --- repo layout shim: find shared modules (V5 batch core, dialogue/dataset tools, project_env) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[3]
for _d in (_ROOT, _ROOT / "transcription_pipeline_versions" / "v5_batch_ingestion_pipeline", _ROOT / "tools" / "dataset", _ROOT / "tools" / "dialogue"):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
# --- end shim ---
import sys
import unittest
from pathlib import Path


VERSION_DIR = Path(__file__).resolve().parents[1]
if str(VERSION_DIR) not in sys.path:
    sys.path.insert(0, str(VERSION_DIR))

from verify_corrections_offline import first_position, read_pcm16_wav  # noqa: E402


class VerifyCorrectionsOfflineTests(unittest.TestCase):
    def test_position_parser_accepts_single_and_range(self) -> None:
        self.assertEqual(first_position("131"), 131)
        self.assertEqual(first_position("671-672"), 671)

    def test_position_parser_rejects_invalid_value(self) -> None:
        with self.assertRaises(ValueError):
            first_position("unknown")


if __name__ == "__main__":
    unittest.main()
