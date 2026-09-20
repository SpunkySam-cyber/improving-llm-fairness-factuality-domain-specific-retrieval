from __future__ import annotations

# --- repo layout shim: find shared modules (V5 batch core, dialogue/dataset tools, project_env) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[3]
for _d in (_ROOT, _ROOT / "transcription_pipeline_versions" / "v5_batch_ingestion_pipeline", _ROOT / "tools" / "dataset", _ROOT / "tools" / "dialogue"):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
# --- end shim ---
import json
import sys
import tempfile
import unittest
from pathlib import Path


VERSION_DIR = Path(__file__).resolve().parents[1]
if str(VERSION_DIR) not in sys.path:
    sys.path.insert(0, str(VERSION_DIR))

from caption_pilot import (  # noqa: E402
    CaptionChoice,
    choose_english_caption,
    parse_json3_caption,
    saved_caption_is_valid,
)


class CaptionSelectionTests(unittest.TestCase):
    def test_prefers_manual_english_json3(self) -> None:
        info = {
            "subtitles": {
                "en": [
                    {"ext": "vtt", "url": "https://example/manual.vtt"},
                    {"ext": "json3", "url": "https://example/manual.json3"},
                ]
            },
            "automatic_captions": {
                "en": [{"ext": "json3", "url": "https://example/auto.json3"}]
            },
        }
        choice = choose_english_caption(info)
        self.assertEqual(
            choice,
            CaptionChoice("manual", "en", "json3", "https://example/manual.json3"),
        )

    def test_falls_back_to_automatic_english(self) -> None:
        info = {
            "subtitles": {},
            "automatic_captions": {
                "en": [{"ext": "json3", "url": "https://example/auto.json3"}]
            },
        }
        choice = choose_english_caption(info)
        self.assertIsNotNone(choice)
        self.assertEqual(choice.kind, "automatic")

    def test_json3_parser_preserves_timestamps_and_text(self) -> None:
        payload = json.dumps(
            {
                "events": [
                    {
                        "tStartMs": 1250,
                        "dDurationMs": 2000,
                        "segs": [{"utf8": "Bismillah"}, {"utf8": " al-Rahman"}],
                    }
                ]
            }
        ).encode()
        segments = parse_json3_caption(payload)
        self.assertEqual(segments[0]["start_seconds"], 1.25)
        self.assertEqual(segments[0]["duration_seconds"], 2.0)
        self.assertEqual(segments[0]["text"], "Bismillah al-Rahman")

    def test_resume_accepts_only_nonempty_saved_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transcript.txt"
            path.write_text("usable transcript\n", encoding="utf-8")
            row = {
                "caption_status": "available",
                "caption_path": str(path),
            }
            self.assertTrue(saved_caption_is_valid(row))
            path.write_text("", encoding="utf-8")
            self.assertFalse(saved_caption_is_valid(row))


if __name__ == "__main__":
    unittest.main()
