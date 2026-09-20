from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


VERSION_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = VERSION_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from database import initialize_database  # noqa: E402
from export_chunk_database_view import load_chunk_view, render_chunk_view  # noqa: E402
from import_existing_pilot import (  # noqa: E402
    PilotImportError,
    import_bundle,
    load_pilot_bundle,
    verify_import,
)


class Stage4ImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pilot_dir = self.root / "pilot"
        benchmark = self.pilot_dir / "videos" / "test-video" / "fresh_benchmark"
        benchmark.mkdir(parents=True)
        manifest = {
            "videos": [
                {
                    "video_id": "test-video",
                    "title": "Test lecture",
                    "url": "https://www.youtube.com/watch?v=test-video",
                    "channel": "Test channel",
                    "channel_id": "UC_TEST",
                    "duration_seconds": 10,
                    "upload_date": "20260101",
                }
            ]
        }
        (self.pilot_dir / "pilot_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        self.text = "one two three four five"
        (benchmark / "assemblyai_transcript.txt").write_text(
            self.text + "\n", encoding="utf-8"
        )
        metadata = {"transcript_id": "provider-1", "word_count": 5}
        (benchmark / "assemblyai_metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        response = {
            "id": "provider-1",
            "status": "completed",
            "language_code": "en",
            "text": self.text,
            "words": [
                {"text": word, "start": index * 100, "end": index * 100 + 80}
                for index, word in enumerate(self.text.split())
            ],
        }
        (benchmark / "assemblyai_response.json").write_text(
            json.dumps(response), encoding="utf-8"
        )
        self.connection = initialize_database(self.root / "test.sqlite")

    def tearDown(self) -> None:
        self.connection.close()
        self.temp.cleanup()

    def test_import_is_idempotent_and_preserves_timestamps(self) -> None:
        bundle = load_pilot_bundle(self.pilot_dir, "test-video")
        first = import_bundle(self.connection, bundle)
        second = import_bundle(self.connection, bundle)
        self.assertEqual(first["transcript_id"], second["transcript_id"])
        self.assertEqual(
            self.connection.execute("SELECT COUNT(*) FROM videos").fetchone()[0], 1
        )
        self.assertEqual(
            self.connection.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0],
            1,
        )
        chunk = self.connection.execute(
            "SELECT start_time_ms, end_time_ms FROM transcript_chunks"
        ).fetchone()
        self.assertEqual(tuple(chunk), (0, 480))
        self.assertEqual(
            self.connection.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0],
            1,
        )
        verification = verify_import(self.connection, first)
        self.assertTrue(verification["continuous_word_coverage"])
        self.assertTrue(verification["all_chunk_checksums_match"])
        self.assertEqual(verification["model_call_count"], 0)

    def test_mismatched_saved_response_is_rejected(self) -> None:
        response_path = (
            self.pilot_dir
            / "videos"
            / "test-video"
            / "fresh_benchmark"
            / "assemblyai_response.json"
        )
        response = json.loads(response_path.read_text(encoding="utf-8"))
        response["text"] = "different"
        response_path.write_text(json.dumps(response), encoding="utf-8")
        with self.assertRaises(PilotImportError):
            load_pilot_bundle(self.pilot_dir, "test-video")

    def test_database_view_contains_all_chunk_columns_and_png(self) -> None:
        bundle = load_pilot_bundle(self.pilot_dir, "test-video")
        import_bundle(self.connection, bundle)
        view = load_chunk_view(self.connection, "test-video")
        table_columns = [
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(transcript_chunks)")
        ]
        self.assertEqual(view["columns"], table_columns)
        self.assertEqual(set(view["row"]), set(table_columns))
        image_path = self.root / "view.png"
        render_chunk_view(view, image_path)
        self.assertTrue(image_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
