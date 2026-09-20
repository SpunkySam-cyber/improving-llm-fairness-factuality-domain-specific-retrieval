from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


VERSION_DIR = Path(__file__).resolve().parents[1]
if str(VERSION_DIR) not in sys.path:
    sys.path.insert(0, str(VERSION_DIR))

from build_manifest import (  # noqa: E402
    LectureRules,
    SourceConfig,
    connect_manifest,
    ingest_entries,
    load_config,
    manifest_summary,
    normalize_title,
    screen_video,
    select_pilot,
)


class ScreeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = LectureRules(
            minimum_duration_seconds=600,
            excluded_title_patterns=(r"\btrailer\b", r"\bshorts?\b"),
        )
        self.patterns = (
            r"\babdal\s+hakim\s+murad\b",
            r"\btimothy\s+winter\b",
        )

    def test_normalize_title_is_case_and_punctuation_insensitive(self) -> None:
        self.assertEqual(
            normalize_title("The Qur’an — Abdal Hakim Murad"),
            "the qur an abdal hakim murad",
        )

    def test_accepts_full_lecture(self) -> None:
        result = screen_video(
            title="The Path to Allah - Abdal Hakim Murad",
            duration_seconds=2269,
            speaker_patterns=self.patterns,
            rules=self.rules,
        )
        self.assertEqual(result.candidate_status, "candidate")
        self.assertTrue(result.speaker_match)

    def test_rejects_short_promo(self) -> None:
        result = screen_video(
            title="Campaign Trailer - Abdal Hakim Murad",
            duration_seconds=172,
            speaker_patterns=self.patterns,
            rules=self.rules,
        )
        self.assertEqual(result.candidate_status, "excluded")
        self.assertEqual(result.exclusion_reason, "below_minimum_duration")

    def test_marks_missing_duration_for_review(self) -> None:
        result = screen_video(
            title="Lecture - Timothy Winter",
            duration_seconds=None,
            speaker_patterns=self.patterns,
            rules=self.rules,
        )
        self.assertEqual(result.candidate_status, "needs_review")

    def test_rejects_other_speakers(self) -> None:
        result = screen_video(
            title="Friday Sermon - Another Speaker",
            duration_seconds=1800,
            speaker_patterns=self.patterns,
            rules=self.rules,
        )
        self.assertEqual(result.candidate_status, "excluded")
        self.assertEqual(result.exclusion_reason, "speaker_not_in_title")


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp.name) / "pipeline.sqlite"
        self.connection = connect_manifest(self.db_path)
        self.rules = LectureRules(600, (r"\btrailer\b",))
        self.source_a = SourceConfig(
            "Source A",
            "https://www.youtube.com/@source-a/videos",
            (r"\babdal\s+hakim\s+murad\b",),
        )
        self.source_b = SourceConfig(
            "Source B",
            "https://www.youtube.com/@source-b/videos",
            (r"\babdal\s+hakim\s+murad\b",),
        )

    def tearDown(self) -> None:
        self.connection.close()
        self.temp.cleanup()

    @staticmethod
    def entry(video_id: str, index: int, duration: int = 1800) -> dict[str, object]:
        return {
            "id": video_id,
            "title": f"Lecture {index} - Abdal Hakim Murad",
            "duration": duration,
            "channel": "Test Channel",
            "channel_id": "UC_TEST",
        }

    def test_exact_video_id_is_deduplicated_across_sources(self) -> None:
        entry = self.entry("same-video", 1)
        first = ingest_entries(self.connection, self.source_a, self.rules, [entry])
        second = ingest_entries(self.connection, self.source_b, self.rules, [entry])
        self.assertEqual(first["inserted"], 1)
        self.assertEqual(second["exact_duplicates"], 1)
        self.assertEqual(manifest_summary(self.connection)["unique_videos"], 1)
        links = self.connection.execute("SELECT COUNT(*) FROM source_videos").fetchone()[0]
        self.assertEqual(links, 2)

    def test_source_name_fills_missing_flat_playlist_channel(self) -> None:
        entry = self.entry("no-channel", 1)
        entry.pop("channel")
        entry.pop("channel_id")
        ingest_entries(self.connection, self.source_a, self.rules, [entry])
        stored = self.connection.execute(
            "SELECT channel FROM videos WHERE video_id = 'no-channel'"
        ).fetchone()[0]
        self.assertEqual(stored, "Source A")

    def test_pilot_contains_unique_candidates_only(self) -> None:
        entries = [self.entry(f"video-{index}", index) for index in range(12)]
        entries.append(
            {
                "id": "other-speaker",
                "title": "Lecture - Another Speaker",
                "duration": 1800,
            }
        )
        ingest_entries(self.connection, self.source_a, self.rules, entries)
        pilot = select_pilot(self.connection, 10)
        self.assertEqual(len(pilot), 10)
        self.assertEqual(len({row["video_id"] for row in pilot}), 10)
        self.assertTrue(all(row["candidate_status"] == "candidate" for row in pilot))

    def test_pilot_can_skip_multi_speaker_title(self) -> None:
        entries = [
            self.entry("solo-1", 1),
            {
                "id": "multi",
                "title": "Event - Abdal Hakim Murad & Another Speaker",
                "duration": 1800,
            },
            self.entry("solo-2", 2),
        ]
        ingest_entries(self.connection, self.source_a, self.rules, entries)
        pilot = select_pilot(self.connection, 2, (r"\bmurad\s*&\s+[a-z]",))
        self.assertEqual([row["video_id"] for row in pilot], ["solo-1", "solo-2"])

    def test_existing_reviewed_pilot_is_preserved_when_source_is_added(self) -> None:
        original = [self.entry(f"original-{index}", index) for index in range(3)]
        ingest_entries(self.connection, self.source_a, self.rules, original)
        first_pilot = select_pilot(self.connection, 2)
        first_ids = {row["video_id"] for row in first_pilot}

        newer_source_entries = [
            self.entry(f"new-source-{index}", index) for index in range(3)
        ]
        ingest_entries(
            self.connection, self.source_b, self.rules, newer_source_entries
        )
        second_pilot = select_pilot(self.connection, 2)

        self.assertEqual({row["video_id"] for row in second_pilot}, first_ids)

    def test_config_validation_and_loading(self) -> None:
        config_path = Path(self.temp.name) / "sources.json"
        config_path.write_text(
            json.dumps(
                {
                    "pilot_size": 5,
                    "lecture_rules": {"minimum_duration_seconds": 900},
                    "sources": [
                        {
                            "name": "Test",
                            "url": "https://www.youtube.com/@test/videos",
                            "speaker_patterns": ["murad"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        config = load_config(config_path)
        self.assertEqual(config.pilot_size, 5)
        self.assertEqual(config.lecture_rules.minimum_duration_seconds, 900)
        self.assertEqual(config.sources[0].name, "Test")


if __name__ == "__main__":
    unittest.main()
