from __future__ import annotations

import hashlib
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


VERSION_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = VERSION_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from database import initialize_database  # noqa: E402


EXPECTED_TABLES = {
    "schema_migrations",
    "videos",
    "transcripts",
    "transcript_chunks",
    "pipeline_runs",
    "model_calls",
    "corpus_terms",
    "corpus_variants",
    "corpus_evidence",
    "correction_proposals",
    "correction_decisions",
}


class DatabaseSchemaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp.name) / "test.sqlite"
        self.connection = initialize_database(self.db_path)

    def tearDown(self) -> None:
        self.connection.close()
        self.temp.cleanup()

    def insert_video_and_transcript(self) -> int:
        self.connection.execute(
            """
            INSERT INTO videos(video_id, canonical_url, title, content_type)
            VALUES ('video-1', 'https://youtu.be/video-1', 'Test lecture', 'lecture')
            """
        )
        text = "one two three four"
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cursor = self.connection.execute(
            """
            INSERT INTO transcripts(
                video_id, source_type, provider, transcript_text, word_count,
                text_sha256
            ) VALUES ('video-1', 'assemblyai', 'AssemblyAI', ?, 4, ?)
            """,
            (text, digest),
        )
        return int(cursor.lastrowid)

    def test_all_required_tables_are_created(self) -> None:
        tables = {
            row[0]
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        self.assertTrue(EXPECTED_TABLES.issubset(tables))

    def test_initialization_is_idempotent(self) -> None:
        self.connection.close()
        self.connection = initialize_database(self.db_path)
        count = self.connection.execute(
            "SELECT COUNT(*) FROM schema_migrations"
        ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_foreign_keys_reject_orphan_transcript(self) -> None:
        digest = "0" * 64
        with self.assertRaises(sqlite3.IntegrityError):
            self.connection.execute(
                """
                INSERT INTO transcripts(
                    video_id, source_type, transcript_text, word_count, text_sha256
                ) VALUES ('missing', 'assemblyai', 'text', 1, ?)
                """,
                (digest,),
            )

    def test_chunk_boundaries_and_counts_are_enforced(self) -> None:
        transcript_id = self.insert_video_and_transcript()
        digest = hashlib.sha256(b"one two three four").hexdigest()
        self.connection.execute(
            """
            INSERT INTO transcript_chunks(
                transcript_id, chunk_index, chunk_text, start_word, end_word,
                word_count, chunk_sha256
            ) VALUES (?, 0, 'one two three four', 0, 4, 4, ?)
            """,
            (transcript_id, digest),
        )
        with self.assertRaises(sqlite3.IntegrityError):
            self.connection.execute(
                """
                INSERT INTO transcript_chunks(
                    transcript_id, chunk_index, chunk_text, start_word, end_word,
                    word_count, chunk_sha256
                ) VALUES (?, 1, 'bad count', 3, 6, 2, ?)
                """,
                (transcript_id, hashlib.sha256(b"bad count").hexdigest()),
            )

    def test_invalid_content_type_is_rejected(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            self.connection.execute(
                """
                INSERT INTO videos(video_id, canonical_url, title, content_type)
                VALUES ('video-2', 'https://youtu.be/video-2', 'Bad type', 'podcast')
                """
            )

    def test_variant_categories_remain_separate(self) -> None:
        term_id = self.connection.execute(
            """
            INSERT INTO corpus_terms(canonical_text, normalized_text, category)
            VALUES ('Tasawwuf', 'tasawwuf', 'islamic_term')
            """
        ).lastrowid
        for variant_type in (
            "accepted_spelling",
            "observed_asr_error",
            "generated_possible_error",
        ):
            self.connection.execute(
                """
                INSERT INTO corpus_variants(
                    term_id, variant_text, normalized_variant, variant_type
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    term_id,
                    f"example {variant_type}",
                    f"example_{variant_type}",
                    variant_type,
                ),
            )
        count = self.connection.execute(
            "SELECT COUNT(*) FROM corpus_variants WHERE term_id = ?",
            (term_id,),
        ).fetchone()[0]
        self.assertEqual(count, 3)

    def test_verified_evidence_requires_reviewer_and_time(self) -> None:
        self.connection.execute(
            """
            INSERT INTO videos(video_id, canonical_url, title)
            VALUES ('video-1', 'https://youtu.be/video-1', 'Test lecture')
            """
        )
        term_id = self.connection.execute(
            """
            INSERT INTO corpus_terms(canonical_text, normalized_text, category)
            VALUES ('Ummah', 'ummah', 'islamic_term')
            """
        ).lastrowid
        with self.assertRaises(sqlite3.IntegrityError):
            self.connection.execute(
                """
                INSERT INTO corpus_evidence(
                    term_id, video_id, start_time_ms, context_excerpt,
                    evidence_kind, review_status
                ) VALUES (?, 'video-1', 1000, 'the ummah', 'spoken_audio', 'verified')
                """,
                (term_id,),
            )


if __name__ == "__main__":
    unittest.main()

