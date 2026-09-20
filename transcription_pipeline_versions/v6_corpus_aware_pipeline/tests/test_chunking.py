from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


VERSION_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = VERSION_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chunking import (  # noqa: E402
    ChunkingConfig,
    ChunkingError,
    CorrectionConflictError,
    CorrectionSpan,
    TranscriptChunk,
    WordTiming,
    chunk_and_persist_transcript,
    create_chunks,
    merge_corrections,
    reconstruct_words,
    sha256_text,
)
from database import initialize_database  # noqa: E402


def numbered_text(word_count: int) -> str:
    return " ".join(f"word{index}" for index in range(word_count))


class ChunkCreationTests(unittest.TestCase):
    def test_empty_text_has_no_chunks(self) -> None:
        self.assertEqual(create_chunks("  \n\t "), [])

    def test_exactly_400_words_has_one_chunk(self) -> None:
        chunks = create_chunks(numbered_text(400))
        self.assertEqual(len(chunks), 1)
        self.assertEqual((chunks[0].start_word, chunks[0].end_word), (0, 400))
        self.assertEqual(chunks[0].overlap_before_words, 0)
        self.assertEqual(chunks[0].overlap_after_words, 0)

    def test_401_words_has_two_chunks_with_100_word_overlap(self) -> None:
        chunks = create_chunks(numbered_text(401))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(
            [(chunk.start_word, chunk.end_word) for chunk in chunks],
            [(0, 400), (300, 401)],
        )
        self.assertEqual(chunks[0].overlap_after_words, 100)
        self.assertEqual(chunks[1].overlap_before_words, 100)

    def test_one_hour_estimate_produces_27_chunks(self) -> None:
        chunks = create_chunks(numbered_text(8057))
        self.assertEqual(len(chunks), 27)
        self.assertEqual((chunks[-1].start_word, chunks[-1].end_word), (7800, 8057))
        self.assertEqual(reconstruct_words(chunks), tuple(numbered_text(8057).split()))

    def test_chunk_text_preserves_internal_whitespace_and_checksum(self) -> None:
        text = "  first\nsecond\tthird   fourth  "
        chunks = create_chunks(text, ChunkingConfig(max_words=3, overlap_words=1))
        self.assertEqual(chunks[0].text, "first\nsecond\tthird")
        self.assertEqual(chunks[0].sha256, sha256_text(chunks[0].text))

    def test_optional_word_timings_reach_chunk_boundaries(self) -> None:
        timings = [WordTiming(index * 100, index * 100 + 80) for index in range(5)]
        chunks = create_chunks(
            numbered_text(5),
            ChunkingConfig(max_words=4, overlap_words=1),
            timings,
        )
        self.assertEqual((chunks[0].start_time_ms, chunks[0].end_time_ms), (0, 380))
        self.assertEqual((chunks[1].start_time_ms, chunks[1].end_time_ms), (300, 480))

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ChunkingError):
            ChunkingConfig(max_words=100, overlap_words=100)

    def test_conflicting_overlap_is_detected(self) -> None:
        chunks = create_chunks(
            numbered_text(5), ChunkingConfig(max_words=4, overlap_words=2)
        )
        second = chunks[1]
        corrupted = TranscriptChunk(
            chunk_index=second.chunk_index,
            text=second.text.replace("word2", "wrong", 1),
            start_word=second.start_word,
            end_word=second.end_word,
            overlap_before_words=second.overlap_before_words,
            overlap_after_words=second.overlap_after_words,
            sha256=second.sha256,
        )
        with self.assertRaises(ChunkingError):
            reconstruct_words([chunks[0], corrupted])

    def test_incorrect_overlap_metadata_is_detected(self) -> None:
        chunks = create_chunks(
            numbered_text(5), ChunkingConfig(max_words=4, overlap_words=2)
        )
        second = chunks[1]
        incorrect = TranscriptChunk(
            chunk_index=second.chunk_index,
            text=second.text,
            start_word=second.start_word,
            end_word=second.end_word,
            overlap_before_words=1,
            overlap_after_words=second.overlap_after_words,
            sha256=second.sha256,
        )
        with self.assertRaises(ChunkingError):
            reconstruct_words([chunks[0], incorrect])


class CorrectionMergeTests(unittest.TestCase):
    def test_duplicate_overlap_correction_is_applied_once(self) -> None:
        source = "The um will have ikhtilaf in it."
        correction = CorrectionSpan(1, 2, "ummah")
        merged = merge_corrections(source, [correction, correction])
        self.assertEqual(merged, "The ummah will have ikhtilaf in it.")

    def test_conflicting_duplicate_corrections_are_rejected(self) -> None:
        with self.assertRaises(CorrectionConflictError):
            merge_corrections(
                "one two three",
                [CorrectionSpan(1, 2, "second"), CorrectionSpan(1, 2, "two")],
            )

    def test_different_overlapping_spans_are_rejected(self) -> None:
        with self.assertRaises(CorrectionConflictError):
            merge_corrections(
                "one two three four",
                [CorrectionSpan(1, 3, "replacement"), CorrectionSpan(2, 4, "other")],
            )


class PersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp.name) / "test.sqlite"
        self.connection = initialize_database(self.db_path)

    def tearDown(self) -> None:
        self.connection.close()
        self.temp.cleanup()

    def insert_transcript(self, text: str) -> int:
        self.connection.execute(
            """
            INSERT INTO videos(video_id, canonical_url, title, content_type)
            VALUES ('video-1', 'https://youtu.be/video-1', 'Test', 'lecture')
            """
        )
        cursor = self.connection.execute(
            """
            INSERT INTO transcripts(
                video_id, source_type, transcript_text, word_count, text_sha256
            ) VALUES ('video-1', 'imported', ?, ?, ?)
            """,
            (text, len(text.split()), hashlib.sha256(text.encode()).hexdigest()),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def test_chunks_are_persisted_and_repeat_run_is_idempotent(self) -> None:
        text = numbered_text(805)
        transcript_id = self.insert_transcript(text)
        first = chunk_and_persist_transcript(self.connection, transcript_id)
        second = chunk_and_persist_transcript(self.connection, transcript_id)
        stored_count = self.connection.execute(
            "SELECT COUNT(*) FROM transcript_chunks WHERE transcript_id = ?",
            (transcript_id,),
        ).fetchone()[0]
        self.assertEqual(len(first), 3)
        self.assertEqual(first, second)
        self.assertEqual(stored_count, 3)

    def test_changed_transcript_checksum_is_rejected(self) -> None:
        transcript_id = self.insert_transcript("one two three")
        self.connection.execute(
            "UPDATE transcripts SET text_sha256 = ? WHERE transcript_id = ?",
            ("0" * 64, transcript_id),
        )
        with self.assertRaises(ChunkingError):
            chunk_and_persist_transcript(self.connection, transcript_id)


if __name__ == "__main__":
    unittest.main()
