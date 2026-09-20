from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from typing import Iterable, Sequence


WORD_PATTERN = re.compile(r"\S+")


class ChunkingError(ValueError):
    """Raised when text, timings, chunks, or corrections are inconsistent."""


class CorrectionConflictError(ChunkingError):
    """Raised when corrections disagree or overlap at global word positions."""


@dataclass(frozen=True)
class ChunkingConfig:
    max_words: int = 400
    overlap_words: int = 100

    def __post_init__(self) -> None:
        if self.max_words <= 0:
            raise ChunkingError("max_words must be greater than zero")
        if self.overlap_words < 0:
            raise ChunkingError("overlap_words cannot be negative")
        if self.overlap_words >= self.max_words:
            raise ChunkingError("overlap_words must be smaller than max_words")

    @property
    def step_words(self) -> int:
        return self.max_words - self.overlap_words


@dataclass(frozen=True)
class WordToken:
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class WordTiming:
    start_time_ms: int
    end_time_ms: int

    def __post_init__(self) -> None:
        if self.start_time_ms < 0:
            raise ChunkingError("word start_time_ms cannot be negative")
        if self.end_time_ms < self.start_time_ms:
            raise ChunkingError("word end_time_ms cannot be before its start")


@dataclass(frozen=True)
class TranscriptChunk:
    chunk_index: int
    text: str
    start_word: int
    end_word: int
    overlap_before_words: int
    overlap_after_words: int
    sha256: str
    start_time_ms: int | None = None
    end_time_ms: int | None = None

    @property
    def word_count(self) -> int:
        return self.end_word - self.start_word


@dataclass(frozen=True)
class CorrectionSpan:
    start_word: int
    end_word: int
    replacement_text: str


def tokenize_words(text: str) -> tuple[WordToken, ...]:
    return tuple(
        WordToken(match.group(0), match.start(), match.end())
        for match in WORD_PATTERN.finditer(text)
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_timings(
    timings: Sequence[WordTiming | None] | None,
    word_count: int,
) -> None:
    if timings is None:
        return
    if len(timings) != word_count:
        raise ChunkingError(
            f"Expected {word_count} word timings, received {len(timings)}"
        )

    previous_start: int | None = None
    for timing in timings:
        if timing is None:
            continue
        if previous_start is not None and timing.start_time_ms < previous_start:
            raise ChunkingError("word timings must be in chronological order")
        previous_start = timing.start_time_ms


def create_chunks(
    text: str,
    config: ChunkingConfig = ChunkingConfig(),
    word_timings: Sequence[WordTiming | None] | None = None,
) -> list[TranscriptChunk]:
    tokens = tokenize_words(text)
    total_words = len(tokens)
    _validate_timings(word_timings, total_words)
    if total_words == 0:
        return []

    starts = list(range(0, total_words, config.step_words))
    if len(starts) > 1 and starts[-1] + config.overlap_words >= total_words:
        starts.pop()

    bounds = [
        (start_word, min(start_word + config.max_words, total_words))
        for start_word in starts
    ]
    chunks: list[TranscriptChunk] = []
    for chunk_index, (start_word, end_word) in enumerate(bounds):
        start_char = tokens[start_word].start_char
        end_char = tokens[end_word - 1].end_char
        chunk_text = text[start_char:end_char]

        previous_end = bounds[chunk_index - 1][1] if chunk_index else start_word
        next_start = (
            bounds[chunk_index + 1][0]
            if chunk_index + 1 < len(bounds)
            else end_word
        )
        overlap_before = max(0, previous_end - start_word)
        overlap_after = max(0, end_word - next_start)

        start_time_ms = None
        end_time_ms = None
        if word_timings is not None:
            first_timing = word_timings[start_word]
            last_timing = word_timings[end_word - 1]
            if first_timing is not None:
                start_time_ms = first_timing.start_time_ms
            if last_timing is not None:
                end_time_ms = last_timing.end_time_ms

        chunks.append(
            TranscriptChunk(
                chunk_index=chunk_index,
                text=chunk_text,
                start_word=start_word,
                end_word=end_word,
                overlap_before_words=overlap_before,
                overlap_after_words=overlap_after,
                sha256=sha256_text(chunk_text),
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
            )
        )
    return chunks


def reconstruct_words(chunks: Sequence[TranscriptChunk]) -> tuple[str, ...]:
    """Reconstruct a word sequence by global position, deduplicating overlap."""
    if not chunks:
        return ()

    ordered = sorted(chunks, key=lambda chunk: chunk.chunk_index)
    for index, chunk in enumerate(ordered):
        if index == 0:
            if chunk.start_word != 0 or chunk.overlap_before_words != 0:
                raise ChunkingError("the first chunk must start at word zero")
        else:
            previous = ordered[index - 1]
            if chunk.start_word <= previous.start_word:
                raise ChunkingError("chunk word positions must move forward")
            if chunk.start_word > previous.end_word:
                raise ChunkingError("chunks contain a gap in global word coverage")
            actual_overlap = previous.end_word - chunk.start_word
            if (
                previous.overlap_after_words != actual_overlap
                or chunk.overlap_before_words != actual_overlap
            ):
                raise ChunkingError(
                    "stored overlap metadata does not match global word bounds"
                )
        if index == len(ordered) - 1 and chunk.overlap_after_words != 0:
            raise ChunkingError("the final chunk cannot overlap a later chunk")

    words_by_position: dict[int, str] = {}
    expected_chunk_index = 0
    for chunk in ordered:
        if chunk.chunk_index != expected_chunk_index:
            raise ChunkingError("chunk indexes must be contiguous from zero")
        expected_chunk_index += 1

        words = tuple(match.group(0) for match in WORD_PATTERN.finditer(chunk.text))
        if len(words) != chunk.word_count:
            raise ChunkingError(
                f"Chunk {chunk.chunk_index} text does not match its word bounds"
            )
        for offset, word in enumerate(words):
            global_position = chunk.start_word + offset
            existing = words_by_position.get(global_position)
            if existing is not None and existing != word:
                raise ChunkingError(
                    f"Conflicting overlap at global word {global_position}"
                )
            words_by_position[global_position] = word

    highest_position = max(words_by_position)
    expected_positions = set(range(highest_position + 1))
    if set(words_by_position) != expected_positions:
        raise ChunkingError("chunks do not provide continuous global word coverage")
    return tuple(words_by_position[index] for index in range(highest_position + 1))


def merge_corrections(
    source_text: str,
    corrections: Iterable[CorrectionSpan],
) -> str:
    """Apply corrections once to the source using global half-open word bounds."""
    tokens = tokenize_words(source_text)
    unique: dict[tuple[int, int], str] = {}
    for correction in corrections:
        if not 0 <= correction.start_word < correction.end_word <= len(tokens):
            raise ChunkingError(
                "correction bounds must fall within the source word positions"
            )
        if not correction.replacement_text.strip():
            raise ChunkingError("replacement_text cannot be empty")

        key = (correction.start_word, correction.end_word)
        existing = unique.get(key)
        if existing is not None and existing != correction.replacement_text:
            raise CorrectionConflictError(
                f"Conflicting replacements for words {key[0]}:{key[1]}"
            )
        unique[key] = correction.replacement_text

    ordered = sorted(
        (
            CorrectionSpan(start_word, end_word, replacement)
            for (start_word, end_word), replacement in unique.items()
        ),
        key=lambda correction: (correction.start_word, correction.end_word),
    )
    for previous, current in zip(ordered, ordered[1:]):
        if current.start_word < previous.end_word:
            raise CorrectionConflictError(
                "Corrections with different global spans cannot overlap"
            )

    merged = source_text
    for correction in reversed(ordered):
        start_char = tokens[correction.start_word].start_char
        end_char = tokens[correction.end_word - 1].end_char
        merged = (
            merged[:start_char]
            + correction.replacement_text.strip()
            + merged[end_char:]
        )
    return merged


def persist_chunks(
    connection: sqlite3.Connection,
    transcript_id: int,
    chunks: Sequence[TranscriptChunk],
) -> int:
    """Store immutable chunks and verify repeat runs produce identical rows."""
    transcript = connection.execute(
        "SELECT word_count FROM transcripts WHERE transcript_id = ?",
        (transcript_id,),
    ).fetchone()
    if transcript is None:
        raise ChunkingError(f"Transcript {transcript_id} does not exist")

    reconstructed = reconstruct_words(chunks)
    if len(reconstructed) != transcript["word_count"]:
        raise ChunkingError(
            "chunk coverage does not match the transcript's stored word_count"
        )

    inserted = 0
    with connection:
        for chunk in chunks:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO transcript_chunks(
                    transcript_id, chunk_index, chunk_text, start_word, end_word,
                    word_count, overlap_before_words, overlap_after_words,
                    start_time_ms, end_time_ms, chunk_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transcript_id,
                    chunk.chunk_index,
                    chunk.text,
                    chunk.start_word,
                    chunk.end_word,
                    chunk.word_count,
                    chunk.overlap_before_words,
                    chunk.overlap_after_words,
                    chunk.start_time_ms,
                    chunk.end_time_ms,
                    chunk.sha256,
                ),
            )
            if cursor.rowcount == 1:
                inserted += 1
                continue

            existing = connection.execute(
                """
                SELECT chunk_text, start_word, end_word, word_count,
                       overlap_before_words, overlap_after_words, start_time_ms,
                       end_time_ms, chunk_sha256
                FROM transcript_chunks
                WHERE transcript_id = ? AND chunk_index = ?
                """,
                (transcript_id, chunk.chunk_index),
            ).fetchone()
            expected = (
                chunk.text,
                chunk.start_word,
                chunk.end_word,
                chunk.word_count,
                chunk.overlap_before_words,
                chunk.overlap_after_words,
                chunk.start_time_ms,
                chunk.end_time_ms,
                chunk.sha256,
            )
            if existing is None or tuple(existing) != expected:
                raise ChunkingError(
                    f"Stored chunk {chunk.chunk_index} differs from regenerated data"
                )
    return inserted


def chunk_and_persist_transcript(
    connection: sqlite3.Connection,
    transcript_id: int,
    config: ChunkingConfig = ChunkingConfig(),
    word_timings: Sequence[WordTiming | None] | None = None,
) -> list[TranscriptChunk]:
    transcript = connection.execute(
        """
        SELECT transcript_text, word_count, text_sha256
        FROM transcripts
        WHERE transcript_id = ?
        """,
        (transcript_id,),
    ).fetchone()
    if transcript is None:
        raise ChunkingError(f"Transcript {transcript_id} does not exist")

    text = transcript["transcript_text"]
    if sha256_text(text) != transcript["text_sha256"]:
        raise ChunkingError("stored transcript checksum does not match its text")
    actual_word_count = len(tokenize_words(text))
    if actual_word_count != transcript["word_count"]:
        raise ChunkingError("stored transcript word_count does not match its text")

    chunks = create_chunks(text, config, word_timings)
    persist_chunks(connection, transcript_id, chunks)
    return chunks
