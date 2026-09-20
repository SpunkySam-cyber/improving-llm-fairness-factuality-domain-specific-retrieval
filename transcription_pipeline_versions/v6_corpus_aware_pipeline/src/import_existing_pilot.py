from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chunking import (
    ChunkingConfig,
    ChunkingError,
    TranscriptChunk,
    WordTiming,
    chunk_and_persist_transcript,
    reconstruct_words,
    sha256_text,
    tokenize_words,
)
from database import DEFAULT_DATABASE, initialize_database


VERSION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PILOT_DIR = (
    VERSION_DIR.parent / "v5_batch_ingestion_pipeline" / "audio_transcription_pilot"
)
DEFAULT_VIDEO_ID = "6TnPRgY_loc"
DEFAULT_REPORT = VERSION_DIR / "reports" / "stage4_import_report.json"


class PilotImportError(RuntimeError):
    """Raised when a saved pilot bundle is incomplete or inconsistent."""


@dataclass(frozen=True)
class PilotBundle:
    video: dict[str, Any]
    transcript_text: str
    assemblyai_metadata: dict[str, Any]
    assemblyai_response: dict[str, Any]
    word_timings: tuple[WordTiming, ...]
    source_files: dict[str, str]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotImportError(f"Cannot read valid JSON from {path}") from exc
    if not isinstance(value, dict):
        raise PilotImportError(f"Expected a JSON object in {path}")
    return value


def load_pilot_bundle(pilot_dir: Path, video_id: str) -> PilotBundle:
    pilot_dir = pilot_dir.resolve()
    manifest_path = pilot_dir / "pilot_manifest.json"
    manifest = _read_json(manifest_path)
    video = next(
        (
            item
            for item in manifest.get("videos", [])
            if isinstance(item, dict) and item.get("video_id") == video_id
        ),
        None,
    )
    if video is None:
        raise PilotImportError(f"Video {video_id} is absent from {manifest_path}")

    benchmark_dir = pilot_dir / "videos" / video_id / "fresh_benchmark"
    transcript_path = benchmark_dir / "assemblyai_transcript.txt"
    metadata_path = benchmark_dir / "assemblyai_metadata.json"
    response_path = benchmark_dir / "assemblyai_response.json"
    try:
        transcript_text = transcript_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise PilotImportError(f"Cannot read {transcript_path}") from exc
    metadata = _read_json(metadata_path)
    response = _read_json(response_path)

    response_text = response.get("text")
    if not isinstance(response_text, str) or response_text != transcript_text:
        raise PilotImportError(
            "The transcript text does not exactly match the saved AssemblyAI response"
        )
    if response.get("status") != "completed":
        raise PilotImportError("The saved AssemblyAI response is not completed")
    if metadata.get("transcript_id") != response.get("id"):
        raise PilotImportError("AssemblyAI transcript IDs do not match")

    tokens = tokenize_words(transcript_text)
    response_words = response.get("words")
    if not isinstance(response_words, list) or len(response_words) != len(tokens):
        raise PilotImportError("AssemblyAI words do not match transcript word count")

    timings: list[WordTiming] = []
    for index, (token, response_word) in enumerate(zip(tokens, response_words)):
        if not isinstance(response_word, dict) or response_word.get("text") != token.text:
            raise PilotImportError(
                f"AssemblyAI word {index} does not match the transcript"
            )
        try:
            timings.append(
                WordTiming(
                    start_time_ms=int(response_word["start"]),
                    end_time_ms=int(response_word["end"]),
                )
            )
        except (KeyError, TypeError, ValueError, ChunkingError) as exc:
            raise PilotImportError(
                f"AssemblyAI word {index} has invalid timestamps"
            ) from exc

    reported_word_count = metadata.get("word_count")
    if reported_word_count != len(tokens):
        raise PilotImportError(
            "AssemblyAI metadata word count does not match the transcript"
        )

    return PilotBundle(
        video=video,
        transcript_text=transcript_text,
        assemblyai_metadata=metadata,
        assemblyai_response=response,
        word_timings=tuple(timings),
        source_files={
            "manifest": str(manifest_path),
            "transcript": str(transcript_path),
            "metadata": str(metadata_path),
            "response": str(response_path),
        },
    )


def import_bundle(
    connection: sqlite3.Connection,
    bundle: PilotBundle,
    config: ChunkingConfig = ChunkingConfig(),
) -> dict[str, Any]:
    video = bundle.video
    video_id = str(video["video_id"])
    now_source = "v5/audio_transcription_pilot"

    with connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO videos(
                video_id, canonical_url, title, channel_name, channel_id,
                duration_seconds, upload_date, content_type, source_collection
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'lecture', ?)
            """,
            (
                video_id,
                video["url"],
                video["title"],
                video.get("channel"),
                video.get("channel_id"),
                video.get("duration_seconds"),
                video.get("upload_date"),
                now_source,
            ),
        )
        stored_video = connection.execute(
            """
            SELECT canonical_url, title, content_type
            FROM videos WHERE video_id = ?
            """,
            (video_id,),
        ).fetchone()
        expected_video = (video["url"], video["title"], "lecture")
        if stored_video is None or tuple(stored_video) != expected_video:
            raise PilotImportError("Existing video record conflicts with pilot metadata")

        transcript_hash = sha256_text(bundle.transcript_text)
        connection.execute(
            """
            INSERT OR IGNORE INTO transcripts(
                video_id, source_type, provider, provider_transcript_id,
                language_code, transcript_text, word_count, text_sha256
            ) VALUES (?, 'assemblyai', 'AssemblyAI', ?, ?, ?, ?, ?)
            """,
            (
                video_id,
                bundle.assemblyai_response["id"],
                bundle.assemblyai_response.get("language_code") or "en",
                bundle.transcript_text,
                len(bundle.word_timings),
                transcript_hash,
            ),
        )
        transcript = connection.execute(
            """
            SELECT transcript_id, provider_transcript_id, transcript_text,
                   word_count, text_sha256
            FROM transcripts
            WHERE video_id = ? AND source_type = 'assemblyai' AND text_sha256 = ?
            """,
            (video_id, transcript_hash),
        ).fetchone()
        if transcript is None:
            raise PilotImportError("Failed to read the imported transcript")
        if (
            transcript["provider_transcript_id"]
            != bundle.assemblyai_response["id"]
            or transcript["transcript_text"] != bundle.transcript_text
            or transcript["word_count"] != len(bundle.word_timings)
        ):
            raise PilotImportError("Existing transcript conflicts with pilot artifacts")
        transcript_id = int(transcript["transcript_id"])

    chunks = chunk_and_persist_transcript(
        connection,
        transcript_id,
        config=config,
        word_timings=bundle.word_timings,
    )

    run_id = f"v6-import-{video_id}-{transcript_hash[:12]}"
    configuration = json.dumps(
        {
            "source_collection": now_source,
            "source_artifacts": bundle.source_files,
            "chunking": {
                "max_words": config.max_words,
                "overlap_words": config.overlap_words,
                "step_words": config.step_words,
            },
            "network_calls": 0,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    with connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO pipeline_runs(
                run_id, pipeline_version, run_type, output_transcript_id,
                configuration_json, status, started_at, completed_at
            ) VALUES (
                ?, 'v6', 'import', ?, ?, 'completed',
                strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            )
            """,
            (run_id, transcript_id, configuration),
        )

    return {
        "stage": 4,
        "operation": "offline_existing_transcript_import",
        "video_id": video_id,
        "title": video["title"],
        "canonical_url": video["url"],
        "duration_seconds": video.get("duration_seconds"),
        "provider": "AssemblyAI",
        "provider_transcript_id": bundle.assemblyai_response["id"],
        "run_id": run_id,
        "transcript_id": transcript_id,
        "word_count": len(bundle.word_timings),
        "transcript_sha256": transcript_hash,
        "chunk_count": len(chunks),
        "chunking": {
            "max_words": config.max_words,
            "overlap_words": config.overlap_words,
            "step_words": config.step_words,
        },
        "first_chunk": {
            "chunk_index": chunks[0].chunk_index,
            "start_word": chunks[0].start_word,
            "end_word": chunks[0].end_word,
            "start_time_ms": chunks[0].start_time_ms,
            "end_time_ms": chunks[0].end_time_ms,
        },
        "last_chunk": {
            "chunk_index": chunks[-1].chunk_index,
            "start_word": chunks[-1].start_word,
            "end_word": chunks[-1].end_word,
            "start_time_ms": chunks[-1].start_time_ms,
            "end_time_ms": chunks[-1].end_time_ms,
        },
        "network_calls": 0,
        "api_cost_usd": 0.0,
        "source_files": bundle.source_files,
    }


def verify_import(
    connection: sqlite3.Connection,
    report: dict[str, Any],
) -> dict[str, Any]:
    transcript = connection.execute(
        """
        SELECT transcript_text, word_count, text_sha256
        FROM transcripts
        WHERE transcript_id = ?
        """,
        (report["transcript_id"],),
    ).fetchone()
    if transcript is None:
        raise PilotImportError("Imported transcript is missing during verification")
    rows = connection.execute(
        """
        SELECT chunk_index, chunk_text, start_word, end_word,
               overlap_before_words, overlap_after_words, chunk_sha256,
               start_time_ms, end_time_ms
        FROM transcript_chunks
        WHERE transcript_id = ?
        ORDER BY chunk_index
        """,
        (report["transcript_id"],),
    ).fetchall()
    chunks = [
        TranscriptChunk(
            chunk_index=row["chunk_index"],
            text=row["chunk_text"],
            start_word=row["start_word"],
            end_word=row["end_word"],
            overlap_before_words=row["overlap_before_words"],
            overlap_after_words=row["overlap_after_words"],
            sha256=row["chunk_sha256"],
            start_time_ms=row["start_time_ms"],
            end_time_ms=row["end_time_ms"],
        )
        for row in rows
    ]
    source_words = tuple(token.text for token in tokenize_words(transcript["transcript_text"]))
    reconstructed_words = reconstruct_words(chunks)
    checksum_matches = all(chunk.sha256 == sha256_text(chunk.text) for chunk in chunks)
    foreign_key_violations = len(connection.execute("PRAGMA foreign_key_check").fetchall())
    model_call_count = connection.execute(
        "SELECT COUNT(*) FROM model_calls WHERE run_id = ?",
        (report["run_id"],),
    ).fetchone()[0]
    checks = {
        "database_integrity": connection.execute("PRAGMA integrity_check").fetchone()[0],
        "foreign_key_violations": foreign_key_violations,
        "stored_chunk_count": len(chunks),
        "chunk_count_matches_report": len(chunks) == report["chunk_count"],
        "continuous_word_coverage": reconstructed_words == source_words,
        "reconstructed_word_count": len(reconstructed_words),
        "transcript_checksum_matches": (
            sha256_text(transcript["transcript_text"]) == transcript["text_sha256"]
        ),
        "all_chunk_checksums_match": checksum_matches,
        "all_chunks_have_start_and_end_timestamps": all(
            chunk.start_time_ms is not None and chunk.end_time_ms is not None
            for chunk in chunks
        ),
        "model_call_count": model_call_count,
    }
    if (
        checks["database_integrity"] != "ok"
        or foreign_key_violations != 0
        or not checks["chunk_count_matches_report"]
        or not checks["continuous_word_coverage"]
        or not checks["transcript_checksum_matches"]
        or not checksum_matches
        or not checks["all_chunks_have_start_and_end_timestamps"]
        or model_call_count != 0
    ):
        raise PilotImportError("Post-import database verification failed")
    return checks


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import one completed v5 AssemblyAI pilot transcript into v6."
    )
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT_DIR)
    parser.add_argument("--video-id", default=DEFAULT_VIDEO_ID)
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = load_pilot_bundle(args.pilot_dir, args.video_id)
    connection = initialize_database(args.db)
    try:
        report = import_bundle(connection, bundle)
        report["verification"] = verify_import(connection, report)
    finally:
        connection.close()
    write_report(report, args.report)
    print(f"Imported video: {report['video_id']}")
    print(f"Transcript words: {report['word_count']}")
    print(f"Stored chunks: {report['chunk_count']}")
    print("Network calls: 0")
    print(f"Report: {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
