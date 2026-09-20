"""Save usable English YouTube captions for the pilot or all candidates.

Manual English captions are preferred. Automatic English captions are used as
the fallback. This stage downloads caption data only; it never downloads media
and never calls AssemblyAI or an LLM.
"""

from __future__ import annotations

# --- repo layout shim: find shared modules (V5 batch core, dialogue/dataset tools, project_env) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
for _d in (_ROOT, _ROOT / "transcription_pipeline_versions" / "v5_batch_ingestion_pipeline", _ROOT / "tools" / "dataset", _ROOT / "tools" / "dialogue"):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
# --- end shim ---
import argparse
import json
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yt_dlp

from build_manifest import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DB_PATH,
    ManifestError,
    connect_manifest,
    load_config,
    select_pilot,
)


VERSION_DIR = Path(__file__).resolve().parents[2] / "transcription_pipeline_versions" / "v5_batch_ingestion_pipeline"  # data/output dir (shared with V5 batch core)
DEFAULT_OUTPUT_DIR = VERSION_DIR / "pilot"
DEFAULT_CANDIDATE_OUTPUT_DIR = VERSION_DIR / "captions"
ENGLISH_LANGUAGE_PRIORITY = ("en-orig", "en", "en-US", "en-GB")
FORMAT_PRIORITY = ("json3", "vtt", "srv3", "ttml")


@dataclass(frozen=True)
class CaptionChoice:
    kind: str
    language: str
    extension: str
    url: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_caption_columns(connection: sqlite3.Connection) -> None:
    existing = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(videos)").fetchall()
    }
    additions = {
        "caption_status": "TEXT",
        "caption_kind": "TEXT",
        "caption_language": "TEXT",
        "caption_path": "TEXT",
        "caption_word_count": "INTEGER",
        "caption_checked_at": "TEXT",
        "caption_error": "TEXT",
    }
    for name, sql_type in additions.items():
        if name not in existing:
            connection.execute(f"ALTER TABLE videos ADD COLUMN {name} {sql_type}")
    connection.commit()


def _ordered_english_keys(mapping: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for preferred in ENGLISH_LANGUAGE_PRIORITY:
        if preferred in mapping and preferred not in keys:
            keys.append(preferred)
    for key in sorted(mapping):
        if key.casefold().startswith("en") and key not in keys:
            keys.append(key)
    return keys


def _select_format(entries: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    for extension in FORMAT_PRIORITY:
        for entry in entries:
            if entry.get("ext") == extension and entry.get("url"):
                return entry
    return next((entry for entry in entries if entry.get("url")), None)


def choose_english_caption(info: dict[str, Any]) -> CaptionChoice | None:
    for kind, field in (("manual", "subtitles"), ("automatic", "automatic_captions")):
        mapping = info.get(field)
        if not isinstance(mapping, dict):
            continue
        for language in _ordered_english_keys(mapping):
            raw_entries = mapping.get(language)
            if not isinstance(raw_entries, list):
                continue
            entry = _select_format(
                [item for item in raw_entries if isinstance(item, dict)]
            )
            if entry is not None:
                return CaptionChoice(
                    kind=kind,
                    language=language,
                    extension=str(entry.get("ext") or "unknown"),
                    url=str(entry["url"]),
                )
    return None


def download_caption(choice: CaptionChoice) -> bytes:
    request = urllib.request.Request(
        choice.url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/146.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def parse_json3_caption(payload: bytes) -> list[dict[str, Any]]:
    try:
        data = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"Invalid JSON3 caption payload: {exc}") from exc
    segments: list[dict[str, Any]] = []
    for event in data.get("events") or []:
        if not isinstance(event, dict):
            continue
        pieces = event.get("segs") or []
        text = "".join(
            str(piece.get("utf8") or "")
            for piece in pieces
            if isinstance(piece, dict)
        )
        text = " ".join(text.replace("\n", " ").split())
        if not text:
            continue
        start_ms = int(event.get("tStartMs") or 0)
        duration_ms = int(event.get("dDurationMs") or 0)
        segments.append(
            {
                "start_seconds": round(start_ms / 1000, 3),
                "duration_seconds": round(duration_ms / 1000, 3),
                "text": text,
            }
        )
    if not segments:
        raise ManifestError("The JSON3 caption contained no text segments.")
    return segments


def parse_vtt_caption(payload: bytes) -> list[dict[str, Any]]:
    text = payload.decode("utf-8-sig", errors="replace")
    cue_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "WEBVTT" or "-->" in stripped:
            continue
        if stripped.isdigit() or stripped.startswith(("Kind:", "Language:")):
            continue
        stripped = re.sub(r"<[^>]+>", "", stripped)
        if not cue_lines or cue_lines[-1] != stripped:
            cue_lines.append(stripped)
    transcript = " ".join(cue_lines).strip()
    if not transcript:
        raise ManifestError("The VTT caption contained no transcript text.")
    return [{"start_seconds": None, "duration_seconds": None, "text": transcript}]


def caption_segments(choice: CaptionChoice, payload: bytes) -> list[dict[str, Any]]:
    if choice.extension == "json3":
        return parse_json3_caption(payload)
    if choice.extension == "vtt":
        return parse_vtt_caption(payload)
    raise ManifestError(
        f"Caption format {choice.extension!r} is not yet supported for normalization."
    )


def save_caption_outputs(
    output_dir: Path,
    video: sqlite3.Row,
    info: dict[str, Any],
    choice: CaptionChoice,
    payload: bytes,
    segments: Sequence[dict[str, Any]],
) -> tuple[Path, int]:
    video_dir = output_dir / video["video_id"]
    video_dir.mkdir(parents=True, exist_ok=True)
    raw_path = video_dir / f"youtube_caption.{choice.extension}"
    raw_path.write_bytes(payload)
    segments_path = video_dir / "youtube_segments.json"
    segments_path.write_text(
        json.dumps(list(segments), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    transcript = " ".join(str(segment["text"]) for segment in segments).strip()
    transcript_path = video_dir / "youtube_transcript.txt"
    transcript_path.write_text(transcript + "\n", encoding="utf-8")
    metadata = {
        "video_id": video["video_id"],
        "title": video["title"],
        "url": video["canonical_url"],
        "channel": info.get("channel") or video["channel"],
        "duration_seconds": info.get("duration") or video["duration_seconds"],
        "caption_kind": choice.kind,
        "caption_language": choice.language,
        "caption_format": choice.extension,
        "caption_word_count": len(transcript.split()),
        "retrieved_at": utc_now(),
    }
    (video_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return transcript_path, len(transcript.split())


def update_caption_state(
    connection: sqlite3.Connection,
    video_id: str,
    *,
    status: str,
    pipeline_status: str,
    choice: CaptionChoice | None = None,
    transcript_path: Path | None = None,
    word_count: int | None = None,
    error: str | None = None,
) -> None:
    transcript_source = None
    if choice is not None:
        transcript_source = f"youtube_{choice.kind}"
    connection.execute(
        """
        UPDATE videos
        SET caption_status = ?, caption_kind = ?, caption_language = ?,
            caption_path = ?, caption_word_count = ?, caption_checked_at = ?,
            caption_error = ?, transcript_source = COALESCE(?, transcript_source),
            pipeline_status = ?, updated_at = ?
        WHERE video_id = ?
        """,
        (
            status,
            choice.kind if choice else None,
            choice.language if choice else None,
            str(transcript_path) if transcript_path else None,
            word_count,
            utc_now(),
            error,
            transcript_source,
            pipeline_status,
            utc_now(),
            video_id,
        ),
    )
    connection.commit()


def saved_caption_is_valid(video: sqlite3.Row) -> bool:
    if video["caption_status"] != "available" or not video["caption_path"]:
        return False
    path = Path(video["caption_path"])
    return path.is_file() and path.stat().st_size > 0


def select_caption_rows(
    connection: sqlite3.Connection,
    scope: str,
    video_ids: Sequence[str] | None = None,
) -> list[sqlite3.Row]:
    if video_ids is not None:
        if not video_ids:
            return []
        placeholders = ",".join("?" for _ in video_ids)
        return connection.execute(
            f"""
            SELECT video_id, title, canonical_url, channel, duration_seconds,
                   caption_status, caption_path
            FROM videos
            WHERE video_id IN ({placeholders})
            ORDER BY discovery_index ASC, upload_date DESC, video_id ASC
            """,
            tuple(video_ids),
        ).fetchall()
    where = (
        "selected_for_pilot = 1"
        if scope == "pilot"
        else "candidate_status = 'candidate'"
    )
    return connection.execute(
        f"""
        SELECT video_id, title, canonical_url, channel, duration_seconds,
               caption_status, caption_path
        FROM videos
        WHERE {where}
        ORDER BY discovery_index ASC, upload_date DESC, video_id ASC
        """
    ).fetchall()


def process_captions(
    connection: sqlite3.Connection,
    output_dir: Path,
    *,
    scope: str = "pilot",
    resume: bool = True,
    limit: int | None = None,
    delay_seconds: float = 0.0,
    video_ids: Sequence[str] | None = None,
) -> dict[str, int]:
    rows = select_caption_rows(connection, scope, video_ids)
    if not rows:
        raise ManifestError(f"No {scope} videos are available in the manifest.")
    if limit is not None:
        rows = rows[:limit]
    counts = {
        "manual": 0,
        "automatic": 0,
        "unavailable": 0,
        "errors": 0,
        "skipped_saved": 0,
    }
    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
        "socket_timeout": 30,
        "retries": 2,
        "extractor_retries": 2,
    }
    last_request_at: float | None = None
    with yt_dlp.YoutubeDL(options) as ydl:
        for index, video in enumerate(rows, start=1):
            if resume and saved_caption_is_valid(video):
                counts["skipped_saved"] += 1
                print(
                    f"[{index}/{len(rows)}] Already saved: {video['title']}",
                    flush=True,
                )
                continue
            if last_request_at is not None and delay_seconds > 0:
                remaining = delay_seconds - (time.monotonic() - last_request_at)
                if remaining > 0:
                    time.sleep(remaining)
            print(f"[{index}/{len(rows)}] Checking captions: {video['title']}", flush=True)
            last_request_at = time.monotonic()
            try:
                info = ydl.extract_info(video["canonical_url"], download=False)
                if not isinstance(info, dict):
                    raise ManifestError("yt-dlp returned no video metadata.")
                choice = choose_english_caption(info)
                if choice is None:
                    counts["unavailable"] += 1
                    update_caption_state(
                        connection,
                        video["video_id"],
                        status="unavailable",
                        pipeline_status="caption_unavailable",
                    )
                    print("  No usable English captions.")
                    continue
                payload = download_caption(choice)
                segments = caption_segments(choice, payload)
                transcript_path, word_count = save_caption_outputs(
                    output_dir, video, info, choice, payload, segments
                )
                counts[choice.kind] += 1
                update_caption_state(
                    connection,
                    video["video_id"],
                    status="available",
                    pipeline_status="caption_saved",
                    choice=choice,
                    transcript_path=transcript_path,
                    word_count=word_count,
                )
                print(
                    f"  Saved {choice.kind} {choice.language} captions: "
                    f"{word_count} words."
                )
            except (ManifestError, OSError, urllib.error.URLError, yt_dlp.utils.DownloadError) as exc:
                counts["errors"] += 1
                update_caption_state(
                    connection,
                    video["video_id"],
                    status="error",
                    pipeline_status="caption_error",
                    error=str(exc)[:1000],
                )
                print(f"  Caption error: {exc}", file=sys.stderr)
    return counts


def process_pilot(connection: sqlite3.Connection, output_dir: Path) -> dict[str, int]:
    """Backward-compatible wrapper used by the original pilot workflow."""
    return process_captions(connection, output_dir, scope="pilot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--scope", choices=("pilot", "candidates"), default="pilot"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--limit", type=int, help="process at most this many selected rows"
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=5.0,
        help="minimum delay between YouTube metadata requests (default: 5)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-download captions even when a valid saved transcript exists",
    )
    parser.add_argument(
        "--queue-jsonl",
        type=Path,
        help="process only video_id values listed in this JSONL review queue",
    )
    return parser.parse_args()


def main() -> int:
    # Windows PowerShell may expose a legacy cp1252 stream; channel titles can
    # contain Arabic, emoji, and other Unicode characters.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    try:
        config = load_config(args.config.resolve())
        connection = connect_manifest(args.db.resolve())
        try:
            ensure_caption_columns(connection)
            if args.scope == "pilot":
                select_pilot(
                    connection,
                    config.pilot_size,
                    config.lecture_rules.pilot_excluded_title_patterns,
                )
            default_output = (
                DEFAULT_OUTPUT_DIR
                if args.scope == "pilot"
                else DEFAULT_CANDIDATE_OUTPUT_DIR
            )
            output_dir = (args.output_dir or default_output).resolve()
            video_ids = None
            if args.queue_jsonl:
                video_ids = [
                    str(json.loads(line)["video_id"])
                    for line in args.queue_jsonl.resolve().read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                ]
            counts = process_captions(
                connection,
                output_dir,
                scope=args.scope,
                resume=not args.refresh,
                limit=args.limit,
                delay_seconds=max(0.0, args.delay_seconds),
                video_ids=video_ids,
            )
        finally:
            connection.close()
        print(f"\nCaption {args.scope} summary")
        for key, value in counts.items():
            print(f"  {key}: {value}")
        print("No video or audio media was downloaded.")
    except (ManifestError, OSError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
