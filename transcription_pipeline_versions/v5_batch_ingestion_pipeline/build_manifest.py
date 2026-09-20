"""Build a resumable, metadata-only lecture manifest from YouTube sources.

This stage deliberately performs no media downloads, transcription, or LLM
calls. It inventories source channels, applies conservative lecture filters,
deduplicates exact YouTube video IDs, and selects a small pilot for review.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


VERSION_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = VERSION_DIR / "sources.json"
DEFAULT_DB_PATH = VERSION_DIR / "pipeline.sqlite"
DEFAULT_SNAPSHOT_DIR = VERSION_DIR / "metadata_snapshots"


class ManifestError(RuntimeError):
    """Raised when source discovery or manifest validation fails."""


@dataclass(frozen=True)
class LectureRules:
    minimum_duration_seconds: int
    excluded_title_patterns: tuple[str, ...]
    pilot_excluded_title_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceConfig:
    name: str
    url: str
    speaker_patterns: tuple[str, ...]


@dataclass(frozen=True)
class ManifestConfig:
    pilot_size: int
    lecture_rules: LectureRules
    sources: tuple[SourceConfig, ...]


@dataclass(frozen=True)
class ScreeningResult:
    speaker_match: bool
    candidate_status: str
    exclusion_reason: str | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_title(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def canonical_video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _compile_patterns(patterns: Sequence[str]) -> tuple[re.Pattern[str], ...]:
    try:
        return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)
    except re.error as exc:
        raise ManifestError(f"Invalid regular expression in configuration: {exc}") from exc


def screen_video(
    *,
    title: str,
    duration_seconds: int | None,
    speaker_patterns: Sequence[str],
    rules: LectureRules,
) -> ScreeningResult:
    title_patterns = _compile_patterns(speaker_patterns)
    excluded_patterns = _compile_patterns(rules.excluded_title_patterns)
    speaker_match = any(pattern.search(title) for pattern in title_patterns)
    if not speaker_match:
        return ScreeningResult(False, "excluded", "speaker_not_in_title")
    if duration_seconds is None:
        return ScreeningResult(True, "needs_review", "duration_unavailable")
    if duration_seconds < rules.minimum_duration_seconds:
        return ScreeningResult(True, "excluded", "below_minimum_duration")
    for pattern in excluded_patterns:
        if pattern.search(title):
            return ScreeningResult(
                True,
                "excluded",
                f"excluded_title_pattern:{pattern.pattern}",
            )
    return ScreeningResult(True, "candidate", None)


def load_config(path: Path) -> ManifestConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"Configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Invalid JSON configuration: {exc}") from exc

    rules_raw = raw.get("lecture_rules") or {}
    rules = LectureRules(
        minimum_duration_seconds=int(rules_raw.get("minimum_duration_seconds", 600)),
        excluded_title_patterns=tuple(rules_raw.get("excluded_title_patterns") or ()),
        pilot_excluded_title_patterns=tuple(
            rules_raw.get("pilot_excluded_title_patterns") or ()
        ),
    )
    sources = tuple(
        SourceConfig(
            name=str(item["name"]).strip(),
            url=str(item["url"]).strip(),
            speaker_patterns=tuple(item.get("speaker_patterns") or ()),
        )
        for item in raw.get("sources") or ()
    )
    pilot_size = int(raw.get("pilot_size", 10))
    if pilot_size < 1:
        raise ManifestError("pilot_size must be at least 1.")
    if rules.minimum_duration_seconds < 0:
        raise ManifestError("minimum_duration_seconds cannot be negative.")
    if not sources:
        raise ManifestError("At least one source must be configured.")
    for source in sources:
        if not source.name or not source.url or not source.speaker_patterns:
            raise ManifestError(
                "Every source requires a name, URL, and speaker_patterns."
            )
        _compile_patterns(source.speaker_patterns)
    _compile_patterns(rules.excluded_title_patterns)
    _compile_patterns(rules.pilot_excluded_title_patterns)
    return ManifestConfig(pilot_size=pilot_size, lecture_rules=rules, sources=sources)


def connect_manifest(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        INSERT OR IGNORE INTO schema_meta(key, value) VALUES ('schema_version', '1');

        CREATE TABLE IF NOT EXISTS sources (
            source_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            url TEXT NOT NULL UNIQUE,
            speaker_patterns_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_scanned_at TEXT
        );

        CREATE TABLE IF NOT EXISTS videos (
            video_id TEXT PRIMARY KEY,
            canonical_url TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            normalized_title TEXT NOT NULL,
            channel TEXT,
            channel_id TEXT,
            duration_seconds INTEGER,
            upload_date TEXT,
            discovery_index INTEGER NOT NULL,
            speaker_match INTEGER NOT NULL CHECK (speaker_match IN (0, 1)),
            candidate_status TEXT NOT NULL
                CHECK (candidate_status IN ('candidate', 'excluded', 'needs_review')),
            exclusion_reason TEXT,
            pipeline_status TEXT NOT NULL DEFAULT 'discovered',
            selected_for_pilot INTEGER NOT NULL DEFAULT 0
                CHECK (selected_for_pilot IN (0, 1)),
            transcript_source TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS source_videos (
            source_id INTEGER NOT NULL REFERENCES sources(source_id),
            video_id TEXT NOT NULL REFERENCES videos(video_id),
            first_seen_at TEXT NOT NULL,
            PRIMARY KEY (source_id, video_id)
        );

        CREATE INDEX IF NOT EXISTS idx_videos_candidate
            ON videos(candidate_status, selected_for_pilot, discovery_index);
        CREATE INDEX IF NOT EXISTS idx_videos_normalized_title
            ON videos(normalized_title);
        """
    )
    return connection


def upsert_source(connection: sqlite3.Connection, source: SourceConfig) -> int:
    now = utc_now()
    connection.execute(
        """
        INSERT INTO sources(name, url, speaker_patterns_json, created_at, last_scanned_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            name = excluded.name,
            speaker_patterns_json = excluded.speaker_patterns_json,
            last_scanned_at = excluded.last_scanned_at
        """,
        (source.name, source.url, json.dumps(source.speaker_patterns), now, now),
    )
    row = connection.execute(
        "SELECT source_id FROM sources WHERE url = ?", (source.url,)
    ).fetchone()
    if row is None:
        raise ManifestError(f"Failed to save source: {source.url}")
    return int(row["source_id"])


def _coerce_duration(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        duration = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(duration, 0)


def ingest_entries(
    connection: sqlite3.Connection,
    source: SourceConfig,
    rules: LectureRules,
    entries: Iterable[dict[str, Any]],
) -> dict[str, int]:
    source_id = upsert_source(connection, source)
    counts = {
        "discovered": 0,
        "inserted": 0,
        "exact_duplicates": 0,
        "candidate": 0,
        "excluded": 0,
        "needs_review": 0,
        "invalid": 0,
    }
    now = utc_now()
    for discovery_index, entry in enumerate(entries, start=1):
        video_id = str(entry.get("id") or "").strip()
        title = str(entry.get("title") or "").strip()
        if not video_id or not title:
            counts["invalid"] += 1
            continue
        counts["discovered"] += 1
        existed = connection.execute(
            "SELECT 1 FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone() is not None
        duration = _coerce_duration(entry.get("duration"))
        screening = screen_video(
            title=title,
            duration_seconds=duration,
            speaker_patterns=source.speaker_patterns,
            rules=rules,
        )
        counts[screening.candidate_status] += 1
        if existed:
            counts["exact_duplicates"] += 1
        else:
            counts["inserted"] += 1

        channel = str(
            entry.get("channel")
            or entry.get("uploader")
            or entry.get("playlist_channel")
            or source.name
        ).strip()
        channel_id = str(
            entry.get("channel_id") or entry.get("playlist_channel_id") or ""
        ).strip()
        upload_date = str(entry.get("upload_date") or "").strip() or None
        connection.execute(
            """
            INSERT INTO videos(
                video_id, canonical_url, title, normalized_title, channel,
                channel_id, duration_seconds, upload_date, discovery_index,
                speaker_match, candidate_status, exclusion_reason,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                canonical_url = excluded.canonical_url,
                title = excluded.title,
                normalized_title = excluded.normalized_title,
                channel = CASE WHEN excluded.channel <> '' THEN excluded.channel ELSE videos.channel END,
                channel_id = CASE WHEN excluded.channel_id <> '' THEN excluded.channel_id ELSE videos.channel_id END,
                duration_seconds = COALESCE(excluded.duration_seconds, videos.duration_seconds),
                upload_date = COALESCE(excluded.upload_date, videos.upload_date),
                discovery_index = MIN(videos.discovery_index, excluded.discovery_index),
                speaker_match = excluded.speaker_match,
                candidate_status = excluded.candidate_status,
                exclusion_reason = excluded.exclusion_reason,
                updated_at = excluded.updated_at
            """,
            (
                video_id,
                canonical_video_url(video_id),
                title,
                normalize_title(title),
                channel,
                channel_id,
                duration,
                upload_date,
                discovery_index,
                int(screening.speaker_match),
                screening.candidate_status,
                screening.exclusion_reason,
                now,
                now,
            ),
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO source_videos(source_id, video_id, first_seen_at)
            VALUES (?, ?, ?)
            """,
            (source_id, video_id, now),
        )
    connection.commit()
    return counts


def discover_source(source: SourceConfig) -> list[dict[str, Any]]:
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--flat-playlist",
        "--dump-json",
        "--ignore-errors",
        "--no-warnings",
        source.url,
    ]
    result = subprocess.run(
        command,
        cwd=VERSION_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    entries: list[dict[str, Any]] = []
    for line_number, line in enumerate(result.stdout.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ManifestError(
                f"yt-dlp returned invalid JSON on line {line_number}: {exc}"
            ) from exc
        if isinstance(item, dict):
            entries.append(item)
    if result.returncode != 0 and not entries:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        raise ManifestError(f"Could not inventory {source.url}: {detail}")
    if not entries:
        raise ManifestError(f"No videos were discovered for {source.url}.")
    return entries


def save_snapshot(
    directory: Path, source: SourceConfig, entries: Sequence[dict[str, Any]]
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", source.name.casefold()).strip("-")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"{slug}-{stamp}.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def select_pilot(
    connection: sqlite3.Connection,
    pilot_size: int,
    excluded_title_patterns: Sequence[str] = (),
) -> list[sqlite3.Row]:
    # A reviewed pilot must remain stable when new source channels are added.
    # Keep eligible existing selections and fill only empty pilot slots.
    existing = connection.execute(
        """
        SELECT video_id, title
        FROM videos
        WHERE selected_for_pilot = 1 AND candidate_status = 'candidate'
        ORDER BY discovery_index ASC, upload_date DESC, video_id ASC
        """
    ).fetchall()
    excluded = _compile_patterns(excluded_title_patterns)
    kept = [
        row
        for row in existing
        if not any(pattern.search(row["title"]) for pattern in excluded)
    ][:pilot_size]
    kept_ids = {row["video_id"] for row in kept}
    connection.execute("UPDATE videos SET selected_for_pilot = 0")
    if kept_ids:
        connection.executemany(
            "UPDATE videos SET selected_for_pilot = 1 WHERE video_id = ?",
            ((video_id,) for video_id in kept_ids),
        )
    rows = connection.execute(
        """
        SELECT video_id, title
        FROM videos
        WHERE candidate_status = 'candidate' AND selected_for_pilot = 0
        ORDER BY discovery_index ASC, upload_date DESC, video_id ASC
        """,
    ).fetchall()
    rows = [
        row
        for row in rows
        if not any(pattern.search(row["title"]) for pattern in excluded)
    ][: max(0, pilot_size - len(kept_ids))]
    connection.executemany(
        "UPDATE videos SET selected_for_pilot = 1 WHERE video_id = ?",
        ((row["video_id"],) for row in rows),
    )
    connection.commit()
    return connection.execute(
        """
        SELECT video_id, title, canonical_url, channel, duration_seconds,
               candidate_status, pipeline_status
        FROM videos
        WHERE selected_for_pilot = 1
        ORDER BY discovery_index ASC, upload_date DESC, video_id ASC
        """
    ).fetchall()


def manifest_summary(connection: sqlite3.Connection) -> dict[str, int]:
    summary: dict[str, int] = {}
    summary["unique_videos"] = int(
        connection.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    )
    for status in ("candidate", "excluded", "needs_review"):
        summary[status] = int(
            connection.execute(
                "SELECT COUNT(*) FROM videos WHERE candidate_status = ?", (status,)
            ).fetchone()[0]
        )
    summary["pilot_selected"] = int(
        connection.execute(
            "SELECT COUNT(*) FROM videos WHERE selected_for_pilot = 1"
        ).fetchone()[0]
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="do not retain the raw metadata JSONL discovery snapshot",
    )
    parser.add_argument(
        "--source",
        action="append",
        help="process only a configured source name; may be repeated",
    )
    parser.add_argument(
        "--pilot-size",
        type=int,
        help="override the configured number of candidates selected for the pilot",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config.resolve())
        requested = {name.casefold() for name in (args.source or [])}
        sources = tuple(
            source
            for source in config.sources
            if not requested or source.name.casefold() in requested
        )
        if not sources:
            raise ManifestError("No configured sources matched --source.")
        pilot_size = args.pilot_size or config.pilot_size
        if pilot_size < 1:
            raise ManifestError("pilot size must be at least 1.")

        connection = connect_manifest(args.db.resolve())
        try:
            for source in sources:
                print(f"Inventorying metadata: {source.name}", flush=True)
                entries = discover_source(source)
                if not args.no_snapshot:
                    snapshot = save_snapshot(
                        args.snapshot_dir.resolve(), source, entries
                    )
                    print(f"  Snapshot: {snapshot}")
                counts = ingest_entries(
                    connection, source, config.lecture_rules, entries
                )
                print(
                    "  Discovered {discovered}; candidates {candidate}; "
                    "excluded {excluded}; review {needs_review}; exact duplicates "
                    "{exact_duplicates}.".format(**counts)
                )
            pilot = select_pilot(
                connection,
                pilot_size,
                config.lecture_rules.pilot_excluded_title_patterns,
            )
            summary = manifest_summary(connection)
            print("\nManifest summary")
            for key, value in summary.items():
                print(f"  {key}: {value}")
            print("\nPilot selection (metadata only; no media downloaded)")
            for index, row in enumerate(pilot, start=1):
                minutes = (row["duration_seconds"] or 0) / 60
                print(
                    f"  {index:02d}. {row['title']} [{minutes:.1f} min] "
                    f"{row['canonical_url']}"
                )
        finally:
            connection.close()
    except (ManifestError, OSError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
