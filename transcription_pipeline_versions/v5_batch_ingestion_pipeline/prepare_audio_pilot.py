"""Select and download a reproducible 10-lecture audio transcription pilot.

This script does not call an ASR or LLM API. It creates a stratified pilot from
the deduplicated lecture pool, downloads audio with yt-dlp, validates it with
ffprobe, and records checksums and run state for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from build_manifest import DEFAULT_DB_PATH, connect_manifest


VERSION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VERSION_DIR.parents[1]
DEFAULT_OUTPUT_DIR = VERSION_DIR / "audio_transcription_pilot"
DEFAULT_FFMPEG_DIR = (
    PROJECT_ROOT / "ffmpeg" / "ffmpeg-8.1.2-essentials_build" / "bin"
)
POT_PROVIDER_SERVER_DIR = (
    VERSION_DIR / "third_party" / "bgutil-ytdlp-pot-provider" / "server"
)
SELECTION_SEED = "ahm-audio-pilot-v1"
ANCHOR_VIDEO_ID = "cIPqGAiLSAY"
PILOT_SIZE = 10
NO_CAPTION_TARGET = 2
DURATION_BUCKETS = (
    ("10-19m", 600, 1199, 2),
    ("20-39m", 1200, 2399, 2),
    ("40-59m", 2400, 3599, 2),
    ("60-89m", 3600, 5399, 2),
    ("90-120m", 5400, 7200, 2),
)

# These formats are unsuitable for a speech-transcription benchmark. The rules
# are intentionally narrow so lecture titles discussing dhikr/qasida remain.
PILOT_EXCLUSION_PATTERNS = (
    r"(?:^|\s)dhikr rihla\b",
    r"\bdhikr\s*&\s*qasidahs\b",
    r"^qasida(?:h)? burdah by\b",
    r"\bep\s*\d+\b",
    r"\bepisode\s*\d+\b",
    r"\bwith\s+(?:shaykh|sheikh|dr\.?|professor)\b",
    r"\bmawlid\b",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def title_is_pilot_eligible(title: str) -> bool:
    normalized = " ".join(title.casefold().split())
    return not any(re.search(pattern, normalized) for pattern in PILOT_EXCLUSION_PATTERNS)


def duration_bucket(duration_seconds: int) -> str | None:
    for name, minimum, maximum, _quota in DURATION_BUCKETS:
        if minimum <= duration_seconds <= maximum:
            return name
    return None


def stable_key(video_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}:{video_id}".encode("utf-8")).hexdigest()


def load_pool(connection: sqlite3.Connection) -> list[dict[str, object]]:
    rows = connection.execute(
        """
        SELECT video_id, title, canonical_url url, channel, channel_id,
               duration_seconds, upload_date, caption_status
        FROM videos
        WHERE candidate_status = 'candidate'
          AND content_status = 'lecture_candidate'
          AND duplicate_of_video_id IS NULL
        ORDER BY video_id
        """
    ).fetchall()
    pool: list[dict[str, object]] = []
    for row in rows:
        duration = int(row["duration_seconds"] or 0)
        bucket = duration_bucket(duration)
        if bucket is None or not title_is_pilot_eligible(row["title"]):
            continue
        item = dict(row)
        item["duration_bucket"] = bucket
        pool.append(item)
    return pool


def first_with_new_channel(
    candidates: Iterable[dict[str, object]], used_channels: set[str]
) -> dict[str, object] | None:
    candidates = list(candidates)
    return next(
        (item for item in candidates if str(item["channel"]) not in used_channels),
        candidates[0] if candidates else None,
    )


def select_pilot(pool: list[dict[str, object]]) -> list[dict[str, object]]:
    by_id = {str(item["video_id"]): item for item in pool}
    if ANCHOR_VIDEO_ID not in by_id:
        raise ValueError(f"Benchmark anchor is not eligible: {ANCHOR_VIDEO_ID}")
    anchor = dict(by_id[ANCHOR_VIDEO_ID])
    anchor["selection_reason"] = "existing benchmark anchor"
    selected = [anchor]
    used_ids = {ANCHOR_VIDEO_ID}
    used_channels = {str(anchor["channel"])}

    # Include two videos without captions to prove the fresh-ASR workflow does
    # not depend on YouTube text. Use different duration strata and channels.
    for target_bucket in ("20-39m", "60-89m"):
        candidates = sorted(
            (
                item
                for item in pool
                if item["duration_bucket"] == target_bucket
                and item["caption_status"] == "unavailable"
                and item["video_id"] not in used_ids
            ),
            key=lambda item: stable_key(str(item["video_id"])),
        )
        choice = first_with_new_channel(candidates, used_channels)
        if choice is None:
            raise ValueError(f"No no-caption candidate available in {target_bucket}")
        choice = dict(choice)
        choice["selection_reason"] = "fresh-ASR no-caption coverage"
        selected.append(choice)
        used_ids.add(str(choice["video_id"]))
        used_channels.add(str(choice["channel"]))

    for bucket_name, _minimum, _maximum, quota in DURATION_BUCKETS:
        while sum(item["duration_bucket"] == bucket_name for item in selected) < quota:
            candidates = sorted(
                (
                    item
                    for item in pool
                    if item["duration_bucket"] == bucket_name
                    and item["video_id"] not in used_ids
                ),
                key=lambda item: stable_key(str(item["video_id"])),
            )
            preferred = [
                item
                for item in candidates
                if item["caption_status"] == "available"
                and str(item["channel"]) not in used_channels
            ]
            choice = (
                preferred[0]
                if preferred
                else first_with_new_channel(candidates, used_channels)
            )
            if choice is None:
                raise ValueError(f"Unable to fill pilot quota for {bucket_name}")
            choice = dict(choice)
            choice["selection_reason"] = "seeded duration/channel stratification"
            selected.append(choice)
            used_ids.add(str(choice["video_id"]))
            used_channels.add(str(choice["channel"]))

    selected.sort(key=lambda item: (int(item["duration_seconds"]), str(item["video_id"])))
    for rank, item in enumerate(selected, start=1):
        item["selection_rank"] = rank
        item["download_status"] = "pending"
        item["audio_path"] = None
        item["audio_sha256"] = None
        item["audio_bytes"] = None
        item["audio_duration_seconds"] = None
        item["download_checked_at"] = None
        item["download_error"] = None

    if len(selected) != PILOT_SIZE:
        raise ValueError(f"Expected {PILOT_SIZE} selections, got {len(selected)}")
    if len({item["video_id"] for item in selected}) != PILOT_SIZE:
        raise ValueError("Pilot selection contains duplicate video IDs")
    if len({item["channel"] for item in selected}) != PILOT_SIZE:
        raise ValueError("Pilot selection does not cover 10 distinct channels")
    if sum(item["caption_status"] == "unavailable" for item in selected) != NO_CAPTION_TARGET:
        raise ValueError("Pilot does not contain exactly two no-caption videos")
    return selected


def new_manifest(connection: sqlite3.Connection) -> dict[str, object]:
    pool = load_pool(connection)
    selected = select_pilot(pool)
    raw_unique_count = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM videos
            WHERE candidate_status = 'candidate'
              AND content_status = 'lecture_candidate'
              AND duplicate_of_video_id IS NULL
            """
        ).fetchone()[0]
    )
    return {
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "selection_method": "seeded stratification; no quality-score cherry-picking",
        "selection_seed": SELECTION_SEED,
        "source_pool_unique_lecture_candidates": raw_unique_count,
        "pilot_eligible_after_narrow_format_and_duration_rules": len(pool),
        "pilot_size": PILOT_SIZE,
        "criteria": {
            "duration_bucket_quotas": [
                {"name": name, "minimum": minimum, "maximum": maximum, "quota": quota}
                for name, minimum, maximum, quota in DURATION_BUCKETS
            ],
            "distinct_channels_required": PILOT_SIZE,
            "no_caption_videos_required": NO_CAPTION_TARGET,
            "benchmark_anchor_video_id": ANCHOR_VIDEO_ID,
            "excluded_title_patterns": list(PILOT_EXCLUSION_PATTERNS),
            "records_deleted": False,
        },
        "download_summary": {"pending": PILOT_SIZE, "downloaded": 0, "error": 0},
        "total_selected_duration_seconds": sum(
            int(item["duration_seconds"]) for item in selected
        ),
        "videos": selected,
    }


def find_executable(name: str, explicit: Path | None = None) -> Path:
    if explicit:
        resolved = explicit.resolve()
        if resolved.is_file():
            return resolved
        raise FileNotFoundError(f"Executable not found: {resolved}")
    found = shutil.which(name)
    if not found:
        raise FileNotFoundError(f"Required executable is not on PATH: {name}")
    return Path(found)


def audio_metadata(ffprobe: Path, audio_path: Path) -> dict[str, object]:
    command = [
        str(ffprobe),
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,format_name",
        "-of",
        "json",
        str(audio_path),
    ]
    result: subprocess.CompletedProcess[str] | None = None
    last_error: subprocess.CalledProcessError | None = None
    # Windows occasionally returns 0xC0000142 while initializing the bundled
    # ffprobe process. Retrying validation is safe because it is read-only.
    for attempt in range(3):
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            break
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1)
    if result is None:
        assert last_error is not None
        raise last_error
    data = json.loads(result.stdout)["format"]
    audio_bytes = audio_path.read_bytes()
    return {
        "audio_path": audio_path.resolve().relative_to(VERSION_DIR).as_posix(),
        "audio_sha256": hashlib.sha256(audio_bytes).hexdigest(),
        "audio_bytes": len(audio_bytes),
        "audio_duration_seconds": round(float(data["duration"]), 3),
        "audio_format": data.get("format_name"),
    }


def update_summary(manifest: dict[str, object]) -> None:
    videos = manifest["videos"]
    manifest["updated_at"] = utc_now()
    manifest["download_summary"] = {
        status: sum(item["download_status"] == status for item in videos)
        for status in ("pending", "downloaded", "error")
    }


def download_video(
    video: dict[str, object],
    output_dir: Path,
    yt_dlp: Path,
    ffmpeg_dir: Path,
    ffprobe: Path,
) -> None:
    video_dir = output_dir / "videos" / str(video["video_id"])
    video_dir.mkdir(parents=True, exist_ok=True)
    audio_path = video_dir / "audio.mp3"
    if audio_path.is_file() and audio_path.stat().st_size > 0:
        video.update(audio_metadata(ffprobe, audio_path))
        video["download_status"] = "downloaded"
        video["download_checked_at"] = utc_now()
        video["download_error"] = None
        return

    common_command = [
        str(yt_dlp),
        "--js-runtimes",
        "node",
        "--no-playlist",
        "--continue",
        "--no-overwrites",
        "--retries",
        "5",
        "--fragment-retries",
        "5",
        "--sleep-requests",
        "1",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "64K",
        "--ffmpeg-location",
        str(ffmpeg_dir),
        "--write-info-json",
    ]
    attempts: list[tuple[str, list[str], Path, Path]] = [
        (
            "default",
            [],
            video_dir / "audio.%(ext)s",
            audio_path,
        ),
        (
            "web_safari_hls_fallback",
            [
                "--extractor-args",
                "youtube:player_client=web_safari",
                "--format",
                "bestaudio/best",
                "--no-continue",
            ],
            video_dir / "audio_fallback.%(ext)s",
            video_dir / "audio_fallback.mp3",
        ),
    ]
    if (POT_PROVIDER_SERVER_DIR / "build" / "main.js").is_file():
        attempts.append(
            (
                "mweb_with_local_pot_provider",
                [
                    "--verbose",
                    "--extractor-args",
                    "youtube:player_client=mweb",
                    "--extractor-args",
                    (
                        "youtubepot-bgutilscript:server_home="
                        f"{POT_PROVIDER_SERVER_DIR.as_posix()}"
                    ),
                    "--format",
                    "bestaudio/best",
                    "--no-continue",
                ],
                video_dir / "audio_pot.%(ext)s",
                video_dir / "audio_pot.mp3",
            )
        )
    logs: list[str] = []
    result: subprocess.CompletedProcess[str] | None = None
    for attempt_name, extra_args, output_template, expected_path in attempts:
        command = [
            *common_command,
            *extra_args,
            "--output",
            str(output_template),
            str(video["url"]),
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        logs.append(
            f"ATTEMPT: {attempt_name}\n"
            f"COMMAND: yt-dlp [options] {video['url']}\n\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        if result.returncode == 0 and expected_path.is_file():
            if expected_path != audio_path:
                expected_path.replace(audio_path)
            break
    (video_dir / "download.log").write_text(
        "\n\n".join(logs),
        encoding="utf-8",
    )
    video["download_checked_at"] = utc_now()
    assert result is not None
    if result.returncode != 0 or not audio_path.is_file():
        video["download_status"] = "error"
        error_lines = (result.stderr or result.stdout).strip().splitlines()
        video["download_error"] = error_lines[-1] if error_lines else "yt-dlp failed"
        return
    video.update(audio_metadata(ffprobe, audio_path))
    video["download_status"] = "downloaded"
    video["download_error"] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--reselect", action="store_true")
    parser.add_argument("--yt-dlp", type=Path)
    parser.add_argument("--ffmpeg-dir", type=Path, default=DEFAULT_FFMPEG_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "pilot_manifest.json"
    try:
        if manifest_path.exists() and not args.reselect:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        else:
            connection = connect_manifest(args.db.resolve())
            try:
                manifest = new_manifest(connection)
            finally:
                connection.close()
            write_json(manifest_path, manifest)
        print(f"Selected videos: {len(manifest['videos'])}")
        print(f"Selected hours: {manifest['total_selected_duration_seconds'] / 3600:.2f}")
        print(f"Manifest: {manifest_path}")
        if args.select_only:
            return 0

        yt_dlp = find_executable("yt-dlp", args.yt_dlp)
        ffmpeg_dir = args.ffmpeg_dir.resolve()
        ffprobe = find_executable("ffprobe", ffmpeg_dir / "ffprobe.exe")
        for video in manifest["videos"]:
            print(
                f"[{video['selection_rank']}/{PILOT_SIZE}] "
                f"{video['video_id']} - {video['title']}",
                flush=True,
            )
            download_video(video, output_dir, yt_dlp, ffmpeg_dir, ffprobe)
            update_summary(manifest)
            write_json(manifest_path, manifest)
            print(f"  status: {video['download_status']}", flush=True)
        summary = manifest["download_summary"]
        print(f"Downloaded: {summary['downloaded']}")
        print(f"Errors: {summary['error']}")
        return 0 if summary["downloaded"] == PILOT_SIZE else 1
    except (OSError, sqlite3.Error, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
