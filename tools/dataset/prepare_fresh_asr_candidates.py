"""Prepare audio for the six accessible candidates without English captions."""

from __future__ import annotations

# --- repo layout shim: find shared modules (V5 batch core, dialogue/dataset tools, project_env) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
for _d in (_ROOT, _ROOT / "transcription_pipeline_versions" / "v5_batch_ingestion_pipeline", _ROOT / "tools" / "dataset", _ROOT / "tools" / "dialogue"):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
# --- end shim ---
import json
import sqlite3
from pathlib import Path

from prepare_audio_pilot import (
    DEFAULT_FFMPEG_DIR,
    download_video,
    find_executable,
    update_summary,
    utc_now,
    write_json,
)


VERSION_DIR = Path(__file__).resolve().parents[2] / "transcription_pipeline_versions" / "v5_batch_ingestion_pipeline"  # data/output dir (shared with V5 batch core)
DB_PATH = VERSION_DIR / "worldview_corpus" / "worldview_discovery.sqlite"
OUTPUT_DIR = VERSION_DIR / "worldview_corpus" / "fresh_asr_candidates"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
VIDEO_IDS = (
    "2RqQFR53iTo",
    "4YzZ4ePmMY4",
    "E_HJNkh6TjI",
    "Ej1MBgHlnTk",
    "bKxT0ovSWYM",
    "k5RsJh9hGJ8",
)


def new_manifest() -> dict[str, object]:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT video_id, title, canonical_url url, channel, channel_id,
                   duration_seconds, upload_date, caption_status
            FROM videos WHERE video_id IN ({}) ORDER BY video_id
            """.format(",".join("?" for _ in VIDEO_IDS)),
            VIDEO_IDS,
        ).fetchall()
    finally:
        connection.close()
    by_id = {row["video_id"]: dict(row) for row in rows}
    if set(by_id) != set(VIDEO_IDS):
        raise RuntimeError("One or more fixed fresh-ASR candidates are missing")
    videos: list[dict[str, object]] = []
    for rank, video_id in enumerate(VIDEO_IDS, start=1):
        item = by_id[video_id]
        item.update(
            {
                "selection_rank": rank,
                "selection_reason": "accessible_candidate_without_english_captions",
                "download_status": "pending",
                "download_error": None,
                "audio_path": None,
                "audio_sha256": None,
                "audio_bytes": None,
                "audio_duration_seconds": None,
                "audio_format": None,
            }
        )
        videos.append(item)
    return {
        "state": "prepared",
        "created_at": utc_now(),
        "purpose": "fresh ASR for accessible expanded-corpus candidates without English captions",
        "total_selected_duration_seconds": sum(int(item["duration_seconds"] or 0) for item in videos),
        "videos": videos,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = (
        json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if MANIFEST_PATH.exists()
        else new_manifest()
    )
    write_json(MANIFEST_PATH, manifest)
    yt_dlp = find_executable("yt-dlp", None)
    ffmpeg_dir = DEFAULT_FFMPEG_DIR.resolve()
    ffprobe = find_executable("ffprobe", ffmpeg_dir / "ffprobe.exe")
    for index, video in enumerate(manifest["videos"], start=1):
        print(f"[{index}/{len(manifest['videos'])}] {video['video_id']}: {video['title']}", flush=True)
        download_video(video, OUTPUT_DIR, yt_dlp, ffmpeg_dir, ffprobe)
        update_summary(manifest)
        write_json(MANIFEST_PATH, manifest)
        print(f"  {video['download_status']}", flush=True)
    failures = sum(item["download_status"] != "downloaded" for item in manifest["videos"])
    print(json.dumps(manifest["download_summary"], indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
