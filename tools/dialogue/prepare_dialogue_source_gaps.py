"""Download the two frozen-corpus dialogue records lacking timestamped text."""

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
QUEUE = VERSION_DIR / "worldview_corpus" / "frozen_v3" / "conversation_queue.jsonl"
OUTPUT_DIR = VERSION_DIR / "worldview_corpus" / "dialogue_source_gaps"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
VIDEO_IDS = ("9UeF4Na28rw", "Tqnbvsojmek")


def new_manifest() -> dict[str, object]:
    records = {
        item["video_id"]: item
        for item in (
            json.loads(line)
            for line in QUEUE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if item["video_id"] in VIDEO_IDS
    }
    if set(records) != set(VIDEO_IDS):
        raise RuntimeError("dialogue source-gap records are missing from frozen queue")
    videos: list[dict[str, object]] = []
    for rank, video_id in enumerate(VIDEO_IDS, start=1):
        source = records[video_id]
        videos.append(
            {
                "video_id": video_id,
                "title": source["title"],
                "url": source["url"],
                "channel": source["channel"],
                "channel_id": source["channel_id"],
                "duration_seconds": source["duration_seconds"],
                "upload_date": source.get("upload_date"),
                "caption_status": source.get("caption_status"),
                "selection_rank": rank,
                "selection_reason": "frozen_dialogue_record_missing_timestamped_source",
                "download_status": "pending",
                "download_error": None,
                "audio_path": None,
                "audio_sha256": None,
                "audio_bytes": None,
                "audio_duration_seconds": None,
                "audio_format": None,
            }
        )
    return {
        "state": "prepared",
        "created_at": utc_now(),
        "purpose": "fresh ASR for two frozen dialogue source gaps",
        "total_selected_duration_seconds": sum(
            int(item["duration_seconds"] or 0) for item in videos
        ),
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
        print(f"[{index}/{len(manifest['videos'])}] {video['video_id']}", flush=True)
        download_video(video, OUTPUT_DIR, yt_dlp, ffmpeg_dir, ffprobe)
        update_summary(manifest)
        write_json(MANIFEST_PATH, manifest)
        print(f"  {video['download_status']}", flush=True)
    failures = sum(
        item["download_status"] != "downloaded" for item in manifest["videos"]
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
