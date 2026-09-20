"""Download and process an approved 10-lecture expansion batch resumably.

The script records approval in the selection manifest, downloads and validates
each audio file, runs the existing AssemblyAI + two-call Gemini pipeline, and
writes machine-readable and Markdown summaries. Completed work is skipped on
reruns, so an interrupted batch can safely resume.
"""

from __future__ import annotations

import argparse
import atexit
import json
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prepare_audio_pilot import (
    DEFAULT_FFMPEG_DIR,
    POT_PROVIDER_SERVER_DIR,
    VERSION_DIR,
    download_video,
    find_executable,
)


EXPANSION_DIR = VERSION_DIR / "lecture_expansion"
SINGLE_RUNNER = VERSION_DIR / "run_fresh_benchmark.py"
POT_PROVIDER_PORT = 4416


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def local_port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def stop_pot_provider(
    process: subprocess.Popen[bytes], log_handle: Any
) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    log_handle.close()


def start_pot_provider(output_dir: Path) -> tuple[subprocess.Popen[bytes], Any] | None:
    entrypoint = POT_PROVIDER_SERVER_DIR / "build" / "main.js"
    if not entrypoint.is_file() or local_port_is_open(POT_PROVIDER_PORT):
        return None
    log_handle = (output_dir / "pot_provider.log").open("ab")
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        ["node", str(entrypoint), "--port", str(POT_PROVIDER_PORT)],
        cwd=POT_PROVIDER_SERVER_DIR,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=creation_flags,
    )
    for _ in range(120):
        if local_port_is_open(POT_PROVIDER_PORT):
            return process, log_handle
        if process.poll() is not None:
            log_handle.close()
            raise RuntimeError(
                f"Local proof-of-origin provider exited with {process.returncode}"
            )
        time.sleep(0.5)
    stop_pot_provider(process, log_handle)
    raise RuntimeError("Local proof-of-origin provider did not start within 60 seconds")


def batch_dir(batch_number: int) -> Path:
    return EXPANSION_DIR / f"batch_{batch_number:02d}"


def summary_path(video: dict[str, Any]) -> Path | None:
    audio_path = video.get("audio_path")
    if not audio_path:
        return None
    resolved_audio = VERSION_DIR / Path(str(audio_path))
    return resolved_audio.parent / "fresh_benchmark" / "benchmark_summary.json"


def normalized_summary_path(video: dict[str, Any]) -> Path | None:
    path = summary_path(video)
    return path.resolve() if path else None


def completed(video: dict[str, Any]) -> bool:
    path = normalized_summary_path(video)
    if not path or not path.is_file():
        return False
    try:
        return load_json(path).get("state") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def record_approval(manifest: dict[str, Any]) -> None:
    if manifest.get("state") == "proposed_awaiting_user_approval":
        approved_at = utc_now()
        manifest["state"] = "approved_for_processing"
        manifest["approved_at"] = approved_at
        manifest["approval_note"] = (
            "User explicitly approved keeping and processing all ten selected "
            "lectures in the current batch."
        )
        criteria = manifest.setdefault("criteria", {})
        criteria["audio_downloads_allowed"] = True
        criteria["paid_api_calls_allowed"] = True
        for video in manifest["videos"]:
            video["selection_status"] = "approved_for_processing"


def download_counts(videos: list[dict[str, Any]]) -> dict[str, int]:
    return {
        key: sum(video.get("download_status") == key for video in videos)
        for key in ("not_started", "pending", "downloaded", "error")
    }


def processing_counts(videos: list[dict[str, Any]]) -> dict[str, int]:
    complete_count = sum(completed(video) for video in videos)
    error_count = sum(video.get("transcription_status") == "error" for video in videos)
    return {
        "completed": complete_count,
        "error": error_count,
        "remaining": len(videos) - complete_count,
    }


def update_status(
    status_path: Path,
    manifest: dict[str, Any],
    state: str,
    current_stage: str | None,
    current_video_id: str | None,
    failures: list[dict[str, Any]],
) -> None:
    videos = manifest["videos"]
    old: dict[str, Any] = {}
    if status_path.is_file():
        try:
            old = load_json(status_path)
        except (OSError, json.JSONDecodeError):
            old = {}
    status = {
        "state": state,
        "batch_number": manifest["batch_number"],
        "started_at": old.get("started_at", utc_now()),
        "updated_at": utc_now(),
        "current_stage": current_stage,
        "current_video_id": current_video_id,
        "download_counts": download_counts(videos),
        "processing_counts": processing_counts(videos),
        "failed_videos": failures,
    }
    if state in {"completed", "completed_with_errors"}:
        status["completed_at"] = utc_now()
    write_json(status_path, status)


def add_numbers(summaries: list[dict[str, Any]], section: str, field: str) -> float:
    return sum(float(item.get(section, {}).get(field, 0) or 0) for item in summaries)


def write_batch_report(output_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    videos = sorted(manifest["videos"], key=lambda item: item["selection_rank"])
    results: list[dict[str, Any]] = []
    for video in videos:
        path = normalized_summary_path(video)
        if path and path.is_file():
            try:
                summary = load_json(path)
                if summary.get("state") == "completed":
                    results.append(summary)
            except (OSError, json.JSONDecodeError):
                pass

    guardrail_failures = sum(
        not bool(item.get("metrics", {}).get("preservation_guardrails_passed"))
        for item in results
    )
    manual_review_chunks = sum(
        len(item.get("manual_review_required_chunks", [])) for item in results
    )
    report = {
        "state": "completed" if len(results) == len(videos) else "incomplete",
        "generated_at": utc_now(),
        "batch_number": manifest["batch_number"],
        "selected_videos": len(videos),
        "completed_videos": len(results),
        "selected_duration_seconds": manifest["total_selected_duration_seconds"],
        "selected_duration_hours": round(
            float(manifest["total_selected_duration_seconds"]) / 3600, 3
        ),
        "assemblyai": {
            "model": "universal-2",
            "total_words": int(add_numbers(results, "assemblyai", "word_count")),
            "mean_confidence": round(
                add_numbers(results, "assemblyai", "confidence") / len(results), 6
            )
            if results
            else None,
        },
        "openrouter": {
            "model": "google/gemini-3.1-flash-lite",
            "requests": int(add_numbers(results, "openrouter_usage", "requests")),
            "network_requests": int(
                add_numbers(results, "openrouter_usage", "network_requests")
            ),
            "cache_hits": int(add_numbers(results, "openrouter_usage", "cache_hits")),
            "prompt_tokens": int(
                add_numbers(results, "openrouter_usage", "prompt_tokens")
            ),
            "completion_tokens": int(
                add_numbers(results, "openrouter_usage", "completion_tokens")
            ),
            "total_tokens": int(add_numbers(results, "openrouter_usage", "total_tokens")),
            "reported_cost_credits": round(
                add_numbers(results, "openrouter_usage", "reported_cost_credits"), 10
            ),
        },
        "quality_controls": {
            "maximum_change_volume_pct": 8.0,
            "minimum_source_similarity_pct": 92.0,
            "guardrail_failures": guardrail_failures,
            "manual_review_required_chunks": manual_review_chunks,
            "manual_review_remains_required": True,
            "source_similarity_is_not_transcription_accuracy": True,
        },
        "videos": [
            {
                "selection_rank": video["selection_rank"],
                "video_id": video["video_id"],
                "title": video["title"],
                "channel": video["channel"],
                "duration_seconds": video["duration_seconds"],
                "download_status": video.get("download_status"),
                "transcription_status": "completed" if completed(video) else video.get("transcription_status"),
                "manual_review_required_chunks": len(
                    load_json(normalized_summary_path(video)).get(
                        "manual_review_required_chunks", []
                    )
                )
                if completed(video) and normalized_summary_path(video)
                else 0,
                "benchmark_summary": str(normalized_summary_path(video))
                if normalized_summary_path(video)
                else None,
            }
            for video in videos
        ],
    }
    write_json(output_dir / "batch_report.json", report)

    lines = [
        f"# Lecture expansion batch {manifest['batch_number']:02d}",
        "",
        f"- State: {report['state']}",
        f"- Completed: {report['completed_videos']}/{report['selected_videos']}",
        f"- Selected audio: {report['selected_duration_hours']:.3f} hours",
        f"- AssemblyAI words: {report['assemblyai']['total_words']:,}",
        f"- Gemini/OpenRouter requests: {report['openrouter']['requests']:,}",
        f"- Gemini/OpenRouter tokens: {report['openrouter']['total_tokens']:,}",
        f"- OpenRouter reported cost: {report['openrouter']['reported_cost_credits']:.10f} credits",
        f"- Preservation guardrail failures: {guardrail_failures}",
        f"- Source-retained chunks requiring manual review: {manual_review_chunks}",
        "",
        "Source similarity is a preservation measurement, not a transcription-accuracy measurement. Raw and corrected transcripts are retained for every video.",
        "",
        "## Videos",
        "",
    ]
    for item in report["videos"]:
        lines.append(
            f"{item['selection_rank']}. `{item['video_id']}` — {item['title']} — {item['transcription_status']}"
        )
    (output_dir / "batch_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, required=True, choices=range(1, 5))
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Record explicit user approval before allowing downloads/API calls.",
    )
    parser.add_argument("--yt-dlp", type=Path)
    parser.add_argument("--ffmpeg-dir", type=Path, default=DEFAULT_FFMPEG_DIR)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate aggregate reports without downloads or API calls.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = batch_dir(args.batch)
    manifest_path = output_dir / "selection_manifest.json"
    status_path = output_dir / "batch_status.json"
    if not manifest_path.is_file():
        print(f"ERROR: Selection manifest does not exist: {manifest_path}", file=sys.stderr)
        return 1

    manifest = load_json(manifest_path)
    if manifest.get("batch_number") != args.batch or len(manifest.get("videos", [])) != 10:
        print("ERROR: Manifest batch number or size is invalid.", file=sys.stderr)
        return 1
    if args.report_only:
        report = write_batch_report(output_dir, manifest)
        print(
            f"REPORT WRITTEN {report['completed_videos']}/{report['selected_videos']} complete",
            flush=True,
        )
        return 0 if report["state"] == "completed" else 1
    if args.approve:
        record_approval(manifest)
        write_json(manifest_path, manifest)
    if manifest.get("state") not in {
        "approved_for_processing",
        "processing",
        "completed",
        "completed_with_errors",
    }:
        print("ERROR: Batch has not been explicitly approved.", file=sys.stderr)
        return 1

    videos = sorted(manifest["videos"], key=lambda item: item["selection_rank"])
    failures: list[dict[str, Any]] = []
    manifest["state"] = "processing"
    manifest["processing_started_at"] = manifest.get("processing_started_at", utc_now())
    write_json(manifest_path, manifest)
    update_status(status_path, manifest, "running", "download", None, failures)

    yt_dlp = find_executable("yt-dlp", args.yt_dlp)
    ffmpeg_dir = args.ffmpeg_dir.resolve()
    ffprobe = find_executable("ffprobe", ffmpeg_dir / "ffprobe.exe")
    provider = start_pot_provider(output_dir)
    if provider:
        atexit.register(stop_pot_provider, *provider)
        print("Local proof-of-origin provider ready on port 4416", flush=True)
    print(f"Batch {args.batch:02d}: download/validation stage", flush=True)
    for video in videos:
        video_id = str(video["video_id"])
        update_status(status_path, manifest, "running", "download", video_id, failures)
        print(f"DOWNLOAD {video['selection_rank']}/10 {video_id}", flush=True)
        try:
            video["download_status"] = "pending"
            download_video(video, output_dir, yt_dlp, ffmpeg_dir, ffprobe)
        except Exception as exc:
            video["download_status"] = "error"
            video["download_error"] = f"{type(exc).__name__}: {exc}"
        if video.get("download_status") != "downloaded":
            failures.append(
                {
                    "stage": "download",
                    "video_id": video_id,
                    "error": video.get("download_error", "download failed"),
                }
            )
            print(f"DOWNLOAD ERROR {video_id}: {video.get('download_error')}", flush=True)
        else:
            print(f"DOWNLOADED {video_id}", flush=True)
        write_json(manifest_path, manifest)
        update_status(status_path, manifest, "running", "download", None, failures)

    if provider:
        stop_pot_provider(*provider)
        atexit.unregister(stop_pot_provider)

    print("TRANSCRIPTION/CORRECTION stage", flush=True)
    for video in videos:
        video_id = str(video["video_id"])
        if video.get("download_status") != "downloaded":
            continue
        if completed(video):
            video["transcription_status"] = "completed"
            write_json(manifest_path, manifest)
            print(f"SKIP COMPLETED {video_id}", flush=True)
            continue
        video["transcription_status"] = "running"
        video["transcription_started_at"] = utc_now()
        write_json(manifest_path, manifest)
        update_status(status_path, manifest, "running", "transcribe_and_correct", video_id, failures)
        print(f"PROCESS {video['selection_rank']}/10 {video_id} - {video['title']}", flush=True)
        result = subprocess.run(
            [
                sys.executable,
                str(SINGLE_RUNNER),
                "--manifest",
                str(manifest_path),
                "--video-id",
                video_id,
            ],
            cwd=VERSION_DIR,
        )
        if result.returncode == 0 and completed(video):
            video["transcription_status"] = "completed"
            video["transcription_completed_at"] = utc_now()
            print(f"COMPLETED {video_id}", flush=True)
        else:
            video["transcription_status"] = "error"
            error = f"One-video runner returned {result.returncode}"
            video["transcription_error"] = error
            failures.append({"stage": "transcribe_and_correct", "video_id": video_id, "error": error})
            print(f"PROCESSING ERROR {video_id}: {error}", flush=True)
        write_json(manifest_path, manifest)
        write_batch_report(output_dir, manifest)
        update_status(status_path, manifest, "running", "transcribe_and_correct", None, failures)

    report = write_batch_report(output_dir, manifest)
    manifest["state"] = "completed" if report["state"] == "completed" else "completed_with_errors"
    manifest["processing_completed_at"] = utc_now()
    write_json(manifest_path, manifest)
    update_status(status_path, manifest, manifest["state"], None, None, failures)
    print(
        f"BATCH FINISHED {report['completed_videos']}/{report['selected_videos']} complete",
        flush=True,
    )
    return 0 if report["state"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
