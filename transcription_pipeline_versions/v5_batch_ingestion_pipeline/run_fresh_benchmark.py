"""Run one fresh AssemblyAI + Gemini 3.1 Flash-Lite benchmark.

The default target is the shortest prepared audio-pilot lecture. The script is
deliberately gated to one video, resumes from an existing AssemblyAI transcript,
and caches successful OpenRouter responses by request hash to avoid repeat cost.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


VERSION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VERSION_DIR.parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
PILOT_DIR = VERSION_DIR / "audio_transcription_pilot"
DEFAULT_MANIFEST = PILOT_DIR / "pilot_manifest.json"
V4_PIPELINE_PATH = (
    PROJECT_ROOT
    / "transcription_pipeline_versions"
    / "v4_assemblyai_two_call_pipeline"
    / "two_call_pipeline.py"
)
MODEL = "google/gemini-3.1-flash-lite"
MAX_CHANGE_VOLUME_PCT = 8.0
MIN_SOURCE_SIMILARITY_PCT = 92.0

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_env import load_env_file, require_env  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_v4_pipeline() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smartlabs_v4_pipeline", V4_PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import pipeline: {V4_PIPELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def select_video(manifest: dict[str, Any], video_id: str | None) -> dict[str, Any]:
    videos = manifest["videos"]
    if video_id:
        matches = [item for item in videos if item["video_id"] == video_id]
        if not matches:
            raise ValueError(f"Video is not in the fixed pilot: {video_id}")
        return matches[0]
    return min(videos, key=lambda item: (item["duration_seconds"], item["video_id"]))


def resolve_audio(video: dict[str, Any]) -> Path:
    if video.get("download_status") != "downloaded" or not video.get("audio_path"):
        raise ValueError(f"Audio is not ready for {video['video_id']}")
    audio_path = VERSION_DIR / video["audio_path"]
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise ValueError(f"Audio file is missing or empty: {audio_path}")
    actual_hash = sha256_file(audio_path)
    if actual_hash != video.get("audio_sha256"):
        raise ValueError(f"Audio checksum mismatch: {audio_path}")
    return audio_path


def transcribe_with_assemblyai(
    audio_path: Path, transcript_path: Path, response_path: Path, metadata_path: Path
) -> dict[str, Any]:
    if transcript_path.is_file() and metadata_path.is_file():
        text = transcript_path.read_text(encoding="utf-8-sig")
        if text.strip():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["resumed_from_saved_transcript"] = True
            return metadata

    import assemblyai as aai

    load_env_file(ENV_PATH)
    aai.settings.api_key = require_env("ASSEMBLYAI_API_KEY", env_path=ENV_PATH)
    config = aai.TranscriptionConfig(
        speech_models=["universal-2"],
        punctuate=True,
        format_text=True,
    )
    print(f"AssemblyAI: transcribing {audio_path.name} with universal-2...", flush=True)
    started_at = utc_now()
    started = time.time()
    transcript = aai.Transcriber(config=config).transcribe(str(audio_path))
    elapsed = time.time() - started
    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")
    text = (transcript.text or "").strip()
    if not text:
        raise RuntimeError("AssemblyAI returned an empty transcript")

    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(text + "\n", encoding="utf-8")
    response = dict(transcript.json_response or {})
    response.pop("audio_url", None)
    write_json(response_path, response)
    metadata = {
        "started_at": started_at,
        "completed_at": utc_now(),
        "elapsed_seconds": round(elapsed, 3),
        "transcript_id": transcript.id,
        "status": str(transcript.status),
        "requested_speech_model": "universal-2",
        "speech_model_used": str(transcript.speech_model_used or transcript.speech_model),
        "audio_duration_seconds": transcript.audio_duration,
        "confidence": transcript.confidence,
        "language_code": str(transcript.language_code),
        "word_count": len(text.split()),
        "character_count": len(text),
        "sdk_reported_cost": None,
        "billing_note": "AssemblyAI SDK response does not report per-transcript cost.",
        "resumed_from_saved_transcript": False,
    }
    write_json(metadata_path, metadata)
    print(
        f"AssemblyAI completed in {elapsed:.1f}s: {metadata['word_count']} words",
        flush=True,
    )
    return metadata


def request_cache_key(kwargs: dict[str, Any]) -> str:
    request = {
        "model": kwargs["model"],
        "system_prompt": kwargs["system_prompt"],
        "user_message": kwargs["user_message"],
        "max_tokens": kwargs["max_tokens"],
        "stage": kwargs["stage"],
    }
    payload = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def install_openrouter_cache(
    pipeline: ModuleType, cache_dir: Path, request_log_path: Path
) -> list[dict[str, Any]]:
    original_call = pipeline.call_openrouter
    events: list[dict[str, Any]] = []
    cache_dir.mkdir(parents=True, exist_ok=True)

    def cached_call(**kwargs: Any) -> tuple[str, str]:
        key = request_cache_key(kwargs)
        cache_path = cache_dir / f"{key}.json"
        event: dict[str, Any] = {
            "timestamp": utc_now(),
            "request_sha256": key,
            "stage": kwargs["stage"],
            "requested_model": kwargs["model"],
            "system_prompt_sha256": hashlib.sha256(
                kwargs["system_prompt"].encode("utf-8")
            ).hexdigest(),
            "user_message_sha256": hashlib.sha256(
                kwargs["user_message"].encode("utf-8")
            ).hexdigest(),
        }
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            kwargs["usage_sink"].append(pipeline.ApiUsage(**cached["usage"]))
            event.update({"cache_hit": True, "usage": cached["usage"]})
            events.append(event)
            with request_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            return cached["content"], cached["actual_model"]

        before = len(kwargs["usage_sink"])
        content, actual_model = original_call(**kwargs)
        if len(kwargs["usage_sink"]) != before + 1:
            raise RuntimeError("OpenRouter usage record was not captured")
        usage = asdict(kwargs["usage_sink"][-1])
        cached = {
            "created_at": utc_now(),
            "request_sha256": key,
            "actual_model": actual_model,
            "content": content,
            "usage": usage,
        }
        write_json(cache_path, cached)
        event.update({"cache_hit": False, "usage": usage})
        events.append(event)
        with request_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return content, actual_model

    pipeline.call_openrouter = cached_call
    return events


def score_summary(pipeline: ModuleType, source: str, corrected: str, reference: str | None) -> dict[str, Any]:
    audit = pipeline.audit_change(source, corrected)
    result: dict[str, Any] = {
        "source_words": audit.source_words,
        "corrected_words": audit.candidate_words,
        "change_volume_pct": audit.change_volume_pct,
        "source_similarity_pct": audit.source_similarity_pct,
        "preservation_guardrails_passed": audit.accepted,
    }
    if reference:
        original = pipeline.score_transcript(reference, source)
        edited = pipeline.score_transcript(reference, corrected)
        result["youtube_caption_diagnostic"] = {
            "reference_is_ground_truth": False,
            "raw_vocabulary_overlap_pct": original.vocabulary_overlap_pct,
            "corrected_vocabulary_overlap_pct": edited.vocabulary_overlap_pct,
            "vocabulary_delta_pp": round(
                edited.vocabulary_overlap_pct - original.vocabulary_overlap_pct, 6
            ),
            "raw_sequence_agreement_pct": original.sequence_agreement_pct,
            "corrected_sequence_agreement_pct": edited.sequence_agreement_pct,
            "sequence_delta_pp": round(
                edited.sequence_agreement_pct - original.sequence_agreement_pct, 6
            ),
        }
    return result


def summarize_usage(events: list[dict[str, Any]]) -> dict[str, Any]:
    usages = [event["usage"] for event in events]
    return {
        "requests": len(usages),
        "network_requests": sum(not event["cache_hit"] for event in events),
        "cache_hits": sum(event["cache_hit"] for event in events),
        "prompt_tokens": sum(int(item["prompt_tokens"]) for item in usages),
        "completion_tokens": sum(int(item["completion_tokens"]) for item in usages),
        "total_tokens": sum(int(item["total_tokens"]) for item in usages),
        "reported_cost_credits": round(
            sum(float(item["cost_credits"] or 0.0) for item in usages), 10
        ),
        "actual_models": sorted({str(item["actual_model"]) for item in usages}),
    }


def write_readme(
    output_dir: Path, video: dict[str, Any], summary: dict[str, Any]
) -> None:
    metrics = summary["metrics"]
    usage = summary["openrouter_usage"]
    diagnostic = metrics.get("youtube_caption_diagnostic")
    diagnostic_lines = ""
    if diagnostic:
        diagnostic_lines = f"""
- YouTube-caption sequence agreement: {diagnostic['raw_sequence_agreement_pct']:.2f}% raw to {diagnostic['corrected_sequence_agreement_pct']:.2f}% corrected ({diagnostic['sequence_delta_pp']:+.2f} pp)
"""
    text = f"""# One-video fresh transcription benchmark

- Video: `{video['video_id']}` — {video['title']}
- Audio duration: {video['audio_duration_seconds'] / 60:.2f} minutes
- AssemblyAI model: Universal-2
- Detection model: `{MODEL}`
- Correction model: `{MODEL}`
- Raw transcript words: {metrics['source_words']}
- Corrected transcript words: {metrics['corrected_words']}
- Change volume: {metrics['change_volume_pct']:.2f}%
- Source similarity: {metrics['source_similarity_pct']:.2f}%
- Preservation guardrails passed: {metrics['preservation_guardrails_passed']}
- OpenRouter requests: {usage['requests']} ({usage['cache_hits']} cache hits)
- OpenRouter tokens: {usage['total_tokens']}
- OpenRouter reported cost: {usage['reported_cost_credits']:.10f} credits
{diagnostic_lines}
The YouTube caption was not supplied to AssemblyAI or Gemini. Its comparison is
diagnostic only and is not treated as human ground truth. Selected Islamic-term
changes still require review against the lecture audio before this model is
approved for the remaining nine pilot lectures.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--video-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
        video = select_video(manifest, args.video_id)
        audio_path = resolve_audio(video)
        output_dir = audio_path.parent / "fresh_benchmark"
        output_dir.mkdir(parents=True, exist_ok=True)
        run_status_path = output_dir / "run_status.json"
        write_json(
            run_status_path,
            {
                "state": "running",
                "started_at": utc_now(),
                "video_id": video["video_id"],
                "assemblyai_model": "universal-2",
                "openrouter_model": MODEL,
            },
        )
        raw_path = output_dir / "assemblyai_transcript.txt"
        assembly_metadata = transcribe_with_assemblyai(
            audio_path,
            raw_path,
            output_dir / "assemblyai_response.json",
            output_dir / "assemblyai_metadata.json",
        )

        pipeline = load_v4_pipeline()
        pipeline.DETECTION_MODEL = MODEL
        pipeline.CORRECTION_MODEL = MODEL
        # The pilot keeps the untouched AssemblyAI transcript beside the corrected
        # version for manual review, so permit moderately larger targeted edits.
        pipeline.MAX_CHANGE_VOLUME_PCT = MAX_CHANGE_VOLUME_PCT
        pipeline.MIN_SOURCE_SIMILARITY_PCT = MIN_SOURCE_SIMILARITY_PCT
        pipeline.FALLBACK_REJECTED_DETECTIONS_TO_EMPTY = True
        pipeline.FALLBACK_REJECTED_CORRECTIONS_TO_SOURCE = True
        request_log_path = output_dir / "openrouter_requests.jsonl"
        events = install_openrouter_cache(
            pipeline, output_dir / "openrouter_cache", request_log_path
        )
        reference_path = VERSION_DIR / "captions" / video["video_id"] / "youtube_transcript.txt"
        configure_args = argparse.Namespace(
            audio=None,
            source=raw_path,
            output_dir=output_dir,
            no_reference=not reference_path.is_file(),
            reference=reference_path if reference_path.is_file() else None,
        )
        pipeline.configure_run_paths(configure_args)
        pipeline.run_pipeline(None)

        source = raw_path.read_text(encoding="utf-8-sig")
        corrected_path = output_dir / "two_call_corrected_transcript.txt"
        corrected = corrected_path.read_text(encoding="utf-8-sig")
        reference = (
            reference_path.read_text(encoding="utf-8-sig")
            if reference_path.is_file()
            else None
        )
        summary = {
            "state": "completed",
            "completed_at": utc_now(),
            "video": {
                "video_id": video["video_id"],
                "title": video["title"],
                "url": video["url"],
                "channel": video["channel"],
                "audio_duration_seconds": video["audio_duration_seconds"],
                "audio_sha256": video["audio_sha256"],
            },
            "assemblyai": assembly_metadata,
            "openrouter": {
                "detection_model_requested": MODEL,
                "correction_model_requested": MODEL,
            },
            "openrouter_usage": summarize_usage(events),
            "metrics": score_summary(pipeline, source, corrected, reference),
            "manual_review_required_chunks": pipeline.LAST_FALLBACK_CHUNKS,
            "artifacts": {
                "raw_transcript": raw_path.name,
                "corrected_transcript": corrected_path.name,
                "word_list": "two_call_word_list.txt",
                "diff": "two_call_diff.txt",
                "score": "two_call_score.txt",
                "assemblyai_response": "assemblyai_response.json",
                "openrouter_request_log": "openrouter_requests.jsonl",
            },
        }
        write_json(output_dir / "benchmark_summary.json", summary)
        write_json(run_status_path, summary)
        write_readme(output_dir, video, summary)
        print(f"Benchmark completed: {output_dir}")
        print(
            f"OpenRouter cost: {summary['openrouter_usage']['reported_cost_credits']:.10f} credits"
        )
        return 0
    except Exception as exc:  # Preserve state for unattended/background runs.
        try:
            if "run_status_path" in locals():
                write_json(
                    run_status_path,
                    {
                        "state": "error",
                        "failed_at": utc_now(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
        finally:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
