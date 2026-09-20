"""Verify the six fresh-ASR candidates before adding them to the corpus."""

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
import re
import sys
from pathlib import Path
from typing import Any

VERSION_DIR = Path(__file__).resolve().parents[2] / "transcription_pipeline_versions" / "v5_batch_ingestion_pipeline"  # data/output dir (shared with V5 batch core)
PROJECT_ROOT = VERSION_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_env import load_env_file, require_env
from run_fresh_benchmark import (
    ENV_PATH,
    MODEL,
    install_openrouter_cache,
    load_v4_pipeline,
    summarize_usage,
    utc_now,
    write_json,
)


ROOT = VERSION_DIR / "worldview_corpus" / "fresh_asr_candidates"
MANIFEST_PATH = ROOT / "manifest.json"
OUTPUT_DIR = ROOT / "content_verification"
WORKING_MANIFEST = (
    VERSION_DIR / "worldview_corpus" / "working_v2" / "corpus_manifest.jsonl"
)

SYSTEM_PROMPT = """You are screening a research corpus about Shaykh Abdal Hakim Murad (Timothy Winter). Use only the supplied metadata and transcript samples.

Return exactly one JSON object with these keys:
- video_id: string
- murad_speaks: "yes", "no", or "uncertain"
- speaker_evidence: concise explanation grounded in the supplied text
- content_type: "lecture", "qa", "interview", "discussion", "mixed_event", or "uncertain"
- content_evidence: concise explanation grounded in the supplied text
- confidence: number from 0 to 1
- needs_manual_review: boolean

A lecture is a substantially continuous solo talk; Q&A has explicit audience questions followed by answers; interview is interviewer-led; discussion has multiple peers; mixed_event combines talks with other event material. A title alone is insufficient. Do not include markdown or text outside the JSON object."""


def transcript_samples(text: str) -> str:
    words = text.split()
    if not words:
        raise ValueError("empty transcript")
    size = min(900, len(words))
    starts = [0, max(0, len(words) // 2 - size // 2), max(0, len(words) - size)]
    labels = ["BEGINNING", "MIDDLE", "END"]
    return "\n\n".join(
        f"{label} (words {start + 1}-{start + size}):\n"
        + " ".join(words[start : start + size])
        for label, start in zip(labels, starts, strict=True)
    )


def clean_json(content: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I)
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("model response is not an object")
    return value


def normalized_word_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def similarity(left: str, right: str) -> float:
    a = normalized_word_set(left)
    b = normalized_word_set(right)
    return len(a & b) / max(1, len(a | b))


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    videos = manifest["videos"]
    existing = [
        json.loads(line)
        for line in WORKING_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    existing_texts: dict[str, str] = {}
    for item in existing:
        caption_path = item.get("caption_path")
        if not caption_path:
            continue
        path = VERSION_DIR / str(caption_path)
        if path.is_file():
            existing_texts[str(item["video_id"])] = path.read_text(
                encoding="utf-8-sig"
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = load_v4_pipeline()
    load_env_file(ENV_PATH)
    api_key = require_env("OPENROUTER_API_KEY", env_path=ENV_PATH)
    events = install_openrouter_cache(
        pipeline, OUTPUT_DIR / "openrouter_cache", OUTPUT_DIR / "requests.jsonl"
    )
    results: list[dict[str, Any]] = []
    for index, video in enumerate(videos, start=1):
        video_id = str(video["video_id"])
        benchmark = (
            VERSION_DIR
            / str(video["audio_path"])
        ).parent / "fresh_benchmark"
        transcript_path = benchmark / "two_call_corrected_transcript.txt"
        transcript = transcript_path.read_text(encoding="utf-8-sig")
        print(f"[{index}/{len(videos)}] {video_id}", flush=True)
        content, actual_model = pipeline.call_openrouter(
            api_key=api_key,
            model=MODEL,
            system_prompt=SYSTEM_PROMPT,
            user_message=(
                f"video_id: {video_id}\n"
                f"title: {video['title']}\nchannel: {video['channel']}\n"
                f"duration_seconds: {video['duration_seconds']}\n\n"
                f"TRANSCRIPT SAMPLES\n{transcript_samples(transcript)}"
            ),
            max_tokens=700,
            stage="Fresh ASR content verification",
            usage_sink=[],
        )
        value = clean_json(content)
        if value.get("video_id") != video_id:
            raise ValueError(f"wrong video_id returned for {video_id}")
        confidence = float(value.get("confidence", 0))
        if value.get("murad_speaks") not in {"yes", "no", "uncertain"}:
            raise ValueError(f"invalid speaker result for {video_id}")
        if value.get("content_type") not in {
            "lecture", "qa", "interview", "discussion", "mixed_event", "uncertain"
        }:
            raise ValueError(f"invalid content type for {video_id}")
        duplicate_matches = sorted(
            (
                {
                    "existing_video_id": old_id,
                    "unique_word_jaccard": round(similarity(transcript, old_text), 6),
                }
                for old_id, old_text in existing_texts.items()
            ),
            key=lambda item: item["unique_word_jaccard"],
            reverse=True,
        )[:3]
        value.update(
            {
                "title": video["title"],
                "url": video["url"],
                "channel": video["channel"],
                "duration_seconds": video["duration_seconds"],
                "confidence": confidence,
                "actual_model": actual_model,
                "screened_at": utc_now(),
                "screening_method": "gemini_fresh_asr_samples_v1",
                "source_transcript": str(transcript_path.relative_to(VERSION_DIR)),
                "accepted_for_working_corpus": (
                    value["murad_speaks"] == "yes"
                    and value["content_type"] != "uncertain"
                    and confidence >= 0.8
                    and not bool(value.get("needs_manual_review"))
                ),
                "nearest_existing_transcripts": duplicate_matches,
                "probable_duplicate": bool(
                    duplicate_matches
                    and duplicate_matches[0]["unique_word_jaccard"] >= 0.85
                ),
            }
        )
        results.append(value)

    output_path = OUTPUT_DIR / "verification_results.jsonl"
    output_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in results)
        + "\n",
        encoding="utf-8",
    )
    report = {
        "state": "completed",
        "completed_at": utc_now(),
        "candidates": len(results),
        "accepted": sum(bool(item["accepted_for_working_corpus"]) for item in results),
        "probable_duplicates": sum(bool(item["probable_duplicate"]) for item in results),
        "content_types": {
            content_type: sum(item["content_type"] == content_type for item in results)
            for content_type in sorted({str(item["content_type"]) for item in results})
        },
        "openrouter_usage": summarize_usage(events),
        "duplicate_method": "unique-word Jaccard; >=0.85 is flagged for manual duplicate review",
    }
    write_json(OUTPUT_DIR / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
