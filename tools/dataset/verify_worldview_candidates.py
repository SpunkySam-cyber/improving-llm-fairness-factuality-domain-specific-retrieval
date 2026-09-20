"""Verify speaker evidence and content type for captioned worldview candidates.

The verifier uses timestamped samples from the beginning, middle, and end of
each saved YouTube caption. Results are provisional research screening, not
voice-biometric proof. Every OpenRouter response is cached by request hash.
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
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any


VERSION_DIR = Path(__file__).resolve().parents[2] / "transcription_pipeline_versions" / "v5_batch_ingestion_pipeline"  # data/output dir (shared with V5 batch core)
PROJECT_ROOT = VERSION_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_env import load_env_file, require_env
from run_fresh_benchmark import (
    MODEL,
    install_openrouter_cache,
    load_v4_pipeline,
    summarize_usage,
    utc_now,
    write_json,
)


ENV_PATH = PROJECT_ROOT / ".env"
CORPUS_DIR = VERSION_DIR / "worldview_corpus"
DISCOVERY_DB = CORPUS_DIR / "worldview_discovery.sqlite"
REVIEW_QUEUE = CORPUS_DIR / "discovery_review" / "new_candidate_review.jsonl"
CROSS_DUPLICATES = (
    CORPUS_DIR / "discovery_review" / "cross_corpus_duplicates.json"
)
CAPTION_DIR = CORPUS_DIR / "candidate_captions"
OUTPUT_DIR = CORPUS_DIR / "content_verification"

SYSTEM_PROMPT = """You are screening a research corpus about Shaykh Abdal Hakim Murad (Timothy Winter). Use only the supplied title, channel, and timestamped caption samples.

Return exactly one JSON object with these keys:
- video_id: string
- murad_speaks: "yes", "no", or "uncertain"
- speaker_evidence: short explanation tied to supplied timestamps
- content_type: "lecture", "qa", "interview", "discussion", "mixed_event", or "uncertain"
- content_evidence: short explanation tied to supplied timestamps
- confidence: number from 0 to 1
- needs_manual_review: boolean

Use "yes" only when the caption context explicitly introduces or addresses Murad and contains speech attributable to him, or when a sustained solo talk is clearly presented as his. A title alone is insufficient. Use "uncertain" when speaker turns cannot be reliably attributed. A lecture is a substantially continuous solo talk; Q&A has explicit questions followed by answers; interview is interviewer-led; discussion has multiple peers; mixed_event combines a talk with performances or other speakers. Do not include markdown or any text outside the JSON object."""


def canonical_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def candidate_ids() -> list[str]:
    queue_ids = {str(item["video_id"]) for item in load_jsonl(REVIEW_QUEUE)}
    duplicate_ids = {
        str(item["new_video_id"])
        for item in json.loads(CROSS_DUPLICATES.read_text(encoding="utf-8"))[
            "pairs"
        ]
        if item["decision"] == "probable_duplicate"
    }
    connection = sqlite3.connect(DISCOVERY_DB)
    try:
        available = {
            row[0]
            for row in connection.execute(
                "SELECT video_id FROM videos WHERE caption_status = 'available'"
            )
        }
    finally:
        connection.close()
    return sorted((queue_ids & available) - duplicate_ids)


def timestamp_label(segment: dict[str, Any]) -> str:
    start = segment.get("start_seconds")
    return "unknown" if start is None else f"{float(start):.1f}s"


def sample_segments(segments: list[dict[str, Any]]) -> str:
    if not segments:
        raise ValueError("caption has no segments")
    window_size = 18
    anchors = [0, len(segments) // 3, (2 * len(segments)) // 3, max(0, len(segments) - window_size)]
    windows: list[str] = []
    seen: set[int] = set()
    for anchor in anchors:
        start = min(max(0, anchor), max(0, len(segments) - window_size))
        if start in seen:
            continue
        seen.add(start)
        selected = segments[start : start + window_size]
        text = " ".join(str(item.get("text") or "").strip() for item in selected).strip()
        windows.append(
            f"[{timestamp_label(selected[0])}–{timestamp_label(selected[-1])}] {text}"
        )
    return "\n".join(windows)


def clean_json_response(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("verification response is not an object")
    return value


def validate_result(value: dict[str, Any], video_id: str) -> dict[str, Any]:
    if value.get("video_id") != video_id:
        raise ValueError("verification response has the wrong video_id")
    if value.get("murad_speaks") not in {"yes", "no", "uncertain"}:
        raise ValueError("invalid murad_speaks")
    if value.get("content_type") not in {
        "lecture",
        "qa",
        "interview",
        "discussion",
        "mixed_event",
        "uncertain",
    }:
        raise ValueError("invalid content_type")
    confidence = float(value.get("confidence"))
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    value["confidence"] = confidence
    value["needs_manual_review"] = bool(value.get("needs_manual_review"))
    value["screening_method"] = "gemini_caption_context_v1"
    value["screened_at"] = utc_now()
    value["accepted_for_working_corpus"] = (
        value["murad_speaks"] == "yes"
        and value["content_type"] != "uncertain"
        and confidence >= 0.80
        and not value["needs_manual_review"]
    )
    return value


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = candidate_ids()
    if len(ids) != 42:
        raise RuntimeError(f"Expected 42 captioned nonduplicate candidates, found {len(ids)}")

    connection = sqlite3.connect(DISCOVERY_DB)
    connection.row_factory = sqlite3.Row
    try:
        metadata = {
            row["video_id"]: dict(row)
            for row in connection.execute(
                """
                SELECT video_id, title, canonical_url url, channel, duration_seconds
                FROM videos WHERE video_id IN ({})
                """.format(",".join("?" for _ in ids)),
                ids,
            )
        }
    finally:
        connection.close()

    pipeline = load_v4_pipeline()
    pipeline.DETECTION_MODEL = MODEL
    load_env_file(ENV_PATH)
    api_key = require_env("OPENROUTER_API_KEY", env_path=ENV_PATH)
    events = install_openrouter_cache(
        pipeline, OUTPUT_DIR / "openrouter_cache", OUTPUT_DIR / "openrouter_requests.jsonl"
    )
    results: list[dict[str, Any]] = []
    for index, video_id in enumerate(ids, start=1):
        item = metadata[video_id]
        segments_path = CAPTION_DIR / video_id / "youtube_segments.json"
        segments = json.loads(segments_path.read_text(encoding="utf-8"))
        user_message = f"""VIDEO METADATA
video_id: {video_id}
title: {item['title']}
channel: {item['channel']}
duration_seconds: {item['duration_seconds']}

TIMESTAMPED CAPTION SAMPLES
{sample_segments(segments)}

Classify speaker evidence and content type. Return exactly the required JSON object."""
        print(f"[{index}/{len(ids)}] Verifying {video_id}: {item['title']}", flush=True)
        content, actual_model = pipeline.call_openrouter(
            api_key=api_key,
            model=MODEL,
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            max_tokens=700,
            stage="Worldview content verification",
            usage_sink=[],
        )
        result = validate_result(clean_json_response(content), video_id)
        result.update(
            {
                "title": item["title"],
                "url": item["url"],
                "channel": item["channel"],
                "duration_seconds": item["duration_seconds"],
                "actual_model": actual_model,
                "caption_samples_only": True,
                "voice_biometric_verification": False,
            }
        )
        results.append(result)

    (OUTPUT_DIR / "verification_results.jsonl").write_bytes(
        ("\n".join(canonical_line(item) for item in results) + "\n").encode("utf-8")
    )
    counts: dict[str, int] = {}
    for item in results:
        key = str(item["content_type"])
        counts[key] = counts.get(key, 0) + 1
    report = {
        "state": "completed",
        "completed_at": utc_now(),
        "candidate_count": len(results),
        "accepted_for_working_corpus": sum(
            bool(item["accepted_for_working_corpus"]) for item in results
        ),
        "manual_review_required": sum(
            bool(item["needs_manual_review"]) for item in results
        ),
        "murad_speaks": {
            state: sum(item["murad_speaks"] == state for item in results)
            for state in ("yes", "no", "uncertain")
        },
        "content_types": dict(sorted(counts.items())),
        "method_note": (
            "Caption-context screening is auditable but is not voice-biometric proof. "
            "Uncertain and low-confidence records require manual audio review."
        ),
        "openrouter_usage": summarize_usage(events),
    }
    write_json(OUTPUT_DIR / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
