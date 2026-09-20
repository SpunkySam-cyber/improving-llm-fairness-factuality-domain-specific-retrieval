"""Generate auditable, pending-review dialogue JSONL from timestamped sources."""

from __future__ import annotations

# --- repo layout shim: find shared modules (V5 batch core, dialogue/dataset tools, project_env) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
for _d in (_ROOT, _ROOT / "transcription_pipeline_versions" / "v5_batch_ingestion_pipeline", _ROOT / "tools" / "dataset", _ROOT / "tools" / "dialogue"):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
# --- end shim ---
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


VERSION_DIR = Path(__file__).resolve().parents[2] / "transcription_pipeline_versions" / "v5_batch_ingestion_pipeline"  # data/output dir (shared with V5 batch core)
PROJECT_ROOT = VERSION_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dialogue_jsonl import validate_record
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


CORPUS = VERSION_DIR / "worldview_corpus"
QUEUE = CORPUS / "frozen_v3" / "conversation_queue.jsonl"
OUTPUT = CORPUS / "frozen_v3" / "dialogue_jsonl"
WINDOW_SEGMENTS = 360
OVERLAP_SEGMENTS = 60

SYSTEM_PROMPT = """You identify question/prompt and answer boundaries in timestamped transcripts for a research corpus about Shaykh Abdal Hakim Murad (Timothy Winter).

Return exactly one JSON object: {"exchanges": [...]}. Each exchange must contain:
- question_start_index, question_end_index
- answer_start_index, answer_end_index
- questioner: string or null
- speaker_evidence: short explanation based only on this chunk
- confidence: number 0 to 1

Indices must refer to the supplied segment indices. Extract only a complete question or conversational prompt spoken by someone other than Murad, followed by a complete response attributable to Murad. In a discussion, a peer statement counts only when it clearly invites Murad's response. Do not label Murad's own rhetorical questions as interviewer questions. Omit introductions, solo lecture passages, incomplete boundary-crossing exchanges, and uncertain speaker assignments. Do not rewrite or quote the text; return indices only. Use an empty exchanges array when none qualify. No markdown."""


def canonical(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_json(content: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I)
    value = json.loads(cleaned)
    if not isinstance(value, dict) or not isinstance(value.get("exchanges"), list):
        raise ValueError("response is not an object with an exchanges array")
    return value


def load_caption_segments(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        {
            "start_seconds": float(item["start_seconds"]),
            "end_seconds": float(item["start_seconds"])
            + max(0.001, float(item.get("duration_seconds") or 0)),
            "text": " ".join(str(item.get("text") or "").split()),
        }
        for item in raw
        if str(item.get("text") or "").strip()
    ]


def load_fresh_asr_segments(response_path: Path) -> list[dict[str, Any]]:
    words = json.loads(response_path.read_text(encoding="utf-8"))["words"]
    segments: list[dict[str, Any]] = []
    for start in range(0, len(words), 35):
        group = words[start : start + 35]
        if not group:
            continue
        segments.append(
            {
                "start_seconds": float(group[0]["start"]) / 1000,
                "end_seconds": max(
                    float(group[-1]["end"]) / 1000,
                    float(group[0]["start"]) / 1000 + 0.001,
                ),
                "text": " ".join(str(item["text"]) for item in group),
            }
        )
    return segments


def segment_source(record: dict[str, Any]) -> tuple[list[dict[str, Any]], Path, str]:
    caption_path = record.get("caption_path")
    if caption_path:
        path = VERSION_DIR / str(caption_path)
        if path.name == "youtube_transcript.txt":
            path = path.with_name("youtube_segments.json")
        if path.is_file() and path.name == "youtube_segments.json":
            return load_caption_segments(path), path, "youtube_caption_segments"
        if path.is_file() and record.get("caption_status") == "fresh_asr_completed":
            response = path.with_name("assemblyai_response.json")
            if response.is_file():
                return load_fresh_asr_segments(response), response, "fresh_asr_words"
    raise FileNotFoundError(f"no timestamped source for {record['video_id']}")


def chunk_starts(length: int) -> list[int]:
    if length <= WINDOW_SEGMENTS:
        return [0]
    step = WINDOW_SEGMENTS - OVERLAP_SEGMENTS
    starts = list(range(0, max(1, length - OVERLAP_SEGMENTS), step))
    final = max(0, length - WINDOW_SEGMENTS)
    if starts[-1] != final:
        starts.append(final)
    return sorted(set(starts))


def render_chunk(segments: list[dict[str, Any]], start: int, end: int) -> str:
    return "\n".join(
        f"{index}|{item['start_seconds']:.3f}-{item['end_seconds']:.3f}|{item['text']}"
        for index, item in enumerate(segments[start:end], start=start)
    )


def joined_text(segments: list[dict[str, Any]], start: int, end: int) -> str:
    return " ".join(item["text"] for item in segments[start : end + 1]).strip()


def overlap_ratio(left: dict[str, Any], right: dict[str, Any]) -> float:
    intersection = max(
        0.0,
        min(left["answer_end_seconds"], right["answer_end_seconds"])
        - max(left["answer_start_seconds"], right["answer_start_seconds"]),
    )
    shorter = min(
        left["answer_end_seconds"] - left["answer_start_seconds"],
        right["answer_end_seconds"] - right["answer_start_seconds"],
    )
    return intersection / max(0.001, shorter)


def deduplicate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in sorted(
        records,
        key=lambda item: (
            item["answer_start_seconds"],
            -float(item["speaker_attribution_confidence"]),
        ),
    ):
        match = next((item for item in selected if overlap_ratio(item, record) >= 0.7), None)
        if match is None:
            selected.append(record)
        elif float(record["speaker_attribution_confidence"]) > float(
            match["speaker_attribution_confidence"]
        ):
            selected[selected.index(match)] = record
    return sorted(selected, key=lambda item: item["question_start_seconds"])


def candidate_record(
    video: dict[str, Any],
    segments: list[dict[str, Any]],
    item: dict[str, Any],
    source_path: Path,
    source_kind: str,
) -> dict[str, Any] | None:
    try:
        qs, qe, ans, ane = (
            int(item[name])
            for name in (
                "question_start_index",
                "question_end_index",
                "answer_start_index",
                "answer_end_index",
            )
        )
        confidence = float(item["confidence"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 <= qs <= qe < ans <= ane < len(segments)) or confidence < 0.8:
        return None
    question = joined_text(segments, qs, qe)
    answer = joined_text(segments, ans, ane)
    if len(question.split()) < 3 or len(answer.split()) < 8:
        return None
    answer_start = segments[ans]["start_seconds"]
    source_question_end = segments[qe]["end_seconds"]
    question_end = min(source_question_end, answer_start)
    if question_end <= segments[qs]["start_seconds"]:
        return None
    method = (
        "gemini_caption_context_pending_manual"
        if source_kind == "youtube_caption_segments"
        else "gemini_fresh_asr_context_pending_manual"
    )
    record = {
        "video_id": video["video_id"],
        "url": video["url"],
        "title": video["title"],
        "content_type": video["content_type"],
        "exchange_index": 0,
        "questioner": item.get("questioner"),
        "question": question,
        "answer": answer,
        "question_start_seconds": segments[qs]["start_seconds"],
        "question_end_seconds": question_end,
        "answer_start_seconds": answer_start,
        "answer_end_seconds": segments[ane]["end_seconds"],
        "answer_speaker": "Abdal Hakim Murad",
        "speaker_attribution_method": method,
        "speaker_attribution_evidence": str(item.get("speaker_evidence") or ""),
        "speaker_attribution_confidence": confidence,
        "source_transcript": (
            "fresh_asr_raw" if source_kind == "fresh_asr_words" else "youtube_caption"
        ),
        "source_timestamp_file": str(source_path.relative_to(VERSION_DIR)),
        "source_segment_indices": {
            "question": [qs, qe],
            "answer": [ans, ane],
        },
        "caption_timestamp_overlap_adjusted": question_end < source_question_end,
        "question_sha256": sha256_text(question),
        "answer_sha256": sha256_text(answer),
        "review_status": "pending",
        "generated_at": utc_now(),
    }
    return record


def main() -> int:
    videos = [
        json.loads(line)
        for line in QUEUE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pipeline = load_v4_pipeline()
    load_env_file(ENV_PATH)
    api_key = require_env("OPENROUTER_API_KEY", env_path=ENV_PATH)
    events = install_openrouter_cache(
        pipeline, OUTPUT / "openrouter_cache", OUTPUT / "openrouter_requests.jsonl"
    )
    combined: list[dict[str, Any]] = []
    source_errors: list[dict[str, str]] = []
    chunk_errors: list[dict[str, Any]] = []
    per_video: list[dict[str, Any]] = []
    for video_index, video in enumerate(videos, start=1):
        video_id = str(video["video_id"])
        try:
            segments, source_path, source_kind = segment_source(video)
        except (OSError, KeyError, ValueError) as exc:
            source_errors.append({"video_id": video_id, "error": str(exc)})
            print(f"[{video_index}/{len(videos)}] SKIP {video_id}: {exc}", flush=True)
            continue
        candidates: list[dict[str, Any]] = []
        starts = chunk_starts(len(segments))
        print(
            f"[{video_index}/{len(videos)}] {video_id}: "
            f"{len(segments)} segments, {len(starts)} chunks",
            flush=True,
        )
        for chunk_number, start in enumerate(starts, start=1):
            end = min(len(segments), start + WINDOW_SEGMENTS)
            user_message = f"""VIDEO
video_id: {video_id}
title: {video['title']}
content_type: {video['content_type']}

CHUNK {chunk_number}/{len(starts)}
Only indices {start} through {end - 1} are present. Omit exchanges cut by a chunk boundary.

TIMESTAMPED SEGMENTS
{render_chunk(segments, start, end)}"""
            try:
                content, _ = pipeline.call_openrouter(
                    api_key=api_key,
                    model=MODEL,
                    system_prompt=SYSTEM_PROMPT,
                    user_message=user_message,
                    max_tokens=1600,
                    stage="Dialogue boundary extraction",
                    usage_sink=[],
                )
                response = clean_json(content)
                for item in response["exchanges"]:
                    if not isinstance(item, dict):
                        continue
                    record = candidate_record(
                        video, segments, item, source_path, source_kind
                    )
                    if record is not None:
                        candidates.append(record)
            except Exception as exc:
                chunk_errors.append(
                    {
                        "video_id": video_id,
                        "chunk_number": chunk_number,
                        "start_index": start,
                        "end_index": end - 1,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        records = deduplicate(candidates)
        for exchange_index, record in enumerate(records, start=1):
            record["exchange_index"] = exchange_index
            errors = validate_record(record)
            if errors:
                raise RuntimeError(
                    f"generated invalid record for {video_id}: {'; '.join(errors)}"
                )
        video_dir = OUTPUT / "videos" / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        (video_dir / "exchanges.jsonl").write_text(
            ("\n".join(canonical(item) for item in records) + ("\n" if records else "")),
            encoding="utf-8",
        )
        combined.extend(records)
        per_video.append(
            {
                "video_id": video_id,
                "content_type": video["content_type"],
                "source_kind": source_kind,
                "segments": len(segments),
                "chunks": len(starts),
                "exchanges": len(records),
            }
        )

    combined.sort(key=lambda item: (item["video_id"], item["exchange_index"]))
    (OUTPUT / "all_exchanges.jsonl").write_text(
        "\n".join(canonical(item) for item in combined) + ("\n" if combined else ""),
        encoding="utf-8",
    )
    report = {
        "state": "completed_with_source_gaps" if source_errors else "completed",
        "completed_at": utc_now(),
        "queued_videos": len(videos),
        "processed_videos": len(per_video),
        "source_gaps": source_errors,
        "chunk_errors": chunk_errors,
        "videos_with_exchanges": sum(item["exchanges"] > 0 for item in per_video),
        "total_exchanges": len(combined),
        "review_status": "pending_manual_speaker_and_boundary_review",
        "quality_note": (
            "Texts and timestamps are reconstructed from saved source segments; Gemini "
            "selects boundaries and speaker context but does not rewrite the text."
        ),
        "per_video": per_video,
        "openrouter_usage": summarize_usage(events),
    }
    write_json(OUTPUT / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not chunk_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
