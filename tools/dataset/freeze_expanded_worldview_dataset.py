"""Freeze the verified expanded worldview corpus without modifying prior freezes."""

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
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VERSION_DIR = Path(__file__).resolve().parents[2] / "transcription_pipeline_versions" / "v5_batch_ingestion_pipeline"  # data/output dir (shared with V5 batch core)
CORPUS = VERSION_DIR / "worldview_corpus"
BASE = CORPUS / "working_v2" / "corpus_manifest.jsonl"
CAPTION_VERIFICATION = (
    CORPUS / "content_verification" / "final_verification_results.jsonl"
)
FRESH_MANIFEST = CORPUS / "fresh_asr_candidates" / "manifest.json"
FRESH_VERIFICATION = (
    CORPUS
    / "fresh_asr_candidates"
    / "content_verification"
    / "verification_results.jsonl"
)
DATABASE = CORPUS / "worldview_discovery.sqlite"
OUTPUT = CORPUS / "frozen_v3"
DUPLICATE_ID = "k5RsJh9hGJ8"
DUPLICATE_OF = "rTJ9U013RZU"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_lines(records: list[dict[str, Any]]) -> str:
    payload = ("\n".join(canonical(item) for item in records) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    base = load_jsonl(BASE)
    caption_results = load_jsonl(CAPTION_VERIFICATION)
    fresh_results = load_jsonl(FRESH_VERIFICATION)
    fresh_by_id = {
        str(item["video_id"]): item
        for item in json.loads(FRESH_MANIFEST.read_text(encoding="utf-8"))["videos"]
    }
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    try:
        ids = [str(item["video_id"]) for item in caption_results]
        metadata = {
            row["video_id"]: dict(row)
            for row in connection.execute(
                "SELECT video_id, canonical_url, title, channel, channel_id, "
                "duration_seconds, upload_date, caption_kind, caption_language, "
                "caption_word_count FROM videos WHERE video_id IN ({})".format(
                    ",".join("?" for _ in ids)
                ),
                ids,
            )
        }
    finally:
        connection.close()

    additions: list[dict[str, Any]] = []
    for result in caption_results:
        if not result["accepted_for_working_corpus"]:
            continue
        video_id = str(result["video_id"])
        item = metadata[video_id]
        content_type = str(result["content_type"])
        additions.append(
            {
                "video_id": video_id,
                "url": item["canonical_url"],
                "title": item["title"],
                "channel": item["channel"],
                "channel_id": item["channel_id"],
                "duration_seconds": item["duration_seconds"],
                "upload_date": item["upload_date"],
                "source_names": ["expanded worldview discovery"],
                "content_type": content_type,
                "content_reason": "verified:gemini_caption_context_v1",
                "caption_status": "available",
                "caption_kind": item["caption_kind"],
                "caption_language": item["caption_language"],
                "caption_word_count": item["caption_word_count"],
                "caption_path": (
                    f"worldview_corpus/candidate_captions/{video_id}/youtube_transcript.txt"
                ),
                "output_format": (
                    "transcript_text" if content_type == "lecture" else "dialogue_jsonl"
                ),
                "requires_speaker_attribution": content_type != "lecture",
                "verification_confidence": result["confidence"],
            }
        )

    for result in fresh_results:
        video_id = str(result["video_id"])
        if (
            not result["accepted_for_working_corpus"]
            or result["probable_duplicate"]
            or video_id == DUPLICATE_ID
        ):
            continue
        source = fresh_by_id[video_id]
        content_type = str(result["content_type"])
        additions.append(
            {
                "video_id": video_id,
                "url": source["url"],
                "title": source["title"],
                "channel": source["channel"],
                "channel_id": source["channel_id"],
                "duration_seconds": source["duration_seconds"],
                "upload_date": source.get("upload_date"),
                "source_names": ["expanded worldview discovery"],
                "content_type": content_type,
                "content_reason": "verified:gemini_fresh_asr_samples_v1",
                "caption_status": "fresh_asr_completed",
                "caption_kind": None,
                "caption_language": "en",
                "caption_word_count": None,
                "caption_path": result["source_transcript"],
                "output_format": (
                    "transcript_text" if content_type == "lecture" else "dialogue_jsonl"
                ),
                "requires_speaker_attribution": content_type != "lecture",
                "verification_confidence": result["confidence"],
                "raw_transcript_path": (
                    str(result["source_transcript"]).replace(
                        "two_call_corrected_transcript.txt", "assemblyai_transcript.txt"
                    )
                ),
            }
        )

    all_records = base + additions
    ids = [str(item["video_id"]) for item in all_records]
    if len(ids) != len(set(ids)):
        repeats = sorted(key for key, count in Counter(ids).items() if count > 1)
        raise RuntimeError(f"duplicate video IDs in freeze: {repeats}")
    all_records.sort(key=lambda item: str(item["video_id"]))
    additions.sort(key=lambda item: str(item["video_id"]))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "corpus_manifest.jsonl").write_text(
        "\n".join(canonical(item) for item in all_records) + "\n", encoding="utf-8"
    )
    conversation = [
        item for item in all_records if item["content_type"] != "lecture"
    ]
    (OUTPUT / "conversation_queue.jsonl").write_text(
        "\n".join(canonical(item) for item in conversation) + "\n", encoding="utf-8"
    )
    (OUTPUT / "new_additions.jsonl").write_text(
        "\n".join(canonical(item) for item in additions) + "\n", encoding="utf-8"
    )
    duplicate_audit = {
        "excluded_video_id": DUPLICATE_ID,
        "retained_video_id": DUPLICATE_OF,
        "decision": "probable_duplicate_excluded",
        "sequence_agreement": 0.9564992900477605,
        "five_word_shingle_jaccard": 0.7342117219445706,
        "new_transcript_coverage": 0.8548003173763554,
        "old_transcript_coverage": 0.8388268881391123,
        "decision_method": "fresh ASR vs existing caption manual transcript comparison",
    }
    (OUTPUT / "duplicate_adjudications.jsonl").write_text(
        canonical(duplicate_audit) + "\n", encoding="utf-8"
    )
    hours = round(sum(float(item["duration_seconds"]) for item in all_records) / 3600, 3)
    by_type = Counter(str(item["content_type"]) for item in all_records)
    report = {
        "state": "frozen",
        "frozen_at": utc_now(),
        "method_version": "worldview-corpus-freeze-v3",
        "base_records": len(base),
        "captioned_additions": sum(item["caption_status"] == "available" for item in additions),
        "fresh_asr_additions": sum(item["caption_status"] == "fresh_asr_completed" for item in additions),
        "total_additions": len(additions),
        "records": len(all_records),
        "hours": hours,
        "by_content_type": dict(sorted(by_type.items())),
        "conversation_records": len(conversation),
        "unresolved_access_errors_excluded": ["_x0mleDj7eE", "8QQKDFgYCWc", "TjQxx2DdPkI"],
        "duplicate_uploads_excluded": [DUPLICATE_ID],
        "manifest_sha256": sha256_lines(all_records),
        "immutability_note": "working_v2 and the earlier 572-lecture freeze were not modified",
    }
    (OUTPUT / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
