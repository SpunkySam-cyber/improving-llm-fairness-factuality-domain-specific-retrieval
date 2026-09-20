"""Build the broader Abdal Hakim Murad worldview-corpus working manifest.

The existing lecture freeze remains immutable.  This exporter reads the working
SQLite manifest and writes a separate corpus that includes lectures plus
conversational material in which Murad speaks.  Conversation records are queued
for speaker-aware transcription and later question/answer JSONL extraction.
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
import argparse
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


VERSION_DIR = Path(__file__).resolve().parents[2] / "transcription_pipeline_versions" / "v5_batch_ingestion_pipeline"  # data/output dir (shared with V5 batch core)
DEFAULT_DB = VERSION_DIR / "pipeline.sqlite"
DEFAULT_OUTPUT = VERSION_DIR / "worldview_corpus" / "working_v2"
METHOD_VERSION = "worldview-working-manifest-v2"

QA_REASONS = {
    "questions_answers",
    "adjudicated:questions_answers",
    "adjudicated:moderated_multi_speaker_qa",
}
INTERVIEW_REASONS = {
    "interview",
    "adjudicated:interview_dialogue",
}
DISCUSSION_REASONS = {
    "book_review",
    "in_conversation",
    "podcast",
    "adjudicated:author_conversation",
    "adjudicated:featured_guest_dialogue",
    "adjudicated:live_discussion",
}
MIXED_EVENT_REASONS = {
    "adjudicated:composite_lecture_performance",
    "adjudicated:devotional_recitation_performance",
    "adjudicated:multiple_alumni_speakers",
}

# This record discusses Murad's writing but, according to the completed manual
# adjudication, does not contain his speech.  It remains in the audit database
# but is not training/research text attributable to him.
NO_MURAD_SPEECH_REASONS = {"adjudicated:derivative_essay_discussion"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    lines = [canonical_line(record) for record in records]
    path.write_bytes((("\n".join(lines) + "\n") if lines else "").encode("utf-8"))


def classify(content_status: str, reason: str | None) -> tuple[str, bool]:
    if content_status == "lecture_candidate":
        return "lecture", True
    if reason in QA_REASONS:
        return "qa", True
    if reason in INTERVIEW_REASONS:
        return "interview", True
    if reason in DISCUSSION_REASONS:
        return "discussion", True
    if reason in MIXED_EVENT_REASONS:
        return "mixed_event", True
    if reason in NO_MURAD_SPEECH_REASONS:
        return "not_murad_speech", False
    return "needs_review", False


def relative_caption_path(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value).resolve()
    try:
        return path.relative_to(VERSION_DIR).as_posix()
    except ValueError:
        return str(path)


def load_records(connection: sqlite3.Connection) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT v.video_id, v.title, v.canonical_url url, v.channel, v.channel_id,
               v.duration_seconds, v.upload_date, v.content_status,
               v.content_reason, v.caption_status, v.caption_path,
               v.caption_word_count,
               GROUP_CONCAT(DISTINCT s.name) source_names
        FROM videos v
        LEFT JOIN source_videos sv ON sv.video_id = v.video_id
        LEFT JOIN sources s ON s.source_id = sv.source_id
        WHERE v.candidate_status = 'candidate'
          AND v.duplicate_of_video_id IS NULL
          AND v.duration_seconds >= 600
        GROUP BY v.video_id
        ORDER BY v.video_id
        """
    ).fetchall()

    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in rows:
        content_type, keep = classify(row["content_status"], row["content_reason"])
        record = {
            "video_id": row["video_id"],
            "title": row["title"],
            "url": row["url"],
            "channel": row["channel"],
            "channel_id": row["channel_id"],
            "duration_seconds": row["duration_seconds"],
            "upload_date": row["upload_date"],
            "content_type": content_type,
            "content_reason": row["content_reason"],
            "output_format": "transcript_text" if content_type == "lecture" else "dialogue_jsonl",
            "requires_speaker_attribution": content_type != "lecture",
            "caption_status": row["caption_status"],
            "caption_path": relative_caption_path(row["caption_path"]),
            "caption_word_count": row["caption_word_count"],
            "source_names": sorted(filter(None, (row["source_names"] or "").split(","))),
        }
        (included if keep else excluded).append(record)
    return included, excluded


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(record["content_type"] for record in records)
    seconds: dict[str, int] = defaultdict(int)
    for record in records:
        seconds[str(record["content_type"])] += int(record["duration_seconds"] or 0)
    return {
        "records": len(records),
        "hours": round(sum(seconds.values()) / 3600, 2),
        "by_content_type": {
            content_type: {
                "records": counts[content_type],
                "hours": round(seconds[content_type] / 3600, 2),
            }
            for content_type in sorted(counts)
        },
        "at_least_20_minutes": sum(
            int(record["duration_seconds"] or 0) >= 1200 for record in records
        ),
        "at_least_60_minutes": sum(
            int(record["duration_seconds"] or 0) >= 3600 for record in records
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(args.db.resolve())
    try:
        included, excluded = load_records(connection)
    finally:
        connection.close()

    conversation_queue = [
        record for record in included if record["content_type"] != "lecture"
    ]
    write_jsonl(output / "corpus_manifest.jsonl", included)
    write_jsonl(output / "conversation_queue.jsonl", conversation_queue)
    write_jsonl(output / "excluded_audit.jsonl", excluded)
    manifest_bytes = (output / "corpus_manifest.jsonl").read_bytes()
    report = {
        "state": "working_manifest",
        "created_at": utc_now(),
        "method_version": METHOD_VERSION,
        "preserves_existing_lecture_freeze": True,
        "minimum_existing_record_duration_seconds": 600,
        "minimum_new_discovery_duration_seconds": 1200,
        "summary": summarize(included),
        "conversation_queue": summarize(conversation_queue),
        "excluded_after_reclassification": summarize(excluded),
        "corpus_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "next_stage": (
            "Run expanded 20-minute discovery searches, deduplicate against this "
            "manifest, then transcribe conversation records with speaker attribution."
        ),
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"Conversation queue: {len(conversation_queue)}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
