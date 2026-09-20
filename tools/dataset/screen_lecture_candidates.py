"""Conservatively classify lectures, obvious non-lectures, and review cases."""

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
import json
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from build_manifest import DEFAULT_DB_PATH, connect_manifest


VERSION_DIR = Path(__file__).resolve().parents[2] / "transcription_pipeline_versions" / "v5_batch_ingestion_pipeline"  # data/output dir (shared with V5 batch core)
DEFAULT_REPORT = VERSION_DIR / "final_dataset_report.json"

NONLECTURE_PATTERNS = {
    "interview": r"\binterview(?:s|ed|ing)?\b",
    "in_conversation": r"\bin\s+conversation\b",
    "podcast": r"\bpodcast\b",
    "panel": r"\bpanel(?:ist|ists)?\b",
    "book_review": r"\bbook\s+review\b",
    "questions_answers": r"\bq\s*(?:&|and)\s*a\b|\bquestions?\s+and\s+answers?\b",
}

REVIEW_PATTERNS = {
    "named_co_speaker_after_murad": (
        r"\b(?:abdal|abdul)\s+hak(?:i|ee)m\s+murad\s*(?:&|and|with)\s+[a-z]"
    ),
    "named_co_speaker_before_murad": (
        r"[a-z]\s*(?:&|and)\s+(?:shaykh\s+|sheikh\s+|dr\s+|professor\s+)?"
        r"(?:abdal|abdul)\s+hak(?:i|ee)m\s+murad\b"
    ),
    "roundtable": r"\broundtable\b|\bround\s+table\b",
    "discussion": r"\bdiscussion\b",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalized_title(title: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", title).casefold().split())


def classify_title(title: str) -> tuple[str, str | None]:
    normalized = normalized_title(title)
    for reason, pattern in NONLECTURE_PATTERNS.items():
        if re.search(pattern, normalized, re.IGNORECASE):
            return "excluded_nonlecture", reason
    for reason, pattern in REVIEW_PATTERNS.items():
        if re.search(pattern, normalized, re.IGNORECASE):
            return "needs_review", reason
    return "lecture_candidate", None


def ensure_columns(connection: sqlite3.Connection) -> None:
    existing = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(videos)").fetchall()
    }
    for name, sql_type in {
        "content_status": "TEXT",
        "content_reason": "TEXT",
        "content_checked_at": "TEXT",
    }.items():
        if name not in existing:
            connection.execute(f"ALTER TABLE videos ADD COLUMN {name} {sql_type}")
    connection.commit()


def classify_manifest(connection: sqlite3.Connection) -> dict[str, object]:
    rows = connection.execute(
        """
        SELECT video_id, title, canonical_url, duration_seconds, caption_status,
               duplicate_of_video_id
        FROM videos
        WHERE candidate_status = 'candidate'
        ORDER BY video_id
        """
    ).fetchall()
    now = utc_now()
    decisions: list[dict[str, object]] = []
    for row in rows:
        status, reason = classify_title(row["title"])
        connection.execute(
            """
            UPDATE videos
            SET content_status = ?, content_reason = ?, content_checked_at = ?
            WHERE video_id = ?
            """,
            (status, reason, now, row["video_id"]),
        )
        decisions.append(
            {
                "video_id": row["video_id"],
                "title": row["title"],
                "url": row["canonical_url"],
                "duration_seconds": row["duration_seconds"],
                "caption_status": row["caption_status"],
                "duplicate_of_video_id": row["duplicate_of_video_id"],
                "content_status": status,
                "content_reason": reason,
            }
        )
    connection.commit()

    counts = {
        "all_candidates": len(decisions),
        "lecture_candidates": sum(
            item["content_status"] == "lecture_candidate" for item in decisions
        ),
        "excluded_nonlectures": sum(
            item["content_status"] == "excluded_nonlecture" for item in decisions
        ),
        "needs_content_review": sum(
            item["content_status"] == "needs_review" for item in decisions
        ),
        "marked_duplicates": sum(
            item["duplicate_of_video_id"] is not None for item in decisions
        ),
        "usable_unique_with_captions": sum(
            item["content_status"] == "lecture_candidate"
            and item["duplicate_of_video_id"] is None
            and item["caption_status"] == "available"
            for item in decisions
        ),
    }
    return {
        "generated_at": now,
        "classification_method": "conservative title rules; no records deleted",
        "counts": counts,
        "excluded": [
            item for item in decisions if item["content_status"] == "excluded_nonlecture"
        ],
        "needs_review": [
            item for item in decisions if item["content_status"] == "needs_review"
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        connection = connect_manifest(args.db.resolve())
        try:
            ensure_columns(connection)
            report = classify_manifest(connection)
        finally:
            connection.close()
        args.report.resolve().write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for key, value in report["counts"].items():
            print(f"{key}: {value}")
        print(f"Report: {args.report.resolve()}")
        return 0
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
