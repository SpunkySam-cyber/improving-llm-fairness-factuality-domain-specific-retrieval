"""Validate JSONL records for Q&A, interviews, and discussions."""

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
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "video_id",
    "url",
    "content_type",
    "exchange_index",
    "question",
    "answer",
    "question_start_seconds",
    "question_end_seconds",
    "answer_start_seconds",
    "answer_end_seconds",
    "answer_speaker",
    "speaker_attribution_method",
}
CONTENT_TYPES = {"qa", "interview", "discussion", "mixed_event"}
ATTRIBUTION_METHODS = {
    "assemblyai_diarization_plus_manual_mapping",
    "manual",
    "gemini_caption_context_pending_manual",
    "gemini_fresh_asr_context_pending_manual",
}
REVIEW_STATUSES = {"pending", "verified", "edited", "rejected"}


def validate_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - record.keys())
    if missing:
        errors.append("missing fields: " + ", ".join(missing))
    if record.get("content_type") not in CONTENT_TYPES:
        errors.append("invalid content_type")
    if record.get("answer_speaker") != "Abdal Hakim Murad":
        errors.append("answer_speaker must be Abdal Hakim Murad")
    if record.get("speaker_attribution_method") not in ATTRIBUTION_METHODS:
        errors.append("invalid speaker_attribution_method")
    if record.get("review_status") is not None and record.get("review_status") not in REVIEW_STATUSES:
        errors.append("invalid review_status")
    if not str(record.get("question") or "").strip():
        errors.append("question is empty")
    if not str(record.get("answer") or "").strip():
        errors.append("answer is empty")
    for prefix in ("question", "answer"):
        start = record.get(f"{prefix}_start_seconds")
        end = record.get(f"{prefix}_end_seconds")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            errors.append(f"{prefix} timestamps must be numeric")
        elif start < 0 or end <= start:
            errors.append(f"invalid {prefix} timestamp range")
    if (
        isinstance(record.get("question_end_seconds"), (int, float))
        and isinstance(record.get("answer_start_seconds"), (int, float))
        and record["answer_start_seconds"] < record["question_end_seconds"]
    ):
        errors.append("answer starts before question ends")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    failures = 0
    records = 0
    for line_number, line in enumerate(
        args.path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        records += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"line {line_number}: invalid JSON: {exc}")
            failures += 1
            continue
        errors = validate_record(record)
        if errors:
            print(f"line {line_number}: " + "; ".join(errors))
            failures += 1
    print(f"Validated {records} records; failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
