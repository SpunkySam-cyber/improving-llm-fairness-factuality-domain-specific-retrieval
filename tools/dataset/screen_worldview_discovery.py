"""Compare expanded YouTube discovery results with the existing corpus.

This is a metadata-only screen. It never declares a semantic duplicate from a
weak title match: strong matches are labelled ``probable_duplicate`` and all
other unseen IDs remain in the review queue until captions/audio are checked.
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
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


VERSION_DIR = Path(__file__).resolve().parents[2] / "transcription_pipeline_versions" / "v5_batch_ingestion_pipeline"  # data/output dir (shared with V5 batch core)
DEFAULT_EXISTING_DB = VERSION_DIR / "pipeline.sqlite"
DEFAULT_DISCOVERY_DB = VERSION_DIR / "worldview_corpus" / "worldview_discovery.sqlite"
DEFAULT_CORPUS = VERSION_DIR / "worldview_corpus" / "working_v2" / "corpus_manifest.jsonl"
DEFAULT_OUTPUT = VERSION_DIR / "worldview_corpus" / "discovery_review"


def normalize_title(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).casefold()
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(
        r"\b(?:shaykh|sheikh|shaikh|dr|professor|abdul|abdal|hakeem|hakim|murad|timothy|winter)\b",
        " ",
        value,
    )
    value = re.sub(r"\b(?:hd|uncut|remastered|youtube)\b", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def infer_content_type(title: str) -> str:
    folded = title.casefold()
    if re.search(r"\bq\s*(?:&|and)?\s*a\b|questions?\s*(?:&|and)\s*answers?", folded):
        return "qa"
    if re.search(r"\binterview(?:s|ed)?\b", folded):
        return "interview"
    if re.search(r"\bconversation\b|\bin conversation\b|\bdialogue\b|\bpanel\b", folded):
        return "discussion"
    if re.search(r"\bpodcast\b", folded):
        return "discussion"
    if re.search(r"\bwith\b|\bdiscussion\b|\bshow\b", folded):
        return "needs_format_review"
    return "lecture_candidate"


def load_corpus(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def best_match(
    item: dict[str, Any], existing: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, float, float]:
    normalized = normalize_title(str(item["title"]))
    best: dict[str, Any] | None = None
    best_score = 0.0
    best_duration_ratio = 0.0
    for candidate in existing:
        candidate_title = normalize_title(str(candidate["title"]))
        if not normalized or not candidate_title:
            continue
        score = SequenceMatcher(None, normalized, candidate_title, autojunk=False).ratio()
        left = int(item.get("duration_seconds") or 0)
        right = int(candidate.get("duration_seconds") or 0)
        duration_ratio = min(left, right) / max(left, right) if left and right else 0.0
        ranking = score * 0.75 + duration_ratio * 0.25
        current = best_score * 0.75 + best_duration_ratio * 0.25
        if ranking > current:
            best = candidate
            best_score = score
            best_duration_ratio = duration_ratio
    return best, best_score, best_duration_ratio


def canonical_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_bytes(
        (("\n".join(canonical_line(record) for record in records) + "\n") if records else "").encode("utf-8")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-db", type=Path, default=DEFAULT_EXISTING_DB)
    parser.add_argument("--discovery-db", type=Path, default=DEFAULT_DISCOVERY_DB)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    old = sqlite3.connect(args.existing_db.resolve())
    discovery = sqlite3.connect(args.discovery_db.resolve())
    discovery.row_factory = sqlite3.Row
    try:
        all_old_ids = {row[0] for row in old.execute("SELECT video_id FROM videos")}
        rows = discovery.execute(
            """
            SELECT video_id, title, canonical_url url, channel, channel_id,
                   duration_seconds, upload_date
            FROM videos
            WHERE candidate_status = 'candidate'
            ORDER BY video_id
            """
        ).fetchall()
    finally:
        old.close()
        discovery.close()

    existing = load_corpus(args.corpus.resolve())
    unseen = [dict(row) for row in rows if row["video_id"] not in all_old_ids]
    review: list[dict[str, Any]] = []
    probable_duplicates: list[dict[str, Any]] = []
    for item in unseen:
        match, title_similarity, duration_ratio = best_match(item, existing)
        exact_normalized_title = bool(
            match
            and normalize_title(str(item["title"]))
            == normalize_title(str(match["title"]))
        )
        probable = bool(
            match
            and (
                (exact_normalized_title and duration_ratio >= 0.80)
                or (title_similarity >= 0.92 and duration_ratio >= 0.90)
            )
        )
        record = {
            **item,
            "provisional_content_type": infer_content_type(str(item["title"])),
            "review_status": "probable_duplicate" if probable else "needs_content_review",
            "best_existing_match": (
                {
                    "video_id": match["video_id"],
                    "title": match["title"],
                    "duration_seconds": match["duration_seconds"],
                    "title_similarity": round(title_similarity, 6),
                    "duration_ratio": round(duration_ratio, 6),
                }
                if match
                else None
            ),
            "duplicate_decision_requires_caption_or_audio_check": True,
        }
        (probable_duplicates if probable else review).append(record)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "new_candidate_review.jsonl", review)
    write_jsonl(output / "probable_duplicates.jsonl", probable_duplicates)
    counts = Counter(record["provisional_content_type"] for record in review)
    report = {
        "discovery_candidates": len(rows),
        "already_known_video_ids": len(rows) - len(unseen),
        "unseen_video_ids": len(unseen),
        "probable_metadata_duplicates": len(probable_duplicates),
        "new_candidates_requiring_content_review": len(review),
        "review_queue_by_provisional_type": dict(sorted(counts.items())),
        "paid_api_calls_made": 0,
        "next_step": (
            "Fetch detailed metadata/captions for the review queue, verify that Murad "
            "speaks, and use transcript similarity to resolve probable duplicates."
        ),
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
