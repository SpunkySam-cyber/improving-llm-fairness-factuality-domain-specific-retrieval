"""Find duplicate lectures using saved caption content, without API calls."""

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
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from build_manifest import DEFAULT_DB_PATH, ManifestError, connect_manifest


VERSION_DIR = Path(__file__).resolve().parents[2] / "transcription_pipeline_versions" / "v5_batch_ingestion_pipeline"  # data/output dir (shared with V5 batch core)
DEFAULT_REPORT_PATH = VERSION_DIR / "duplicate_report.json"
# Five-word-shingle overlap of 0.80 means most of the transcript is shared.
# Records are marked, never deleted, so this decision remains reversible.
AUTO_DUPLICATE_THRESHOLD = 0.80
REVIEW_THRESHOLD = 0.75
SHINGLE_SIZE = 5


@dataclass(frozen=True)
class CaptionRecord:
    video_id: str
    title: str
    duration_seconds: int | None
    caption_kind: str | None
    caption_path: Path
    caption_word_count: int
    source_rank: int
    tokens: tuple[str, ...]
    fingerprint: str
    shingles: frozenset[int]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_tokens(text: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return tuple(re.findall(r"[^\W_]+(?:['’][^\W_]+)?", normalized))


def transcript_fingerprint(tokens: Sequence[str]) -> str:
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()


def transcript_shingles(
    tokens: Sequence[str], size: int = SHINGLE_SIZE
) -> frozenset[int]:
    if not tokens:
        return frozenset()
    if len(tokens) < size:
        windows: Iterable[Sequence[str]] = (tokens,)
    else:
        windows = (tokens[index : index + size] for index in range(len(tokens) - size + 1))
    return frozenset(
        int.from_bytes(
            hashlib.blake2b(" ".join(window).encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        for window in windows
    )


def jaccard_similarity(left: frozenset[int], right: frozenset[int]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def plausible_pair(left: CaptionRecord, right: CaptionRecord) -> bool:
    if not left.tokens or not right.tokens:
        return False
    word_ratio = min(len(left.tokens), len(right.tokens)) / max(
        len(left.tokens), len(right.tokens)
    )
    if word_ratio < 0.75:
        return False
    if left.duration_seconds and right.duration_seconds:
        duration_ratio = min(left.duration_seconds, right.duration_seconds) / max(
            left.duration_seconds, right.duration_seconds
        )
        if duration_ratio < 0.80:
            return False
    return True


def ensure_duplicate_columns(connection: sqlite3.Connection) -> None:
    existing = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(videos)").fetchall()
    }
    additions = {
        "transcript_fingerprint": "TEXT",
        "duplicate_of_video_id": "TEXT",
        "duplicate_similarity": "REAL",
        "duplicate_method": "TEXT",
        "duplicate_checked_at": "TEXT",
    }
    for name, sql_type in additions.items():
        if name not in existing:
            connection.execute(f"ALTER TABLE videos ADD COLUMN {name} {sql_type}")
    connection.commit()


def load_records(connection: sqlite3.Connection) -> list[CaptionRecord]:
    rows = connection.execute(
        """
        SELECT v.video_id, v.title, v.duration_seconds, v.caption_kind,
               v.caption_path, COALESCE(v.caption_word_count, 0) caption_word_count,
               COALESCE(MIN(s.source_id), 999999) source_rank
        FROM videos v
        LEFT JOIN source_videos sv ON sv.video_id = v.video_id
        LEFT JOIN sources s ON s.source_id = sv.source_id
        WHERE v.candidate_status = 'candidate'
          AND v.caption_status = 'available'
          AND v.caption_path IS NOT NULL
        GROUP BY v.video_id
        ORDER BY v.video_id
        """
    ).fetchall()
    records: list[CaptionRecord] = []
    for row in rows:
        path = Path(row["caption_path"])
        if not path.is_file() or path.stat().st_size == 0:
            continue
        tokens = normalize_tokens(path.read_text(encoding="utf-8"))
        if not tokens:
            continue
        records.append(
            CaptionRecord(
                video_id=row["video_id"],
                title=row["title"],
                duration_seconds=row["duration_seconds"],
                caption_kind=row["caption_kind"],
                caption_path=path,
                caption_word_count=row["caption_word_count"],
                source_rank=row["source_rank"],
                tokens=tokens,
                fingerprint=transcript_fingerprint(tokens),
                shingles=transcript_shingles(tokens),
            )
        )
    return records


def canonical_key(record: CaptionRecord) -> tuple[int, int, int, str]:
    return (
        0 if record.caption_kind == "manual" else 1,
        record.source_rank,
        -record.caption_word_count,
        record.video_id,
    )


def find_pairs(records: Sequence[CaptionRecord]) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    for index, left in enumerate(records):
        for right in records[index + 1 :]:
            if left.fingerprint == right.fingerprint:
                similarity = 1.0
                method = "exact_normalized_transcript"
            elif plausible_pair(left, right):
                similarity = jaccard_similarity(left.shingles, right.shingles)
                if similarity < REVIEW_THRESHOLD:
                    continue
                method = "caption_5gram_jaccard"
            else:
                continue
            canonical, duplicate = sorted((left, right), key=canonical_key)
            pairs.append(
                {
                    "canonical_video_id": canonical.video_id,
                    "canonical_title": canonical.title,
                    "duplicate_video_id": duplicate.video_id,
                    "duplicate_title": duplicate.title,
                    "similarity": round(similarity, 6),
                    "method": method,
                    "decision": (
                        "duplicate"
                        if similarity >= AUTO_DUPLICATE_THRESHOLD
                        else "review"
                    ),
                }
            )
    return sorted(
        pairs,
        key=lambda item: (-float(item["similarity"]), str(item["duplicate_video_id"])),
    )


def apply_high_confidence_duplicates(
    connection: sqlite3.Connection,
    records: Sequence[CaptionRecord],
    pairs: Sequence[dict[str, object]],
) -> int:
    now = utc_now()
    connection.execute(
        """
        UPDATE videos
        SET transcript_fingerprint = NULL, duplicate_of_video_id = NULL,
            duplicate_similarity = NULL, duplicate_method = NULL,
            duplicate_checked_at = ?
        WHERE candidate_status = 'candidate'
        """,
        (now,),
    )
    connection.executemany(
        "UPDATE videos SET transcript_fingerprint = ? WHERE video_id = ?",
        ((record.fingerprint, record.video_id) for record in records),
    )
    accepted: dict[str, dict[str, object]] = {}
    for pair in pairs:
        if pair["decision"] != "duplicate":
            continue
        duplicate_id = str(pair["duplicate_video_id"])
        current = accepted.get(duplicate_id)
        if current is None or float(pair["similarity"]) > float(current["similarity"]):
            accepted[duplicate_id] = pair
    connection.executemany(
        """
        UPDATE videos
        SET duplicate_of_video_id = ?, duplicate_similarity = ?,
            duplicate_method = ?, duplicate_checked_at = ?
        WHERE video_id = ?
        """,
        (
            (
                pair["canonical_video_id"],
                pair["similarity"],
                pair["method"],
                now,
                duplicate_id,
            )
            for duplicate_id, pair in accepted.items()
        ),
    )
    connection.commit()
    return len(accepted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        connection = connect_manifest(args.db.resolve())
        try:
            ensure_duplicate_columns(connection)
            records = load_records(connection)
            pairs = find_pairs(records)
            duplicate_count = apply_high_confidence_duplicates(
                connection, records, pairs
            )
        finally:
            connection.close()
        report = {
            "generated_at": utc_now(),
            "caption_records_compared": len(records),
            "auto_duplicate_threshold": AUTO_DUPLICATE_THRESHOLD,
            "review_threshold": REVIEW_THRESHOLD,
            "high_confidence_duplicates": duplicate_count,
            "review_pairs": sum(pair["decision"] == "review" for pair in pairs),
            "pairs": pairs,
        }
        args.report.resolve().write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Caption records compared: {len(records)}")
        print(f"High-confidence duplicates: {duplicate_count}")
        print(f"Pairs requiring review: {report['review_pairs']}")
        print(f"Report: {args.report.resolve()}")
    except (ManifestError, OSError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
