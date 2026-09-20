"""Resolve the bounded review queues and freeze an auditable lecture dataset.

This script performs no network or paid API calls. Decisions are reversible in the
working manifest, while every frozen snapshot is stored as an immutable database
record plus portable JSONL/CSV/Markdown artifacts.
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
import csv
import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Sequence

from build_manifest import DEFAULT_DB_PATH, connect_manifest
from deduplicate_captions import (
    CaptionRecord,
    canonical_key,
    jaccard_similarity,
    load_records,
)


VERSION_DIR = Path(__file__).resolve().parents[2] / "transcription_pipeline_versions" / "v5_batch_ingestion_pipeline"  # data/output dir (shared with V5 batch core)
DEFAULT_DUPLICATE_REPORT = VERSION_DIR / "duplicate_report.json"
DEFAULT_OUTPUT_DIR = VERSION_DIR / "frozen_dataset"
METHOD_VERSION = "review-freeze-v1"

# These are the complete 12 records emitted by the conservative title screen on
# 2026-08-11. The rationale is deliberately explicit for paper reproducibility.
CONTENT_ADJUDICATIONS = {
    "3BaN64_IKkQ": (
        "excluded_nonlecture",
        "moderated_multi_speaker_qa",
        "Transcript alternates direct questions and answers with Sejad Mekić.",
    ),
    "5DrqAkcFwl8": (
        "excluded_nonlecture",
        "featured_guest_dialogue",
        "Paul Yunus Pringle is a featured speaker and the transcript is a dialogue.",
    ),
    "65TlTjOnDDg": (
        "excluded_nonlecture",
        "composite_lecture_performance",
        "Full transcript combines Murad material with another named performer and devotional performance.",
    ),
    "FCdazjRlEwU": (
        "excluded_nonlecture",
        "questions_answers",
        "Title says Q/A and transcript is a moderated question-and-answer session.",
    ),
    "HuWM88HbOBY": (
        "excluded_nonlecture",
        "live_discussion",
        "Friday Night Live transcript is a host-led discussion with repeated questions.",
    ),
    "ManKUIl57ic": (
        "excluded_nonlecture",
        "interview_dialogue",
        "Transcript opens with and continues through interviewer questions.",
    ),
    "RszvkBGOWpQ": (
        "excluded_nonlecture",
        "multiple_alumni_speakers",
        "Transcript contains successive alumni accounts rather than one Murad lecture.",
    ),
    "aPTcjQ-lxwE": (
        "lecture_candidate",
        "solo_lecture_hosted_by_organization",
        "Transcript is a continuous solo lecture; HAKIM in the title is the host organization.",
    ),
    "afdBhrdxSbQ": (
        "excluded_nonlecture",
        "derivative_essay_discussion",
        "Transcript discusses Murad's essay in a two-voice format and is not his lecture speech.",
    ),
    "l7jLp7sNAEE": (
        "excluded_nonlecture",
        "live_discussion",
        "Friday Night Live transcript is a host-led discussion with another named speaker.",
    ),
    "tGaAVQl-K9E": (
        "excluded_nonlecture",
        "devotional_recitation_performance",
        "Full transcript is a composite recitation/performance rather than a Murad lecture.",
    ),
    "viY0DplG2BA": (
        "excluded_nonlecture",
        "author_conversation",
        "Tea Over Books transcript is a conversation with author William Barylo.",
    ),
}


@dataclass(frozen=True)
class DuplicateEvidence:
    canonical_video_id: str
    duplicate_video_id: str
    five_gram_jaccard: float
    smaller_transcript_containment: float
    word_count_ratio: float
    duration_ratio: float
    sequence_ratio: float
    decision: str
    rule: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_freeze_id(now: str) -> str:
    return "ahm-lectures-" + now.replace("+00:00", "Z").replace(":", "").replace("-", "")


def secondary_duplicate_evidence(
    canonical: CaptionRecord, duplicate: CaptionRecord
) -> DuplicateEvidence:
    intersection = len(canonical.shingles & duplicate.shingles)
    smaller_count = min(len(canonical.shingles), len(duplicate.shingles))
    containment = intersection / smaller_count if smaller_count else 0.0
    word_ratio = min(len(canonical.tokens), len(duplicate.tokens)) / max(
        len(canonical.tokens), len(duplicate.tokens)
    )
    if canonical.duration_seconds and duplicate.duration_seconds:
        duration_ratio = min(canonical.duration_seconds, duplicate.duration_seconds) / max(
            canonical.duration_seconds, duplicate.duration_seconds
        )
    else:
        duration_ratio = 0.0
    jaccard = jaccard_similarity(canonical.shingles, duplicate.shingles)
    sequence_ratio = SequenceMatcher(
        None, canonical.tokens, duplicate.tokens, autojunk=False
    ).ratio()
    accepted = (
        jaccard >= 0.75
        and containment >= 0.85
        and word_ratio >= 0.85
        and duration_ratio >= 0.80
        and sequence_ratio >= 0.90
    )
    return DuplicateEvidence(
        canonical_video_id=canonical.video_id,
        duplicate_video_id=duplicate.video_id,
        five_gram_jaccard=round(jaccard, 6),
        smaller_transcript_containment=round(containment, 6),
        word_count_ratio=round(word_ratio, 6),
        duration_ratio=round(duration_ratio, 6),
        sequence_ratio=round(sequence_ratio, 6),
        decision="duplicate" if accepted else "unresolved",
        rule=(
            "jaccard>=0.75; containment>=0.85; word_ratio>=0.85; "
            "duration_ratio>=0.80; sequence_ratio>=0.90"
        ),
    )


def ensure_audit_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS content_review_audit (
            review_run_id TEXT NOT NULL,
            video_id TEXT NOT NULL REFERENCES videos(video_id),
            prior_status TEXT,
            decision TEXT NOT NULL,
            reason TEXT NOT NULL,
            evidence TEXT NOT NULL,
            reviewed_at TEXT NOT NULL,
            PRIMARY KEY (review_run_id, video_id)
        );

        CREATE TABLE IF NOT EXISTS duplicate_review_audit (
            review_run_id TEXT NOT NULL,
            canonical_video_id TEXT NOT NULL REFERENCES videos(video_id),
            duplicate_video_id TEXT NOT NULL REFERENCES videos(video_id),
            evidence_json TEXT NOT NULL,
            reviewed_at TEXT NOT NULL,
            PRIMARY KEY (review_run_id, canonical_video_id, duplicate_video_id)
        );

        CREATE TABLE IF NOT EXISTS dataset_freezes (
            freeze_id TEXT PRIMARY KEY,
            frozen_at TEXT NOT NULL,
            method_version TEXT NOT NULL,
            criteria_json TEXT NOT NULL,
            counts_json TEXT NOT NULL,
            dataset_sha256 TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS frozen_dataset_items (
            freeze_id TEXT NOT NULL REFERENCES dataset_freezes(freeze_id),
            video_id TEXT NOT NULL REFERENCES videos(video_id),
            caption_sha256 TEXT NOT NULL,
            transcript_word_count INTEGER NOT NULL,
            manifest_json TEXT NOT NULL,
            PRIMARY KEY (freeze_id, video_id)
        );
        """
    )
    connection.commit()


def apply_content_adjudications(
    connection: sqlite3.Connection, review_run_id: str, reviewed_at: str
) -> list[dict[str, str | None]]:
    rows = connection.execute(
        """
        SELECT video_id, title, content_status
        FROM videos
        WHERE content_status = 'needs_review'
        ORDER BY video_id
        """
    ).fetchall()
    queued_ids = {row["video_id"] for row in rows}
    expected_ids = set(CONTENT_ADJUDICATIONS)
    if queued_ids != expected_ids:
        missing = sorted(queued_ids - expected_ids)
        stale = sorted(expected_ids - queued_ids)
        raise ValueError(
            f"Content review queue changed; unconfigured={missing}, not_queued={stale}"
        )
    decisions: list[dict[str, str | None]] = []
    for row in rows:
        decision, reason, evidence = CONTENT_ADJUDICATIONS[row["video_id"]]
        connection.execute(
            """
            INSERT INTO content_review_audit(
                review_run_id, video_id, prior_status, decision, reason,
                evidence, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                review_run_id,
                row["video_id"],
                row["content_status"],
                decision,
                reason,
                evidence,
                reviewed_at,
            ),
        )
        connection.execute(
            """
            UPDATE videos
            SET content_status = ?, content_reason = ?, content_checked_at = ?
            WHERE video_id = ?
            """,
            (decision, f"adjudicated:{reason}", reviewed_at, row["video_id"]),
        )
        decisions.append(
            {
                "video_id": row["video_id"],
                "title": row["title"],
                "prior_status": row["content_status"],
                "decision": decision,
                "reason": reason,
                "evidence": evidence,
            }
        )
    return decisions


def reviewed_duplicate_pairs(report_path: Path) -> list[dict[str, object]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return [pair for pair in report["pairs"] if pair["decision"] == "review"]


def _connected_components(edges: Sequence[tuple[str, str]]) -> list[set[str]]:
    graph: dict[str, set[str]] = {}
    for left, right in edges:
        graph.setdefault(left, set()).add(right)
        graph.setdefault(right, set()).add(left)
    components: list[set[str]] = []
    visited: set[str] = set()
    for node in sorted(graph):
        if node in visited:
            continue
        stack = [node]
        component: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            stack.extend(graph[current] - visited)
        components.append(component)
    return components


def apply_duplicate_adjudications(
    connection: sqlite3.Connection,
    review_run_id: str,
    reviewed_at: str,
    report_path: Path,
) -> list[DuplicateEvidence]:
    records = load_records(connection)
    by_id = {record.video_id: record for record in records}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    review_pairs = [pair for pair in report["pairs"] if pair["decision"] == "review"]
    if len(review_pairs) != 10:
        raise ValueError(f"Expected 10 duplicate review pairs, found {len(review_pairs)}")

    evidence_rows: list[DuplicateEvidence] = []
    for pair in review_pairs:
        left = by_id[str(pair["canonical_video_id"])]
        right = by_id[str(pair["duplicate_video_id"])]
        evidence = secondary_duplicate_evidence(left, right)
        evidence_rows.append(evidence)
        connection.execute(
            """
            INSERT INTO duplicate_review_audit(
                review_run_id, canonical_video_id, duplicate_video_id,
                evidence_json, reviewed_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                review_run_id,
                evidence.canonical_video_id,
                evidence.duplicate_video_id,
                json.dumps(asdict(evidence), ensure_ascii=False, sort_keys=True),
                reviewed_at,
            ),
        )
    unresolved = [row for row in evidence_rows if row.decision != "duplicate"]
    if unresolved:
        ids = [(row.canonical_video_id, row.duplicate_video_id) for row in unresolved]
        raise ValueError(f"Secondary duplicate checks left unresolved pairs: {ids}")

    accepted_edges = [
        (str(pair["canonical_video_id"]), str(pair["duplicate_video_id"]))
        for pair in report["pairs"]
        if pair["decision"] == "duplicate"
    ] + [
        (row.canonical_video_id, row.duplicate_video_id) for row in evidence_rows
    ]
    accepted_nodes = {node for edge in accepted_edges for node in edge}
    connection.executemany(
        """
        UPDATE videos
        SET duplicate_of_video_id = NULL, duplicate_similarity = NULL,
            duplicate_method = NULL, duplicate_checked_at = ?
        WHERE video_id = ?
        """,
        ((reviewed_at, video_id) for video_id in accepted_nodes),
    )
    edge_similarity = {
        frozenset((str(pair["canonical_video_id"]), str(pair["duplicate_video_id"]))): float(
            pair["similarity"]
        )
        for pair in report["pairs"]
        if pair["decision"] == "duplicate"
    }
    edge_similarity.update(
        {
            frozenset((row.canonical_video_id, row.duplicate_video_id)): row.sequence_ratio
            for row in evidence_rows
        }
    )
    for component in _connected_components(accepted_edges):
        canonical = min((by_id[video_id] for video_id in component), key=canonical_key)
        for video_id in sorted(component - {canonical.video_id}):
            similarities = [
                similarity
                for edge, similarity in edge_similarity.items()
                if video_id in edge and canonical.video_id in edge
            ]
            similarity = max(similarities) if similarities else None
            connection.execute(
                """
                UPDATE videos
                SET duplicate_of_video_id = ?, duplicate_similarity = ?,
                    duplicate_method = ?, duplicate_checked_at = ?
                WHERE video_id = ?
                """,
                (
                    canonical.video_id,
                    similarity,
                    "reviewed_caption_similarity_component",
                    reviewed_at,
                    video_id,
                ),
            )
    return evidence_rows


def relative_caption_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(VERSION_DIR).as_posix()
    except ValueError:
        return str(resolved)


def build_frozen_items(connection: sqlite3.Connection) -> list[dict[str, object]]:
    rows = connection.execute(
        """
        SELECT v.video_id, v.title, v.canonical_url url, v.channel, v.channel_id,
               v.duration_seconds, v.upload_date, v.caption_kind,
               v.caption_language, v.caption_path, v.caption_word_count,
               v.content_status, v.content_reason,
               GROUP_CONCAT(DISTINCT s.name) source_names
        FROM videos v
        LEFT JOIN source_videos sv ON sv.video_id = v.video_id
        LEFT JOIN sources s ON s.source_id = sv.source_id
        WHERE v.candidate_status = 'candidate'
          AND v.content_status = 'lecture_candidate'
          AND v.duplicate_of_video_id IS NULL
          AND v.caption_status = 'available'
          AND v.caption_path IS NOT NULL
        GROUP BY v.video_id
        ORDER BY v.video_id
        """
    ).fetchall()
    items: list[dict[str, object]] = []
    for row in rows:
        caption_path = Path(row["caption_path"])
        if not caption_path.is_file():
            raise ValueError(f"Missing frozen caption: {caption_path}")
        caption_bytes = caption_path.read_bytes()
        if not caption_bytes:
            raise ValueError(f"Empty frozen caption: {caption_path}")
        items.append(
            {
                "video_id": row["video_id"],
                "title": row["title"],
                "url": row["url"],
                "channel": row["channel"],
                "channel_id": row["channel_id"],
                "duration_seconds": row["duration_seconds"],
                "upload_date": row["upload_date"],
                "caption_kind": row["caption_kind"],
                "caption_language": row["caption_language"],
                "caption_path": relative_caption_path(caption_path),
                "caption_word_count": row["caption_word_count"],
                "caption_sha256": hashlib.sha256(caption_bytes).hexdigest(),
                "content_status": row["content_status"],
                "content_reason": row["content_reason"],
                "source_names": sorted((row["source_names"] or "").split(",")),
            }
        )
    return items


def canonical_json_line(item: dict[str, object]) -> str:
    return json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def dataset_digest(items: Sequence[dict[str, object]]) -> str:
    payload = "\n".join(canonical_json_line(item) for item in items) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dataset_counts(connection: sqlite3.Connection, frozen_count: int) -> dict[str, object]:
    scalar = lambda sql: int(connection.execute(sql).fetchone()[0])
    caption_rows = connection.execute(
        """
        SELECT COALESCE(caption_status, 'not_checked') status, COUNT(*) count
        FROM videos WHERE candidate_status = 'candidate'
        GROUP BY COALESCE(caption_status, 'not_checked') ORDER BY status
        """
    ).fetchall()
    return {
        "all_candidates": scalar(
            "SELECT COUNT(*) FROM videos WHERE candidate_status = 'candidate'"
        ),
        "lecture_candidates": scalar(
            "SELECT COUNT(*) FROM videos WHERE candidate_status = 'candidate' AND content_status = 'lecture_candidate'"
        ),
        "excluded_nonlectures": scalar(
            "SELECT COUNT(*) FROM videos WHERE candidate_status = 'candidate' AND content_status = 'excluded_nonlecture'"
        ),
        "unresolved_content_review": scalar(
            "SELECT COUNT(*) FROM videos WHERE candidate_status = 'candidate' AND content_status = 'needs_review'"
        ),
        "marked_duplicates": scalar(
            "SELECT COUNT(*) FROM videos WHERE candidate_status = 'candidate' AND duplicate_of_video_id IS NOT NULL"
        ),
        "caption_status": {row["status"]: int(row["count"]) for row in caption_rows},
        "frozen_usable_unique_with_captions": frozen_count,
    }


def write_freeze_artifacts(
    output_root: Path,
    freeze_id: str,
    frozen_at: str,
    items: Sequence[dict[str, object]],
    report: dict[str, object],
) -> Path:
    freeze_dir = output_root / freeze_id
    freeze_dir.mkdir(parents=True, exist_ok=False)
    jsonl_path = freeze_dir / "dataset_manifest.jsonl"
    # Write bytes so Windows does not translate canonical LF separators to
    # CRLF. This makes the published dataset SHA-256 equal the file SHA-256.
    jsonl_path.write_bytes(
        ("\n".join(canonical_json_line(item) for item in items) + "\n").encode(
            "utf-8"
        )
    )
    fieldnames = [
        "video_id",
        "title",
        "url",
        "channel",
        "duration_seconds",
        "upload_date",
        "caption_kind",
        "caption_language",
        "caption_word_count",
        "caption_sha256",
        "caption_path",
        "content_reason",
        "source_names",
    ]
    with (freeze_dir / "dataset_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for item in items:
            row = dict(item)
            row["source_names"] = " | ".join(item["source_names"])
            writer.writerow(row)
    (freeze_dir / "freeze_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Keep the familiar top-level report name synchronized with the newest
    # completed freeze, so an earlier screening-only count is not mistaken for
    # the final dataset count.
    (VERSION_DIR / "final_dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    counts = report["counts"]
    markdown = f"""# Frozen Abdal Hakim Murad lecture dataset

- Freeze ID: `{freeze_id}`
- Frozen at: `{frozen_at}`
- Method: `{METHOD_VERSION}`
- Final unique lectures with saved captions: **{counts['frozen_usable_unique_with_captions']}**
- Excluded non-lectures: {counts['excluded_nonlectures']}
- Marked duplicate uploads: {counts['marked_duplicates']}
- Unresolved content reviews: {counts['unresolved_content_review']}
- Dataset SHA-256: `{report['dataset_sha256']}`

The snapshot contains metadata and caption references; no video or audio was
downloaded. `dataset_manifest.jsonl` is the canonical machine-readable manifest.
Every caption entry includes a SHA-256 checksum so later changes can be detected.
All review decisions and thresholds are recorded in `freeze_report.json` and the
SQLite audit tables.
"""
    (freeze_dir / "README.md").write_text(markdown, encoding="utf-8")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "LATEST.json").write_text(
        json.dumps(
            {
                "freeze_id": freeze_id,
                "frozen_at": frozen_at,
                "path": freeze_dir.relative_to(VERSION_DIR).as_posix(),
                "dataset_sha256": report["dataset_sha256"],
                "count": len(items),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return freeze_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--duplicate-report", type=Path, default=DEFAULT_DUPLICATE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--freeze-id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frozen_at = utc_now()
    freeze_id = args.freeze_id or safe_freeze_id(frozen_at)
    review_run_id = f"{freeze_id}-review"
    output_root = args.output_dir.resolve()
    if (output_root / freeze_id).exists():
        print(f"ERROR: freeze already exists: {output_root / freeze_id}")
        return 1
    connection = connect_manifest(args.db.resolve())
    try:
        ensure_audit_tables(connection)
        connection.execute("BEGIN")
        content_decisions = apply_content_adjudications(
            connection, review_run_id, frozen_at
        )
        duplicate_decisions = apply_duplicate_adjudications(
            connection,
            review_run_id,
            frozen_at,
            args.duplicate_report.resolve(),
        )
        items = build_frozen_items(connection)
        counts = dataset_counts(connection, len(items))
        if counts["unresolved_content_review"] != 0:
            raise ValueError("Cannot freeze while content reviews remain unresolved")
        digest = dataset_digest(items)
        criteria = {
            "candidate_status": "candidate",
            "minimum_duration_seconds": 600,
            "content_status": "lecture_candidate",
            "duplicate_of_video_id": None,
            "caption_status": "available",
            "caption_file_required": True,
            "records_deleted": False,
        }
        report = {
            "freeze_id": freeze_id,
            "frozen_at": frozen_at,
            "method_version": METHOD_VERSION,
            "dataset_sha256": digest,
            "criteria": criteria,
            "counts": counts,
            "content_review": {
                "queue_size": len(content_decisions),
                "resolved": len(content_decisions),
                "decisions": content_decisions,
            },
            "duplicate_review": {
                "pair_count": len(duplicate_decisions),
                "resolved_pairs": sum(
                    decision.decision == "duplicate" for decision in duplicate_decisions
                ),
                "unique_new_duplicate_records": len(
                    {decision.duplicate_video_id for decision in duplicate_decisions}
                ),
                "decisions": [asdict(decision) for decision in duplicate_decisions],
            },
            "integrity": {
                "caption_files_verified": len(items),
                "caption_files_missing_or_empty": 0,
                "per_caption_sha256_recorded": True,
            },
        }
        connection.execute(
            """
            INSERT INTO dataset_freezes(
                freeze_id, frozen_at, method_version, criteria_json,
                counts_json, dataset_sha256
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                freeze_id,
                frozen_at,
                METHOD_VERSION,
                json.dumps(criteria, ensure_ascii=False, sort_keys=True),
                json.dumps(counts, ensure_ascii=False, sort_keys=True),
                digest,
            ),
        )
        connection.executemany(
            """
            INSERT INTO frozen_dataset_items(
                freeze_id, video_id, caption_sha256,
                transcript_word_count, manifest_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                (
                    freeze_id,
                    item["video_id"],
                    item["caption_sha256"],
                    item["caption_word_count"],
                    canonical_json_line(item),
                )
                for item in items
            ),
        )
        freeze_dir = write_freeze_artifacts(
            output_root, freeze_id, frozen_at, items, report
        )
        connection.commit()
    except (OSError, sqlite3.Error, ValueError, KeyError) as exc:
        connection.rollback()
        print(f"ERROR: {exc}")
        return 1
    finally:
        connection.close()

    print(f"Content cases resolved: {len(content_decisions)}")
    print(f"Duplicate pairs resolved: {len(duplicate_decisions)}")
    print(f"Marked duplicate records: {counts['marked_duplicates']}")
    print(f"Frozen usable lectures: {len(items)}")
    print(f"Dataset SHA-256: {digest}")
    print(f"Freeze directory: {freeze_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
