"""Select one reproducible 10-lecture expansion batch without paid work.

The script reads the frozen, deduplicated lecture manifest; excludes the ten
existing audio-pilot videos and any earlier expansion batches; then selects two
lectures from each duration stratum. It writes a proposed manifest and a short
Markdown report. It never downloads audio or calls an external API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


VERSION_DIR = Path(__file__).resolve().parent
FROZEN_MANIFEST = VERSION_DIR / "worldview_corpus" / "frozen_v3" / "corpus_manifest.jsonl"
PILOT_MANIFEST = VERSION_DIR / "audio_transcription_pilot" / "pilot_manifest.json"
OUTPUT_ROOT = VERSION_DIR / "lecture_expansion"
SELECTION_SEED = "ahm-lecture-expansion-v1"
BATCH_SIZE = 10
DURATION_BUCKETS = (
    ("10-19m", 600, 1199, 2),
    ("20-39m", 1200, 2399, 2),
    ("40-59m", 2400, 3599, 2),
    ("60-89m", 3600, 5399, 2),
    ("90m-plus", 5400, None, 2),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bucket_for(duration: int) -> str | None:
    for name, minimum, maximum, _quota in DURATION_BUCKETS:
        if duration >= minimum and (maximum is None or duration <= maximum):
            return name
    return None


def stable_key(batch_number: int, video_id: str) -> str:
    value = f"{SELECTION_SEED}:batch-{batch_number:02d}:{video_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def clean_display(value: str) -> str:
    """Repair common legacy title mojibake for the human-readable report only."""
    return (
        value.replace("â€˜", "'")
        .replace("â€™", "'")
        .replace("â€“", "-")
        .replace("â€”", "-")
    )


def earlier_batch_ids(batch_number: int) -> set[str]:
    selected: set[str] = set()
    for number in range(1, batch_number):
        path = OUTPUT_ROOT / f"batch_{number:02d}" / "selection_manifest.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Batch {number:02d} manifest is required before selecting batch {batch_number:02d}: {path}"
            )
        selected.update(str(item["video_id"]) for item in load_json(path)["videos"])
    return selected


def pick_batch(batch_number: int) -> tuple[list[dict], dict]:
    frozen = load_jsonl(FROZEN_MANIFEST)
    lectures = [item for item in frozen if item.get("content_type") == "lecture"]
    pilot = load_json(PILOT_MANIFEST)
    pilot_ids = {str(item["video_id"]) for item in pilot["videos"]}
    used_ids = pilot_ids | earlier_batch_ids(batch_number)
    used_channels = {str(item.get("channel") or "") for item in pilot["videos"]}

    eligible: list[dict] = []
    for item in lectures:
        duration = int(item.get("duration_seconds") or 0)
        bucket = bucket_for(duration)
        if not bucket or str(item["video_id"]) in used_ids:
            continue
        candidate = dict(item)
        candidate["duration_bucket"] = bucket
        eligible.append(candidate)

    selected: list[dict] = []
    batch_channels: set[str] = set()
    for bucket_name, minimum, maximum, quota in DURATION_BUCKETS:
        candidates = sorted(
            (item for item in eligible if item["duration_bucket"] == bucket_name),
            key=lambda item: stable_key(batch_number, str(item["video_id"])),
        )
        for _ in range(quota):
            remaining = [item for item in candidates if item not in selected]
            if not remaining:
                raise ValueError(f"Unable to fill {bucket_name} quota for batch {batch_number}")
            preferred = [
                item
                for item in remaining
                if str(item.get("channel") or "") not in used_channels
                and str(item.get("channel") or "") not in batch_channels
            ]
            distinct_in_batch = [
                item
                for item in remaining
                if str(item.get("channel") or "") not in batch_channels
            ]
            choice = dict((preferred or distinct_in_batch or remaining)[0])
            choice["selection_reason"] = "seeded duration stratification with channel-diversity preference"
            choice["batch_number"] = batch_number
            choice["selection_status"] = "proposed_awaiting_user_approval"
            choice["download_status"] = "not_started"
            choice["transcription_status"] = "not_started"
            selected.append(choice)
            batch_channels.add(str(choice.get("channel") or ""))

    selected.sort(key=lambda item: (int(item["duration_seconds"]), str(item["video_id"])))
    for rank, item in enumerate(selected, start=1):
        item["selection_rank"] = rank

    if len(selected) != BATCH_SIZE:
        raise ValueError(f"Expected {BATCH_SIZE} videos, selected {len(selected)}")
    if len({str(item["video_id"]) for item in selected}) != BATCH_SIZE:
        raise ValueError("Duplicate video ID in selection")
    if any(item.get("content_type") != "lecture" for item in selected):
        raise ValueError("Non-lecture content entered the batch")
    for bucket_name, _minimum, _maximum, quota in DURATION_BUCKETS:
        actual = sum(item["duration_bucket"] == bucket_name for item in selected)
        if actual != quota:
            raise ValueError(f"Bucket {bucket_name}: expected {quota}, got {actual}")

    context = {
        "frozen_records": len(frozen),
        "frozen_lectures": len(lectures),
        "existing_pilot_videos": len(pilot_ids),
        "previous_expansion_videos": len(used_ids - pilot_ids),
        "eligible_remaining_lectures": len(eligible),
    }
    return selected, context


def markdown_report(manifest: dict) -> str:
    lines = [
        f"# Lecture expansion batch {manifest['batch_number']:02d} - proposed selection",
        "",
        "Status: **Awaiting user approval**",
        "",
        "No audio was downloaded and no paid API was called.",
        "",
        "| # | Duration bucket | Duration | Channel | Captions | Lecture |",
        "|---:|---|---:|---|---|---|",
    ]
    for item in manifest["videos"]:
        seconds = int(item["duration_seconds"])
        duration = f"{seconds // 60}:{seconds % 60:02d}"
        title = clean_display(str(item["title"])).replace("|", "\\|")
        channel = clean_display(str(item.get("channel") or "")).replace("|", "\\|")
        lines.append(
            f"| {item['selection_rank']} | {item['duration_bucket']} | {duration} | "
            f"{channel} | {item.get('caption_status', 'unknown')} | "
            f"[{title}]({item['url']}) |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Lecture-only records: {manifest['validation']['lecture_only_records']}/10",
            f"- Unique video IDs: {manifest['validation']['unique_video_ids']}/10",
            f"- Distinct channels: {manifest['validation']['distinct_channels']}/10",
            "- Duration distribution: two videos in each of five buckets",
            "- Existing pilot overlap: zero",
            "- Paid API calls: zero",
            "- Audio downloads: zero",
            "",
        ]
    )
    return "\n".join(lines)


def build(batch_number: int) -> Path:
    if batch_number < 1 or batch_number > 4:
        raise ValueError("batch number must be between 1 and 4")
    selected, context = pick_batch(batch_number)
    output_dir = OUTPUT_ROOT / f"batch_{batch_number:02d}"
    manifest_path = output_dir / "selection_manifest.json"
    report_path = output_dir / "selection_report.md"
    if manifest_path.exists() or report_path.exists():
        raise FileExistsError(
            f"Batch {batch_number:02d} selection already exists; refusing to overwrite it"
        )

    manifest = {
        "state": "proposed_awaiting_user_approval",
        "created_at": utc_now(),
        "batch_number": batch_number,
        "batch_size": BATCH_SIZE,
        "selection_seed": SELECTION_SEED,
        "selection_method": "fixed-seed duration stratification with channel-diversity preference",
        "source_manifest": str(FROZEN_MANIFEST),
        "criteria": {
            "content_type": "lecture",
            "duration_bucket_quotas": [
                {"name": name, "minimum": minimum, "maximum": maximum, "quota": quota}
                for name, minimum, maximum, quota in DURATION_BUCKETS
            ],
            "exclude_existing_audio_pilot": True,
            "exclude_previous_expansion_batches": True,
            "records_deleted": False,
            "audio_downloads_allowed": False,
            "paid_api_calls_allowed": False,
        },
        "source_context": context,
        "validation": {
            "lecture_only_records": sum(item["content_type"] == "lecture" for item in selected),
            "unique_video_ids": len({item["video_id"] for item in selected}),
            "distinct_channels": len({item.get("channel") for item in selected}),
            "existing_pilot_overlap": 0,
        },
        "total_selected_duration_seconds": sum(int(item["duration_seconds"]) for item in selected),
        "videos": selected,
    }
    write_json(manifest_path, manifest)
    report_path.write_text(markdown_report(manifest), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True, choices=range(1, 5))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(build(args.batch))
