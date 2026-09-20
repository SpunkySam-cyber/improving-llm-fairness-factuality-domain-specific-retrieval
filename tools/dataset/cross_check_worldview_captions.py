"""Compare newly saved discovery captions against the original caption corpus."""

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

from build_manifest import connect_manifest
from deduplicate_captions import (
    AUTO_DUPLICATE_THRESHOLD,
    REVIEW_THRESHOLD,
    jaccard_similarity,
    load_records,
    plausible_pair,
)


VERSION_DIR = Path(__file__).resolve().parents[2] / "transcription_pipeline_versions" / "v5_batch_ingestion_pipeline"  # data/output dir (shared with V5 batch core)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-db", type=Path, default=VERSION_DIR / "pipeline.sqlite")
    parser.add_argument(
        "--discovery-db",
        type=Path,
        default=VERSION_DIR / "worldview_corpus" / "worldview_discovery.sqlite",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=VERSION_DIR / "worldview_corpus" / "discovery_review" / "cross_corpus_duplicates.json",
    )
    args = parser.parse_args()

    old_connection = connect_manifest(args.existing_db.resolve())
    new_connection = connect_manifest(args.discovery_db.resolve())
    try:
        old_records = load_records(old_connection)
        new_records = load_records(new_connection)
    finally:
        old_connection.close()
        new_connection.close()

    pairs: list[dict[str, object]] = []
    for new in new_records:
        for old in old_records:
            if new.fingerprint == old.fingerprint:
                similarity = 1.0
                method = "exact_normalized_transcript"
            elif plausible_pair(new, old):
                similarity = jaccard_similarity(new.shingles, old.shingles)
                if similarity < REVIEW_THRESHOLD:
                    continue
                method = "caption_5gram_jaccard"
            else:
                continue
            pairs.append(
                {
                    "new_video_id": new.video_id,
                    "new_title": new.title,
                    "existing_video_id": old.video_id,
                    "existing_title": old.title,
                    "similarity": round(similarity, 6),
                    "method": method,
                    "decision": (
                        "probable_duplicate"
                        if similarity >= AUTO_DUPLICATE_THRESHOLD
                        else "review"
                    ),
                }
            )
    pairs.sort(key=lambda item: (-float(item["similarity"]), str(item["new_video_id"])))
    report = {
        "existing_caption_records": len(old_records),
        "new_caption_records": len(new_records),
        "probable_duplicate_pairs": sum(
            pair["decision"] == "probable_duplicate" for pair in pairs
        ),
        "review_pairs": sum(pair["decision"] == "review" for pair in pairs),
        "pairs": pairs,
    }
    args.report.resolve().write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "pairs"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
