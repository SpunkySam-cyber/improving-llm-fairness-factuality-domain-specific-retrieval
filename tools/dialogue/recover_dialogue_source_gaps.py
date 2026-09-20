"""Search YouTube for accessible reuploads of two unavailable dialogue sources.

This script is deliberately conservative: it discovers, scores, and probes
candidates, but never substitutes a different recording without strong title
and duration evidence. All results are cached in one JSON report.
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
import json
import re
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from prepare_audio_pilot import find_executable, utc_now, write_json


VERSION_DIR = Path(__file__).resolve().parents[2] / "transcription_pipeline_versions" / "v5_batch_ingestion_pipeline"  # data/output dir (shared with V5 batch core)
ROOT = VERSION_DIR / "worldview_corpus" / "dialogue_source_gaps"
MANIFEST_PATH = ROOT / "manifest.json"
REPORT_PATH = ROOT / "recovery_search_report.json"

SEARCHES = {
    "9UeF4Na28rw": [
        '"Travelling Home" "Essays on Islam in Europe" Abdal Hakim Murad',
        '"Travelling Home" Abdal Hakim Murad book review',
        '"Travelling Home" Timothy Winter interview',
    ],
    "Tqnbvsojmek": [
        '"Contentions" "Kamil Khan Mumtaz" Timothy Winter',
        '"Contentions" Abdal Hakim Murad book review',
        '"Contentions" Timothy Winter interview',
    ],
}


def normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def tokens(value: str) -> set[str]:
    stop = {"a", "an", "and", "by", "dr", "in", "of", "on", "the", "with"}
    return {item for item in normalize(value).split() if item not in stop}


def title_scores(target: str, candidate: str) -> tuple[float, float]:
    left = normalize(target)
    right = normalize(candidate)
    sequence = SequenceMatcher(None, left, right).ratio()
    a = tokens(target)
    b = tokens(candidate)
    jaccard = len(a & b) / max(1, len(a | b))
    return sequence, jaccard


def duration_score(target: int, candidate: int | None) -> float:
    if not candidate or target <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(candidate - target) / target)


def run_json(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return json.loads(result.stdout)


def search(yt_dlp: Path, query: str) -> list[dict[str, Any]]:
    payload = run_json(
        [
            str(yt_dlp),
            "--quiet",
            "--no-warnings",
            "--flat-playlist",
            "--dump-single-json",
            f"ytsearch20:{query}",
        ]
    )
    return [item for item in payload.get("entries") or [] if isinstance(item, dict)]


def probe(yt_dlp: Path, video_id: str) -> tuple[bool, str | None, dict[str, Any] | None]:
    try:
        payload = run_json(
            [
                str(yt_dlp),
                "--quiet",
                "--no-warnings",
                "--simulate",
                "--dump-single-json",
                f"https://www.youtube.com/watch?v={video_id}",
            ]
        )
        return True, None, payload
    except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return False, str(exc), None


def main() -> int:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    targets = {str(item["video_id"]): item for item in manifest["videos"]}
    yt_dlp = find_executable("yt-dlp", None)
    report: dict[str, Any] = {
        "state": "running",
        "started_at": utc_now(),
        "method": "three YouTube searches per target, title/duration scoring, accessibility probe",
        "targets": {},
    }
    write_json(REPORT_PATH, report)
    for target_id, target in targets.items():
        discovered: dict[str, dict[str, Any]] = {}
        query_errors: list[dict[str, str]] = []
        for query in SEARCHES[target_id]:
            try:
                for item in search(yt_dlp, query):
                    video_id = str(item.get("id") or "")
                    if not video_id or video_id == target_id:
                        continue
                    candidate = discovered.setdefault(
                        video_id,
                        {
                            "video_id": video_id,
                            "url": f"https://www.youtube.com/watch?v={video_id}",
                            "title": str(item.get("title") or ""),
                            "channel": item.get("channel") or item.get("uploader"),
                            "duration_seconds": item.get("duration"),
                            "matched_queries": [],
                        },
                    )
                    candidate["matched_queries"].append(query)
            except Exception as exc:
                query_errors.append({"query": query, "error": str(exc)})

        ranked: list[dict[str, Any]] = []
        for item in discovered.values():
            sequence, jaccard = title_scores(str(target["title"]), item["title"])
            duration = duration_score(
                int(target["duration_seconds"]),
                int(item["duration_seconds"]) if item.get("duration_seconds") else None,
            )
            unique_target_tokens = tokens(str(target["title"]))
            unique_coverage = len(unique_target_tokens & tokens(item["title"])) / max(
                1, len(unique_target_tokens)
            )
            combined = 0.35 * sequence + 0.35 * jaccard + 0.2 * duration + 0.1 * unique_coverage
            item.update(
                {
                    "title_sequence_similarity": round(sequence, 6),
                    "title_token_jaccard": round(jaccard, 6),
                    "duration_similarity": round(duration, 6),
                    "target_token_coverage": round(unique_coverage, 6),
                    "combined_score": round(combined, 6),
                }
            )
            ranked.append(item)
        ranked.sort(key=lambda item: (-item["combined_score"], item["video_id"]))

        # Probe only the strongest ten to minimize network traffic.
        for item in ranked[:10]:
            accessible, error, payload = probe(yt_dlp, item["video_id"])
            item["accessible"] = accessible
            item["access_error"] = error
            if payload:
                item["probed_title"] = payload.get("title")
                item["probed_duration_seconds"] = payload.get("duration")
                item["probed_channel"] = payload.get("channel") or payload.get("uploader")

        strong = [
            item
            for item in ranked
            if item.get("accessible")
            and item["title_token_jaccard"] >= 0.72
            and item["duration_similarity"] >= 0.75
        ]
        target_report = {
            "target": target,
            "queries": SEARCHES[target_id],
            "query_errors": query_errors,
            "discovered_candidates": len(ranked),
            "strong_accessible_matches": strong,
            "top_candidates": ranked[:15],
            "decision": (
                "strong_match_found_requires_transcript_confirmation"
                if strong
                else "no_safe_automatic_substitution"
            ),
        }
        report["targets"][target_id] = target_report
        write_json(REPORT_PATH, report)

    report["state"] = "completed"
    report["completed_at"] = utc_now()
    report["safe_matches_found"] = sum(
        bool(item["strong_accessible_matches"])
        for item in report["targets"].values()
    )
    write_json(REPORT_PATH, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
