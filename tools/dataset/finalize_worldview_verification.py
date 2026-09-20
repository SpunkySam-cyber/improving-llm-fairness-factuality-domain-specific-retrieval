"""Apply explicit manual adjudications to provisional caption verification."""

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
from collections import Counter
from pathlib import Path
from typing import Any


VERSION_DIR = Path(__file__).resolve().parents[2] / "transcription_pipeline_versions" / "v5_batch_ingestion_pipeline"  # data/output dir (shared with V5 batch core)
INPUT_DIR = VERSION_DIR / "worldview_corpus" / "content_verification"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> int:
    results = load_jsonl(INPUT_DIR / "verification_results.jsonl")
    overrides = {
        item["video_id"]: item
        for item in load_jsonl(INPUT_DIR / "manual_adjudications.jsonl")
    }
    final: list[dict[str, Any]] = []
    for provisional in results:
        item = dict(provisional)
        override = overrides.get(item["video_id"])
        if override:
            item["provisional_result_preserved"] = {
                "murad_speaks": item["murad_speaks"],
                "content_type": item["content_type"],
                "speaker_evidence": item["speaker_evidence"],
                "content_evidence": item["content_evidence"],
                "accepted_for_working_corpus": item["accepted_for_working_corpus"],
            }
            item.update(override)
            item["needs_manual_review"] = False
        item["verification_state"] = "adjudicated" if override else "screened"
        final.append(item)
    (INPUT_DIR / "final_verification_results.jsonl").write_bytes(
        ("\n".join(canonical(item) for item in final) + "\n").encode("utf-8")
    )
    content_types = Counter(item["content_type"] for item in final)
    report = {
        "candidate_count": len(final),
        "accepted_for_working_corpus": sum(bool(item["accepted_for_working_corpus"]) for item in final),
        "rejected": sum(not bool(item["accepted_for_working_corpus"]) for item in final),
        "manual_adjudications": len(overrides),
        "content_types": dict(sorted(content_types.items())),
        "method_note": "Provisional Gemini caption-context screening is preserved. Explicit timestamped manual adjudications are layered on top and never overwrite the audit trail.",
    }
    (INPUT_DIR / "final_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
