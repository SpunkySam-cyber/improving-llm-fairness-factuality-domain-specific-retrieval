"""Apply documented human dialogue edits without losing automated originals."""

from __future__ import annotations

# --- repo layout shim: find shared modules (V5 batch core, dialogue/dataset tools, project_env) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
for _d in (_ROOT, _ROOT / "transcription_pipeline_versions" / "v5_batch_ingestion_pipeline", _ROOT / "tools" / "dataset", _ROOT / "tools" / "dialogue"):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
# --- end shim ---
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dialogue_jsonl import validate_record


VERSION_DIR = Path(__file__).resolve().parents[2] / "transcription_pipeline_versions" / "v5_batch_ingestion_pipeline"  # data/output dir (shared with V5 batch core)
ROOT = VERSION_DIR / "worldview_corpus" / "frozen_v3" / "dialogue_jsonl"
ALL_EXCHANGES = ROOT / "all_exchanges.jsonl"
ADJUDICATIONS = ROOT / "manual_adjudications.jsonl"
REPORT = ROOT / "report.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


REVIEW_1_ANSWER = """Iftars sometimes tend to be rather excessively flamboyant affairs that have been brought into the hospitality industry, so a kind of generic buffet with generic, predictable food present everywhere around the world in a generic hotel. That's really not very interesting for me. The memorable iftars have been iftars with kind of simple people who are breaking the fast in a joyful and spontaneous way. So this was brought home to me in Cairo, for instance, where I lived for three years back in the '80s, where one of the well-known Egyptian publishers during Ramadan would throw an iftar celebration. I think it was at the Marriott Hotel, one of the grandest venues. It used to be the palace of the Empress Eugénie. It's a very historic, interesting place. And so you had maybe two or three hundred literary figures in Egypt, and Tawfiq al-Hakim, you might have heard, used to be there, and some others. And they would be waiting and waiting, and then the time came and everybody charged for the buffet, and I thought, I'll wait. And at the end they left almost nothing for me. I think there was some bread and some sweet things. And so that was not a particularly luminous experience. I remember it quite easily. But then, of course, in Egyptian villages, eating something really simple with local people, with water from their well and geese running around, and the feeling of the setting of the sun, which is the most atmospheric time of day, that was very extraordinary. But I remember also in the communist period, for instance, visiting Uzbekistan, and Ramadan happened to include Karl Marx's birthday. Perhaps you remember from Yugoslavia's communist times, they made a big ceremonial of these things. So we watched some of the rituals and lots of girls dancing and formations and marching, and extremely kind of dead and empty, people just going through it because Moscow insisted on it. But then that evening I found in Tashkent that there was one mosque that was still functioning. And so I found a taxi driver who could take me to this place, which is not the place the tourists ever go to see. And I thought, well, there'll be a few old guys here, and it was absolutely packed, absolutely full. You can imagine, a great ancient people like the Uzbeks with a history of Islam; only one place was open. So the tarawih that I remember was so beautiful: long, long rak'ahs, and after every four rak'ahs they would stop and everybody would get a dish of tea, and there would be a talk by one of the muftis. The tarawih went on for about three or four hours, I reckon. Well, it must have been wintertime then. I remember the night was long. But that was very atmospheric, and you felt the greyness of communism and atheism on the streets outside and the stupid Marxist celebration, and then you saw real people and the light. It was like stepping into a different world. So I remember that very clearly."""


REVIEWS = [
    {
        "video_id": "3BaN64_IKkQ",
        "exchange_index": 1,
        "review_number": 1,
        "decision": "edited",
        "reviewer": "project_researcher",
        "review_source": "manual_audio_review_user_feedback",
        "question": (
            "So can you please narrate to us some of your most memorable iftars? "
            "It doesn't have to be of recent years—any time that you remember."
        ),
        "answer": REVIEW_1_ANSWER,
        "question_start_seconds": 235.239,
        "question_end_seconds": 244.92,
        "answer_start_seconds": 244.92,
        "answer_end_seconds": 444.52,
        "source_segment_indices": {"question": [86, 89], "answer": [90, 168]},
        "review_notes": (
            "Corrected ifar/ifs to iftar/iftars; identified Empress Eugénie and "
            "Tawfiq al-Hakim; moved 'Yes, I mean' out of the question; extended "
            "Murad's answer from 359.12 seconds through 'So I remember that very "
            "clearly' at approximately 444.52 seconds. Rolling YouTube cues overlap "
            "with the next speaker from about 442.12 seconds."
        ),
    }
]


def main() -> int:
    rows = load_jsonl(ALL_EXCHANGES)
    existing_adjudications = load_jsonl(ADJUDICATIONS) if ADJUDICATIONS.exists() else []
    by_review_key = {
        (str(item["video_id"]), int(item["exchange_index"])): item
        for item in existing_adjudications
    }
    for review in REVIEWS:
        key = (str(review["video_id"]), int(review["exchange_index"]))
        match_index = next(
            (
                index
                for index, item in enumerate(rows)
                if (str(item["video_id"]), int(item["exchange_index"])) == key
            ),
            None,
        )
        if match_index is None:
            raise RuntimeError(f"dialogue record not found: {key}")
        original = rows[match_index]
        reviewed_at = utc_now()
        updated = dict(original)
        for field in (
            "question",
            "answer",
            "question_start_seconds",
            "question_end_seconds",
            "answer_start_seconds",
            "answer_end_seconds",
            "source_segment_indices",
        ):
            updated[field] = review[field]
        updated.update(
            {
                "question_sha256": sha256_text(str(review["question"])),
                "answer_sha256": sha256_text(str(review["answer"])),
                "review_status": "edited",
                "speaker_attribution_method": "manual",
                "speaker_attribution_confidence": 1.0,
                "human_review": {
                    "review_number": review["review_number"],
                    "reviewed_at": reviewed_at,
                    "reviewer": review["reviewer"],
                    "review_source": review["review_source"],
                    "decision": review["decision"],
                    "notes": review["review_notes"],
                },
            }
        )
        errors = validate_record(updated)
        if errors:
            raise RuntimeError(f"reviewed record is invalid: {'; '.join(errors)}")
        rows[match_index] = updated
        by_review_key[key] = {
            "video_id": key[0],
            "exchange_index": key[1],
            "review_number": review["review_number"],
            "reviewed_at": reviewed_at,
            "decision": review["decision"],
            "reviewer": review["reviewer"],
            "review_source": review["review_source"],
            "review_notes": review["review_notes"],
            "original_automated_record": original,
            "human_edited_record": updated,
        }

    rows.sort(key=lambda item: (str(item["video_id"]), int(item["exchange_index"])))
    ALL_EXCHANGES.write_text(
        "\n".join(canonical(item) for item in rows) + "\n", encoding="utf-8"
    )
    for video_id in sorted({str(item["video_id"]) for item in rows}):
        video_rows = [item for item in rows if item["video_id"] == video_id]
        output = ROOT / "videos" / video_id / "exchanges.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "\n".join(canonical(item) for item in video_rows) + "\n", encoding="utf-8"
        )
    adjudications = sorted(by_review_key.values(), key=lambda item: (item["video_id"], item["exchange_index"]))
    ADJUDICATIONS.write_text(
        "\n".join(canonical(item) for item in adjudications) + "\n", encoding="utf-8"
    )
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    counts = {
        state: sum(item.get("review_status") == state for item in rows)
        for state in ("pending", "verified", "edited", "rejected")
    }
    report["review_status"] = "manual_review_in_progress"
    report["manual_review_summary"] = counts
    report["last_manual_review_at"] = utc_now()
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
