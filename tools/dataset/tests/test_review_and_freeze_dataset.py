from __future__ import annotations

# --- repo layout shim: find shared modules (V5 batch core, dialogue/dataset tools, project_env) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[3]
for _d in (_ROOT, _ROOT / "transcription_pipeline_versions" / "v5_batch_ingestion_pipeline", _ROOT / "tools" / "dataset", _ROOT / "tools" / "dialogue"):
    if str(_d) not in _sys.path:
        _sys.path.append(str(_d))
# --- end shim ---
import sys
import unittest
from pathlib import Path


VERSION_DIR = Path(__file__).resolve().parents[1]
if str(VERSION_DIR) not in sys.path:
    sys.path.insert(0, str(VERSION_DIR))

from deduplicate_captions import CaptionRecord, transcript_fingerprint, transcript_shingles  # noqa: E402
from review_and_freeze_dataset import (  # noqa: E402
    CONTENT_ADJUDICATIONS,
    _connected_components,
    dataset_digest,
    secondary_duplicate_evidence,
)


def record(video_id: str, words: str, duration: int = 100) -> CaptionRecord:
    tokens = tuple(words.split())
    return CaptionRecord(
        video_id=video_id,
        title=video_id,
        duration_seconds=duration,
        caption_kind="automatic",
        caption_path=Path(f"{video_id}.txt"),
        caption_word_count=len(tokens),
        source_rank=1,
        tokens=tokens,
        fingerprint=transcript_fingerprint(tokens),
        shingles=transcript_shingles(tokens),
    )


class ReviewAndFreezeTests(unittest.TestCase):
    def test_all_twelve_content_cases_are_configured(self) -> None:
        self.assertEqual(len(CONTENT_ADJUDICATIONS), 12)
        self.assertEqual(
            sum(value[0] == "lecture_candidate" for value in CONTENT_ADJUDICATIONS.values()),
            1,
        )

    def test_secondary_duplicate_rule_accepts_small_caption_variation(self) -> None:
        base = " ".join(f"word{i}" for i in range(100))
        changed = base.replace("word50", "replacement")
        evidence = secondary_duplicate_evidence(record("a", base), record("b", changed))
        self.assertEqual(evidence.decision, "duplicate")

    def test_secondary_duplicate_rule_rejects_different_transcripts(self) -> None:
        left = " ".join(f"left{i}" for i in range(100))
        right = " ".join(f"right{i}" for i in range(100))
        evidence = secondary_duplicate_evidence(record("a", left), record("b", right))
        self.assertEqual(evidence.decision, "unresolved")

    def test_connected_components_merge_triplicate_edges(self) -> None:
        components = _connected_components([("a", "b"), ("b", "c"), ("x", "y")])
        self.assertIn({"a", "b", "c"}, components)
        self.assertIn({"x", "y"}, components)

    def test_dataset_digest_is_order_sensitive_and_stable(self) -> None:
        items = [{"video_id": "a"}, {"video_id": "b"}]
        self.assertEqual(dataset_digest(items), dataset_digest(items))
        self.assertNotEqual(dataset_digest(items), dataset_digest(list(reversed(items))))


if __name__ == "__main__":
    unittest.main()
