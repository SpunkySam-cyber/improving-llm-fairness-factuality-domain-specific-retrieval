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

from build_worldview_manifest import classify  # noqa: E402
from dialogue_jsonl import validate_record  # noqa: E402
from screen_worldview_discovery import infer_content_type, normalize_title  # noqa: E402


class WorldviewClassificationTests(unittest.TestCase):
    def test_preserves_lectures(self) -> None:
        self.assertEqual(classify("lecture_candidate", None), ("lecture", True))

    def test_restores_qa_and_interview(self) -> None:
        self.assertEqual(
            classify("excluded_nonlecture", "questions_answers"), ("qa", True)
        )
        self.assertEqual(
            classify("excluded_nonlecture", "interview"), ("interview", True)
        )

    def test_does_not_attribute_derivative_discussion_to_murad(self) -> None:
        self.assertEqual(
            classify(
                "excluded_nonlecture", "adjudicated:derivative_essay_discussion"
            ),
            ("not_murad_speech", False),
        )


class DialogueRecordTests(unittest.TestCase):
    def test_valid_exchange(self) -> None:
        record = {
            "video_id": "abc",
            "url": "https://www.youtube.com/watch?v=abc",
            "content_type": "qa",
            "exchange_index": 1,
            "question": "What is tradition?",
            "answer": "Tradition is...",
            "question_start_seconds": 10.0,
            "question_end_seconds": 15.0,
            "answer_start_seconds": 15.1,
            "answer_end_seconds": 30.0,
            "answer_speaker": "Abdal Hakim Murad",
            "speaker_attribution_method": "manual",
        }
        self.assertEqual(validate_record(record), [])

    def test_rejects_wrong_answer_speaker(self) -> None:
        record = {
            "content_type": "qa",
            "question": "Question",
            "answer": "Answer",
            "question_start_seconds": 1,
            "question_end_seconds": 2,
            "answer_start_seconds": 2,
            "answer_end_seconds": 3,
            "answer_speaker": "Unknown",
        }
        self.assertIn(
            "answer_speaker must be Abdal Hakim Murad", validate_record(record)
        )


class DiscoveryScreenTests(unittest.TestCase):
    def test_normalizes_speaker_variants(self) -> None:
        self.assertEqual(
            normalize_title("Shaykh Abdal Hakim Murad: Islam & Freedom (HD)"),
            "islam freedom",
        )

    def test_infers_conversation_formats(self) -> None:
        self.assertEqual(infer_content_type("Live Q&A with Shaykh"), "qa")
        self.assertEqual(infer_content_type("A conversation with Murad"), "discussion")


if __name__ == "__main__":
    unittest.main()
