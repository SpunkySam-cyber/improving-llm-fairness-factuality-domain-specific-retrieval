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

from screen_lecture_candidates import classify_title  # noqa: E402


class LectureScreeningTests(unittest.TestCase):
    def test_keeps_ordinary_lecture(self) -> None:
        self.assertEqual(
            classify_title("The Path to Allah – Abdal Hakim Murad"),
            ("lecture_candidate", None),
        )

    def test_excludes_explicit_interview(self) -> None:
        status, reason = classify_title(
            "Interview with Shaykh Abdal Hakim Murad"
        )
        self.assertEqual(status, "excluded_nonlecture")
        self.assertEqual(reason, "interview")

    def test_flags_named_co_speaker_without_deleting(self) -> None:
        status, reason = classify_title(
            "Abdal Hakim Murad & Isam Bachiri: Commemorating the Miraj"
        )
        self.assertEqual(status, "needs_review")
        self.assertEqual(reason, "named_co_speaker_after_murad")

    def test_does_not_treat_topic_ampersand_as_co_speaker(self) -> None:
        self.assertEqual(
            classify_title("Fasting & Restraint – Abdal Hakim Murad"),
            ("lecture_candidate", None),
        )


if __name__ == "__main__":
    unittest.main()
