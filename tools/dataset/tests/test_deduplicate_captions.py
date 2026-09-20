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

from deduplicate_captions import (  # noqa: E402
    jaccard_similarity,
    normalize_tokens,
    transcript_fingerprint,
    transcript_shingles,
)


class DuplicateCaptionTests(unittest.TestCase):
    def test_normalization_ignores_case_and_punctuation(self) -> None:
        left = normalize_tokens("Bismillah, AL-Rahman!")
        right = normalize_tokens("bismillah al rahman")
        self.assertEqual(left, right)

    def test_exact_fingerprint_is_stable(self) -> None:
        left = transcript_fingerprint(normalize_tokens("A short lecture."))
        right = transcript_fingerprint(normalize_tokens("a SHORT lecture"))
        self.assertEqual(left, right)

    def test_shingle_similarity_separates_different_text(self) -> None:
        base = transcript_shingles(normalize_tokens("one two three four five six seven"))
        same = transcript_shingles(normalize_tokens("one two three four five six seven"))
        other = transcript_shingles(normalize_tokens("alpha beta gamma delta epsilon zeta"))
        self.assertEqual(jaccard_similarity(base, same), 1.0)
        self.assertEqual(jaccard_similarity(base, other), 0.0)


if __name__ == "__main__":
    unittest.main()
