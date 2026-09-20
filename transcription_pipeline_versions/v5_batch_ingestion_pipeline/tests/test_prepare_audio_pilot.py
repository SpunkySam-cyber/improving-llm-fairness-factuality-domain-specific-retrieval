from __future__ import annotations

import sys
import unittest
from pathlib import Path


VERSION_DIR = Path(__file__).resolve().parents[1]
if str(VERSION_DIR) not in sys.path:
    sys.path.insert(0, str(VERSION_DIR))

from prepare_audio_pilot import (  # noqa: E402
    duration_bucket,
    stable_key,
    title_is_pilot_eligible,
)


class PrepareAudioPilotTests(unittest.TestCase):
    def test_duration_buckets_cover_pilot_range(self) -> None:
        self.assertEqual(duration_bucket(600), "10-19m")
        self.assertEqual(duration_bucket(2399), "20-39m")
        self.assertEqual(duration_bucket(5400), "90-120m")
        self.assertIsNone(duration_bucket(7201))

    def test_excludes_performance_without_excluding_lecture_about_dhikr(self) -> None:
        self.assertFalse(title_is_pilot_eligible("Dhikr Rihla Abdul Hakim Murad"))
        self.assertFalse(
            title_is_pilot_eligible("Qasidah Burdah by Sheikh Abdal Hakim Murad")
        )
        self.assertTrue(
            title_is_pilot_eligible("Dhikr & Fikr - Abdal Hakim Murad: Ramadan Therapy")
        )

    def test_seeded_key_is_stable(self) -> None:
        self.assertEqual(stable_key("abc"), stable_key("abc"))
        self.assertNotEqual(stable_key("abc"), stable_key("def"))


if __name__ == "__main__":
    unittest.main()
