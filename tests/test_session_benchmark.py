"""The pure parts of tools/evaluate/session_benchmark.py: the summary of a session's per-frame records."""

import unittest

from tools.evaluate.session_benchmark import percentiles, stage_medians, summarise


def _record(**kwargs):
    base = dict(status="regressed", reprojection=30.0, dressed=True, garment_pixels=1000, uncovered=0.1, shoulder_ratio=1.0, front_or_back=True)
    return {**base, **kwargs}


class SummaryTest(unittest.TestCase):
    def test_percentiles_skip_missing_and_non_finite_values(self):
        got = percentiles([1.0, 2.0, None, float("nan"), 3.0], points=(50,))
        self.assertEqual(got, {"p50": 2.0})
        self.assertEqual(percentiles([None]), {})

    def test_stage_medians_come_from_the_profile_line(self):
        text = "something\nmedian ms a frame: mediapipe 14  regressor 1  fit 13  total 120\n"
        self.assertEqual(stage_medians(text), {"mediapipe": 14.0, "regressor": 1.0, "fit": 13.0, "total": 120.0})
        self.assertEqual(stage_medians("nothing here"), {})

    def test_stage_names_can_hold_a_space(self):
        text = "median ms a frame: mediapipe 14  erase start 3  render 78  erase wait 16  composite 15  total 145"
        self.assertEqual(stage_medians(text), {"mediapipe": 14.0, "erase start": 3.0, "render": 78.0, "erase wait": 16.0, "composite": 15.0, "total": 145.0})

    def test_the_summary_counts_dressed_frames_and_shoulders_outside_the_band_in_front_and_back_views_only(self):
        records = [
            _record(),
            _record(shoulder_ratio=2.0),  # front, far too wide
            _record(shoulder_ratio=2.0, front_or_back=False),  # a profile: not counted
            _record(dressed=False, garment_pixels=None, uncovered=None, status="no person", reprojection=None, shoulder_ratio=None, front_or_back=None),
        ]
        summary = summarise(records, {"total": 100.0}, 12.34)
        self.assertEqual(summary["frames"], 4)
        self.assertEqual(summary["dressed_share"], 0.75)
        self.assertEqual(summary["status_counts"], {"regressed": 3, "no person": 1})
        self.assertEqual(summary["front_or_back_frames"], 2)
        self.assertEqual(summary["shoulder_ratio_outside_band"], 1)
        self.assertEqual(summary["garment_pixels_median"], 1000)
        self.assertEqual(summary["stage_median_ms"], {"total": 100.0})
        self.assertEqual(summary["wall_seconds"], 12.3)

    def test_an_empty_session_does_not_crash(self):
        summary = summarise([], {}, 0.0)
        self.assertEqual(summary["frames"], 0)
        self.assertEqual(summary["dressed_share"], 0.0)


if __name__ == "__main__":
    unittest.main()
