"""tools/evaluate/yaw_by_pose.py on synthetic records."""

import unittest

from tools.evaluate.yaw_by_pose import agreement, missing_runs, summarise, trace, wrap


class WrapTest(unittest.TestCase):
    def test_wraps_into_half_open_range(self):
        self.assertAlmostEqual(wrap(190.0), -170.0)
        self.assertAlmostEqual(wrap(-190.0), 170.0)
        self.assertAlmostEqual(wrap(180.0), -180.0)


class TraceTest(unittest.TestCase):
    def test_signed_extremes_and_their_times(self):
        records = [{"body_yaw": 0.0}, {"body_yaw": -57.0}, {"body_yaw": None}, {"body_yaw": 92.0}]
        out = trace(records, "body_yaw", fps=1.0)
        self.assertEqual(out["frames_with_yaw"], 3)
        self.assertAlmostEqual(out["lowest_degrees"], -57.0)
        self.assertEqual(out["lowest_at_seconds"], 1.0)
        self.assertAlmostEqual(out["highest_degrees"], 92.0)
        self.assertEqual(out["highest_at_seconds"], 3.0)

    def test_counts_back_views_over_135_degrees(self):
        records = [{"body_yaw": 163.0}, {"body_yaw": -170.0}, {"body_yaw": 20.0}]
        self.assertEqual(trace(records)["frames_over_135_degrees"], 2)

    def test_no_values_gives_a_count_of_zero(self):
        self.assertEqual(trace([{"body_yaw": None}]), {"frames_with_yaw": 0})


class MissingRunsTest(unittest.TestCase):
    def test_reports_only_stretches_of_a_second_or_more(self):
        yaw = [{"body_yaw": 0.0}] + [{"body_yaw": None}] * 3 + [{"body_yaw": 5.0}, {"body_yaw": None}, {"body_yaw": 1.0}]
        self.assertEqual(missing_runs(yaw, fps=3.0), [{"from_seconds": 0.33, "to_seconds": 1.0}])

    def test_a_gap_at_the_end_is_reported(self):
        yaw = [{"body_yaw": 0.0}] + [{"body_yaw": None}] * 4
        self.assertEqual(len(missing_runs(yaw, fps=2.0)), 1)


class AgreementTest(unittest.TestCase):
    def test_wraps_across_the_seam(self):
        records = [{"body_yaw": 175.0, "mediapipe_yaw": -175.0}, {"body_yaw": 0.0, "mediapipe_yaw": None}]
        out = agreement(records)
        self.assertEqual(out["frames"], 1)
        self.assertAlmostEqual(out["median_gap_degrees"], 10.0)

    def test_summarise_has_both_yaw_sources(self):
        out = summarise([{"body_yaw": 0.0, "mediapipe_yaw": 0.0}] * 4)
        self.assertEqual(out["frames"], 4)
        self.assertIn("missing_runs", out["body_yaw"])
        self.assertIn("missing_runs", out["mediapipe_yaw"])


if __name__ == "__main__":
    unittest.main()
