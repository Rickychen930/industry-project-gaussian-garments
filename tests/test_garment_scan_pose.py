"""ScanPoseBinder.size_factors: how the wearer/scan size ratios scale the garment."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from tryon import paths

from tryon.body.fitter import SMPLX_ROOT, BodyFitter
from tryon.garment.scan_pose import ScanPoseBinder, scan_pose_file
from tryon.garment.skinning import SkinnedGarment


class SizeFactorsTest(unittest.TestCase):
    def factors(self, mode, height=1.2, chest=1.3, widen=False):
        return ScanPoseBinder.size_factors(mode, height, chest, widen)

    def test_current_scales_all_axes_by_the_height_ratio(self):
        np.testing.assert_allclose(self.factors("current"), [1.2, 1.2, 1.2])

    def test_current_widens_only_when_the_renderer_asks(self):
        np.testing.assert_allclose(self.factors("current", widen=True), [1.2 * 1.3, 1.2, 1.2 * 1.3])
        np.testing.assert_allclose(self.factors("current", chest=0.8, widen=True), [1.2, 1.2, 1.2])

    def test_no_chest_ignores_the_chest_ratio(self):
        np.testing.assert_allclose(self.factors("no_chest", widen=True), [1.2, 1.2, 1.2])

    def test_none_is_identity(self):
        np.testing.assert_allclose(self.factors("none"), [1, 1, 1])

    def test_shrink_never_grows_the_garment(self):
        np.testing.assert_allclose(self.factors("shrink", height=1.2), [1, 1, 1])
        np.testing.assert_allclose(self.factors("shrink", height=0.8), [0.8, 0.8, 0.8])

    def test_damped_is_the_square_root(self):
        np.testing.assert_allclose(self.factors("damped", height=1.44), [1.2, 1.2, 1.2])

    def test_height_y_leaves_the_width_alone(self):
        np.testing.assert_allclose(self.factors("height_y"), [1, 1.2, 1])

    def test_widen_never_narrows(self):
        np.testing.assert_allclose(self.factors("widen", chest=0.8), [1.2, 1.2, 1.2])

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            self.factors("bogus")


REFINED = ("00122_outer", "00148_upper", "00176_outer", "10004_outer")
POSE_KEYS = ("betas", "body_pose", "global_orient", "offset")


def _write_pose(path, seed, refined=None):
    rng = np.random.default_rng(seed)
    extra = {} if refined is None else {"refined": np.asarray(refined)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, betas=rng.normal(size=10), body_pose=rng.normal(size=63), global_orient=rng.normal(size=3), offset=rng.normal(size=3), **extra)


class ScanPoseSourceTest(unittest.TestCase):
    """Where `ScanPoseBinder` takes the scan pose from: repository asset, cached file, nothing."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.assets = root / "assets"
        self.checkpoint = root / "garments" / "g1"
        patch = mock.patch.object(paths, "SCAN_POSES", self.assets)
        patch.start()
        self.addCleanup(patch.stop)

    def binder(self):
        return ScanPoseBinder(np.zeros((1, 3)), np.zeros((0, 3), int), self.checkpoint)

    def test_nothing_means_no_scan_pose(self):
        self.assertEqual(scan_pose_file(self.checkpoint), (None, "missing"))
        self.assertIsNone(self.binder()._scan_pose())
        self.assertEqual(scan_pose_file(None), (None, "missing"))

    def test_the_cached_file_is_used_without_an_asset(self):
        _write_pose(self.checkpoint / "stage1" / "scan_pose.npz", 1)
        self.assertEqual(scan_pose_file(self.checkpoint)[1], "cached")
        pose = self.binder()._scan_pose()
        np.testing.assert_array_equal(pose["betas"], np.load(self.checkpoint / "stage1" / "scan_pose.npz")["betas"])
        self.assertFalse(pose["refined"])

    def test_the_asset_wins_over_the_cached_file_and_carries_the_refined_flag(self):
        _write_pose(self.checkpoint / "stage1" / "scan_pose.npz", 1)
        _write_pose(self.assets / "g1.npz", 2, refined=1)
        self.assertEqual(scan_pose_file(self.checkpoint)[1], "asset")
        pose = self.binder()._scan_pose()
        for key in POSE_KEYS:
            np.testing.assert_array_equal(pose[key], np.load(self.assets / "g1.npz")[key])
        self.assertTrue(pose["refined"])


class RepositoryScanPosesTest(unittest.TestCase):
    def test_the_four_misaligned_garments_ship_refined_and_the_rest_do_not(self):
        files = sorted(paths.SCAN_POSES.glob("*.npz"))
        self.assertEqual(len(files), 15)
        for file in files:
            data = np.load(file)
            for key in POSE_KEYS:
                self.assertIn(key, data.files)
            refined = "refined" in data.files and int(data["refined"]) == 1
            self.assertEqual(refined, file.stem in REFINED, file.stem)

    def test_the_binder_reads_the_flag_from_the_asset(self):
        for name in REFINED + ("00152_outer",):
            binder = ScanPoseBinder(np.zeros((1, 3)), np.zeros((0, 3), int), Path("/nonexistent") / name)
            self.assertEqual(binder._scan_pose()["refined"], name in REFINED, name)

    def test_assets_equal_the_poses_cached_in_the_checkpoints(self):
        cached = [d for d in paths.GARMENTS.glob("*") if (d / "stage1" / "scan_pose.npz").exists()] if paths.GARMENTS.is_dir() else []
        if not cached:
            self.skipTest("no garment checkpoints with a cached scan pose")
        for directory in cached:
            asset = paths.SCAN_POSES / f"{directory.name}.npz"
            if not asset.exists():
                continue
            old, new = np.load(directory / "stage1" / "scan_pose.npz"), np.load(asset)
            for key in POSE_KEYS:
                np.testing.assert_array_equal(old[key], new[key], f"{directory.name}.{key}")
            self.assertEqual("refined" in old.files and int(old["refined"]), "refined" in new.files and int(new["refined"]))


def _arm_down(theta):
    """A body with the shoulders `theta` degrees below horizontal."""
    fitter = BodyFitter()
    with torch.no_grad():
        fitter.body_pose[0, (16 - 1) * 3 + 2] = -np.radians(theta)
        fitter.body_pose[0, (17 - 1) * 3 + 2] = +np.radians(theta)
    return fitter


@unittest.skipUnless(SMPLX_ROOT.is_dir(), "no SMPL-X body models")
class RefineArmsTest(unittest.TestCase):
    """`ScanPoseBinder.refine_arms`: arms bound at the wrong angle turn into the sleeves."""

    @classmethod
    def setUpClass(cls):
        truth = _arm_down(30.0)
        with torch.no_grad():
            body = truth._forward().vertices[0].numpy().astype(np.float64)
        weights = truth.model.lbs_weights.detach().numpy()
        cls.is_arm = weights[:, list(SkinnedGarment.ARM_JOINTS)].sum(axis=1) > 0.5
        cls.sleeve_points = body[cls.is_arm]  # a sleeve lying on the true arms
        cls.sleeve = np.ones(len(cls.sleeve_points), bool)

    def binder(self, mode):
        binder = ScanPoseBinder(np.zeros((1, 3)), np.zeros((0, 3), int), None)
        binder.REFINE_ARMS = mode
        return binder

    def distance(self, fitter):
        with torch.no_grad():
            body = fitter._forward().vertices[0].numpy().astype(np.float64)
        return ScanPoseBinder.sleeve_arm_distance(body, self.is_arm, self.sleeve_points, self.sleeve)

    def test_arms_bound_at_the_wrong_angle_move_into_the_sleeves(self):
        fitter = _arm_down(75.0)
        before = self.distance(fitter)
        self.assertGreater(before, 0.05)
        self.assertTrue(self.binder("always").refine_arms(fitter, self.sleeve_points, self.sleeve))
        self.assertLess(self.distance(fitter), 0.5 * before)

    def test_arms_already_in_their_sleeves_are_left_alone(self):
        fitter = _arm_down(30.0)
        pose = fitter.body_pose.detach().clone()
        self.assertFalse(self.binder("always").refine_arms(fitter, self.sleeve_points, self.sleeve))
        self.assertTrue(torch.equal(fitter.body_pose, pose))

    def test_off_and_unflagged_do_nothing(self):
        for mode, flagged in ((None, True), ("refined", False)):
            fitter = _arm_down(75.0)
            pose = fitter.body_pose.detach().clone()
            self.assertFalse(self.binder(mode).refine_arms(fitter, self.sleeve_points, self.sleeve, flagged))
            self.assertTrue(torch.equal(fitter.body_pose, pose))

    def test_a_flagged_pose_is_refined_in_refined_mode(self):
        fitter = _arm_down(75.0)
        self.assertTrue(self.binder("refined").refine_arms(fitter, self.sleeve_points, self.sleeve, True))

    def test_poor_mode_waits_for_a_poor_fit(self):
        self.assertFalse(self.binder("poor").refine_arms(_arm_down(34.0), self.sleeve_points, self.sleeve))
        self.assertTrue(self.binder("poor").refine_arms(_arm_down(75.0), self.sleeve_points, self.sleeve))


if __name__ == "__main__":
    unittest.main()
