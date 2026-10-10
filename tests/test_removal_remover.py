"""Erasing the wearer's real clothing before compositing.

Two kinds of check. Synthetic: a frame with a known solid-colour "garment"
region and a hand-built mask (and, where a parser is involved, a small
fake one), so the erase step can be judged against an exact answer with no
network involved. Real: the project's own annotated photo, parsed by the
pretrained SCHP parser, judged by (a) how many exposed real-garment pixels
actually changed and (b) the parser's own clothing probability in that
region before vs after.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ATR = ROOT / "models" / "pretrained" / "parsing_atr.onnx"
REAL_FRAME = ROOT / "data" / "annotated" / "Take1" / "0004" / "rgb_images" / "00001.png"


def bare_remover(schp=None, fashn=None):
    """A ClothingRemover without loading any parser: `schp` / `fashn` are
    fakes exposing the methods the real parsers do."""
    from tryon.removal.remover import ClothingRemover

    remover = ClothingRemover.__new__(ClothingRemover)
    remover.schp, remover.fashn, remover.lip = schp, fashn, None
    remover.threshold, remover.dilate_px, remover.inpaint_radius = 0.5, 6, 5
    return remover


class RemoveSyntheticTest(unittest.TestCase):
    """The erase step alone, with a hand-built mask - no parser."""

    @classmethod
    def setUpClass(cls):
        cls.remover = bare_remover()

    def _frame_with_red_shirt(self):
        frame = np.full((120, 100, 3), (60, 180, 60), np.uint8)  # green background
        frame[30:80, 35:65] = (0, 0, 220)  # a solid red "shirt" (BGR)
        mask = np.zeros((120, 100), bool)
        mask[30:80, 35:65] = True
        return frame, mask

    def test_masked_region_no_longer_contains_the_garment_colour(self):
        frame, mask = self._frame_with_red_shirt()
        clean = self.remover.remove(frame, mask)
        red = (clean[..., 2] > 150) & (clean[..., 1] < 100) & (clean[..., 0] < 100)
        self.assertEqual(int(red[mask].sum()), 0, "red garment pixels survived the erase")

    def test_unmasked_pixels_are_untouched(self):
        frame, mask = self._frame_with_red_shirt()
        clean = self.remover.remove(frame, mask)
        np.testing.assert_array_equal(clean[~mask], frame[~mask])

    def test_fill_comes_from_the_surroundings(self):
        # With a green background all around the hole, Telea's fill of the
        # hole must itself be green-ish - the whole point of inpainting
        # over e.g. painting the hole black.
        frame, mask = self._frame_with_red_shirt()
        clean = self.remover.remove(frame, mask)
        filled = clean[mask].astype(int)
        self.assertGreater(filled[:, 1].mean(), filled[:, 2].mean() + 50, "fill is not background-coloured")

    def test_empty_mask_returns_an_untouched_copy(self):
        frame, _ = self._frame_with_red_shirt()
        clean = self.remover.remove(frame, np.zeros(frame.shape[:2], bool))
        np.testing.assert_array_equal(clean, frame)
        self.assertIsNot(clean, frame)


@unittest.skipUnless(ATR.exists(), "run tools/data/download_pretrained.py schp")
@unittest.skipUnless(REAL_FRAME.exists(), "no annotated frame")
class RemoveRealFrameTest(unittest.TestCase):
    """The full step on the project's real annotated photo: the parser
    finds the wearer's jacket, the erase removes it."""

    @classmethod
    def setUpClass(cls):
        import cv2

        from tryon.removal.remover import ClothingRemover

        cls.remover = ClothingRemover(schp_checkpoint=ATR)
        cls.frame = cv2.resize(cv2.imread(str(REAL_FRAME)), (640, 871))
        cls.mask = cls.remover.clothing_mask(cls.frame)
        cls.clean = cls.remover.remove(cls.frame, cls.mask)

    def test_the_parser_finds_a_torso_sized_garment(self):
        # The wearer's jacket in this wide shot is a few % of the frame.
        # Not empty, not the whole frame.
        fraction = self.mask.mean()
        self.assertGreater(fraction, 0.005)
        self.assertLess(fraction, 0.15)

    def test_the_mask_sits_on_the_upper_body(self):
        # The person stands centred; the jacket's mask centroid should be
        # in the upper-middle of the frame, above the mid-height where the
        # trousers start. Coarse on purpose - a placement sanity check.
        ys, xs = np.nonzero(self.mask)
        height, width = self.mask.shape
        self.assertLess(ys.mean(), height * 0.5, "mask centroid below mid-frame - not the upper body")
        self.assertLess(abs(xs.mean() - width / 2), width * 0.2, "mask centroid far from the centred person")

    def test_most_masked_pixels_are_rewritten(self):
        changed = np.abs(self.clean.astype(int) - self.frame.astype(int)).sum(axis=2) > 10
        self.assertGreater(changed[self.mask].mean(), 0.8)

    def test_the_parser_no_longer_sees_clothing_there(self):
        """Judged by the same parser, the erased region should look far
        less like clothing afterwards. Generous bounds."""
        before = self.remover.clothing_probability(self.frame)[self.mask].mean()
        after = self.remover.clothing_probability(self.clean)[self.mask].mean()
        self.assertGreater(before, 0.3)
        self.assertLess(after, before * 0.5)



class ParserWiringTest(unittest.TestCase):
    """How the clothing probability is read off the parser(s)."""

    class Schp:
        """Fakes SCHPParser: 18 ATR classes as a fixed probability field."""

        def __init__(self):
            rng = np.random.default_rng(0)
            self.maps = rng.random((18, 48, 40)).astype(np.float32)
            self.maps /= self.maps.sum(axis=0, keepdims=True)

        def probabilities(self, frame, flip=False):
            self.flip = flip
            return self.maps

        def fast_clothing(self, frame, classes):
            return self.maps[list(classes)].sum(axis=0), self.maps.argmax(axis=0)

    def test_without_a_parser_it_says_how_to_get_one(self):
        from tryon.removal.remover import ClothingRemover

        with self.assertRaisesRegex(RuntimeError, "setup.sh"):
            ClothingRemover()

    def test_the_probability_is_upper_clothes_plus_dress_in_range(self):
        schp = self.Schp()
        remover = bare_remover(schp)
        remover.refine = "none"
        frame = np.zeros((48, 40, 3), np.uint8)
        probability = remover.clothing_probability(frame)
        np.testing.assert_allclose(probability, schp.maps[4] + schp.maps[7], atol=1e-6)
        self.assertEqual(probability.shape, (48, 40))
        self.assertGreaterEqual(float(probability.min()), 0.0)
        self.assertLessEqual(float(probability.max()), 1.0)
        np.testing.assert_array_equal(remover._last_labels, schp.maps.argmax(axis=0))

    def test_the_flip_average_takes_the_full_path(self):
        schp = self.Schp()
        remover = bare_remover(schp)
        remover.refine, remover.SCHP_FLIP = "none", True
        probability = remover.clothing_probability(np.zeros((48, 40, 3), np.uint8))
        self.assertTrue(schp.flip)
        np.testing.assert_allclose(probability, schp.maps[4] + schp.maps[7], atol=1e-6)

    def test_fashn_sums_its_upper_classes_and_maps_labels_to_atr(self):
        from tryon.removal.parsing_fashn import TO_ATR, UPPER_CLASSES

        class Fashn:
            def __init__(self):
                rng = np.random.default_rng(1)
                self.maps = rng.random((len(TO_ATR), 30, 20)).astype(np.float32)
                self.maps /= self.maps.sum(axis=0, keepdims=True)

            def probabilities(self, frame):
                return self.maps

        fashn = Fashn()
        remover = bare_remover(fashn=fashn)
        remover.refine = "none"
        probability = remover.clothing_probability(np.zeros((30, 20, 3), np.uint8))
        np.testing.assert_allclose(probability, fashn.maps[list(UPPER_CLASSES)].sum(axis=0), atol=1e-6)
        np.testing.assert_array_equal(remover._last_labels, TO_ATR[fashn.maps.argmax(axis=0)])


class MaskJobTest(unittest.TestCase):
    """the live loop runs the mask on a worker thread while the CPU renders."""

    def test_result_is_the_frames_own_parse(self):
        from tryon.app.pipeline import MaskJob

        class Remover:
            def parse_clothing(self, frame, keypoints=None):
                return frame[..., 0] > 100

        frame = np.random.default_rng(1).integers(0, 255, (20, 30, 3), dtype=np.uint8)
        job = MaskJob(Remover(), frame)
        job.start()
        np.testing.assert_array_equal(job.result(), frame[..., 0] > 100)
        self.assertIs(job.frame, frame)

    def test_a_worker_exception_is_raised_by_result(self):
        from tryon.app.pipeline import MaskJob

        class Broken:
            def parse_clothing(self, frame, keypoints=None):
                raise ValueError("boom")

        job = MaskJob(Broken(), np.zeros((4, 4, 3), np.uint8))
        job.start()
        with self.assertRaises(ValueError):
            job.result()

    def test_jobs_chained_with_after_run_one_at_a_time_in_order(self):
        import time

        from tryon.app.pipeline import MaskJob

        log, running, overlap = [], [0], []

        class Remover:
            def parse_clothing(self, frame, keypoints=None):
                running[0] += 1
                overlap.append(running[0])
                time.sleep(0.02 if frame[0, 0, 0] == 0 else 0.0)  # the first is the slowest
                log.append(int(frame[0, 0, 0]))
                running[0] -= 1
                return int(frame[0, 0, 0])

        frames = [np.full((2, 2, 3), i, np.uint8) for i in range(4)]
        jobs, previous = [], None
        for frame in frames:
            previous = MaskJob(Remover(), frame, after=previous)
            previous.start()
            jobs.append(previous)
        self.assertEqual([job.result() for job in jobs], [0, 1, 2, 3])
        self.assertEqual(log, [0, 1, 2, 3])
        self.assertEqual(max(overlap), 1)


class ParseClothingTest(unittest.TestCase):
    """`parse_clothing` returns the parse; `clothing_mask` is it plus the legacy state."""

    class Schp:
        def fast_clothing(self, frame, classes):
            labels = np.zeros(frame.shape[:2], np.uint8)
            labels[frame[..., 0] > 100] = 4
            return (labels == 4).astype(np.float32), labels

    def _remover(self):
        remover = bare_remover(schp=self.Schp())
        remover.dilate_px, remover.refine, remover.tta = 0, "none", False
        return remover

    def test_the_parse_is_returned_and_nothing_is_left_on_the_remover(self):
        remover = self._remover()
        frame = np.zeros((30, 30, 3), np.uint8)
        frame[10:20, 10:20, 0] = 200
        parse = remover.parse_clothing(frame)
        self.assertTrue(parse.mask[10:20, 10:20].all() and not parse.mask[0, 0])
        np.testing.assert_array_equal(parse.labels == 4, parse.mask)
        np.testing.assert_array_equal(parse.clothing, parse.mask)
        self.assertFalse(hasattr(remover, "_last_labels"), "the pure parse shares nothing")

    def test_two_parses_do_not_touch_each_other(self):
        remover = self._remover()
        a, b = np.zeros((30, 30, 3), np.uint8), np.zeros((30, 30, 3), np.uint8)
        a[:10, :, 0] = 200
        b[20:, :, 0] = 200
        parse_a = remover.parse_clothing(a)
        parse_b = remover.parse_clothing(b)
        self.assertTrue(parse_a.mask[:10].all() and not parse_a.mask[20:].any())
        self.assertTrue(parse_b.mask[20:].all() and not parse_b.mask[:10].any())
        self.assertTrue((parse_a.labels[:10] == 4).all(), "frame a's labels survived parsing b")

    def test_clothing_mask_is_the_parses_mask_and_keeps_the_legacy_state(self):
        remover = self._remover()
        frame = np.zeros((30, 30, 3), np.uint8)
        frame[10:20, 10:20, 0] = 200
        mask = remover.clothing_mask(frame)
        np.testing.assert_array_equal(mask, remover.parse_clothing(frame).mask)
        np.testing.assert_array_equal(remover._last_labels == 4, mask)

    def test_remove_and_skin_colour_read_the_given_parse_not_the_last_call(self):
        from tryon.removal.remover import ClothingParse

        remover = self._remover()
        frame = np.full((60, 60, 3), 100, np.uint8)
        frame[20:40, 20:40] = (90, 120, 160)
        face = np.zeros((60, 60), np.uint8)
        face[20:40, 20:40] = 11
        other = np.zeros((60, 60), np.uint8)
        mine = ClothingParse(np.zeros((60, 60), bool), face, np.zeros((60, 60), bool))
        remover._last_labels = other  # another frame's parse, left by the legacy interface
        np.testing.assert_array_equal(remover.skin_colour(frame, mine), [90, 120, 160])
        self.assertIsNone(remover.skin_colour(frame))


class ClothingMaskTest(unittest.TestCase):
    """`clothing_mask`: the thresholded probability, grown by `dilate_px`."""

    def test_the_mask_is_the_thresholded_probability_dilated(self):
        class Schp:
            def fast_clothing(self, frame, classes):
                clothing = np.zeros((60, 60), np.float32)
                clothing[20:40, 20:40] = 0.9
                clothing[45:50, 45:50] = 0.3  # below the threshold
                return clothing, np.zeros((60, 60), np.uint8)

        remover = bare_remover(Schp())
        remover.refine, remover.dilate_px = "none", 3
        mask = remover.clothing_mask(np.zeros((60, 60, 3), np.uint8))
        self.assertTrue(mask[20:40, 20:40].all())
        self.assertTrue(mask[18, 30], "grown by the dilation")
        self.assertFalse(mask[14, 30], "but not further")
        self.assertFalse(mask[45:50, 45:50].any())
        self.assertTrue(remover._last_clothing[20:40, 20:40].all())
        self.assertFalse(remover._last_clothing[18, 30], "the raw clothing is undilated")


class ArmStrokesTest(unittest.TestCase):
    """The keypoint-driven sleeve region of the agnostic mask."""

    def _keypoints(self, visible=1.0):
        points = np.zeros((33, 3), np.float32)
        points[11] = (100, 100, visible)  # left shoulder
        points[12] = (200, 100, visible)  # right shoulder
        points[13] = (60, 160, visible)  # left elbow
        points[14] = (240, 160, visible)  # right elbow
        points[15] = (30, 220, visible)  # left wrist
        points[16] = (270, 220, visible)  # right wrist
        return points

    def test_covers_the_upper_arm_and_extends_past_the_wrist_but_keeps_the_hand(self):
        from tryon.removal.remover import arm_strokes

        mask = arm_strokes(self._keypoints(), (300, 300))
        self.assertTrue(mask[130, 80], "midpoint of the left upper arm should be covered")
        self.assertTrue(mask[190, 45], "midpoint of the left forearm should be covered")
        # extended wrist = elbow + 1.2 * (wrist - elbow) = (24, 232); the
        # hand disc around it is cut back out
        self.assertFalse(mask[232, 24], "the hand itself stays")
        self.assertFalse(mask[100, 150], "the torso between the shoulders is not an arm")
        self.assertGreater(mask.mean(), 0.02)

    def test_an_invisible_arm_is_left_alone(self):
        from tryon.removal.remover import arm_strokes

        points = self._keypoints()
        points[13, 2] = 0.0  # left elbow not seen
        mask = arm_strokes(points, (300, 300))
        self.assertFalse(mask[130, 80])
        self.assertTrue(mask[130, 220], "the visible right arm is still covered")

    def test_no_shoulders_means_no_strokes(self):
        from tryon.removal.remover import arm_strokes

        self.assertFalse(arm_strokes(self._keypoints(visible=0.0), (300, 300)).any())


class HemLeakTest(unittest.TestCase):
    """`hem_leak`: the parser's upper-clothing region below the hip line
    that is really trousers (its own lower-body labels, or "dress" on a
    wearer in trousers) - the round blob below the hem measured on the
    annotated profile camera 0076 (4,865 px below the hips, 3,570 of
    them removed by this; the front camera's mask changes by 6 px)."""

    def _labels(self, dress_blob=True, pants=True):
        labels = np.zeros((100, 60), np.uint8)
        labels[20:60, 15:45] = 4  # upper clothes, hem at row 60
        if pants:
            labels[60:100, 20:40] = 6
        if dress_blob:
            labels[62:75, 22:38] = 7  # the leak: "dress" on the thigh
        labels[56:60, 20:40] = 6  # trousers showing above the hip landmarks
        return labels

    def test_the_blob_and_the_trousers_below_the_hips_are_removed(self):
        from tryon.removal.remover import hem_leak

        keypoints = np.zeros((33, 3))
        keypoints[[23, 24], 1] = 60.0
        keypoints[[23, 24], 2] = 1.0
        leak = hem_leak(self._labels(), keypoints)
        self.assertTrue(leak[62:75, 22:38].all(), "the dress blob below the hips stays")
        self.assertTrue(leak[80:100, 20:40].all(), "trousers below the hips stay")
        self.assertFalse(leak[:61].any(), "nothing at or above the hip line is touched")

    def test_the_hip_line_comes_from_the_parser_without_keypoints(self):
        from tryon.removal.remover import hem_leak

        leak = hem_leak(self._labels(), None)
        self.assertTrue(leak[62:75, 22:38].all())
        self.assertFalse(leak[:56].any(), "the parser's hip line is where its trousers start")

    def test_a_dress_on_a_wearer_without_trousers_is_not_a_leak(self):
        from tryon.removal.remover import hem_leak

        labels = self._labels(dress_blob=False, pants=False)
        labels[20:90, 15:45] = 7  # a real dress, to the knee
        labels[90:100, 20:40] = 12  # legs below it
        leak = hem_leak(labels, None)
        self.assertFalse(leak[labels == 7].any(), "a dress is upper clothing")
        self.assertTrue(leak[92:100, 20:40].all(), "the legs below it are not")

    def test_invisible_hips_fall_back_to_the_parser(self):
        from tryon.removal.remover import hem_leak

        keypoints = np.zeros((33, 3))
        keypoints[[23, 24], 1] = 10.0  # visibility 0: ignored
        self.assertFalse(hem_leak(self._labels(), keypoints)[:56].any())

    def _long_shirt(self):
        """A long shirt over shorts: upper clothes to row 60, its hem labelled dress down to row
        78 at the torso's full width, the shorts and legs beside and below it."""
        labels = np.zeros((120, 60), np.uint8)
        labels[20:60, 15:45] = 4
        labels[60:78, 15:45] = 7  # the hem, below the hip line (row 60)
        labels[60:90, 10:15] = 6  # shorts either side of it, and below
        labels[60:90, 45:50] = 6
        labels[78:90, 15:45] = 6
        labels[90:120, 20:40] = 12
        return labels

    def test_a_long_shirt_hem_below_the_hips_is_not_a_leak(self):
        from tryon.removal.remover import hem_leak

        keypoints = np.zeros((33, 3))
        keypoints[[23, 24], 1] = 60.0
        keypoints[[23, 24], 2] = 1.0
        labels = self._long_shirt()
        leak = hem_leak(labels, keypoints)
        self.assertFalse(leak[labels == 7].any(), "the hem hangs from the torso: it is erased")
        self.assertTrue((labels[61:78, 15:45] == 7).all())
        self.assertTrue(leak[78:90, 15:45].all(), "the shorts below it still are not")
        self.assertTrue(hem_leak(labels, keypoints, hanging=False)[61:78, 15:45].all(), "the rule without the exception drops the hem")

    def test_a_narrow_blob_joined_to_the_torso_is_still_a_leak(self):
        from tryon.removal.remover import hem_leak

        keypoints = np.zeros((33, 3))
        keypoints[[23, 24], 1] = 60.0
        keypoints[[23, 24], 2] = 1.0
        labels = self._labels(dress_blob=False)
        labels[58:75, 27:33] = 7  # a 6 px tail: a fifth of the torso's width
        labels[58:60, 27:33] = 4
        leak = hem_leak(labels, keypoints)
        self.assertTrue(leak[62:75, 27:33].all())


class HemRescueTest(unittest.TestCase):
    """`hem_rescue`: a long shirt's hem the parser labels as LEG (measured on 00152_outer: 4-9k px a
    frame) is given back to the erase by its colour, a real trouser blob under a short jacket (camera 0076) is not."""

    RED, GREY = (30, 30, 200), (150, 140, 130)

    def scene(self, hem_colour):
        frame = np.full((120, 60, 3), (60, 180, 60), np.uint8)
        labels = np.zeros((120, 60), np.uint8)
        frame[20:60, 15:45], labels[20:60, 15:45] = self.RED, 4  # the shirt to the hips (row 60)
        frame[60:85, 15:45] = hem_colour  # below the hip line: parsed as the wearer's legs
        labels[60:85, 15:45] = 12
        mask = np.zeros((120, 60), bool)
        mask[20:85, 12:48] = True
        leak = np.zeros_like(mask)
        leak[61:, :] = np.isin(labels, (12, 13))[61:, :]
        return frame, labels, mask, leak

    def test_a_hem_of_the_garments_colour_below_the_hips_is_given_back(self):
        from tryon.removal.remover import hem_rescue

        frame, labels, mask, leak = self.scene(self.RED)
        rescued = hem_rescue(frame, mask, leak, labels == 4, 60.0)
        self.assertTrue(rescued[62:84, 16:44].all())
        self.assertFalse(rescued[:61].any())

    def test_trousers_under_a_jacket_stay_a_leak(self):
        from tryon.removal.remover import hem_rescue

        frame, labels, mask, leak = self.scene(self.GREY)
        self.assertFalse(hem_rescue(frame, mask, leak, labels == 4, 60.0).any())

    def test_a_match_that_does_not_hang_from_the_mask_stays_a_leak(self):
        from tryon.removal.remover import hem_rescue

        frame, labels, mask, leak = self.scene(self.GREY)
        frame[100:110, 5:15] = self.RED  # red like the shirt, but far from the part of the mask that stays
        labels[100:110, 5:15] = 12
        mask[100:110, 5:15] = True
        leak[100:110, 5:15] = True
        rescued = hem_rescue(frame, mask, leak, labels == 4, 60.0)
        self.assertFalse(rescued[100:110, 5:15].any())

    def test_no_garment_above_the_hips_rescues_nothing(self):
        from tryon.removal.remover import hem_rescue

        frame, labels, mask, leak = self.scene(self.RED)
        self.assertFalse(hem_rescue(frame, mask, leak, np.zeros_like(mask), 60.0).any())


class SkinFillTest(unittest.TestCase):
    """The removed region is filled toward the wearer's skin colour."""

    def test_masked_pixels_move_toward_the_given_skin_colour(self):
        remover = bare_remover()
        frame = np.full((60, 60, 3), (0, 200, 0), np.uint8)  # green
        frame[20:40, 20:40] = (255, 0, 0)  # a blue "garment"
        mask = np.zeros((60, 60), bool)
        mask[20:40, 20:40] = True
        skin = np.array([100, 130, 190])
        out = remover.remove(frame, mask, skin_bgr=skin)
        inside = out[24:36, 24:36].reshape(-1, 3).astype(float).mean(axis=0)
        # closer to the skin colour than to either the garment or the surroundings
        self.assertLess(np.linalg.norm(inside - skin), np.linalg.norm(inside - [255, 0, 0]))
        self.assertLess(np.linalg.norm(inside - skin), np.linalg.norm(inside - [0, 200, 0]))
        np.testing.assert_array_equal(out[~mask], frame[~mask])

    def test_without_a_skin_estimate_it_is_the_plain_inpaint(self):
        remover = bare_remover()
        frame = np.full((40, 40, 3), (0, 200, 0), np.uint8)
        frame[10:30, 10:30] = (255, 0, 0)
        mask = np.zeros((40, 40), bool)
        mask[10:30, 10:30] = True
        self.assertIsNone(remover.skin_colour(frame))  # no parse, no face
        out = remover.remove(frame, mask)
        self.assertLess(int(out[mask][:, 0].max()), 100, "the blue garment is gone")


@unittest.skipUnless((ROOT / "models" / "pretrained" / "big-lama.pt").exists(), "run tools/data/download_pretrained.py lama")
class LamaFillTest(unittest.TestCase):
    """The opt-in LaMa fill: pretrained, so only its wiring is tested -
    the hole is rewritten from the surroundings, nothing else moves."""

    def test_the_hole_is_filled_from_the_surroundings_and_nothing_else_moves(self):
        remover = bare_remover()
        remover.fill = "lama"
        frame = np.full((120, 100, 3), (0, 200, 0), np.uint8)
        frame[40:80, 30:70] = (255, 0, 0)
        mask = np.zeros((120, 100), bool)
        mask[40:80, 30:70] = True
        out = remover.remove(frame, mask)
        self.assertEqual(out.shape, frame.shape)
        self.assertLess(int(out[mask][:, 0].max()), 100, "the blue garment is gone")
        self.assertGreater(float(out[mask][:, 1].mean()), 120, "filled from the green surroundings")
        grown = mask.copy()
        grown[40 - remover.LAMA_MARGIN : 80 + remover.LAMA_MARGIN, 30 - remover.LAMA_MARGIN : 70 + remover.LAMA_MARGIN] = True
        np.testing.assert_array_equal(out[~grown], frame[~grown])

    def test_parsed_skin_survives_the_fill_and_the_background_plate_fills_outside_the_body(self):
        from tryon.removal.background import BackgroundPlate

        remover = bare_remover()
        remover.fill = "lama"
        frame = np.full((120, 100, 3), (0, 200, 0), np.uint8)
        frame[40:80, 30:70] = (255, 0, 0)
        frame[55:65, 45:55] = (120, 150, 210)  # a hand inside the hole
        mask = np.zeros((120, 100), bool)
        mask[40:80, 30:70] = True
        labels = np.zeros((120, 100), np.uint8)
        labels[40:80, 30:70] = 4
        labels[55:65, 45:55] = 14
        remover._last_labels = labels
        remover._last_clothing = labels == 4
        out = remover.remove(frame, mask)
        np.testing.assert_array_equal(out[57:63, 47:53], frame[57:63, 47:53], "the hand is kept")

        # the room behind the left half of the garment was seen earlier
        room = np.full((120, 100, 3), (10, 90, 250), np.uint8)
        plate = BackgroundPlate(margin=1)
        person = np.zeros((120, 100), bool)
        person[:, 55:] = True
        for _ in range(3):
            plate.update(room, person)
        body = np.zeros((120, 100), bool)
        body[:, 50:] = True
        out = remover.remove(frame, mask, plate=plate, body=body)
        np.testing.assert_array_equal(out[40:80, 30:50], room[40:80, 30:50], "outside the body: the observed room")

    def test_arms_are_drawn_back_as_skin_tubes(self):
        remover = bare_remover()
        remover.fill = "lama"
        frame = np.full((120, 100, 3), (0, 200, 0), np.uint8)
        frame[40:80, 30:70] = (255, 0, 0)
        mask = np.zeros((120, 100), bool)
        mask[40:80, 30:70] = True
        arms = np.zeros((120, 100), bool)
        arms[55:65, 20:80] = True  # a horizontal arm through the hole
        skin = np.array([100, 130, 190])
        out = remover.remove(frame, mask, skin_bgr=skin, arms=arms)
        axis = out[60, 35:65].astype(float)  # tube axis: full skin colour
        self.assertLess(np.abs(axis - skin).max(), 40)
        edge = out[56, 35:65].astype(float)  # near the edge: darker skin, still not green
        self.assertLess(edge[:, 0].mean(), axis[:, 0].mean() + 1)
        self.assertLess(edge[:, 1].mean(), 190)
        np.testing.assert_array_equal(out[20, :], frame[20, :])


class CompositeChannelOrderTest(unittest.TestCase):
    """`GarmentRenderer.render` returns BGR + alpha (the rasteriser's
    order); the composite must NOT reverse it. Until 25 Sep 2026 it did,
    and the red plaid 00152_outer was drawn blue in every frame."""

    def test_a_pure_blue_garment_stays_blue_in_the_bgr_frame(self):
        from tryon.app.compositing import composite_garment

        frame = np.zeros((4, 4, 3), np.uint8)
        bgra = np.zeros((4, 4, 4), np.uint8)
        bgra[..., 0] = 255  # blue in BGR
        bgra[..., 3] = 255
        composite_garment(frame, bgra)
        np.testing.assert_array_equal(frame[0, 0], [255, 0, 0])


class InnerLayerTest(unittest.TestCase):
    """`ClothingRemover._inner_layer`: an undershirt under a coat stays."""

    class _Lip:
        def __init__(self, labels):
            self._labels = labels

        def labels(self, frame):
            return self._labels

        fast_labels = labels

    def _remover(self, labels):
        from tryon.removal.remover import ClothingRemover

        remover = ClothingRemover.__new__(ClothingRemover)
        remover.lip = self._Lip(labels)
        return remover

    def test_a_coat_keeps_the_shirt_under_it(self):
        labels = np.zeros((60, 60), np.uint8)
        labels[10:50, 10:50] = 7  # coat
        labels[20:50, 25:35] = 5  # the shirt through its opening
        inner = self._remover(labels)._inner_layer(np.zeros((60, 60, 3), np.uint8))
        self.assertIsNotNone(inner)
        self.assertTrue(inner[30, 30])
        self.assertFalse(inner[15, 15])

    def test_a_single_layer_is_erased_as_before(self):
        labels = np.zeros((60, 60), np.uint8)
        labels[10:50, 10:50] = 5  # a T-shirt and nothing over it
        self.assertIsNone(self._remover(labels)._inner_layer(np.zeros((60, 60, 3), np.uint8)))


class LipRecheckTest(unittest.TestCase):
    """While LIP finds no coat it runs on every LIP_RECHECK-th frame only."""

    def test_no_coat_skips_then_rechecks(self):
        from tryon.removal.remover import ClothingRemover

        calls = []

        class Lip:
            def __init__(self):
                self.labels = np.zeros((20, 20), np.uint8)
                self.labels[5:15, 5:15] = 5  # a shirt, no coat

            def fast_labels(self, frame):
                calls.append(1)
                return self.labels

        remover = ClothingRemover.__new__(ClothingRemover)
        remover.lip = Lip()
        frame = np.zeros((20, 20, 3), np.uint8)
        for _ in range(ClothingRemover.LIP_RECHECK * 2):
            self.assertIsNone(remover._inner_layer(frame))
        self.assertEqual(len(calls), 2)
        # a coat appears: seen at the next check, then every frame
        remover.lip.labels[5:15, 5:15] = 7
        remover.lip.labels[8:12, 8:12] = 5
        results = [remover._inner_layer(frame) for _ in range(ClothingRemover.LIP_RECHECK + 2)]
        self.assertTrue(any(r is not None for r in results))
        first = next(i for i, r in enumerate(results) if r is not None)
        self.assertTrue(all(r is not None for r in results[first:]))


class PlateFillTest(unittest.TestCase):
    """Hole pixels the background plate knows (outside the body) come from
    the plate with either fill; until 7 Oct the default (skin) fill ignored
    the plate the live loop was learning."""

    class Plate:
        def fill(self, frame, hole, body):
            out = frame.copy()
            known = hole & ~body
            out[known] = 200
            return out, hole & body

    def test_the_skin_fill_takes_the_room_from_the_plate(self):
        from tryon.removal.remover import ClothingRemover

        remover = ClothingRemover.__new__(ClothingRemover)
        remover.fill, remover.inpaint_radius = "skin", 3
        frame = np.full((40, 40, 3), 50, np.uint8)
        hole = np.zeros((40, 40), bool)
        hole[10:30, 5:35] = True
        body = np.zeros((40, 40), bool)
        body[:, 15:25] = True
        out = remover.remove(frame, hole, skin_bgr=(90, 120, 160), plate=self.Plate(), body=body)
        self.assertTrue((out[10:30, 5:15] == 200).all(), "the room the plate knows is used")
        self.assertFalse((out[12:28, 17:23] == 200).any(), "the body is filled by the inpaint, not the plate")


class SkinColourTest(unittest.TestCase):
    """`skin_colour`: the median face colour, from the parse just run."""

    def test_the_face_pixels_of_the_last_parse_give_the_colour(self):
        remover = bare_remover()
        frame = np.full((60, 60, 3), (10, 200, 10), np.uint8)
        frame[10:30, 10:30] = (90, 120, 160)
        labels = np.zeros((60, 60), np.uint8)
        labels[10:30, 10:30] = remover.FACE_CLASS
        remover._last_labels = labels
        np.testing.assert_array_equal(remover.skin_colour(frame), [90, 120, 160])

    def test_a_sliver_of_face_is_not_enough(self):
        remover = bare_remover()
        labels = np.zeros((60, 60), np.uint8)
        labels[10:12, 10:12] = remover.FACE_CLASS
        remover._last_labels = labels
        self.assertIsNone(remover.skin_colour(np.zeros((60, 60, 3), np.uint8)))


class CroppedWorkTest(unittest.TestCase):
    """The crops that replaced full-frame work give the full-frame answer."""

    @staticmethod
    def scene():
        import cv2

        rng = np.random.default_rng(4)
        frame = cv2.GaussianBlur(rng.integers(0, 255, (160, 200, 3), np.uint8), (7, 7), 2)
        probability = np.zeros((160, 200), np.float32)
        cv2.circle(probability, (70, 60), 25, 1.0, -1)
        return frame, cv2.GaussianBlur(probability, (9, 9), 3)

    def test_guided_refine_on_a_crop_matches_the_full_frame(self):
        from tryon.removal.remover import _guided_refine, _guided_refine_full

        frame, probability = self.scene()
        full, crop = _guided_refine_full(frame, probability), _guided_refine(frame, probability)
        seeded = probability > 0.05
        np.testing.assert_allclose(crop[seeded], full[seeded], atol=1e-6)
        np.testing.assert_array_equal(crop > 0.5, full > 0.5)
        self.assertLess(np.abs(crop - full).max(), 0.05)

    def test_guided_refine_of_an_empty_map_is_empty(self):
        from tryon.removal.remover import _guided_refine

        frame, _ = self.scene()
        self.assertFalse(_guided_refine(frame, np.zeros((160, 200), np.float32)).any())

    def test_seam_ring_inpaint_on_a_crop_matches_the_full_frame(self):
        import cv2

        remover = bare_remover()
        frame, _ = self.scene()
        ring = np.zeros((160, 200), bool)
        ring[40:44, 30:120] = True
        ring[40:100, 60:63] = True
        full = cv2.inpaint(frame, ring.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        np.testing.assert_array_equal(remover._inpaint_ring(frame, ring), full)
        np.testing.assert_array_equal(remover._inpaint_ring(frame, np.zeros_like(ring)), frame)


class LabelFillTest(unittest.TestCase):
    """The label-aware fill (`LABEL_FILL`): a hole is filled from the nearest source of its own kind."""

    GREEN, BLUE, SKIN, RED = (60, 180, 60), (180, 90, 40), (120, 150, 210), (0, 0, 220)

    @classmethod
    def setUpClass(cls):
        cls.remover = bare_remover()
        cls.remover.LABEL_FILL = True

    def setUp(self):
        from tryon.removal.remover import ClothingRemover

        ClothingRemover._rng = None

    def scene(self):
        """Room on the left half, trousers (label 6) on the right, a red shirt hole across the seam, an arm (14)."""
        frame = np.zeros((120, 160, 3), np.uint8)
        labels = np.zeros((120, 160), np.uint8)
        frame[:] = self.GREEN
        frame[:, 80:] = self.BLUE
        labels[:, 80:] = 6
        mask = np.zeros((120, 160), bool)
        mask[30:90, 50:110] = True
        frame[mask] = self.RED
        labels[mask] = 4
        return frame, labels, mask

    def fill(self, frame, labels, mask, **kwargs):
        from tryon.removal.remover import ClothingParse

        return self.remover.remove(frame, mask, parse=ClothingParse(mask, labels, mask.copy()), **kwargs)

    def test_a_hole_across_two_regions_takes_each_regions_colour_not_a_blend(self):
        frame, labels, mask = self.scene()
        out = self.fill(frame, labels, mask).astype(float)
        rows = slice(55, 65)
        left, right = out[rows, 55:70].reshape(-1, 3), out[rows, 95:105].reshape(-1, 3)
        np.testing.assert_allclose(left.mean(axis=0), self.GREEN, atol=12)
        np.testing.assert_allclose(right.mean(axis=0), self.BLUE, atol=12)
        # no filled pixel is far from both colours (a blend of the two would be)
        filled = out[mask]
        nearest = np.minimum(np.abs(filled - self.GREEN).sum(axis=1), np.abs(filled - self.BLUE).sum(axis=1))
        self.assertLess(np.percentile(nearest, 75), 40)

    def test_pixels_outside_the_hole_are_untouched(self):
        frame, labels, mask = self.scene()
        out = self.fill(frame, labels, mask)
        np.testing.assert_array_equal(out[~mask], frame[~mask])

    def test_the_arm_tube_is_skin_coloured_with_the_visible_arms_grain(self):
        frame, labels, mask = self.scene()
        rng = np.random.default_rng(1)
        arm = np.zeros_like(mask)
        arm[8:20, 10:150] = True  # a bare arm across the top
        frame[arm] = np.clip(np.array(self.SKIN) + rng.normal(0, 4, (int(arm.sum()), 3)), 0, 255).astype(np.uint8)
        labels[arm] = 14
        tube_hole = np.zeros_like(mask)
        tube_hole[8:20, 55:105] = True  # the middle of it erased
        tube = tube_hole.copy()
        out = self.fill(frame, labels, tube_hole, arms=tube)
        patch = out[11:17, 60:100].reshape(-1, 3).astype(float)
        np.testing.assert_allclose(patch.mean(axis=0), self.SKIN, atol=20)
        self.assertGreater(patch.std(axis=0).mean(), 0.5, "the fill is flat")
        self.assertLess(patch.std(axis=0).mean(), 12)

    def test_an_undershirt_is_continued_not_the_necks_skin(self):
        frame, labels, mask = self.scene()
        frame[:, :] = self.GREEN
        labels[:] = 0
        labels[mask] = 4
        neck = np.zeros_like(mask)
        neck[10:28, 70:90] = True
        frame[neck] = self.SKIN
        labels[neck] = 11
        under = np.zeros_like(mask)
        under[92:118, 55:105] = True  # the kept inner layer, below the hole (label 4, not in the mask)
        frame[under] = (240, 240, 240)
        labels[under] = 4
        body = np.zeros_like(mask)
        body[10:118, 45:115] = True
        out = self.fill(frame, labels, mask, body=body).astype(float)
        np.testing.assert_allclose(out[32:40, 70:90].reshape(-1, 3).mean(axis=0), (240, 240, 240), atol=30)

    def test_the_arm_tube_narrows_from_the_shoulder_to_the_wrist(self):
        from tryon.removal.remover import arm_tube

        keypoints = np.zeros((33, 3))
        keypoints[11] = (150, 20, 1)
        keypoints[12] = (50, 20, 1)  # shoulder width 100
        keypoints[13] = (150, 90, 1)
        keypoints[15] = (150, 160, 1)
        keypoints[[14, 16], 2] = 0.0  # the other arm is not seen: no tube
        mask, radius = arm_tube(keypoints, (200, 200), width_fraction=0.3)
        self.assertFalse(mask[:, :100].any(), "an arm that is not seen has no tube")
        near_shoulder, near_wrist = mask[30, :].sum(), mask[155, :].sum()
        self.assertGreater(near_shoulder, 1.25 * near_wrist)
        self.assertAlmostEqual(float(radius[30, 150]), 15.0 * (1 - 0.1 * 10 / 70), delta=1.5)

    def test_the_parsers_real_arm_skin_is_not_refilled_but_a_white_cuff_labelled_arm_is(self):
        frame, labels, mask = self.scene()
        frame[:] = self.GREEN
        labels[:] = 0
        face = np.zeros_like(mask)
        face[2:16, 70:90] = True
        frame[face], labels[face] = self.SKIN, 11
        skin = np.zeros_like(mask)
        skin[40:50, 52:80] = True  # real arm skin, inside the hole
        cuff = np.zeros_like(mask)
        cuff[70:80, 52:80] = True  # a white cuff the parser called arm
        frame[skin], labels[skin] = (110, 140, 200), 14
        frame[cuff], labels[cuff] = (240, 240, 240), 15
        labels[mask & ~skin & ~cuff] = 4
        from tryon.removal.remover import ClothingParse

        clothing = mask & ~skin & ~cuff
        out = self.remover.remove(frame, mask, parse=ClothingParse(mask, labels, clothing))
        np.testing.assert_array_equal(out[43:47, 58:74], frame[43:47, 58:74])
        self.assertTrue((np.abs(out[73:77, 58:74].astype(int) - (240, 240, 240)).sum(axis=-1) > 60).all(), "the white cuff stays erased")

    def test_the_torso_is_the_faces_colour_not_the_arms(self):
        frame, labels, mask = self.scene()
        frame[:] = self.GREEN
        labels[:] = 0
        face = np.zeros_like(mask)
        face[10:30, 70:90] = True
        arm = np.zeros_like(mask)
        arm[30:90, 38:49] = True  # a browner arm beside the torso
        pale, brown = (150, 170, 220), (70, 110, 160)
        frame[face], labels[face] = pale, 11
        frame[arm], labels[arm] = brown, 14
        labels[mask] = 4
        body = np.zeros_like(mask)
        body[20:100, 45:115] = True
        out = self.fill(frame, labels, mask, body=body).astype(float)
        np.testing.assert_allclose(out[32:38, 72:88].reshape(-1, 3).mean(axis=0), pale, atol=25)

    def test_nearest_source_matches_brute_force(self):
        rng = np.random.default_rng(3)
        sources = rng.random((40, 50)) > 0.97
        ny, nx = type(self.remover)._nearest_source(sources)
        ys, xs = np.nonzero(sources)
        for y, x in [(0, 0), (20, 25), (39, 49), (10, 40)]:
            distance = (ys - y) ** 2 + (xs - x) ** 2
            self.assertEqual((ny[y, x] - y) ** 2 + (nx[y, x] - x) ** 2, distance.min())

    def test_the_default_fill_is_unchanged_when_the_label_fill_is_off(self):
        frame, labels, mask = self.scene()
        from tryon.removal.remover import ClothingParse

        off = bare_remover()
        off.LABEL_FILL = False
        a = off.remove(frame, mask, parse=ClothingParse(mask, labels, mask.copy()))
        b = off.remove(frame, mask)
        np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
