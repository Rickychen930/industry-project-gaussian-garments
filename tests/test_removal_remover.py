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
    """live.py runs the mask on a worker thread while the CPU renders."""

    def test_result_matches_the_direct_call(self):
        from tryon.app.pipeline import MaskJob

        class Remover:
            def clothing_mask(self, frame, keypoints=None):
                return frame[..., 0] > 100

        frame = np.random.default_rng(1).integers(0, 255, (20, 30, 3), dtype=np.uint8)
        job = MaskJob(Remover(), frame)
        job.start()
        np.testing.assert_array_equal(job.result(), frame[..., 0] > 100)

    def test_a_worker_exception_is_raised_by_result(self):
        from tryon.app.pipeline import MaskJob

        class Broken:
            def clothing_mask(self, frame, keypoints=None):
                raise ValueError("boom")

        job = MaskJob(Broken(), np.zeros((4, 4, 3), np.uint8))
        job.start()
        with self.assertRaises(ValueError):
            job.result()


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


if __name__ == "__main__":
    unittest.main()
