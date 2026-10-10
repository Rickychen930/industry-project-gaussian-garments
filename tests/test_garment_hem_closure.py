"""Long-range attachment of the hem (cloth.py) and the front closure of open-front garments (skinning.py)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tryon.garment.cloth import LightCloth, _long_range, geodesic_from
from tryon.garment.renderer import GarmentRenderer
from tryon.garment.skinning import SkinnedGarment, closed_front_flag, front_closure, piece_labels


def _chain(count=6, spacing=1.0):
    """Vertices on a line along y, joined into a path; vertex 0 is the anchor."""
    rest = np.stack([np.zeros(count), -spacing * np.arange(count), np.zeros(count)], axis=1)
    edges = np.stack([np.arange(count - 1), np.arange(1, count)], axis=1)
    return rest, edges


class GeodesicTest(unittest.TestCase):
    def test_distance_along_the_edges_not_the_straight_line(self):
        # an L: 0 - 1 - 2 along y, then 2 - 3 along x; vertex 3 is 2.828 away in a line, 3 along the edges
        rest = np.array([[0, 0, 0], [0, -1, 0], [0, -2, 0], [1, -2, 0]], float)
        edges = np.array([[0, 1], [1, 2], [2, 3]])
        distance, anchor_of = geodesic_from(4, edges, rest, [0])
        np.testing.assert_allclose(distance, [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(anchor_of, [0, 0, 0, 0])

    def test_each_vertex_measures_from_its_nearest_anchor(self):
        rest, edges = _chain(5)
        distance, anchor_of = geodesic_from(5, edges, rest, [0, 4])
        np.testing.assert_allclose(distance, [0.0, 1.0, 2.0, 1.0, 0.0])
        np.testing.assert_array_equal(anchor_of, [0, 0, 0, 4, 4])

    def test_a_piece_without_an_anchor_is_unreachable(self):
        rest = np.array([[0, 0, 0], [0, -1, 0], [5, 0, 0], [5, -1, 0]], float)
        distance, anchor_of = geodesic_from(4, np.array([[0, 1], [2, 3]]), rest, [0])
        self.assertTrue(np.isinf(distance[2]) and np.isinf(distance[3]))
        np.testing.assert_array_equal(anchor_of, [0, 0, -1, -1])


class LongRangeTest(unittest.TestCase):
    def test_vertices_end_within_the_geodesic_bound_and_unstretched_ones_do_not_move(self):
        rest, edges = _chain(6)
        distance, anchor_of = geodesic_from(6, edges, rest, [0])
        x = rest.copy()
        x[4] += [0.0, -2.0, 0.0]  # pulled 2 m further down than its rest path allows
        out = _long_range(x, anchor_of, distance)
        self.assertLessEqual(np.linalg.norm(out[4] - out[0]), distance[4] + 1e-12)
        np.testing.assert_array_equal(out[[0, 1, 2, 3, 5]], x[[0, 1, 2, 3, 5]])
        self.assertTrue(np.array_equal(x[4], rest[4] + [0.0, -2.0, 0.0]))  # the input is not touched

    def test_a_raised_anchor_pulls_the_hem_up_and_a_lowered_one_does_nothing(self):
        rest, edges = _chain(4)
        distance, anchor_of = geodesic_from(4, edges, rest, [0])
        lowered = rest + [0.0, -0.5, 0.0]
        lowered[0] = rest[0] + [0.0, -1.0, 0.0]  # the anchor drops 1 m, the rest follows 0.5 m
        np.testing.assert_array_equal(_long_range(lowered, anchor_of, distance), lowered)
        raised = rest.copy()
        raised[0] += [0.0, 1.0, 0.0]  # the anchor rises, the others stay
        out = _long_range(raised, anchor_of, distance)
        self.assertGreater(out[3, 1], raised[3, 1])  # the hem was pulled up
        self.assertAlmostEqual(float(np.linalg.norm(out[3] - out[0])), distance[3])

    def test_the_cloth_applies_it_only_when_lra_is_set(self):
        rest, edges = _chain(4)
        distance, anchor_of = geodesic_from(4, edges, rest, [0])
        cloth = LightCloth()
        cloth.LRA = None
        cloth.set_long_range(anchor_of, distance)
        far = rest + [0.0, -3.0, 0.0]
        far[0] = rest[0]
        np.testing.assert_array_equal(cloth._attach(far), far)  # LRA is None: off
        cloth.LRA = 0.1
        out = cloth._attach(far)
        self.assertLessEqual(np.linalg.norm(out[3] - out[0]), 1.1 * distance[3] + LightCloth.LRA_MARGIN + 1e-9)


def _open_jacket():
    """A back piece (with 'sleeves' out to |x| = 0.7) and two front panels with their inner edges at |x| = 0.05."""
    faces, vertices = [], []

    def grid(x0, x1, nx=8, ny=6):
        base = len(vertices)
        for j in range(ny):
            for i in range(nx):
                vertices.append([x0 + (x1 - x0) * i / (nx - 1), -0.05 * j, 0.0])
        for j in range(ny - 1):
            for i in range(nx - 1):
                a = base + j * nx + i
                faces.extend([[a, a + 1, a + nx], [a + 1, a + nx + 1, a + nx]])

    grid(-0.7, 0.7, 24, 6)  # back + sleeves
    grid(0.05, 0.7)  # one front, inner edge at +0.05
    grid(-0.7, -0.05)  # the other, inner edge at -0.05
    return np.array(vertices), np.array(faces)


class FrontClosureTest(unittest.TestCase):
    def test_the_inner_edges_meet_and_the_far_side_stays(self):
        rest, faces = _open_jacket()
        moved = rest + front_closure(rest, faces, seam=0.01, blend=0.2)
        labels = piece_labels(faces, len(rest))
        left, right = labels == labels[len(rest) - 1], labels == labels[24 * 6 + 1]
        self.assertAlmostEqual(float(moved[right, 0].min()), 0.005, places=6)
        self.assertAlmostEqual(float(moved[left, 0].max()), -0.005, places=6)
        # the panel far from its edge (blend = 0.2 m) and the back piece are where they were
        np.testing.assert_array_equal(moved[rest[:, 0] > 0.3], rest[rest[:, 0] > 0.3])
        back = labels == labels[0]
        np.testing.assert_array_equal(moved[back], rest[back])
        # nothing moves up, down or in depth
        np.testing.assert_array_equal(moved[:, 1:], rest[:, 1:])

    def test_the_move_fades_out_monotonically_across_the_panel(self):
        rest, faces = _open_jacket()
        shift = -front_closure(rest, faces)[:, 0]
        row = np.flatnonzero((rest[:, 1] == 0.0) & (rest[:, 0] > 0.04) & (np.arange(len(rest)) >= 24 * 6))
        row = row[np.argsort(rest[row, 0])]
        self.assertTrue(np.all(np.diff(shift[row]) <= 1e-12))
        self.assertGreater(shift[row[0]], 0.0)
        self.assertEqual(shift[row[-1]], 0.0)

    def test_a_closed_garment_is_the_identity(self):
        # a back and a front piece, no two panels on opposite sides of the midline
        rest, faces = _open_jacket()
        keep = piece_labels(faces, len(rest)) != piece_labels(faces, len(rest))[-1]
        index = np.cumsum(keep) - 1
        mask = keep[faces].all(axis=1)
        np.testing.assert_array_equal(front_closure(rest[keep], index[faces[mask]]), np.zeros((int(keep.sum()), 3)))

    def test_panels_that_already_overlap_do_not_move(self):
        rest, faces = _open_jacket()
        rest[:, 0] = np.where(np.abs(rest[:, 0]) < 0.3, rest[:, 0] * 0.0 + np.sign(rest[:, 0]) * 0.001, rest[:, 0])
        displacement = front_closure(rest, faces, seam=0.01)
        self.assertLessEqual(float(np.abs(displacement).max()), 0.0045)  # at most the half seam

    def test_moved_carries_the_displacement_into_the_physical_rest(self):
        rest = np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90 degrees about z
        garment = SkinnedGarment(rest, np.ones((2, 1)), physical_rest=rest @ rotation.T)
        garment.bind_linear = np.broadcast_to(rotation, (2, 3, 3)).copy()
        offset = np.array([[-0.05, 0.0, 0.0], [0.0, 0.0, 0.0]])
        moved = garment.moved(offset)
        np.testing.assert_allclose(moved.rest[0], [0.05, 0.0, 0.0])
        np.testing.assert_allclose(moved.physical_rest[0], rotation @ [0.05, 0.0, 0.0])
        np.testing.assert_array_equal(garment.rest, rest)  # the original is untouched
        np.testing.assert_array_equal(garment.moved(np.zeros((2, 3))).rest, garment.rest)  # zero offset: identity
        with self.assertRaises(ValueError):
            garment.moved(np.zeros((3, 3)))


class SleeveAnchorsTest(unittest.TestCase):
    def test_the_anchors_are_the_sleeve_side_of_the_armhole_and_only_the_body_is_limited(self):
        from types import SimpleNamespace

        rest, edges = _chain(6)  # 0, 1 sleeve | 2 .. 5 body panel hanging below
        renderer = object.__new__(GarmentRenderer)
        renderer._binding = SimpleNamespace(physical_rest=rest)
        renderer.sleeve_vertices = np.array([True, True, False, False, False, False])
        renderer.scan_binder = SimpleNamespace(edges=edges)
        anchor_of, distance = renderer._long_range_anchors()
        np.testing.assert_array_equal(anchor_of, [-1, -1, 1, 1, 1, 1])
        np.testing.assert_allclose(distance[2:], [1.0, 2.0, 3.0, 4.0])
        self.assertIs(renderer._long_range_anchors()[0], anchor_of)  # once per binding


class ClosedFrontFlagTest(unittest.TestCase):
    def _flag(self, text=None):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        if text is not None:
            Path(folder.name, "stage1").mkdir()
            Path(folder.name, "stage1", "closed_front.json").write_text(text)
        return Path(folder.name)

    def test_an_absent_file_is_off(self):
        self.assertFalse(closed_front_flag(self._flag()))

    def test_true_and_the_object_form_are_on_false_is_off(self):
        self.assertTrue(closed_front_flag(self._flag("true")))
        self.assertTrue(closed_front_flag(self._flag(json.dumps({"closed": True, "note": "worn closed"}))))
        self.assertFalse(closed_front_flag(self._flag(json.dumps({"closed": False}))))

    def test_the_repository_flag_wins_over_the_checkpoint_file_and_names_only_00136_outer(self):
        # the repo asset is keyed by the checkpoint folder's name
        self.assertTrue(closed_front_flag(Path("/nonexistent") / "00136_outer"))
        self.assertFalse(closed_front_flag(Path("/nonexistent") / "00152_outer"))
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        garment = Path(folder.name) / "00136_outer"
        (garment / "stage1").mkdir(parents=True)
        (garment / "stage1" / "closed_front.json").write_text("false")
        self.assertTrue(closed_front_flag(garment))  # the asset says closed, the local file is ignored

    def test_a_malformed_file_raises_instead_of_turning_the_closure_off(self):
        with self.assertRaises(ValueError):
            closed_front_flag(self._flag("{not json"))
        with self.assertRaises(ValueError):
            closed_front_flag(self._flag('"yes"'))

    def test_the_renderer_closes_only_with_the_flag_or_when_forced(self):
        renderer = object.__new__(GarmentRenderer)
        renderer.checkpoint_dir = self._flag()
        self.assertEqual(GarmentRenderer.CLOSE_FRONT, "auto")
        self.assertFalse(renderer.closes_front())
        renderer.checkpoint_dir = self._flag("true")
        self.assertTrue(renderer.closes_front())
        renderer.CLOSE_FRONT = None
        self.assertFalse(renderer.closes_front())
        renderer.CLOSE_FRONT = "always"
        renderer.checkpoint_dir = self._flag()
        self.assertTrue(renderer.closes_front())


if __name__ == "__main__":
    unittest.main()
