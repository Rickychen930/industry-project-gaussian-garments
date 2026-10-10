"""`_torn_faces` (numba) answers as the numpy form it replaced."""

import unittest

import numpy as np

from tryon.garment.renderer import _torn_faces


def _numpy_form(vertices, faces, limits, seam):
    torn = np.zeros(len(faces), dtype=bool)
    for (a, b), limit in zip(((0, 1), (1, 2), (2, 0)), limits):
        torn |= np.linalg.norm(vertices[faces[:, b]] - vertices[faces[:, a]], axis=1) > limit
    if seam is not None:
        torn |= seam
    return torn


class TornFacesKernelTest(unittest.TestCase):
    def test_same_faces_as_the_numpy_form(self):
        rng = np.random.default_rng(0)
        vertices = rng.normal(size=(500, 3))
        faces = rng.integers(0, 500, (3000, 3))
        rest = rng.normal(size=(500, 3))
        limits = [np.linalg.norm(rest[faces[:, b]] - rest[faces[:, a]], axis=1) * 2.0 for a, b in ((0, 1), (1, 2), (2, 0))]
        seam = rng.random(3000) < 0.05
        for use_seam in (None, seam):
            want = _numpy_form(vertices, faces, limits, use_seam)
            got = _torn_faces(vertices, faces, limits[0], limits[1], limits[2], np.zeros(3000, bool) if use_seam is None else use_seam)
            np.testing.assert_array_equal(got, want)
            self.assertTrue(0 < want.sum() < 3000)


if __name__ == "__main__":
    unittest.main()
