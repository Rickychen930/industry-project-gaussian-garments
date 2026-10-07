"""Body collision (tryon/garment/collision.py): points end at least the margin
outside the body, and the body's normals point outward."""

from __future__ import annotations

import unittest

import numpy as np


def _sphere(radius=1.0, subdivisions=3):
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces)


class BodyNormalsTest(unittest.TestCase):
    def test_a_sphere_s_normals_point_outward(self):
        from tryon.garment.collision import body_vertex_normals

        vertices, faces = _sphere()
        normals = body_vertex_normals(vertices, faces)
        radial = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
        self.assertGreater(float(np.einsum("ij,ij->i", normals, radial).min()), 0.99)


    def test_the_compiled_normals_equal_the_numpy_original_bit_for_bit(self):
        from tryon.garment.collision import _body_vertex_normals_numpy, body_vertex_normals

        vertices, faces = _sphere()
        rng = np.random.default_rng(0)
        for _ in range(3):  # a new pose, the same faces (the corner table is reused)
            vertices = vertices + rng.normal(0, 0.02, vertices.shape)
            np.testing.assert_array_equal(body_vertex_normals(vertices, faces), _body_vertex_normals_numpy(vertices, faces))
        other = faces[rng.permutation(len(faces))]  # other topology: the table is rebuilt
        np.testing.assert_array_equal(body_vertex_normals(vertices, other), _body_vertex_normals_numpy(vertices, other))


class PushOutOfBodyTest(unittest.TestCase):
    def test_every_point_ends_at_least_the_margin_outside(self):
        from tryon.garment.collision import BodySurface, push_out_of_body

        vertices, faces = _sphere()
        body = BodySurface(vertices, faces)
        rng = np.random.default_rng(0)
        directions = rng.normal(size=(500, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        points = directions * rng.uniform(0.6, 1.3, size=(500, 1))  # inside, on and outside
        margin = 0.02
        outside_before = np.linalg.norm(points, axis=1) > 1.0 + margin + 0.05
        kept = points[outside_before].copy()
        pushed = push_out_of_body(points, body, margin, passes=10)
        nearest, normals = body.nearest(pushed)
        signed = np.einsum("ij,ij->i", pushed - nearest, normals)
        self.assertGreaterEqual(float(signed.min()), margin - 1e-9)
        # points already well outside are left exactly where they were
        np.testing.assert_array_equal(pushed[outside_before], kept)


if __name__ == "__main__":
    unittest.main()
