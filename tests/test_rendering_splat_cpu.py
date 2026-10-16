"""Tests for the hand-written Gaussian splat renderer: projection,
covariance propagation, SH colour and depth-ordered compositing.

Each test states a geometric property, so a failure names the maths that broke.
"""

import unittest

import numpy as np

from tryon.rendering.splat_cpu import (
    face_frames,
    face_scaled_log_scaling,
    look_at,
    _project_points_numpy,
    project_points,
    quaternion_to_rotation,
    render_splats,
    sh_to_rgb,
    world_positions,
)


class QuaternionTests(unittest.TestCase):
    def test_identity_quaternion_is_the_identity_rotation(self):
        matrix = quaternion_to_rotation(np.array([[1.0, 0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(matrix[0], np.eye(3), atol=1e-12)

    def test_ninety_degrees_about_z_maps_x_onto_y(self):
        # wxyz convention, matching the checkpoint's stored rotations.
        half = np.sqrt(0.5)
        matrix = quaternion_to_rotation(np.array([[half, 0.0, 0.0, half]]))[0]
        np.testing.assert_allclose(matrix @ [1, 0, 0], [0, 1, 0], atol=1e-9)

    def test_unnormalised_quaternions_are_normalised_first(self):
        scaled = quaternion_to_rotation(np.array([[2.0, 0.0, 0.0, 0.0]]))[0]
        np.testing.assert_allclose(scaled, np.eye(3), atol=1e-12)

    def test_rotations_are_orthonormal(self):
        rng = np.random.default_rng(0)
        matrices = quaternion_to_rotation(rng.normal(size=(16, 4)))
        for matrix in matrices:
            np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-9)
            self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=9)


class WorldPositionTests(unittest.TestCase):
    """Stored `xyz` are offsets in each face's tangent frame; world position
    is `R_face @ local * scale_face + centre_face` (`MeshGaussianModel.get_xyz`)."""

    def _unit_triangle(self):
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float64
        )
        return vertices, np.array([[0, 1, 2]])

    def test_a_zero_offset_lands_on_the_face_centre(self):
        vertices, faces = self._unit_triangle()
        world = world_positions(
            np.zeros((1, 3)), np.array([0]), vertices, faces
        )
        np.testing.assert_allclose(world[0], vertices.mean(axis=0), atol=1e-12)

    def test_offsets_move_along_the_face_frame_not_the_world_axes(self):
        """A rotated triangle carries its Gaussians: the same offset gives a different world point."""
        vertices, faces = self._unit_triangle()
        rotated = vertices @ np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        ).T
        offset = np.array([[0.3, 0.0, 0.0]])
        flat = world_positions(offset, np.array([0]), vertices, faces)
        turned = world_positions(offset, np.array([0]), rotated, faces)
        self.assertGreater(np.linalg.norm(flat[0] - turned[0]), 0.1)

    def test_a_larger_face_scales_the_offset(self):
        vertices, faces = self._unit_triangle()
        offset = np.array([[0.25, 0.0, 0.0]])
        small = world_positions(offset, np.array([0]), vertices, faces)
        large = world_positions(offset, np.array([0]), vertices * 3.0, faces)
        small_radius = np.linalg.norm(small[0] - vertices.mean(axis=0))
        large_radius = np.linalg.norm(large[0] - (vertices * 3.0).mean(axis=0))
        self.assertAlmostEqual(large_radius / small_radius, 3.0, places=6)


class FaceScalingTests(unittest.TestCase):
    """Stored `scaling` is relative to the bound face and must be multiplied
    by that face's scale (`MeshGaussianModel.remember_scaling`); skipping it
    made splats ~90x too large."""

    def _unit_triangle(self):
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float64
        )
        return vertices, np.array([[0, 1, 2]])

    def test_scaling_is_taken_relative_to_the_bound_face(self):
        vertices, faces = self._unit_triangle()
        log_scaling = np.log(np.full((1, 3), 0.5))
        scaled = face_scaled_log_scaling(
            log_scaling, np.array([0]), vertices, faces
        )
        _, _, scale = face_frames(vertices, faces)
        np.testing.assert_allclose(
            np.exp(scaled[0]), 0.5 * scale[0], rtol=1e-9
        )

    def test_a_bigger_face_carries_bigger_gaussians(self):
        vertices, faces = self._unit_triangle()
        log_scaling = np.log(np.full((1, 3), 0.5))
        small = face_scaled_log_scaling(log_scaling, np.array([0]), vertices, faces)
        large = face_scaled_log_scaling(
            log_scaling, np.array([0]), vertices * 4.0, faces
        )
        ratio = float(np.exp(large[0, 0]) / np.exp(small[0, 0]))
        self.assertAlmostEqual(ratio, 4.0, places=6)






class SplatCoverageTests(unittest.TestCase):
    """A trained garment has fewer Gaussians than pixels, so stored-size splats
    leave gaps and a grainy look; a 1.3x dilation is the default trade of
    coverage against blur."""

    def test_dilation_is_applied_in_log_space(self):
        from tryon.rendering.splat_cpu import face_frames, face_scaled_log_scaling

        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float64
        )
        faces = np.array([[0, 1, 2]])
        log_scaling = np.log(np.full((1, 3), 0.5))
        plain = face_scaled_log_scaling(log_scaling, np.array([0]), vertices, faces)
        grown = face_scaled_log_scaling(
            log_scaling, np.array([0]), vertices, faces, dilate=1.3
        )
        np.testing.assert_allclose(
            np.exp(grown[0]), np.exp(plain[0]) * 1.3, rtol=1e-9
        )
        # And the un-dilated call must be unchanged.
        _, _, scale = face_frames(vertices, faces)
        np.testing.assert_allclose(np.exp(plain[0]), 0.5 * scale[0], rtol=1e-9)

    def test_dilation_covers_more_of_the_gap_between_splats(self):
        from tryon.rendering.splat_cpu import look_at, render_splats

        # A sparse grid of small splats, as a real garment's are: at stored
        # size they leave holes, dilated they close.
        grid = np.stack(
            np.meshgrid(
                np.linspace(-0.2, 0.2, 14), np.linspace(-0.2, 0.2, 14), [0.0]
            ),
            axis=-1,
        ).reshape(-1, 3)
        base = {
            "xyz": grid,
            "rotation": np.tile([1.0, 0.0, 0.0, 0.0], (len(grid), 1)),
            "opacity": np.full((len(grid), 1), 6.0),
            "feature_dc": np.full((len(grid), 1, 3), 0.6),
            "feature_rest": np.zeros((len(grid), 15, 3)),
        }
        view = look_at(np.array([0.0, 0.0, 1.2]), np.zeros(3))
        intrinsics = np.array([[300.0, 0, 64.0], [0, 300.0, 64.0], [0, 0, 1.0]])

        def thin_fraction(dilate):
            base["scaling"] = np.log(np.full((len(grid), 3), 0.006 * dilate))
            _, alpha = render_splats(base, view, intrinsics, (128, 128))
            covered = alpha > 0.02
            if not covered.any():
                return 1.0
            rows = np.flatnonzero(covered.any(axis=1))
            cols = np.flatnonzero(covered.any(axis=0))
            box = alpha[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
            return float(np.mean(box < 0.9))

        self.assertLess(thin_fraction(1.3), thin_fraction(1.0))


class ProjectionTests(unittest.TestCase):
    def test_a_point_on_the_axis_projects_to_the_principal_point(self):
        intrinsics = np.array([[100.0, 0, 64.0], [0, 100.0, 48.0], [0, 0, 1.0]])
        pixels, depth, _ = project_points(
            np.array([[0.0, 0.0, 2.0]]), np.eye(4), intrinsics
        )
        np.testing.assert_allclose(pixels[0], [64.0, 48.0], atol=1e-9)
        self.assertAlmostEqual(float(depth[0]), 2.0, places=9)

    def test_further_points_project_closer_to_the_centre(self):
        """Perspective: the same lateral offset subtends a smaller angle further away."""
        intrinsics = np.array([[100.0, 0, 64.0], [0, 100.0, 48.0], [0, 0, 1.0]])
        near, _, _ = project_points(
            np.array([[0.5, 0.0, 1.0]]), np.eye(4), intrinsics
        )
        far, _, _ = project_points(
            np.array([[0.5, 0.0, 4.0]]), np.eye(4), intrinsics
        )
        self.assertGreater(near[0, 0] - 64.0, far[0, 0] - 64.0)

    def test_points_behind_the_camera_are_reported_invalid(self):
        intrinsics = np.array([[100.0, 0, 64.0], [0, 100.0, 48.0], [0, 0, 1.0]])
        _, _, valid = project_points(
            np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 2.0]]), np.eye(4), intrinsics
        )
        self.assertFalse(bool(valid[0]))
        self.assertTrue(bool(valid[1]))

    def test_the_compiled_projection_equals_the_numpy_original_bit_for_bit(self):
        rng = np.random.default_rng(5)
        points = rng.uniform(-4.0, 4.0, (3000, 3))  # in front of and behind the camera
        view = look_at(np.array([0.3, 0.2, 2.5]), np.array([0.0, 0.1, 0.0]))
        intrinsics = np.array([[420.0, 0, 250.5], [0, 417.0, 301.25], [0, 0, 1.0]])
        for got, expected in zip(project_points(points, view, intrinsics), _project_points_numpy(points, view, intrinsics)):
            np.testing.assert_array_equal(got, expected)
        self.assertFalse(project_points(points, view, intrinsics)[2].all())


class LookAtTests(unittest.TestCase):
    def test_the_camera_sees_the_target_at_its_principal_point(self):
        target = np.array([0.0, 1.0, 0.0])
        view = look_at(np.array([0.0, 1.0, 3.0]), target)
        intrinsics = np.array([[100.0, 0, 64.0], [0, 100.0, 48.0], [0, 0, 1.0]])
        pixels, depth, valid = project_points(target[None], view, intrinsics)
        self.assertTrue(bool(valid[0]))
        np.testing.assert_allclose(pixels[0], [64.0, 48.0], atol=1e-6)
        self.assertAlmostEqual(float(depth[0]), 3.0, places=6)

    def test_orbiting_keeps_the_target_centred(self):
        target = np.array([0.0, 1.0, 0.0])
        intrinsics = np.array([[100.0, 0, 64.0], [0, 100.0, 48.0], [0, 0, 1.0]])
        for angle in (0.0, 45.0, 90.0, 180.0, 270.0):
            radians = np.radians(angle)
            eye = target + 3.0 * np.array([np.sin(radians), 0.0, np.cos(radians)])
            pixels, _, valid = project_points(
                target[None], look_at(eye, target), intrinsics
            )
            self.assertTrue(bool(valid[0]), f"target lost at {angle} degrees")
            np.testing.assert_allclose(pixels[0], [64.0, 48.0], atol=1e-6)


class SphericalHarmonicTests(unittest.TestCase):
    def test_the_dc_term_alone_is_a_view_independent_colour(self):
        dc = np.full((1, 1, 3), 0.5)
        rest = np.zeros((1, 15, 3))
        forward = sh_to_rgb(dc, rest, np.array([[0.0, 0.0, 1.0]]))
        sideways = sh_to_rgb(dc, rest, np.array([[1.0, 0.0, 0.0]]))
        np.testing.assert_allclose(forward, sideways, atol=1e-12)

    def test_higher_orders_make_colour_depend_on_view_direction(self):
        dc = np.zeros((1, 1, 3))
        rest = np.zeros((1, 15, 3))
        rest[0, 0] = 1.0  # first degree-1 band
        forward = sh_to_rgb(dc, rest, np.array([[0.0, 0.0, 1.0]]))
        sideways = sh_to_rgb(dc, rest, np.array([[0.0, 1.0, 0.0]]))
        self.assertGreater(float(np.abs(forward - sideways).max()), 0.05)

    def test_output_is_clipped_to_displayable_range(self):
        dc = np.full((1, 1, 3), 50.0)
        rest = np.zeros((1, 15, 3))
        rgb = sh_to_rgb(dc, rest, np.array([[0.0, 0.0, 1.0]]))
        self.assertLessEqual(float(rgb.max()), 1.0)
        self.assertGreaterEqual(float(rgb.min()), 0.0)


class RenderTests(unittest.TestCase):
    def _one_splat(self, position, colour=(1.0, 0.0, 0.0), scale=0.05, opacity=5.0):
        return {
            "xyz": np.array([position], np.float64),
            "scaling": np.log(np.full((1, 3), scale)),
            "rotation": np.array([[1.0, 0.0, 0.0, 0.0]]),
            "opacity": np.array([[opacity]]),
            "feature_dc": np.array([[list(colour)]], np.float64),
            "feature_rest": np.zeros((1, 15, 3)),
        }

    def _camera(self):
        return look_at(np.array([0.0, 0.0, 3.0]), np.zeros(3)), np.array(
            [[100.0, 0, 32.0], [0, 100.0, 32.0], [0, 0, 1.0]]
        )

    def test_a_single_splat_lands_where_it_projects(self):
        view, intrinsics = self._camera()
        image, alpha = render_splats(self._one_splat([0, 0, 0]), view, intrinsics, (64, 64))
        self.assertGreater(float(alpha[32, 32]), 0.5)
        self.assertGreater(float(alpha.sum()), 0.0)
        # Red splat: the red channel must dominate where it is opaque.
        self.assertGreater(float(image[32, 32, 2]), float(image[32, 32, 0]))

    def test_an_empty_scene_renders_transparent(self):
        view, intrinsics = self._camera()
        empty = {
            "xyz": np.zeros((0, 3)),
            "scaling": np.zeros((0, 3)),
            "rotation": np.zeros((0, 4)),
            "opacity": np.zeros((0, 1)),
            "feature_dc": np.zeros((0, 1, 3)),
            "feature_rest": np.zeros((0, 15, 3)),
        }
        _, alpha = render_splats(empty, view, intrinsics, (32, 32))
        self.assertEqual(float(alpha.max()), 0.0)

    def test_a_nearer_splat_occludes_a_further_one(self):
        """Depth ordering: the front splat's colour wins where they overlap."""
        view, intrinsics = self._camera()
        scene = self._one_splat([0, 0, 0], colour=(1.0, 0.0, 0.0), opacity=8.0)
        behind = self._one_splat([0, 0, -0.5], colour=(0.0, 0.0, 1.0), opacity=8.0)
        for key in scene:
            scene[key] = np.concatenate([scene[key], behind[key]], axis=0)
        image, _ = render_splats(scene, view, intrinsics, (64, 64))
        centre = image[32, 32]
        self.assertGreater(float(centre[2]), float(centre[0]))

    def test_a_larger_scale_covers_more_pixels(self):
        view, intrinsics = self._camera()
        _, small = render_splats(
            self._one_splat([0, 0, 0], scale=0.02), view, intrinsics, (64, 64)
        )
        _, large = render_splats(
            self._one_splat([0, 0, 0], scale=0.08), view, intrinsics, (64, 64)
        )
        self.assertGreater(
            int((large > 0.1).sum()), int((small > 0.1).sum())
        )

    def test_moving_the_camera_moves_the_splat_in_frame(self):
        intrinsics = np.array([[100.0, 0, 32.0], [0, 100.0, 32.0], [0, 0, 1.0]])
        scene = self._one_splat([0.4, 0, 0])
        front = look_at(np.array([0.0, 0.0, 3.0]), np.zeros(3))
        side = look_at(np.array([3.0, 0.0, 0.0]), np.zeros(3))
        _, from_front = render_splats(scene, front, intrinsics, (64, 64))
        _, from_side = render_splats(scene, side, intrinsics, (64, 64))
        front_x = float(np.argwhere(from_front > 0.1)[:, 1].mean())
        side_x = float(np.argwhere(from_side > 0.1)[:, 1].mean())
        self.assertGreater(abs(front_x - side_x), 5.0)


if __name__ == "__main__":
    unittest.main()


class CompiledPrePassTest(unittest.TestCase):
    """`_prepare`'s numba kernels against the numpy originals they replace."""

    def _splats(self, n=500, bands=15):
        rng = np.random.default_rng(3)
        return {
            "xyz": rng.normal(size=(n, 3)),
            "scaling": rng.normal(size=(n, 3)) - 3.0,
            "rotation": rng.normal(size=(n, 4)),
            "feature_dc": rng.normal(size=(n, 1, 3)),
            "feature_rest": rng.normal(size=(n, bands, 3)) * 0.2,
            "opacity": rng.normal(size=(n, 1)),
        }

    def test_covariance_and_colour_match_the_numpy_versions(self):
        from tryon.rendering.splat_cpu import _prepare, screen_covariance, sh_to_rgb

        splats = self._splats()
        view = look_at((0.3, -0.2, 6.0), (0.0, 0.0, 0.0))
        intrinsics = np.array([[400.0, 0.0, 160.0], [0.0, 400.0, 120.0], [0.0, 0.0, 1.0]])
        _pixels, _depth, _valid, cov2d, _det, colours, _opacity, _radius = _prepare(splats, view, intrinsics, 3.0)
        expected_cov = screen_covariance(splats["xyz"], splats["scaling"], splats["rotation"], view, intrinsics)
        eye = -view[:3, :3].T @ view[:3, 3]
        expected_rgb = sh_to_rgb(splats["feature_dc"], splats["feature_rest"], splats["xyz"] - eye)
        np.testing.assert_allclose(cov2d, expected_cov, rtol=0, atol=1e-12)
        np.testing.assert_allclose(colours, expected_rgb, rtol=0, atol=1e-12)

    def test_fewer_sh_bands_still_match(self):
        from tryon.rendering.splat_cpu import _prepare, sh_to_rgb

        for bands in (0, 3, 8):
            splats = self._splats(bands=bands)
            view = look_at((0.0, 0.0, 5.0), (0.0, 0.0, 0.0))
            intrinsics = np.array([[300.0, 0.0, 100.0], [0.0, 300.0, 100.0], [0.0, 0.0, 1.0]])
            colours = _prepare(splats, view, intrinsics, 3.0)[5]
            eye = -view[:3, :3].T @ view[:3, 3]
            expected = sh_to_rgb(splats["feature_dc"], splats["feature_rest"], splats["xyz"] - eye)
            np.testing.assert_allclose(colours, expected, rtol=0, atol=1e-12, err_msg=f"bands={bands}")


class RotationToQuaternionCompiledTest(unittest.TestCase):
    """`rotation_to_quaternion` (numba) equals the vectorised numpy original,
    including the three off-trace branches."""

    def test_matches_numpy(self):
        from scipy.spatial.transform import Rotation

        from tryon.rendering.splat_cpu import _rotation_to_quaternion_numpy, rotation_to_quaternion

        m = Rotation.random(2000, random_state=0).as_matrix()
        m[:3] = np.eye(3)
        m[3:6] = np.diag([1.0, -1.0, -1.0])
        m[6:9] = np.diag([-1.0, 1.0, -1.0])
        m[9:12] = np.diag([-1.0, -1.0, 1.0])
        np.testing.assert_array_equal(rotation_to_quaternion(m), _rotation_to_quaternion_numpy(m))
        self.assertEqual(rotation_to_quaternion(np.zeros((0, 3, 3))).shape, (0, 4))


class FaceFramesCompiledTest(unittest.TestCase):
    """`face_frames_numba` (the shared `face_frame` arithmetic) equals the numpy `face_frames`."""

    def test_matches_numpy(self):
        from tryon.rendering.splat_cpu import face_frames, face_frames_numba

        rng = np.random.default_rng(0)
        vertices = rng.normal(size=(300, 3))
        faces = rng.integers(0, 300, size=(500, 3))
        faces[0] = (4, 4, 4)  # degenerate: the 1e-20 floors
        _, orientation, scale = face_frames(vertices, faces)
        frames, scale_c = face_frames_numba(np.ascontiguousarray(vertices), np.ascontiguousarray(faces, np.int64))
        np.testing.assert_allclose(frames, orientation, rtol=0, atol=1e-12)
        np.testing.assert_allclose(scale_c, scale, rtol=0, atol=1e-12)


class PreparedOpacityTest(unittest.TestCase):
    """A caller's own `opacity_alpha` is what `_prepare` blends with; absent, the sigmoid of `opacity`."""

    def test_prepared_sigmoid_is_used_and_equal(self):
        from tryon.rendering.splat_cpu import _prepare

        rng = np.random.default_rng(1)
        n = 50
        splats = {
            "xyz": rng.normal(size=(n, 3)) * 0.1,
            "scaling": np.full((n, 3), -4.0),
            "rotation": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
            "feature_dc": rng.normal(size=(n, 1, 3)),
            "feature_rest": rng.normal(size=(n, 15, 3)),
            "opacity": rng.normal(size=(n, 1)),
        }
        view = np.eye(4)
        view[2, 3] = 5.0
        intrinsics = np.array([[300.0, 0.0, 100.0], [0.0, 300.0, 100.0], [0.0, 0.0, 1.0]])
        plain = _prepare(splats, view, intrinsics, 3.0)[6]
        with_alpha = _prepare({**splats, "opacity_alpha": 1.0 / (1.0 + np.exp(-splats["opacity"].ravel()))}, view, intrinsics, 3.0)[6]
        np.testing.assert_array_equal(plain, with_alpha)
        marked = _prepare({**splats, "opacity_alpha": np.full(n, 0.25)}, view, intrinsics, 3.0)[6]
        np.testing.assert_array_equal(marked, np.full(n, 0.25))
