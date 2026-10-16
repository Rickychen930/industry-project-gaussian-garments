"""A 3D Gaussian splat rasteriser, written out in numpy/numba rather than called.

Renders the trained garments of Gaussian Garments (~95k Gaussians each) on
the CPU. Pipeline:

1. World placement: stored `xyz` are offsets in each face's tangent frame;
   world position is `R_face @ local * s_face + c_face`.
2. Projection: rigid world-to-camera transform, then `u = f * x / z + c`.
3. Covariance: `Sigma = R S S^T R^T`, projected as `J W Sigma W^T J^T`
   (EWA splatting, Zwicker et al. 2001).
4. Colour: spherical harmonics along the view direction.
5. Compositing: sorted back-to-front and alpha-blended.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# real spherical harmonic constants, bands 0-3, in checkpoint order
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


def quaternion_to_rotation(quaternions):
    """Rotation matrices from `wxyz` quaternions, normalised first."""
    q = np.asarray(quaternions, np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(-1, 3, 3)


def face_frames(vertices, faces):
    """Per-face origin, orthonormal frame and scale.

    Mirrors `compute_face_orientation`: axis 0 along edge v0->v1, axis 1
    along the normal, axis 2 completes the frame. Scale is the mean of the
    edge length and the triangle's extent along axis 2.
    """
    vertices = np.asarray(vertices, np.float64)
    triangles = vertices[np.asarray(faces)]
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]

    def normalise(vectors):
        return vectors / np.maximum(
            np.linalg.norm(vectors, axis=1, keepdims=True), 1e-20
        )

    a0 = normalise(v1 - v0)
    a1 = normalise(np.cross(a0, v2 - v0))
    a2 = -normalise(np.cross(a1, a0))
    orientation = np.stack([a0, a1, a2], axis=2)
    scale = (
        np.linalg.norm(v1 - v0, axis=1)
        + np.abs(np.einsum("ij,ij->i", a2, v2 - v0))
    ) / 2
    return triangles.mean(axis=1), orientation, scale


@njit(inline="always")
def face_frame(vertices, faces, f):
    """`face_frames` for one face as scalars: the three axes (a0x ... a2z) and the scale.

    The one copy shared by the compiled kernels (`GarmentRenderer` placement,
    `AppearanceNet.world_xyz`), inlined so each keeps its own loop. numba's
    cache does not notice edits here: clear `__pycache__` of the kernels'
    modules after changing it."""
    i0, i1, i2 = faces[f, 0], faces[f, 1], faces[f, 2]
    v0x, v0y, v0z = vertices[i0, 0], vertices[i0, 1], vertices[i0, 2]
    e1x, e1y, e1z = vertices[i1, 0] - v0x, vertices[i1, 1] - v0y, vertices[i1, 2] - v0z
    e2x, e2y, e2z = vertices[i2, 0] - v0x, vertices[i2, 1] - v0y, vertices[i2, 2] - v0z
    l1 = np.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
    d = max(l1, 1e-20)
    a0x, a0y, a0z = e1x / d, e1y / d, e1z / d
    nx, ny, nz = a0y * e2z - a0z * e2y, a0z * e2x - a0x * e2z, a0x * e2y - a0y * e2x
    d = max(np.sqrt(nx * nx + ny * ny + nz * nz), 1e-20)
    a1x, a1y, a1z = nx / d, ny / d, nz / d
    cx, cy, cz = a1y * a0z - a1z * a0y, a1z * a0x - a1x * a0z, a1x * a0y - a1y * a0x
    d = max(np.sqrt(cx * cx + cy * cy + cz * cz), 1e-20)
    a2x, a2y, a2z = -cx / d, -cy / d, -cz / d
    scale = (l1 + abs(a2x * e2x + a2y * e2y + a2z * e2z)) / 2
    return a0x, a0y, a0z, a1x, a1y, a1z, a2x, a2y, a2z, scale


@njit(cache=True)
def face_frames_numba(vertices, faces):
    """(frames (F, 3, 3), scale (F,)) of every face; compiled `face_frames`.
    Serial on purpose (shading workers call it; see `appearance._world_xyz_numba`)."""
    frames = np.empty((len(faces), 3, 3))
    scale = np.empty(len(faces))
    for f in range(len(faces)):
        a0x, a0y, a0z, a1x, a1y, a1z, a2x, a2y, a2z, scale[f] = face_frame(vertices, faces, f)
        frames[f, 0, 0], frames[f, 1, 0], frames[f, 2, 0] = a0x, a0y, a0z
        frames[f, 0, 1], frames[f, 1, 1], frames[f, 2, 1] = a1x, a1y, a1z
        frames[f, 0, 2], frames[f, 1, 2], frames[f, 2, 2] = a2x, a2y, a2z
    return frames, scale


def world_positions(local_xyz, binding, vertices, faces):
    """Place face-local Gaussian offsets into the world."""
    centres, orientation, scale = face_frames(vertices, faces)
    binding = np.asarray(binding, int)
    rotated = np.einsum(
        "nij,nj->ni", orientation[binding], np.asarray(local_xyz, np.float64)
    )
    return rotated * scale[binding, None] + centres[binding]


def face_scaled_log_scaling(log_scaling, binding, vertices, faces, dilate=1.0):
    """Scale each Gaussian by the face it is bound to, in log space.

    Stored `scaling` is relative to its face; used directly, splats would be
    ~90x too large. Returned in log space for `screen_covariance`.

    `dilate` grows every splat by a constant factor: the stored sizes leave
    gaps on a garment this sparse, and 1.3 closes most of them for the least
    blur.
    """
    _, _, scale = face_frames(vertices, faces)
    return (
        np.asarray(log_scaling, np.float64)
        + np.log(np.maximum(scale[np.asarray(binding, int)], 1e-20))[:, None]
        + np.log(max(float(dilate), 1e-20))
    )






def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """World-to-camera matrix for a camera at `eye` looking at `target`.

    Camera convention: +x right, +y down, +z forward, so a point in front
    has positive depth and `project_points` can divide by it directly.
    """
    eye = np.asarray(eye, np.float64)
    forward = np.asarray(target, np.float64) - eye
    forward = forward / np.maximum(np.linalg.norm(forward), 1e-12)
    up = np.asarray(up, np.float64)
    if abs(float(np.dot(forward, up / np.linalg.norm(up)))) > 0.999:
        up = np.array([0.0, 0.0, 1.0])  # degenerate: looking straight up
    right = np.cross(forward, up)
    right = right / np.maximum(np.linalg.norm(right), 1e-12)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    view = np.eye(4)
    view[:3, :3] = rotation
    view[:3, 3] = -rotation @ eye
    return view


@njit(cache=True, fastmath={"contract"})
def camera_coordinates(px, py, pz, rotation, translation):
    """Camera-space position of a world point, bit for bit equal to `points @ R.T + t`.

    Accelerate's gemm evaluates each dot product as a multiply and two fused
    multiply-adds in the order y, x, z; `contract` lets LLVM fuse them here
    without fusing the callers' arithmetic."""
    a = py * rotation[0, 1]
    a = px * rotation[0, 0] + a
    x = (pz * rotation[0, 2] + a) + translation[0]
    b = py * rotation[1, 1]
    b = px * rotation[1, 0] + b
    y = (pz * rotation[1, 2] + b) + translation[1]
    c = py * rotation[2, 1]
    c = px * rotation[2, 0] + c
    z = (pz * rotation[2, 2] + c) + translation[2]
    return x, y, z


@njit(cache=True)
def _project_numba(points, rotation, translation, fx, fy, cx, cy):
    """`_project_points_numpy` in one serial pass (no BLAS, no temporaries).
    Camera coordinates match the matmul bit for bit; safe on any thread."""
    n = len(points)
    pixels = np.empty((n, 2))
    depth = np.empty(n)
    valid = np.empty(n, np.bool_)
    for i in range(n):
        x, y, z = camera_coordinates(points[i, 0], points[i, 1], points[i, 2], rotation, translation)
        depth[i] = z
        ok = z > 1e-6
        valid[i] = ok
        safe = z if ok else 1.0
        pixels[i, 0] = fx * x / safe + cx
        pixels[i, 1] = fy * y / safe + cy
    return pixels, depth, valid


def project_points(points, view, intrinsics):
    """Pixel coordinates, depth, and which points are in front."""
    return _project_numba(
        np.ascontiguousarray(points, np.float64),
        np.ascontiguousarray(view[:3, :3], np.float64),
        np.ascontiguousarray(view[:3, 3], np.float64),
        float(intrinsics[0, 0]),
        float(intrinsics[1, 1]),
        float(intrinsics[0, 2]),
        float(intrinsics[1, 2]),
    )


def _project_points_numpy(points, view, intrinsics):
    """Numpy `project_points`; oracle for the compiled one."""
    points = np.asarray(points, np.float64)
    camera = points @ view[:3, :3].T + view[:3, 3]
    depth = camera[:, 2]
    valid = depth > 1e-6
    safe = np.where(valid, depth, 1.0)
    pixels = np.empty((len(points), 2))
    pixels[:, 0] = intrinsics[0, 0] * camera[:, 0] / safe + intrinsics[0, 2]
    pixels[:, 1] = intrinsics[1, 1] * camera[:, 1] / safe + intrinsics[1, 2]
    return pixels, depth, valid


def screen_covariance(points, scaling, rotation, view, intrinsics):
    """2D covariance of each projected Gaussian (EWA splatting).

    `Sigma = R S S^T R^T` in world space, linearised about each centre:
    screen covariance is `J W Sigma W^T J^T` with `W` the camera rotation
    and `J` the Jacobian of the perspective division.
    """
    scales = np.exp(np.asarray(scaling, np.float64))
    rotations = quaternion_to_rotation(rotation)
    scaled = rotations * scales[:, None, :]
    world_cov = scaled @ np.transpose(scaled, (0, 2, 1))

    camera = np.asarray(points, np.float64) @ view[:3, :3].T + view[:3, 3]
    z = np.maximum(camera[:, 2], 1e-6)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    jacobian = np.zeros((len(points), 2, 3))
    jacobian[:, 0, 0] = fx / z
    jacobian[:, 0, 2] = -fx * camera[:, 0] / (z * z)
    jacobian[:, 1, 1] = fy / z
    jacobian[:, 1, 2] = -fy * camera[:, 1] / (z * z)

    transform = jacobian @ view[:3, :3]
    cov2d = transform @ world_cov @ np.transpose(transform, (0, 2, 1))
    # small dilation keeps sub-pixel Gaussians visible and the matrix invertible
    cov2d[:, 0, 0] += 0.3
    cov2d[:, 1, 1] += 0.3
    return cov2d


def sh_to_rgb(feature_dc, feature_rest, directions):
    """Evaluate spherical harmonics for the given viewing directions.

    Band 0 is the view-independent colour; bands 1-3 add view dependence.
    Stored coefficients are offsets from 0.5.
    """
    dc = np.asarray(feature_dc, np.float64).reshape(len(feature_dc), 3)
    rest = np.asarray(feature_rest, np.float64)
    directions = np.asarray(directions, np.float64)
    directions = directions / np.maximum(
        np.linalg.norm(directions, axis=1, keepdims=True), 1e-12
    )
    rgb = SH_C0 * dc
    available = rest.shape[1]
    if available >= 3:
        x, y, z = directions[:, 0:1], directions[:, 1:2], directions[:, 2:3]
        rgb = rgb - SH_C1 * y * rest[:, 0] + SH_C1 * z * rest[:, 1]
        rgb = rgb - SH_C1 * x * rest[:, 2]
        if available >= 8:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            rgb = (
                rgb
                + SH_C2[0] * xy * rest[:, 3]
                + SH_C2[1] * yz * rest[:, 4]
                + SH_C2[2] * (2.0 * zz - xx - yy) * rest[:, 5]
                + SH_C2[3] * xz * rest[:, 6]
                + SH_C2[4] * (xx - yy) * rest[:, 7]
            )
            if available >= 15:
                rgb = (
                    rgb
                    + SH_C3[0] * y * (3.0 * xx - yy) * rest[:, 8]
                    + SH_C3[1] * xy * z * rest[:, 9]
                    + SH_C3[2] * y * (4.0 * zz - xx - yy) * rest[:, 10]
                    + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * rest[:, 11]
                    + SH_C3[4] * x * (4.0 * zz - xx - yy) * rest[:, 12]
                    + SH_C3[5] * z * (xx - yy) * rest[:, 13]
                    + SH_C3[6] * x * (xx - 3.0 * yy) * rest[:, 14]
                )
    return np.clip(rgb + 0.5, 0.0, 1.0)


def render_splats_reference(splats, view, intrinsics, size, tile=3.0):
    """Rasterise Gaussians into (BGR float image, alpha).

    Back-to-front painter's algorithm: each Gaussian contributes
    `alpha * exp(-0.5 * d^T Sigma^-1 d)` within `tile` standard deviations of
    its centre. Slow oracle for `render_splats`.
    """
    height, width = size
    image = np.zeros((height, width, 3), np.float64)
    alpha_buffer = np.zeros((height, width), np.float64)
    positions = np.asarray(splats["xyz"], np.float64)
    if len(positions) == 0:
        return image, alpha_buffer

    pixels, depth, valid = project_points(positions, view, intrinsics)
    cov2d = screen_covariance(
        positions, splats["scaling"], splats["rotation"], view, intrinsics
    )
    determinant = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
    valid &= determinant > 1e-12

    eye = -view[:3, :3].T @ view[:3, 3]
    colours = sh_to_rgb(
        splats["feature_dc"], splats["feature_rest"], positions - eye
    )
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(splats["opacity"], np.float64).ravel()))

    # furthest first
    order = np.argsort(-depth)
    order = order[valid[order]]
    radius = tile * np.sqrt(
        np.maximum(cov2d[:, 0, 0], cov2d[:, 1, 1])
    )

    for index in order:
        centre = pixels[index]
        extent = radius[index]
        x0 = max(int(np.floor(centre[0] - extent)), 0)
        x1 = min(int(np.ceil(centre[0] + extent)) + 1, width)
        y0 = max(int(np.floor(centre[1] - extent)), 0)
        y1 = min(int(np.ceil(centre[1] + extent)) + 1, height)
        if x1 <= x0 or y1 <= y0:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        dx = xs - centre[0]
        dy = ys - centre[1]
        a, b, c = cov2d[index, 0, 0], cov2d[index, 0, 1], cov2d[index, 1, 1]
        inverse = determinant[index]
        power = -0.5 * (
            (c * dx * dx - 2 * b * dx * dy + a * dy * dy) / inverse
        )
        weight = opacity[index] * np.exp(np.minimum(power, 0.0))
        weight = np.where(weight > 1 / 255, weight, 0.0)
        if not weight.any():
            continue
        region = image[y0:y1, x0:x1]
        existing = alpha_buffer[y0:y1, x0:x1]
        # BGR, as everywhere in this package
        colour = colours[index][::-1]
        region *= (1 - weight)[..., None]
        region += weight[..., None] * colour
        alpha_buffer[y0:y1, x0:x1] = existing * (1 - weight) + weight
    return image, alpha_buffer


# --- Fast rasteriser --------------------------------------------------------
#
# Compositing is an order-dependent painter's recurrence, so splats are
# binned into TILE x TILE blocks and composited per tile in depth order
# (Kerbl et al. 2023, on the CPU). `render_splats_reference` is the oracle;
# tests require agreement to a quarter of a grey level.

# 6 beat 16 and 32 on both a scattered and a real-garment workload: partly
# overlapped tiles are mostly wasted work at small splat radii.
TILE = 6


@njit(cache=True, parallel=True)
def _screen_covariance_numba(points, scaling, rotation, view_rotation, view_translation, fx, fy):
    """Compiled `screen_covariance`, one splat per iteration (same maths)."""
    n = len(points)
    out = np.empty((n, 2, 2))
    for i in prange(n):
        q0, q1, q2, q3 = rotation[i, 0], rotation[i, 1], rotation[i, 2], rotation[i, 3]
        norm = max(np.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3), 1e-12)
        w, x, y, z = q0 / norm, q1 / norm, q2 / norm, q3 / norm
        r00, r01, r02 = 1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)
        r10, r11, r12 = 2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)
        r20, r21, r22 = 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)
        s0, s1, s2 = np.exp(scaling[i, 0]), np.exp(scaling[i, 1]), np.exp(scaling[i, 2])
        # Sigma = M M^T with M = R diag(s)
        m00, m01, m02 = r00 * s0, r01 * s1, r02 * s2
        m10, m11, m12 = r10 * s0, r11 * s1, r12 * s2
        m20, m21, m22 = r20 * s0, r21 * s1, r22 * s2
        c00 = m00 * m00 + m01 * m01 + m02 * m02
        c01 = m00 * m10 + m01 * m11 + m02 * m12
        c02 = m00 * m20 + m01 * m21 + m02 * m22
        c11 = m10 * m10 + m11 * m11 + m12 * m12
        c12 = m10 * m20 + m11 * m21 + m12 * m22
        c22 = m20 * m20 + m21 * m21 + m22 * m22
        # camera-space centre
        px, py, pz = points[i, 0], points[i, 1], points[i, 2]
        cx = view_rotation[0, 0] * px + view_rotation[0, 1] * py + view_rotation[0, 2] * pz + view_translation[0]
        cy = view_rotation[1, 0] * px + view_rotation[1, 1] * py + view_rotation[1, 2] * pz + view_translation[1]
        cz = view_rotation[2, 0] * px + view_rotation[2, 1] * py + view_rotation[2, 2] * pz + view_translation[2]
        zz = max(cz, 1e-6)
        j00, j02 = fx / zz, -fx * cx / (zz * zz)
        j11, j12 = fy / zz, -fy * cy / (zz * zz)
        # T = J W  (2x3)
        t00 = j00 * view_rotation[0, 0] + j02 * view_rotation[2, 0]
        t01 = j00 * view_rotation[0, 1] + j02 * view_rotation[2, 1]
        t02 = j00 * view_rotation[0, 2] + j02 * view_rotation[2, 2]
        t10 = j11 * view_rotation[1, 0] + j12 * view_rotation[2, 0]
        t11 = j11 * view_rotation[1, 1] + j12 * view_rotation[2, 1]
        t12 = j11 * view_rotation[1, 2] + j12 * view_rotation[2, 2]
        # T Sigma (2x3)
        a0 = t00 * c00 + t01 * c01 + t02 * c02
        a1 = t00 * c01 + t01 * c11 + t02 * c12
        a2 = t00 * c02 + t01 * c12 + t02 * c22
        b0 = t10 * c00 + t11 * c01 + t12 * c02
        b1 = t10 * c01 + t11 * c11 + t12 * c12
        b2 = t10 * c02 + t11 * c12 + t12 * c22
        out[i, 0, 0] = a0 * t00 + a1 * t01 + a2 * t02 + 0.3
        out[i, 0, 1] = a0 * t10 + a1 * t11 + a2 * t12
        out[i, 1, 0] = b0 * t00 + b1 * t01 + b2 * t02
        out[i, 1, 1] = b0 * t10 + b1 * t11 + b2 * t12 + 0.3
    return out


@njit(cache=True, parallel=True)
def _sh_to_rgb_numba(dc, rest, directions, c1, c2, c3):
    """Compiled `sh_to_rgb`; bands present follow `rest`'s width."""
    n = len(dc)
    available = rest.shape[1]
    out = np.empty((n, 3))
    for i in prange(n):
        dx, dy, dz = directions[i, 0], directions[i, 1], directions[i, 2]
        norm = max(np.sqrt(dx * dx + dy * dy + dz * dz), 1e-12)
        x, y, z = dx / norm, dy / norm, dz / norm
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        for ch in range(3):
            v = SH_C0 * dc[i, ch]
            if available >= 3:
                v = v - c1 * y * rest[i, 0, ch] + c1 * z * rest[i, 1, ch] - c1 * x * rest[i, 2, ch]
                if available >= 8:
                    v = (
                        v
                        + c2[0] * xy * rest[i, 3, ch]
                        + c2[1] * yz * rest[i, 4, ch]
                        + c2[2] * (2.0 * zz - xx - yy) * rest[i, 5, ch]
                        + c2[3] * xz * rest[i, 6, ch]
                        + c2[4] * (xx - yy) * rest[i, 7, ch]
                    )
                    if available >= 15:
                        v = (
                            v
                            + c3[0] * y * (3.0 * xx - yy) * rest[i, 8, ch]
                            + c3[1] * xy * z * rest[i, 9, ch]
                            + c3[2] * y * (4.0 * zz - xx - yy) * rest[i, 10, ch]
                            + c3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * rest[i, 11, ch]
                            + c3[4] * x * (4.0 * zz - xx - yy) * rest[i, 12, ch]
                            + c3[5] * z * (xx - yy) * rest[i, 13, ch]
                            + c3[6] * x * (xx - 3.0 * yy) * rest[i, 14, ch]
                        )
            v = v + 0.5
            out[i, ch] = 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
    return out


def _prepare(splats, view, intrinsics, tile):
    """Everything per-splat that does not depend on the pixel (shared by the
    CPU and GL rasterisers); covariance and SH run compiled."""
    positions = np.ascontiguousarray(splats["xyz"], dtype=np.float64)
    pixels, depth, valid = project_points(positions, view, intrinsics)
    view_rotation = np.ascontiguousarray(view[:3, :3], dtype=np.float64)
    view_translation = np.ascontiguousarray(view[:3, 3], dtype=np.float64)
    cov2d = _screen_covariance_numba(
        positions,
        np.ascontiguousarray(splats["scaling"], dtype=np.float64),
        np.ascontiguousarray(splats["rotation"], dtype=np.float64),
        view_rotation,
        view_translation,
        float(intrinsics[0, 0]),
        float(intrinsics[1, 1]),
    )
    determinant = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
    valid &= determinant > 1e-12
    eye = -view[:3, :3].T @ view[:3, 3]
    rest = np.asarray(splats["feature_rest"], np.float64)
    colours = _sh_to_rgb_numba(
        np.ascontiguousarray(np.asarray(splats["feature_dc"], np.float64).reshape(len(positions), 3)),
        np.ascontiguousarray(rest.reshape(len(positions), rest.shape[1], 3) if rest.ndim == 3 else rest.reshape(len(positions), -1, 3)),
        np.ascontiguousarray(positions - eye),
        SH_C1,
        np.asarray(SH_C2, np.float64),
        np.asarray(SH_C3, np.float64),
    )
    opacity = splats.get("opacity_alpha")  # precomputed sigmoid, when the caller kept it
    if opacity is None:
        opacity = 1.0 / (1.0 + np.exp(-np.asarray(splats["opacity"], np.float64).ravel()))
    radius = tile * np.sqrt(np.maximum(cov2d[:, 0, 0], cov2d[:, 1, 1]))
    return pixels, depth, valid, cov2d, determinant, colours, opacity, radius


@njit(cache=True)
def _bucket_sort_by_tile(tile_id, n_tiles):
    """Stable O(n + n_tiles) counting sort of (splat, tile) pair indices by tile_id.

    Groups only; depth order within a tile is sorted in `_composite_tiles`
    (insertion sort lost: real tile groups average ~800 splats).
    """
    counts = np.zeros(n_tiles, dtype=np.int64)
    n = len(tile_id)
    for i in range(n):
        counts[tile_id[i]] += 1
    starts = np.zeros(n_tiles + 1, dtype=np.int64)
    for t in range(n_tiles):
        starts[t + 1] = starts[t] + counts[t]
    cursor = starts[:-1].copy()
    order = np.empty(n, dtype=np.int64)
    for i in range(n):
        t = tile_id[i]
        order[cursor[t]] = i
        cursor[t] += 1
    return order, starts


@njit(cache=True, parallel=True)
def _composite_tiles(
    starts, ends, splat_of, tile_id, depth, tiles_x, height, width,
    pixels, a, b, c, determinant, opacity, colours_bgr, x0, x1, y0, y1,
    image, alpha_buffer,
):
    """The painter's-recurrence loop, one compiled pass over every tile.

    Same maths as `render_splats_reference`. Each tile's group (already
    grouped by `_bucket_sort_by_tile`) is depth-sorted here with `np.argsort`.
    """
    n_tiles = len(starts)
    # tiles write only their own block, so they run in parallel
    for i in prange(n_tiles):
        start, end = starts[i], ends[i]
        t = tile_id[i]
        py = (t // tiles_x) * TILE
        px = (t % tiles_x) * TILE
        th = min(TILE, height - py)
        tw = min(TILE, width - px)
        # the group arrives grouped by tile but not depth-ordered
        group = splat_of[start:end]
        group_depth = depth[group]
        local_order = np.argsort(group_depth)
        # nearest-first with a running transmittance equals the back-to-front blend
        trans = np.ones((th, tw))
        accum_bgr = np.zeros((th, tw, 3))
        accum_alpha = np.zeros((th, tw))
        for k in range(len(local_order)):
            s = group[local_order[k]]
            px_s, py_s = pixels[s, 0], pixels[s, 1]
            as_, bs, cs, det = a[s], b[s], c[s], determinant[s]
            op = opacity[s]
            bx0, bx1, by0, by1 = x0[s], x1[s], y0[s], y1[s]
            # only cells inside this splat's bounding box
            row_start = max(by0 - py, 0)
            row_end = min(by1 - py, th)
            col_start = max(bx0 - px, 0)
            col_end = min(bx1 - px, tw)
            for row in range(row_start, row_end):
                ry = py + row
                dy = ry - py_s
                for col in range(col_start, col_end):
                    rx = px + col
                    dx = rx - px_s
                    power = -0.5 * (cs * dx * dx - 2 * bs * dx * dy + as_ * dy * dy) / det
                    if power > 0.0:
                        power = 0.0
                    weight = op * np.exp(power)
                    if weight <= 1.0 / 255:
                        continue
                    contribution = weight * trans[row, col]
                    accum_bgr[row, col, 0] += contribution * colours_bgr[s, 0]
                    accum_bgr[row, col, 1] += contribution * colours_bgr[s, 1]
                    accum_bgr[row, col, 2] += contribution * colours_bgr[s, 2]
                    accum_alpha[row, col] += contribution
                    trans[row, col] *= 1.0 - weight
        image[py:py + th, px:px + tw, :] += accum_bgr
        alpha_buffer[py:py + th, px:px + tw] += accum_alpha


def render_splats(splats, view, intrinsics, size, tile=3.0):
    """Rasterise Gaussians into (BGR float image, alpha); tiled and compiled.
    Same output as `render_splats_reference`."""
    height, width = size
    image = np.zeros((height, width, 3), np.float64)
    alpha_buffer = np.zeros((height, width), np.float64)
    if len(splats["xyz"]) == 0:
        return image, alpha_buffer
    pixels, depth, valid, cov2d, determinant, colours, opacity, radius = _prepare(
        splats, view, intrinsics, tile
    )
    # per-splat bounding box clipped to the image
    x0 = np.maximum(np.floor(pixels[:, 0] - radius), 0).astype(np.int64)
    x1 = np.minimum(np.ceil(pixels[:, 0] + radius) + 1, width).astype(np.int64)
    y0 = np.maximum(np.floor(pixels[:, 1] - radius), 0).astype(np.int64)
    y1 = np.minimum(np.ceil(pixels[:, 1] + radius) + 1, height).astype(np.int64)
    valid &= (x1 > x0) & (y1 > y0)
    keep = np.flatnonzero(valid)
    if keep.size == 0:
        return image, alpha_buffer
    # enumerate (splat, tile) pairs
    tiles_x = (width + TILE - 1) // TILE
    tiles_y = (height + TILE - 1) // TILE
    n_tiles = tiles_x * tiles_y
    tx0, tx1 = x0[keep] // TILE, (x1[keep] - 1) // TILE
    ty0, ty1 = y0[keep] // TILE, (y1[keep] - 1) // TILE
    counts = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
    splat_of = np.repeat(keep, counts)
    # tile of each pair, row-major within the splat's tile box
    offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    span_x = np.repeat(tx1 - tx0 + 1, counts)
    tile_x = np.repeat(tx0, counts) + offsets % span_x
    tile_y = np.repeat(ty0, counts) + offsets // span_x
    tile_id = (tile_y * tiles_x + tile_x).astype(np.int64)
    # group pairs by tile; depth is sorted per tile inside the kernel
    bucket_order, all_starts = _bucket_sort_by_tile(tile_id, n_tiles)
    splat_of = splat_of[bucket_order].astype(np.int64)
    non_empty = np.flatnonzero(np.diff(all_starts))
    starts = all_starts[non_empty]
    ends = all_starts[non_empty + 1]
    run_tile_id = non_empty.astype(np.int64)
    a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    colours_bgr = np.ascontiguousarray(colours[:, ::-1])  # BGR
    _composite_tiles(
        starts, ends, splat_of, run_tile_id, depth, tiles_x, height, width,
        pixels, a, b, c, determinant, opacity.ravel(), colours_bgr, x0, x1, y0, y1,
        image, alpha_buffer,
    )
    return image, alpha_buffer


# --- Quaternions ------------------------------------------------------------


def _rotation_to_quaternion_reference(matrices):
    """Shepperd's method, one matrix at a time. The oracle for the batch form."""
    m = np.asarray(matrices, np.float64)
    q = np.empty((len(m), 4))
    for i, r in enumerate(m):
        t = np.trace(r)
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            q[i] = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            q[i] = ((r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            q[i] = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s)
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            q[i] = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s)
    return q


@njit(cache=True)
def _rotation_to_quaternion_numba(m):
    """Compiled `_rotation_to_quaternion_numpy`, one matrix per iteration.
    Serial, so it is safe beside another thread's numba parallel region."""
    n = m.shape[0]
    q = np.empty((n, 4))
    for i in range(n):
        r00, r11, r22 = m[i, 0, 0], m[i, 1, 1], m[i, 2, 2]
        trace = r00 + r11 + r22
        if trace > 0:
            s = np.sqrt(max(trace + 1.0, 0.0)) * 2
            q[i, 0], q[i, 1], q[i, 2], q[i, 3] = 0.25 * s, (m[i, 2, 1] - m[i, 1, 2]) / s, (m[i, 0, 2] - m[i, 2, 0]) / s, (m[i, 1, 0] - m[i, 0, 1]) / s
        elif r00 > r11 and r00 > r22:
            s = np.sqrt(max(1.0 + r00 - r11 - r22, 0.0)) * 2
            q[i, 0], q[i, 1], q[i, 2], q[i, 3] = (m[i, 2, 1] - m[i, 1, 2]) / s, 0.25 * s, (m[i, 0, 1] + m[i, 1, 0]) / s, (m[i, 0, 2] + m[i, 2, 0]) / s
        elif r11 > r22:
            s = np.sqrt(max(1.0 + r11 - r00 - r22, 0.0)) * 2
            q[i, 0], q[i, 1], q[i, 2], q[i, 3] = (m[i, 0, 2] - m[i, 2, 0]) / s, (m[i, 0, 1] + m[i, 1, 0]) / s, 0.25 * s, (m[i, 1, 2] + m[i, 2, 1]) / s
        else:
            s = np.sqrt(max(1.0 + r22 - r00 - r11, 0.0)) * 2
            q[i, 0], q[i, 1], q[i, 2], q[i, 3] = (m[i, 1, 0] - m[i, 0, 1]) / s, (m[i, 0, 2] + m[i, 2, 0]) / s, (m[i, 1, 2] + m[i, 2, 1]) / s, 0.25 * s
    return q


def rotation_to_quaternion(matrices):
    """Batch of 3x3 rotations -> (w, x, y, z), Shepperd's method, compiled."""
    m = np.ascontiguousarray(matrices, np.float64)
    if m.shape[0] == 0:
        return np.zeros((0, 4))
    return _rotation_to_quaternion_numba(m)


def _rotation_to_quaternion_numpy(matrices):
    """Batch of 3x3 rotations -> (w, x, y, z), Shepperd's method, vectorised.

    All four branches are evaluated and the right one selected; selection by
    the largest of (trace, r00, r11, r22) keeps square roots non-negative.
    Tested against `_rotation_to_quaternion_reference`.
    """
    m = np.asarray(matrices, np.float64)
    r00, r11, r22 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    trace = r00 + r11 + r22
    # clipped so unused branches never NaN
    s0 = np.sqrt(np.maximum(trace + 1.0, 0.0)) * 2
    s1 = np.sqrt(np.maximum(1.0 + r00 - r11 - r22, 0.0)) * 2
    s2 = np.sqrt(np.maximum(1.0 + r11 - r00 - r22, 0.0)) * 2
    s3 = np.sqrt(np.maximum(1.0 + r22 - r00 - r11, 0.0)) * 2
    with np.errstate(divide="ignore", invalid="ignore"):
        q0 = np.stack([0.25 * s0, (m[:, 2, 1] - m[:, 1, 2]) / s0, (m[:, 0, 2] - m[:, 2, 0]) / s0, (m[:, 1, 0] - m[:, 0, 1]) / s0], 1)
        q1 = np.stack([(m[:, 2, 1] - m[:, 1, 2]) / s1, 0.25 * s1, (m[:, 0, 1] + m[:, 1, 0]) / s1, (m[:, 0, 2] + m[:, 2, 0]) / s1], 1)
        q2 = np.stack([(m[:, 0, 2] - m[:, 2, 0]) / s2, (m[:, 0, 1] + m[:, 1, 0]) / s2, 0.25 * s2, (m[:, 1, 2] + m[:, 2, 1]) / s2], 1)
        q3 = np.stack([(m[:, 1, 0] - m[:, 0, 1]) / s3, (m[:, 0, 2] + m[:, 2, 0]) / s3, (m[:, 1, 2] + m[:, 2, 1]) / s3, 0.25 * s3], 1)
    use0 = trace > 0
    use1 = ~use0 & (r00 > r11) & (r00 > r22)
    use2 = ~use0 & ~use1 & (r11 > r22)
    q = np.where(use0[:, None], q0, np.where(use1[:, None], q1, np.where(use2[:, None], q2, q3)))
    return q
