"""The appearance network's texture space: the garment template as the
reference reads it, the binding of every Gaussian to a texel of the
512 x 512 UV atlas (with its barycentric anchor on the face), and the
ambient-occlusion and normal maps the network takes as input, baked in
that atlas (`bake_maps`, the replacement for the reference's Blender
bake; see `appearance.py`).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from tryon.garment.raycast import build_scene

TEXTURE_SIZE = 512
TEXTURE_MARGIN = 5  # Blender bake margin used by the reference


def read_template(path):
    """`utils/io_utils.py::read_obj`: vertices, uvs, faces, texture_faces."""
    vertices, uvs, faces, texture_faces = [], [], [], []
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                vertices.append(tuple(map(float, parts[1:])))
            elif parts[0] == "vt":
                uvs.append(tuple(map(float, parts[1:])))
            elif parts[0] == "f":
                faces.append(tuple(int(p.split("/")[0]) for p in parts[1:]))
                texture_faces.append(tuple(int(p.split("/")[1]) for p in parts[1:]))
    return {
        "vertices": np.array(vertices, np.float32),
        "uvs": np.array(uvs, np.float32),
        "faces": np.array(faces) - 1,
        "texture_faces": np.array(texture_faces) - 1,
    }


def _rasterize(triangle, scale):
    """`AvatarGaussianModel.rasterize`, line for line: scanline over pixel
    centres, sorted by y, split at the middle vertex."""
    pixels = []

    def itp(end1, end2, target, axis):
        return end1 + (end2 - end1) * (target - end1[axis]) / (end2[axis] - end1[axis])

    triangle = np.array(triangle) * scale
    miny, maxy = np.floor(triangle[:, 1].min()), np.floor(triangle[:, 1].max())
    v1, v2, v3 = triangle[triangle[:, 1].argsort()]
    v4 = itp(v1, v3, v2[1], 1)
    for _y in range(int(miny), int(maxy) + 1):
        y = _y + 0.5
        if y < v1[1] or y >= v3[1]:
            continue
        elif y < v2[1]:
            fmin, fmax = sorted([itp(v1, v2, y, 1)[0], itp(v1, v4, y, 1)[0]])
        else:
            fmin, fmax = sorted([itp(v3, v2, y, 1)[0], itp(v3, v4, y, 1)[0]])
        minx, maxx = np.floor(fmin), np.floor(fmax)
        for _x in range(int(minx), int(maxx) + 1):
            x = _x + 0.5
            if x < fmin or x >= fmax:
                continue
            pixels.append([_x, _y])
    return np.array(pixels)


@njit(cache=True)
def _texture_binding_numba(triangles, res):
    """`_texture_binding_python` compiled: `_rasterize`'s scanline per face,
    later faces overwriting earlier ones (~1.1 s of Python per call, twice
    per garment load). `triangles`: the UV triangles already scaled by
    `res`, float32 as the reference has them. The arithmetic follows NumPy
    1.x's promotion in `_rasterize` exactly - float32 throughout, except
    `y - end1[1]` with the Python float y, which NumPy computes in float64
    and rounds to float32 when it scales the float32 edge; float64
    everywhere put one texel more or less on 2 of 15 garments."""
    bind_map = np.full((res, res), -1, np.int64)
    for index in range(len(triangles)):
        tri = triangles[index]
        # stable sort of the three corners by y (numpy's argsort of 3 values)
        a, b, c = 0, 1, 2
        if tri[b, 1] < tri[a, 1]:
            a, b = b, a
        if tri[c, 1] < tri[b, 1]:
            b, c = c, b
            if tri[b, 1] < tri[a, 1]:
                a, b = b, a
        v1x, v1y = tri[a, 0], tri[a, 1]
        v2x, v2y = tri[b, 0], tri[b, 1]
        v3x, v3y = tri[c, 0], tri[c, 1]
        miny = np.floor(min(tri[0, 1], tri[1, 1], tri[2, 1]))
        maxy = np.floor(max(tri[0, 1], tri[1, 1], tri[2, 1]))
        # v4 = itp(v1, v3, v2[1], 1), all float32
        d = v2y - v1y
        v4x = v1x + ((v3x - v1x) * d) / (v3y - v1y)
        v4y = v1y + ((v3y - v1y) * d) / (v3y - v1y)
        for _y in range(int(miny), int(maxy) + 1):
            y = _y + 0.5
            if y < v1y or y >= v3y:
                continue
            elif y < v2y:
                t1 = np.float32(y - np.float64(v1y))
                e1 = v1x + ((v2x - v1x) * t1) / (v2y - v1y)
                e2 = v1x + ((v4x - v1x) * t1) / (v4y - v1y)
            else:
                t3 = np.float32(y - np.float64(v3y))
                e1 = v3x + ((v2x - v3x) * t3) / (v2y - v3y)
                e2 = v3x + ((v4x - v3x) * t3) / (v4y - v3y)
            fmin, fmax = min(e1, e2), max(e1, e2)
            for _x in range(int(np.floor(fmin)), int(np.floor(fmax)) + 1):
                x = _x + 0.5
                if x < fmin or x >= fmax:
                    continue
                bind_map[_y, _x] = index
    return bind_map


def texture_binding(uvs, texture_faces, res=TEXTURE_SIZE):
    """`get_texture_binding`: face id per texel, -1 where uncovered.
    Compiled (`_texture_binding_numba`); `_texture_binding_python` is the
    line-for-line port it is tested against."""
    # scaled as `_rasterize` does: np.array(triangle) * scale keeps the uvs' dtype
    triangles = np.ascontiguousarray(np.asarray(uvs)[np.asarray(texture_faces)] * res, np.float32)
    return _texture_binding_numba(triangles, res)


def _texture_binding_python(uvs, texture_faces, res=TEXTURE_SIZE):
    """`get_texture_binding`: face id per texel, -1 where uncovered.

    Row = floor(v * res), column = floor(u * res), no vertical flip - the
    same orientation Blender's `image.pixels` (bottom row first) gives the
    baked maps, which is why the reference never flips either.
    """
    bind_map = np.full((res, res), -1, np.int64)
    for index, face in enumerate(texture_faces):
        pixels = _rasterize(uvs[face], res)
        if len(pixels) == 0:
            continue
        bind_map[pixels[:, 1], pixels[:, 0]] = index
    return bind_map


def barycentric_2d(triangles, points):
    """`utils/geometry_utils.py::barycentric_2D` in numpy: (alpha, beta, gamma)."""
    triangles = np.concatenate([triangles, np.zeros(triangles.shape[:2] + (1,))], axis=-1)
    points = np.concatenate([points, np.zeros((len(points), 1))], axis=-1)
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    normal = np.cross(b - a, c - a)
    area = np.linalg.norm(normal, axis=-1)
    normal = normal / area[:, None]
    alpha = (np.cross(b - points, c - points) * normal).sum(-1) / area
    beta = (np.cross(c - points, a - points) * normal).sum(-1) / area
    return alpha, beta, 1 - alpha - beta


class TexelBinding:
    """Everything `AvatarGaussianModel.__init__` derives from the UV atlas."""

    def __init__(self, template, res=TEXTURE_SIZE):
        self.res = res
        self.bind_map = texture_binding(template["uvs"], template["texture_faces"], res)
        self.gaussian_mask = self.bind_map > -1
        self.binding = self.bind_map[self.gaussian_mask]
        self.num_gs = len(self.binding)
        self.gs_u, self.gs_v = np.where(self.gaussian_mask)
        uv_triangles = template["uvs"][template["texture_faces"]][self.binding] * res
        uv_gs = np.array([self.gs_v, self.gs_u]).T + 0.5
        self.bc = np.stack(barycentric_2d(uv_triangles.astype(np.float64), uv_gs), axis=1)  # (N, 3)

    def barycentric_3d(self, triangles):
        """`get_barycentric_3d`: the texel's point on the (deformed) mesh."""
        tri = np.asarray(triangles, np.float64)[self.binding]
        return np.einsum("nk,nkd->nd", self.bc, tri)


def _hemisphere_directions(count, seed=0):
    """Cosine-weighted directions around +z, fixed for reproducibility."""
    rng = np.random.default_rng(seed)
    u1, u2 = rng.random(count), rng.random(count)
    r = np.sqrt(u1)
    phi = 2 * np.pi * u2
    return np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(1 - u1)], axis=1)


def _tangent_frames(normals):
    helper = np.where(np.abs(normals[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    t1 = np.cross(helper, normals)
    t1 /= np.maximum(np.linalg.norm(t1, axis=1, keepdims=True), 1e-12)
    t2 = np.cross(normals, t1)
    return t1, t2


_DILATION_CACHE = {}


def _dilate_map(values, covered, margin):
    """Fill texels within `margin` of a covered texel with the nearest
    covered value, as Blender's bake margin does. The nearest-texel map
    depends only on the (fixed) atlas, so it is computed once per atlas."""
    from scipy import ndimage

    key = (covered.shape, margin, hash(covered.tobytes()))
    if key not in _DILATION_CACHE:
        distance, (rows, cols) = ndimage.distance_transform_edt(~covered, return_indices=True)
        fill = (~covered) & (distance <= margin)
        _DILATION_CACHE[key] = (fill, rows[fill], cols[fill])
    fill, rows, cols = _DILATION_CACHE[key]
    out = values.copy()
    out[fill] = values[rows, cols]
    return out


def bake_maps(vertices, faces, binding: TexelBinding, occluders=(), rays=32, seed=0,
              max_distance=10.0, margin=TEXTURE_MARGIN, per_vertex=False):
    """Ambient occlusion (res, res) and object-space normal (res, res, 3)
    maps in [0, 1], laid out like the reference's PNGs.

    `occluders`: extra (vertices, faces) meshes - the body, other garments
    - that block light in the AO, as the reference passes the body PLY and
    the other garments to Blender.

    `per_vertex`: cast the AO rays from the mesh's vertices (area-weighted
    normals) and interpolate them to the texels barycentrically, instead
    of from every texel: ~8k ray origins instead of ~93k, for the live
    loop (see `AsyncAppearance`). AO varies smoothly across a face, so
    little is lost; measured in `AsyncAppearance`'s docstring.
    """
    import open3d as o3d

    vertices = np.asarray(vertices, np.float64)
    faces = np.asarray(faces)
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-20)

    points = binding.barycentric_3d(triangles)
    normals = face_normals[binding.binding]

    scene = build_scene([(vertices, faces), *occluders])

    if per_vertex:
        area_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        vertex_normals = np.zeros_like(vertices)
        for k in range(3):
            np.add.at(vertex_normals, faces[:, k], area_normals)
        vertex_normals /= np.maximum(np.linalg.norm(vertex_normals, axis=1, keepdims=True), 1e-20)
        origins, directions_about = vertices, vertex_normals
    else:
        origins, directions_about = points, normals
    local = _hemisphere_directions(rays, seed).astype(np.float32)  # (rays, 3)
    t1, t2 = _tangent_frames(directions_about)
    ray_buffer = np.empty((len(origins), rays, 6), np.float32)
    ray_buffer[:, :, :3] = (origins + 1e-4 * directions_about)[:, None].astype(np.float32)
    frame = np.stack([t1, t2, directions_about], axis=1).astype(np.float32)  # (N, 3, 3) rows
    ray_buffer[:, :, 3:] = local[None] @ frame  # (N, rays, 3): local coords -> world
    hit = scene.cast_rays(o3d.core.Tensor(ray_buffer.reshape(-1, 6)))["t_hit"].numpy().reshape(len(origins), rays)
    ambient_values = 1.0 - (hit < max_distance).mean(axis=1)
    if per_vertex:
        ambient_values = np.einsum("nk,nk->n", binding.bc, ambient_values[faces[binding.binding]])

    res = binding.res
    ambient = np.zeros((res, res), np.float64)
    normal = np.zeros((res, res, 3), np.float64)
    ambient[binding.gs_u, binding.gs_v] = ambient_values
    normal[binding.gs_u, binding.gs_v] = normals * 0.5 + 0.5
    if margin > 0:
        ambient = _dilate_map(ambient, binding.gaussian_mask, margin)
        normal = _dilate_map(normal, binding.gaussian_mask, margin)
    return ambient, normal
