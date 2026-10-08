"""Reading a Gaussian Garments checkpoint (Rong et al., 3DV 2025).

A checkpoint directory holds `stage1/template_uv.obj` (the garment mesh,
with UVs) and `stage3/ckpt/net.pt` (the Gaussians: their positions,
rotations, scales and colours, each bound to a face of the template).
This module turns those files into the arrays the renderer works with;
the appearance network's own reading of the template (its original
vertex indexing) is in `texels.read_template`.
"""

from __future__ import annotations

import numpy as np
from numba import njit


BINDING_TOLERANCE = 4


def load_checkpoint(model_dir):
    """Splat tensors and template mesh for one trained garment."""
    import torch

    checkpoint = torch.load(
        model_dir / "stage3" / "ckpt" / "net.pt",
        map_location="cpu",
        weights_only=False,
    )["avatar_net"]
    return checkpoint, load_template(model_dir / "stage1" / "template_uv.obj")


def load_template(path):
    """The garment template, split into vertices by (position, UV) only.

    trimesh splits an OBJ's vertices on every distinct (v, vt, vn) corner.
    Most checkpoints store per-vertex normals, but 00122_outer,
    00134_upper and 10004_outer store ONE NORMAL PER FACE ("f 1/1/1
    3/2/1 2/3/1"), so every face got three vertices of its own: 47k
    vertices for 15.7k faces, a triangle soup with no shared edges. The
    cloth solver then simulated each triangle as a separate piece of
    cloth and the garment fell apart into shreds; the sleeve labels and
    anything else built on mesh adjacency broke with it. Normals are not
    used (the splats carry their own frames), so the vn indices are
    dropped and the split is by (v, vt) - 8,245 vertices for 00122_outer,
    the same scheme 00152_outer's normal-free file always had."""
    import io

    import trimesh

    lines = []
    with open(path) as handle:
        for line in handle:
            if line.startswith("f "):
                corners = [corner.split("/") for corner in line.split()[1:]]
                line = "f " + " ".join("/".join(c[:2]) if len(c) > 1 and c[1] else c[0] for c in corners) + "\n"
            elif line.startswith("vn "):
                continue
            lines.append(line)
    return trimesh.load(io.StringIO("".join(lines)), file_type="obj", process=False)


@njit(cache=True)
def _uv_binding_numba(triangles, resolution):
    """The loop of `_uv_binding_python` compiled: the same pixel-centre test
    in the same float64 arithmetic, later faces overwriting earlier ones
    (~3 s of Python per garment load)."""
    binding = np.full((resolution, resolution), -1, np.int32)
    for index in range(len(triangles)):
        ax, ay = triangles[index, 0, 0], triangles[index, 0, 1]
        bx, by = triangles[index, 1, 0], triangles[index, 1, 1]
        cx, cy = triangles[index, 2, 0], triangles[index, 2, 1]
        x0 = max(int(np.floor(min(ax, bx, cx))), 0)
        x1 = min(int(np.ceil(max(ax, bx, cx))) + 1, resolution)
        y0 = max(int(np.floor(min(ay, by, cy))), 0)
        y1 = min(int(np.ceil(max(ay, by, cy))) + 1, resolution)
        if x1 <= x0 or y1 <= y0:
            continue
        denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(denominator) < 1e-12:
            continue
        for y in range(y0, y1):
            py = y + 0.5
            for x in range(x0, x1):
                px = x + 0.5
                w0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denominator
                w1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denominator
                if w0 >= 0 and w1 >= 0 and (1 - w0 - w1) >= 0:
                    binding[y, x] = index
    return binding


def uv_binding(mesh, count):
    """Recover which face each Gaussian belongs to.

    The checkpoints store splats but not their binding, so it is rebuilt the
    way the model created it: rasterise the UV atlas and take one Gaussian
    per covered texel. Resolution is recovered by trying the standard sizes
    until the count matches exactly - it does match exactly, which is the
    check that the reconstruction is right rather than merely plausible.
    """
    uvs = np.asarray(mesh.visual.uv)
    faces = np.asarray(mesh.faces)
    for resolution in (512, 1024, 256, 2048):
        binding = _uv_binding_numba(np.ascontiguousarray(uvs[faces] * resolution, np.float64), resolution)
        covered = int((binding >= 0).sum())
        if count is None or abs(covered - count) <= BINDING_TOLERANCE:
            return binding[binding >= 0], resolution
    raise SystemExit(
        f"Could not reproduce the {count} stored Gaussians from the UV atlas at "
        "any standard resolution; the checkpoint and template may not match."
    )


def _uv_binding_python(mesh, count):
    """The original per-face numpy loop; the oracle for `uv_binding`."""
    uvs = np.asarray(mesh.visual.uv)
    faces = np.asarray(mesh.faces)
    for resolution in (512, 1024, 256, 2048):
        binding = np.full((resolution, resolution), -1, np.int32)
        triangles = uvs[faces] * resolution
        for index, triangle in enumerate(triangles):
            x0 = max(int(np.floor(triangle[:, 0].min())), 0)
            x1 = min(int(np.ceil(triangle[:, 0].max())) + 1, resolution)
            y0 = max(int(np.floor(triangle[:, 1].min())), 0)
            y1 = min(int(np.ceil(triangle[:, 1].max())) + 1, resolution)
            if x1 <= x0 or y1 <= y0:
                continue
            ys, xs = np.mgrid[y0:y1, x0:x1]
            px, py = xs + 0.5, ys + 0.5  # pixel centres, as the model used
            (ax, ay), (bx, by), (cx, cy) = triangle
            denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(denominator) < 1e-12:
                continue
            w0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denominator
            w1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denominator
            inside = (w0 >= 0) & (w1 >= 0) & ((1 - w0 - w1) >= 0)
            if inside.any():
                binding[ys[inside], xs[inside]] = index
        covered = int((binding >= 0).sum())
        # `count=None` means "whatever this atlas yields" - used by tests and
        # by callers with no checkpoint to match.
        if count is None or abs(covered - count) <= BINDING_TOLERANCE:
            return binding[binding >= 0], resolution
    raise SystemExit(
        f"Could not reproduce the {count} stored Gaussians from the UV atlas at "
        "any standard resolution; the checkpoint and template may not match."
    )


RELAXED_REST_NAME = "relaxed_rest.npz"


def relaxed_rest_path(model_dir):
    """Where a garment's relaxed rest offset lives: next to `scan_pose.npz`."""
    from pathlib import Path

    return Path(model_dir) / "stage1" / RELAXED_REST_NAME


def load_relaxed_rest(model_dir, vertex_count):
    """(vertex_count, 3) offset of the garment's relaxed rest shape from its
    template, in the skinning's zero-pose frame at the scan subject's size
    (`tools/evaluate/relaxed_rest.py`); None when the file is absent or does
    not fit this mesh, so a garment without one binds as before."""
    path = relaxed_rest_path(model_dir)
    if not path.exists():
        return None
    try:
        offset = np.asarray(np.load(path)["offset"], np.float64)
    except (OSError, KeyError, ValueError):
        return None
    return offset if offset.shape == (vertex_count, 3) else None
