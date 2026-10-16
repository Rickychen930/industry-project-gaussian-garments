"""A small network that shades the garment every frame, distilled from the
checkpoint's appearance network (`appearance.py`, the teacher).

The teacher needs an AO bake plus a 55M-parameter StyleUNet, too slow for
every frame. The student reads cheap inputs at 256 x 256 in the UV atlas:

  camera-space normals   posed face normal per texel, in camera frame (3)
  body distance          distance to the nearest body vertex, clipped at 10 cm (1)
  coverage               texels that hold a Gaussian (1)

and predicts the teacher's 51 channels (3 position offset + 48 SH
coefficients), read bilinearly per Gaussian. Trained per garment by
`tools/train/shading_student.py`.

A student may store a per-Gaussian mean of the teacher's output and predict
only the residual: the fine garment-fixed detail cannot fit in a 256 map
without surface coordinates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F

from tryon.garment.async_worker import IntervalWorker


SIZE = 256
IN_CHANNELS = 5
OUT_CHANNELS = 51
BODY_DISTANCE_CLIP = 0.10  # m
FILE_NAME = "shading_student.pt"


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())


class StudentUNet(nn.Module):
    def __init__(self, widths=(32, 64, 128, 192)):
        super().__init__()
        self.down = nn.ModuleList()
        cin = IN_CHANNELS
        for w in widths:
            self.down.append(_block(cin, w))
            cin = w
        self.up = nn.ModuleList()
        for skip, w in zip(reversed(widths[:-1]), reversed(widths[1:])):
            self.up.append(_block(w + skip, skip))
        self.head = nn.Conv2d(widths[0], OUT_CHANNELS, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.down):
            x = block(x)
            if i < len(self.down) - 1:
                skips.append(x)
                x = F.avg_pool2d(x, 2)
        for block in self.up:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skips.pop()], 1))
        return self.head(x)


class StudentInputs:
    """Student input maps for a posed garment, from `texels.TexelBinding` pooled to SIZE x SIZE."""

    def __init__(self, binding, faces):
        self.binding = binding
        self.faces = np.asarray(faces)
        scale = binding.res // SIZE
        self.cell = (binding.gs_u // scale) * SIZE + (binding.gs_v // scale)
        self.count = np.bincount(self.cell, minlength=SIZE * SIZE).astype(np.float32)
        self.coverage = (self.count > 0).astype(np.float32).reshape(SIZE, SIZE)
        # bilinear read of each Gaussian's texel centre: four neighbours and
        # weights, precomputed (grid_sample has no backward on MPS in torch 2.1)
        row = (binding.gs_u + 0.5) * SIZE / binding.res - 0.5
        col = (binding.gs_v + 0.5) * SIZE / binding.res - 0.5
        r0, c0 = np.floor(row).astype(int), np.floor(col).astype(int)
        fr, fc = row - r0, col - c0
        index, weight = [], []
        for dr, dc, w in ((0, 0, (1 - fr) * (1 - fc)), (0, 1, (1 - fr) * fc), (1, 0, fr * (1 - fc)), (1, 1, fr * fc)):
            index.append(np.clip(r0 + dr, 0, SIZE - 1) * SIZE + np.clip(c0 + dc, 0, SIZE - 1))
            weight.append(w)
        self.index = torch.as_tensor(np.stack(index), dtype=torch.long)  # (4, N)
        self.weight = torch.as_tensor(np.stack(weight), dtype=torch.float32)  # (4, N)

    def _pool(self, values):
        out = np.zeros((SIZE * SIZE, values.shape[1]), np.float32)
        for c in range(values.shape[1]):
            out[:, c] = np.bincount(self.cell, weights=values[:, c], minlength=SIZE * SIZE)
        out /= np.maximum(self.count, 1.0)[:, None]
        return out.reshape(SIZE, SIZE, -1)

    def maps(self, vertices, view, body_vertices, per_vertex_distance=False):
        """(IN_CHANNELS, SIZE, SIZE) float32.

        `per_vertex_distance`: query body distance at mesh vertices and
        interpolate to texels (~11x fewer queries; training used per-texel)."""
        vertices = np.asarray(vertices, np.float64)
        triangles = vertices[self.faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
        camera_normals = normals[self.binding.binding] @ np.asarray(view, np.float64)[:3, :3].T
        tree = cKDTree(np.asarray(body_vertices, np.float64))
        if per_vertex_distance:
            at_vertex, _ = tree.query(vertices, k=1, workers=4)
            distance = np.einsum("nk,nk->n", self.binding.bc, at_vertex[self.faces[self.binding.binding]])
        else:
            distance, _ = tree.query(self.binding.barycentric_3d(triangles), k=1, workers=4)
        body = np.clip(distance / BODY_DISTANCE_CLIP, 0.0, 1.0)[:, None]
        pooled = self._pool(np.hstack([camera_normals, body]).astype(np.float32))
        return np.concatenate([pooled.transpose(2, 0, 1), self.coverage[None]]).astype(np.float32)

    def gather_numpy(self, output):
        """`gather` on the CPU: (N, 51) float64 from a (51, SIZE, SIZE) array (MPS round trips cost ~100 ms)."""
        if not hasattr(self, "_sampler"):
            # bilinear read as a sparse (N, SIZE*SIZE) matrix, 4 entries a row
            from scipy.sparse import csr_matrix

            index, weight = self.index.cpu().numpy(), self.weight.cpu().numpy()
            rows = np.tile(np.arange(index.shape[1]), 4)
            self._sampler = csr_matrix((weight.ravel().astype(np.float32), (rows, index.ravel())), shape=(index.shape[1], SIZE * SIZE))
        flat = np.ascontiguousarray(np.asarray(output, np.float32).reshape(output.shape[0], -1).T)  # (SIZE*SIZE, 51)
        return np.asarray(self._sampler @ flat, np.float64)

    def gather(self, output):
        """(N, 51) per Gaussian from the student's (1, 51, SIZE, SIZE) output."""
        if self.index.device != output.device:
            self.index, self.weight = self.index.to(output.device), self.weight.to(output.device)
        flat = output[0].reshape(output.shape[1], -1)  # (51, SIZE*SIZE)
        values = sum(flat[:, self.index[k]] * self.weight[k].to(output.dtype) for k in range(4))
        return values.T


class ShadingStudent:
    """Inference: stands in for `AppearanceNet.forward` with the same result keys."""

    def __init__(self, checkpoint_dir, faces=None, device=None, weights=None):
        from tryon.garment.appearance import AppearanceNet

        # reuse the teacher's static Gaussians and binding without its network
        self.static = AppearanceNet(checkpoint_dir, faces=faces, device="cpu", network=False)
        self.inputs = StudentInputs(self.static.binding, self.static.template["faces"] if faces is None else faces)
        self.device = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        path = Path(weights) if weights else Path(checkpoint_dir) / "stage3" / FILE_NAME
        saved = torch.load(path, map_location="cpu")
        # `--widths` is recorded when non-default
        self.model = (StudentUNet(tuple(saved["widths"])) if saved.get("widths") else StudentUNet()).eval()
        self.model.load_state_dict(saved["model"])
        self.mean = None if saved.get("mean") is None else np.asarray(saved["mean"], np.float64)
        # fp16 on the GPU (the SH it predicts are small corrections)
        self.dtype = torch.float16 if self.device.type == "mps" else torch.float32
        self.model = self.model.to(self.device, self.dtype)
        self.per_vertex_distance = False
        # Scale of the predicted shading added to the static colours (1.0 = teacher's); a measuring flag.
        self.shadow_strength = 1.0

    @torch.no_grad()
    def forward(self, vertices, view, body_vertices):
        vertices = np.asarray(vertices, np.float64)
        maps = self.inputs.maps(vertices, view, body_vertices, per_vertex_distance=self.per_vertex_distance)
        out = self.model(torch.as_tensor(maps)[None].to(self.device, self.dtype))[0].float().cpu().numpy()
        per_gaussian = self.inputs.gather_numpy(out)
        if self.mean is not None:
            per_gaussian += self.mean
        offset, shadow = per_gaussian[:, :3], per_gaussian[:, 3:].reshape(-1, 16, 3)
        shs = self.static.static_features() + self.shadow_strength * shadow
        local_xyz = self.static.xyz + offset
        return {"feature_dc": shs[:, :1], "feature_rest": shs[:, 1:],
                "xyz": self.static.world_xyz(vertices, local_xyz), "local_xyz": local_xyz}

    @staticmethod
    def exists(checkpoint_dir):
        return (Path(checkpoint_dir) / "stage3" / FILE_NAME).exists()


class AsyncStudent(IntervalWorker):
    """`ShadingStudent.forward` on a worker thread every INTERVAL seconds; each
    frame draws the newest result placed on the current mesh (as
    `appearance.AsyncAppearance`). At equal age the teacher is better; the
    student wins by being cheap enough to be fresh."""

    INTERVAL = 0.7  # s: continuous passes starve the body regressor on the GPU

    def __init__(self, student):
        self.student = student
        super().__init__()

    def _make_job(self, vertices, view, body_vertices):
        return (np.array(vertices, np.float64), np.array(view), np.array(body_vertices, np.float64))

    def _compute(self, job):
        result = self.student.forward(*job)
        # contiguous once per pass; slices of one array would be copied every frame
        return {k: np.ascontiguousarray(result[k]) for k in ("feature_dc", "feature_rest", "local_xyz")}

    def latest(self, vertices, frames=None):
        """The newest result on `vertices` (`frames`: see `AppearanceNet.world_xyz`), or None before the first."""
        result = self._newest()
        if result is None:
            return None
        return {"feature_dc": result["feature_dc"], "feature_rest": result["feature_rest"],
                "xyz": self.student.static.world_xyz(vertices, result["local_xyz"], frames)}
