"""The per-frame appearance network shipped in the stage-3 checkpoint.

`net.pt` holds the static Gaussians plus `shadow_net` (a 55M-parameter
StyleUNet) and a small `viewdir_net`. Gaussian Garments (Rong et al.
2409.08189) runs them per frame on an ambient-occlusion map and a normal map
of the deformed garment (baked in the UV atlas) and ADDS the output (48 SH
coefficients and a 3-vector offset per texel) to the static Gaussians. That
is the source of wrinkle shadows and self-occlusion.

Ported from `scene/avatar_net.py`, `scene/avatar_gaussian_model.py` and
`scene/styleunet/styleunet.py` of eth-ait/Gaussian-Garments. Those files
carry non-commercial licences (Inria GRAPHDECO, NVIDIA Source Code
License-NC) and the repo has no LICENSE: treat this port and the weights as
research/evaluation use only.

Reproduced from the reference code path (no CUDA run available):

* texel<->Gaussian binding and barycentric anchors,
* the camera-space normal transform, including the quirk of applying the
  transposed world-to-view matrix (the network was trained with it),
* the view-direction map, including `viewdir_map[gs_u, gs_u]`; harmless
  because `SWGAN_unet.forward` adds the view feature after the last
  `to_rgb`, so it never reaches the output,
* the StyleUNet forward with fixed noise and constant style `1/sqrt(512)`.

Not the reference: the AO and normal maps. The paper bakes them in Blender
Cycles; `bake_maps` uses flat face normals in the file's y-up frame and a
cosine-weighted hemisphere ray cast (open3d, 32 rays, 10 m range),
dilated 5 texels.

Conventions: `view` is the rasteriser's world-to-camera 4x4 (OpenCV
`[R|t]`, +x right, +y down, +z forward). The scan world is y-up with the
subject facing +z.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from numba import njit

from tryon.garment.async_worker import IntervalWorker
from tryon.rendering.splat_cpu import face_frames, face_frames_numba
from tryon.garment.styleunet import SWGAN_unet
from tryon.garment.texels import TEXTURE_SIZE, TexelBinding, bake_maps, read_template

SH_DEGREE = 3
SH_COEFFS = (SH_DEGREE + 1) ** 2  # 16
OUT_CHANNELS = SH_COEFFS * 3 + 3  # 51: 3 xyz offset + 48 SH


@njit(cache=True)
def _world_xyz_numba(vertices, faces, binding, bc, local_xyz, frames, scale):
    """Compiled `AppearanceNet._world_xyz_numpy`: frame @ local * scale + anchor.

    Serial on purpose: shading workers call it while the loop runs
    `_place_splats_numba`, and numba's workqueue aborts on two parallel
    regions at once."""
    n = len(binding)
    xyz = np.empty((n, 3))
    for s in range(n):
        f = binding[s]
        i0, i1, i2 = faces[f, 0], faces[f, 1], faces[f, 2]
        for r in range(3):
            anchor = bc[s, 0] * vertices[i0, r] + bc[s, 1] * vertices[i1, r] + bc[s, 2] * vertices[i2, r]
            xyz[s, r] = (
                frames[f, r, 0] * local_xyz[s, 0] + frames[f, r, 1] * local_xyz[s, 1] + frames[f, r, 2] * local_xyz[s, 2]
            ) * scale[f] + anchor
    return xyz


def _world_to_view_transposed(view):
    """`Camera.world_view_transform` for a w2c 4x4 `view`, as `scene/cameras.py` builds it."""
    view = np.asarray(view, np.float64)
    R, T = view[:3, :3].T, view[:3, 3]
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.T
    Rt[:3, 3] = T
    Rt[3, 3] = 1.0
    return np.float32(Rt).T


def pick_device(device=None):
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AppearanceNet:
    """`shadow_net` + `viewdir_net` from `net.pt` and the static Gaussians they modify."""

    def __init__(self, checkpoint_dir, device=None, texture_size=TEXTURE_SIZE, faces=None, half=False, network=True):
        """`half`: run `shadow_net` in float16 (~2x faster on MPS, under a grey
        level of colour error). Off by default: float32 is the reference.

        `network=False`: only the static Gaussians and texel binding (what
        `shading_student.ShadingStudent` needs)."""
        checkpoint_dir = Path(checkpoint_dir)
        self.device = pick_device(device)
        self.dtype = torch.float16 if half else torch.float32
        self.texture_size = texture_size
        state = torch.load(checkpoint_dir / "stage3" / "ckpt" / "net.pt", map_location="cpu")
        self.sh_degree = int(state.get("activate_sh_degree", SH_DEGREE))
        state = state["avatar_net"]

        self.template = read_template(checkpoint_dir / "stage1" / "template_uv.obj")
        self.binding = TexelBinding(self.template, texture_size)
        stored = state["xyz"].shape[0]
        if self.binding.num_gs != stored:
            raise ValueError(
                f"UV atlas covers {self.binding.num_gs} texels at {texture_size}^2 but the "
                f"checkpoint stores {stored} Gaussians; the texel binding is wrong")
        # The renderer's UV-split mesh and the reference's index the same
        # faces in the same order; accept either by vertex count.
        self.faces_by_vertex_count = {len(self.template["vertices"]): self.template["faces"]}
        if faces is not None:
            faces = np.asarray(faces)
            self.faces_by_vertex_count[int(faces.max()) + 1] = faces

        self.xyz = state["xyz"].numpy().astype(np.float64)  # (N, 3) face-local
        self.feature_dc = state["feature_dc"].numpy().astype(np.float64)  # (N, 1, 3)
        self.feature_rest = state["feature_rest"].numpy().astype(np.float64)  # (N, 15, 3)
        self.opacity = state["opacity"].numpy().astype(np.float64)
        self.scaling = state["scaling"].numpy().astype(np.float64)
        self.rotation = state["rotation"].numpy().astype(np.float64)

        self.num_gs = self.binding.num_gs
        self.last_timing = {}
        if not network:
            self.shadow_net = self.viewdir_net = None
            return
        style_dim = texture_size
        self.shadow_net = SWGAN_unet(texture_size, 4, (self.sh_degree + 1) ** 2 * 3 + 3, texture_size, style_dim, 2)
        self.viewdir_net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(64, 128, 4, 2, 1))
        self.shadow_net.load_state_dict(
            {k[len("shadow_net."):]: v for k, v in state.items() if k.startswith("shadow_net.")}, strict=True)
        self.viewdir_net.load_state_dict(
            {k[len("viewdir_net."):]: v for k, v in state.items() if k.startswith("viewdir_net.")}, strict=True)
        self.shadow_net.eval().to(self.device, self.dtype)
        self.viewdir_net.eval().to(self.device, self.dtype)
        self.shadow_style = torch.ones([1, style_dim], dtype=self.dtype, device=self.device) / math.sqrt(style_dim)
        self.num_gs = self.binding.num_gs
        self.last_timing = {}

    def faces_for(self, vertices):
        try:
            return self.faces_by_vertex_count[len(vertices)]
        except KeyError:
            raise ValueError(
                f"{len(vertices)} vertices match neither the template ({len(self.template['vertices'])}) "
                f"nor the faces this net was given ({list(self.faces_by_vertex_count)})") from None

    def bake(self, vertices, occluders=(), **kwargs):
        """AO and normal maps for a deformed mesh - see `bake_maps`."""
        return bake_maps(vertices, self.faces_for(vertices), self.binding, occluders, **kwargs)

    def static_features(self):
        """`get_features`: (N, 16, 3) static SH, coefficient-major."""
        return np.concatenate([self.feature_dc, self.feature_rest], axis=1)

    def condition_maps(self, normal, view):
        """Camera-space normal channels (3, res, res): the normal map decoded
        from [0, 1], rotated by `world_view_transform` as the reference does,
        and re-normalised."""
        w2v = _world_to_view_transposed(view).astype(np.float64)
        normal = np.asarray(normal, np.float64)  # (H, W, 3)
        covered = normal.sum(-1) > 0
        nw_pad = np.concatenate([normal * 2 - 1, np.ones(normal.shape[:2] + (1,))], axis=-1) * covered[..., None]
        nc = nw_pad @ w2v[:3].T  # (H, W, 3) = (W @ [n; 1])[:3] per texel
        norm = np.linalg.norm(nc, axis=-1, keepdims=True)
        return (nc / np.maximum(norm, 1e-12)).transpose(2, 0, 1)

    def view_feature(self, anchor, orientation, view):
        """`viewdir_net` on the reference's view-direction map; fidelity checks
        only (it never reaches the output, so `forward` skips it)."""
        camera_center = np.linalg.inv(_world_to_view_transposed(view).astype(np.float64))[3, :3]
        dir_pp = anchor - camera_center
        dir_pp = dir_pp / np.linalg.norm(dir_pp, axis=1, keepdims=True)
        local_viewdir = np.einsum("nji,nj->ni", orientation[self.binding.binding], dir_pp)  # R^T d
        H = W = self.texture_size
        viewdir_map = torch.zeros(H, W, 3)
        # as in the reference (sic): gs_u used for both indices
        viewdir_map[self.binding.gs_u, self.binding.gs_u] = torch.as_tensor(local_viewdir, dtype=torch.float32)
        return self.viewdir_net(viewdir_map.permute(2, 0, 1)[None].to(self.device, self.dtype))

    @torch.no_grad()
    def forward(self, vertices, view, ambient=None, normal=None, occluders=(), use_viewdir=False):
        """One frame of `AvatarNet.forward`.

        `vertices`: deformed garment mesh. `view`: world-to-camera 4x4.
        `ambient` (res, res) and `normal` (res, res, 3) in [0, 1]; baked
        from the mesh when not given.

        Returns numpy arrays: `feature_dc` (N, 1, 3) and `feature_rest`
        (N, 15, 3) (static + shadow SH); `xyz` (N, 3) world positions of
        `local_xyz` (stored xyz + offset); `offset`, `shadow_shs`, `anchor`
        and the two maps. Stage timings go to `self.last_timing`.
        """
        timing = {}
        vertices = np.asarray(vertices, np.float64)
        faces = self.faces_for(vertices)
        if ambient is None or normal is None:
            t0 = time.perf_counter()
            ambient, normal = self.bake(vertices, occluders)
            timing["bake"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        _, orientation, scale = face_frames(vertices, faces)  # face_orien_mat, face_scaling
        anchor = self.binding.barycentric_3d(vertices[faces])
        combined = np.concatenate(
            [np.asarray(ambient, np.float32)[None], self.condition_maps(normal, view).astype(np.float32)])
        combined = torch.as_tensor(combined)[None].to(self.device, self.dtype)
        view_feature = self.view_feature(anchor, orientation, view) if use_viewdir else None
        timing["prepare"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        shadow_out = self.shadow_net([self.shadow_style], combined, randomize_noise=False,
                                     view_feature=view_feature)[0]
        if self.device.type == "mps":
            torch.mps.synchronize()
        timing["network"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        # gather covered texels on the host; boolean indexing on MPS costs seconds
        maps = shadow_out[0].float().cpu().numpy()
        per_gaussian = maps[:, self.binding.gs_u, self.binding.gs_v].T.astype(np.float64)  # (N, 51)
        offset = per_gaussian[:, :3]
        shadow_shs = per_gaussian[:, 3:].reshape(self.num_gs, -1, 3)
        shs = self.static_features() + shadow_shs
        local_xyz = self.xyz + offset
        frames = orientation[self.binding.binding]
        xyz = np.einsum("nij,nj->ni", frames, local_xyz) * scale[self.binding.binding, None] + anchor
        timing["gather"] = time.perf_counter() - t0
        self.last_timing = timing
        return {
            "feature_dc": shs[:, :1],
            "feature_rest": shs[:, 1:],
            "xyz": xyz,
            "local_xyz": local_xyz,
            "offset": offset,
            "shadow_shs": shadow_shs,
            "anchor": anchor,
            "ambient": np.asarray(ambient),
            "normal": np.asarray(normal),
        }

    def world_xyz(self, vertices, local_xyz, frames=None):
        """World positions of face-local Gaussian positions (`forward`'s
        "local_xyz") on `vertices`, to place an earlier result on the
        current mesh. `_world_xyz_numpy` is the oracle.

        `frames`: (frames, scale) already computed for `vertices`; computed
        here when omitted."""
        vertices = np.ascontiguousarray(vertices, np.float64)
        faces = np.ascontiguousarray(self.faces_for(vertices), np.int64)
        if frames is None:
            frames = face_frames_numba(vertices, faces)
        return _world_xyz_numba(vertices, faces, np.ascontiguousarray(self.binding.binding, np.int64),
                                np.ascontiguousarray(self.binding.bc, np.float64), np.ascontiguousarray(local_xyz, np.float64), *frames)

    def _world_xyz_numpy(self, vertices, local_xyz):
        vertices = np.asarray(vertices, np.float64)
        faces = self.faces_for(vertices)
        _, orientation, scale = face_frames(vertices, faces)
        anchor = self.binding.barycentric_3d(vertices[faces])
        frames = orientation[self.binding.binding]
        return np.einsum("nij,nj->ni", frames, local_xyz) * scale[self.binding.binding, None] + anchor

    def static_xyz(self, vertices):
        """World positions of the static Gaussians (`get_xyz`); no network."""
        vertices = np.asarray(vertices, np.float64)
        faces = self.faces_for(vertices)
        _, orientation, scale = face_frames(vertices, faces)
        anchor = self.binding.barycentric_3d(vertices[faces])
        frames = orientation[self.binding.binding]
        return np.einsum("nij,nj->ni", frames, self.xyz) * scale[self.binding.binding, None] + anchor


def merge(splats, result, indices=None, keys=("feature_dc", "feature_rest", "xyz")):
    """Copy of `splats` with `keys` replaced by the network's values.

    `indices`: which of the checkpoint's N Gaussians each entry of `splats`
    is, when it holds a subset; omitted means all N in order."""
    count = len(splats["xyz"])
    if indices is None:
        if count != len(result["xyz"]):
            raise ValueError(
                f"splats hold {count} Gaussians but the network produced {len(result['xyz'])}; "
                "pass `indices` for a subset")
        indices = slice(None)
    elif len(indices) != count:
        # a mismatch would crash the compiled rasteriser with out-of-bounds reads
        raise ValueError(f"splats hold {count} Gaussians but {len(indices)} indices were given")
    out = dict(splats)
    for key in keys:
        out[key] = result[key][indices]
    return out


def apply(splats, net: AppearanceNet, vertices, view, indices=None, **forward_kwargs):
    """Run `net` on the frame and merge its output into `splats`."""
    return merge(splats, net.forward(vertices, view, **forward_kwargs), indices)


class AsyncAppearance(IntervalWorker):
    """`AppearanceNet.forward` on a worker thread for the live loop.

    Shading changes slowly and is bound to texels, so a slightly old result
    still fits: `latest(vertices)` returns the newest result re-placed on
    the CURRENT mesh. Uses 8 AO rays (vs 32 in the reference) for speed.
    """

    RAYS = 8
    INTERVAL = 2.0  # s between shading passes

    def __init__(self, checkpoint_dir, faces=None, half=True):
        self.net = AppearanceNet(checkpoint_dir, faces=faces, half=half)
        super().__init__()

    def _closed_on_thread(self):
        self.net = None  # drop the network (switching garments)

    def _make_job(self, vertices, view, occluders):
        return (np.array(vertices, np.float64), np.array(view), list(occluders))

    def _compute(self, job):
        vertices, view, occluders = job
        ambient, normal = self.net.bake(vertices, occluders, rays=self.RAYS)
        result = self.net.forward(vertices, view, ambient=ambient, normal=normal)
        return {k: np.ascontiguousarray(result[k]) for k in ("feature_dc", "feature_rest", "local_xyz")}

    def submit(self, vertices, view, occluders=()):
        """Queue this frame for shading unless a frame is being shaded."""
        super().submit(vertices, view, occluders)

    def latest(self, vertices, frames=None):
        """The newest result on `vertices`, or None before the first.
        `frames`: see `AppearanceNet.world_xyz`."""
        result = self._newest()
        if result is None:
            return None
        return {"feature_dc": result["feature_dc"], "feature_rest": result["feature_rest"],
                "xyz": self.net.world_xyz(vertices, result["local_xyz"], frames)}
