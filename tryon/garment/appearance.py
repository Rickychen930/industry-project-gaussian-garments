"""The per-frame appearance network that ships in the stage-3 checkpoint.

`net.pt` holds more than the 93,491 static Gaussians `tryon/rendering/splat_cpu.py`
renders: 55M parameters of `shadow_net` (a StyleUNet) and a small
`viewdir_net`. At capture time Gaussian Garments (Rong et al. 2409.08189)
ran them on every frame - input: a 512x512 ambient-occlusion map and a
normal map of the deformed garment, baked in the UV atlas - and ADDED the
output to the static Gaussians: 48 spherical-harmonic coefficients and a
3-vector position offset per texel. That is where the studio shading,
wrinkle shadows and cloth self-occlusion come from; the static SH alone
reads flat. This module makes that network usable here.

Ported from the reference implementation, `scene/avatar_net.py`,
`scene/avatar_gaussian_model.py` and `scene/styleunet/styleunet.py` of
https://github.com/eth-ait/Gaussian-Garments (ETH AIT). The Gaussian
model files there carry the Inria GRAPHDECO header ("free for
non-commercial, research and evaluation use"); the StyleUNet is the
rosinality StyleGAN2 port (MIT) as adapted by StyleAvatar/Animatable
Gaussians, whose CUDA kernels carry the NVIDIA Source Code License-NC.
The repository root has no LICENSE file. Treat this port, and the
checkpoint weights, as research/evaluation use only.

What is reproduced exactly (verified against the reference code path,
not against a CUDA run - none is available here):

* the texel<->Gaussian binding (`get_texture_binding` + `rasterize`,
  ported line for line; 93,491 covered texels, asserted at load),
* the barycentric anchor of every Gaussian (`get_barycentric_3d`),
* the camera-space normal transform, INCLUDING the reference's quirk of
  applying the transposed world-to-view matrix (it multiplies by
  `world_view_transform`, which in 3DGS is stored transposed, so the
  normals are rotated by the camera-to-world rotation). The network was
  trained with it, so it is kept,
* the view-direction map, INCLUDING `viewdir_map[gs_u, gs_u]` (row index
  used twice). It does not matter: `SWGAN_unet.forward` adds the view
  feature to `out` AFTER the last `to_rgb`, so it never reaches the
  output (`test_garment_appearance.py` checks this),
* the StyleUNet forward with the stored fixed noise buffers
  (`randomize_noise=False`) and the constant style `1/sqrt(512)`.

What is NOT the reference: the ambient-occlusion and normal maps. The
paper bakes them in Blender Cycles (`bpy.ops.object.bake(type='AO')` and
`type='NORMAL', normal_space='OBJECT'`, margin 5 texels) with the body
mesh present as an occluder. `bake_maps` below replaces that with flat
face normals in the garment's own frame (the OBJ importer keeps the mesh
data in file coordinates and rotates the OBJECT, so 'OBJECT' space is
the file's y-up frame) and a cosine-weighted hemisphere ray cast
(open3d, 32 rays a texel, 10 m range like Blender's default AO
distance), dilated 5 texels. Blender's AO is the same integral
estimated with more samples; the systematic differences are the flat
(not smooth) normal - the template has no `vn`/`s` lines, so Blender's
importer flat-shades it too - and whatever occluders are passed in.

Coordinate conventions: `view` is the rasteriser's world-to-camera 4x4
(+x right, +y down, +z forward), which is the OpenCV `[R|t]` the
checkpoint's `cameras.json` also uses. The scan's world is y-up with the
subject facing +z (camera 0004 sits at z=+2.8 looking down -z).
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
    """`AppearanceNet._world_xyz_numpy` (the barycentric anchor, frame @ local
    * scale + anchor, on the face frames of `face_frames_numba`), one Gaussian
    per iteration: the numpy version cost 22-30 ms a frame at 93k Gaussians
    in the live loop. Serial on purpose: the shading workers
    (`AsyncStudent`, `AsyncAppearance`) call it on their own thread while
    the loop runs `_place_splats_numba`, and numba's workqueue layer aborts
    the process on two parallel regions at once."""
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


# --- Camera, as the reference builds it ------------------------------------


def _world_to_view_transposed(view):
    """`Camera.world_view_transform` for the rasteriser's `view` (w2c 4x4):
    `getWorld2View2(R, T).transpose(0, 1)` with `R = w2c[:3, :3].T`,
    `T = w2c[:3, 3]`, exactly as `scene/cameras.py` does it."""
    view = np.asarray(view, np.float64)
    R, T = view[:3, :3].T, view[:3, 3]
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.T
    Rt[:3, 3] = T
    Rt[3, 3] = 1.0
    return np.float32(Rt).T


# --- The network ------------------------------------------------------------


def pick_device(device=None):
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AppearanceNet:
    """`shadow_net` + `viewdir_net` from `net.pt`, plus the static Gaussians
    they are added to. `forward` is `AvatarNet.forward` for one frame."""

    def __init__(self, checkpoint_dir, device=None, texture_size=TEXTURE_SIZE, faces=None, half=False, network=True):
        """`half`: run `shadow_net` in float16. Measured on the M1 (MPS):
        ~1.0 s a frame against ~2.3 s in float32, output within 2.3e-3
        (max) / 1.8e-4 (mean) of float32 in SH units - under a grey level
        of colour. Off by default: float32 is the reference.

        `network=False`: only the static Gaussians and the texel binding
        (what `shading_student.ShadingStudent` uses) - building and loading
        the 55M-parameter nets was ~0.6 s of a garment switch for nothing."""
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
        # The renderer's mesh (trimesh, UV-split: 8208 vertices) and the
        # reference's (7972 vertices) index the same faces in the same
        # order; accept either by vertex count.
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
        """The 4-channel network input: ambient goes in as is; the normal
        map is decoded from [0, 1], rotated by `world_view_transform` as
        the reference does (see the module docstring) and re-normalised.
        Returns the 3 camera-space normal channels, (3, res, res)."""
        w2v = _world_to_view_transposed(view).astype(np.float64)
        normal = np.asarray(normal, np.float64)  # (H, W, 3)
        covered = normal.sum(-1) > 0
        nw_pad = np.concatenate([normal * 2 - 1, np.ones(normal.shape[:2] + (1,))], axis=-1) * covered[..., None]
        nc = nw_pad @ w2v[:3].T  # (H, W, 3) = (W @ [n; 1])[:3] per texel
        norm = np.linalg.norm(nc, axis=-1, keepdims=True)
        # F.normalize: divide by max(norm, eps)
        return (nc / np.maximum(norm, 1e-12)).transpose(2, 0, 1)

    def view_feature(self, anchor, orientation, view):
        """`viewdir_net` on the reference's view-direction map. Kept for
        fidelity checks only: `SWGAN_unet.forward` never lets it reach
        the output, so `forward` skips it by default."""
        camera_center = np.linalg.inv(_world_to_view_transposed(view).astype(np.float64))[3, :3]
        dir_pp = anchor - camera_center
        dir_pp = dir_pp / np.linalg.norm(dir_pp, axis=1, keepdims=True)
        local_viewdir = np.einsum("nji,nj->ni", orientation[self.binding.binding], dir_pp)  # R^T d
        H = W = self.texture_size
        viewdir_map = torch.zeros(H, W, 3)
        # Reference: `viewdir_map[gs_u, gs_u] = local_viewdir` (sic).
        viewdir_map[self.binding.gs_u, self.binding.gs_u] = torch.as_tensor(local_viewdir, dtype=torch.float32)
        return self.viewdir_net(viewdir_map.permute(2, 0, 1)[None].to(self.device, self.dtype))

    @torch.no_grad()
    def forward(self, vertices, view, ambient=None, normal=None, occluders=(), use_viewdir=False):
        """One frame of `AvatarNet.forward`.

        `vertices`: the deformed garment mesh. `view`: the rasteriser's
        world-to-camera 4x4. `ambient` (res, res) and `normal` (res, res,
        3) in [0, 1]; baked here from the mesh when not given.

        Returns a dict of numpy arrays: `feature_dc` (N, 1, 3) and
        `feature_rest` (N, 15, 3) = static + shadow SH, in the layout
        `tryon/rendering/splat_cpu.py` renders; `xyz` (N, 3) world positions of
        `local_xyz = stored xyz + offset` about the barycentric anchor
        (`get_final_xyz`); `offset` and `shadow_shs` (N, 16, 3) alone;
        `anchor` (N, 3), the barycentric points; the two maps.
        Stage timings land in `self.last_timing`.
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
        # (1, 51, H, W) -> gather the covered texels; a contiguous copy to
        # the host and numpy indexing, since boolean indexing on MPS and a
        # permuted `.cpu()` each cost seconds.
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
        """World positions of face-local Gaussian positions (the stored xyz
        plus a network offset, `forward`'s "local_xyz") on `vertices` - to
        place an earlier frame's result on the current mesh. Compiled
        (`_world_xyz_numba`); `_world_xyz_numpy` is the oracle.

        `frames`: (frames, scale) already computed for these `vertices` and
        this mesh's faces (`GarmentRenderer.splats` has them); omitted, they
        are computed here."""
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
        """World positions of the STATIC Gaussians about the barycentric
        anchor (`get_xyz` of the stage-3 model) - no network."""
        vertices = np.asarray(vertices, np.float64)
        faces = self.faces_for(vertices)
        _, orientation, scale = face_frames(vertices, faces)
        anchor = self.binding.barycentric_3d(vertices[faces])
        frames = orientation[self.binding.binding]
        return np.einsum("nij,nj->ni", frames, self.xyz) * scale[self.binding.binding, None] + anchor


def merge(splats, result, indices=None, keys=("feature_dc", "feature_rest", "xyz")):
    """A copy of `splats` (the dict `GarmentRenderer.splats` /
    `render_splats` use) with `feature_dc`, `feature_rest` and `xyz`
    replaced by the network's. `indices`: which of the checkpoint's N
    Gaussians each entry of `splats` is, when it holds a subset (torn
    faces, body occlusion, subsampling); omitted means all N in order.
    `keys`: which of the three to take (`GarmentRenderer.render` takes the
    positions first, for the occlusion test, and the colours of what survives it)."""
    count = len(splats["xyz"])
    if indices is None:
        if count != len(result["xyz"]):
            raise ValueError(
                f"splats hold {count} Gaussians but the network produced {len(result['xyz'])}; "
                "pass `indices` for a subset")
        indices = slice(None)
    elif len(indices) != count:
        # a mismatch here would reach the compiled rasteriser as arrays of
        # different lengths, which reads out of bounds and crashes the process
        raise ValueError(f"splats hold {count} Gaussians but {len(indices)} indices were given")
    out = dict(splats)
    for key in keys:
        out[key] = result[key][indices]
    return out


def apply(splats, net: AppearanceNet, vertices, view, indices=None, **forward_kwargs):
    """Run `net` on the frame and merge its output into `splats`."""
    return merge(splats, net.forward(vertices, view, **forward_kwargs), indices)


class AsyncAppearance(IntervalWorker):
    """`AppearanceNet.forward` on a worker thread, for the live loop.

    The shading changes slowly with the pose and is bound to the garment's
    texels, so an answer a frame or two old still sits on the right cloth:
    `latest(vertices)` returns the newest finished result with its
    positions re-placed on the CURRENT mesh (`world_xyz`), and hands the
    worker the current mesh when it is idle. Measured on the M1 at 8 AO
    rays: bake ~150 ms (CPU, open3d) + network ~285 ms (MPS, fp16) a
    result, none of it in the frame loop. 8 rays against the reference's
    32: rendered frame within 42.6 dB (PSNR) of the 32-ray one.
    """

    RAYS = 8
    INTERVAL = 2.0  # s between shading passes: 2 s adds nothing to a 0.65 s frame, 1.5 s adds 0.17 s (measured)

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
        # contiguous once per pass, not per frame (see `AsyncStudent._compute`)
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
