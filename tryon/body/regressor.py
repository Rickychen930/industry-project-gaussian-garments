"""Monocular SMPL-X regression with Multi-HMR (Baradel et al., ECCV 2024).

Our body comes from fitting SMPL-X to MediaPipe's 2D keypoints; a network
that regresses the whole body from the image knows far more. On the 16
annotated 4D-DRESS photos with ground-truth SMPL-X (the project's own
tools/evaluate/real_fit.py protocol, 12 joints, Procrustes-aligned MPJPE):
our fit 87 mm; Multi-HMR 672_S 48 mm, 672_B 38 mm, 896_L 25 mm. CPU time a
photo: 0.57 s / 1.7 s / 32 s (672_S 0.63 s on MPS).

Code and weights (NAVER, non-commercial licence) are NOT part of this
repository: clone https://github.com/naver/multi-hmr to
`models/pretrained/multi-hmr` (git-ignored), put the checkpoints from
download.europe.naverlabs.com/ComputerVision/MultiHMR/ in its
`models/multiHMR/`, symlink `models/smplx` to the SMPL-X models, and
`pip install --no-deps roma` (einops is already a dependency).

`predict` returns the most confident person's SMPL-X in THIS project's
convention: Multi-HMR works in the OpenCV camera frame (y down, z into
the scene); our model frame has y up and a person facing the camera
facing +z, which is the same frame turned 180 degrees about x, so the
root rotation is pre-multiplied by diag(1, -1, -1).
"""

from __future__ import annotations

import collections

import sys
import types
from pathlib import Path

import numpy as np

from tryon.paths import REPO_ROOT

ROOT = REPO_ROOT
MULTIHMR_ROOT = ROOT / "models" / "pretrained" / "multi-hmr"
_FLIP = np.diag([1.0, -1.0, -1.0])
# Input square (px); None = the model's own (672). On the 8 annotated
# views 448 px was as good (PA-MPJPE 46.9 vs 50.0 mm, 0.30 vs 0.93 s on a
# loaded CPU; 364 px 54.8 mm, 2 of 8 missed), but on the five release
# videos, where the person is smaller in the frame, garment IoU fell
# 0.716 -> 0.685 - so the native size stays the default.
INPUT_SIZE = None


def available(model_name="multiHMR_672_S"):
    return (MULTIHMR_ROOT / "models" / "multiHMR" / f"{model_name}.pt").exists()


class MultiHMR:
    def __init__(self, model_name="multiHMR_672_S", device="cpu", fov_degrees=60.0, input_size=INPUT_SIZE):
        import torch

        if str(MULTIHMR_ROOT) not in sys.path:
            sys.path.insert(0, str(MULTIHMR_ROOT))
        # its rendering helpers import pyrender with EGL (not on macOS) and are not needed
        render = types.ModuleType("utils.render")
        for name in ("render_meshes", "print_distance_on_image", "render_side_views", "create_scene"):
            setattr(render, name, None)
        render.OPENCV_TO_OPENGL_CAMERA_CONVENTION = None
        sys.modules.setdefault("utils.render", render)
        import os

        cwd = os.getcwd()
        os.chdir(MULTIHMR_ROOT)  # it resolves models/smplx relative to the working directory
        try:
            from model import Model

            checkpoint = torch.load(MULTIHMR_ROOT / "models" / "multiHMR" / f"{model_name}.pt", map_location="cpu", weights_only=False)
            kwargs = dict(vars(checkpoint["args"]))
            kwargs["type"], kwargs["img_size"] = checkpoint["args"].train_return_type, checkpoint["args"].img_size[0]
            self.model = Model(**kwargs)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        finally:
            os.chdir(cwd)
        self.device = torch.device(device)
        self.model = self.model.to(self.device).eval()
        # the ViT takes other sizes (a multiple of its 14 px patch)
        self.size = int(input_size or kwargs["img_size"])
        self.fov = float(fov_degrees)

    def predict(self, frame_bgr, intrinsics=None):
        """{global_orient (3,), body_pose (63,), betas (10,), joints_2d (22, 2)
        frame pixels of the body joints as it projects them, score} or None.
        `intrinsics`: the camera's 3x3 K in frame pixels if known; else a
        `fov_degrees` field of view (a webcam's is unknown)."""
        import cv2
        import torch
        from scipy.spatial.transform import Rotation

        height, width = frame_bgr.shape[:2]
        scale = self.size / max(height, width)
        resized = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        padx, pady = (self.size - resized.shape[1]) // 2, (self.size - resized.shape[0]) // 2
        square = np.zeros((self.size, self.size, 3), np.uint8)
        square[pady : pady + resized.shape[0], padx : padx + resized.shape[1]] = resized
        x = (square.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        x = torch.from_numpy(x.transpose(2, 0, 1))[None].to(self.device)
        k = torch.eye(3)
        if intrinsics is not None:
            k[0, 0], k[1, 1] = intrinsics[0][0] * scale, intrinsics[1][1] * scale
            k[0, 2], k[1, 2] = intrinsics[0][2] * scale + padx, intrinsics[1][2] * scale + pady
        else:
            focal = self.size / (2.0 * np.tan(np.radians(self.fov) / 2.0))
            k[0, 0] = k[1, 1] = focal
            k[0, 2] = k[1, 2] = self.size / 2.0
        with torch.no_grad():
            humans = self.model(x, is_training=False, nms_kernel_size=3, det_thresh=0.3, K=k[None].to(self.device))
        if not humans:
            return None
        person = max(humans, key=lambda h: float(h["scores"]))
        rotvec = person["rotvec"].float().cpu().numpy().reshape(-1, 3)
        root = Rotation.from_matrix(_FLIP @ Rotation.from_rotvec(rotvec[0]).as_matrix()).as_rotvec()
        # its own projection of the 22 body joints, back in frame pixels
        joints_2d = (person["j2d"].float().cpu().numpy()[:22] - [padx, pady]) / scale
        return {
            "joints_2d": joints_2d,
            "image_size": (height, width),
            "global_orient": root,
            "body_pose": rotvec[1:22].reshape(-1),
            "betas": person["shape"].float().cpu().numpy().reshape(-1)[:10],
            "score": float(person["scores"]),
        }


NLF_ROOT = ROOT / "models" / "pretrained" / "nlf"
NLF_WEIGHTS = NLF_ROOT / "nlf_s_multi_0.2.2.torchscript"
NLF_PYTHON = NLF_ROOT / "venv" / "bin" / "python"


def nlf_available():
    return NLF_WEIGHTS.exists() and NLF_PYTHON.exists()


def box_from_landmarks(points, shape, margin=0.25, floor=0.3):
    """[x, y, w, h] around MediaPipe's visible landmarks, grown by `margin`
    of the larger side each way and clipped to the frame - NLF's person box
    without running its YOLO detector (1.5 s a frame on the CPU). None
    without at least 4 visible landmarks."""
    if points is None:
        return None
    visible = points[:, 2] >= floor
    if visible.sum() < 4:
        return None
    xy = points[visible, :2]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    grow = margin * float((hi - lo).max())
    height, width = shape[:2]
    x0, y0 = max(lo[0] - grow, 0.0), max(lo[1] - grow, 0.0)
    x1, y1 = min(hi[0] + grow, width - 1.0), min(hi[1] + grow, height - 1.0)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def box_from_joints(joints_2d, shape, margin=0.3):
    """The person box from a regressor's own projected body joints (22, 2),
    grown by `margin` of the larger side (the joints stop at the neck and
    ankles) - kept by the live loop to re-find the wearer when MediaPipe
    loses them."""
    points = np.ones((len(joints_2d), 3), np.float32)
    points[:, :2] = joints_2d
    return box_from_landmarks(points, shape, margin=margin)


def person_in_box(person_mask, box, fraction=0.15):
    """Whether at least `fraction` of `box` [x, y, w, h] is person pixels."""
    if person_mask is None or box is None:
        return False
    x, y, w, h = (int(round(v)) for v in box)
    region = np.asarray(person_mask)[max(y, 0) : max(y + h, 0), max(x, 0) : max(x + w, 0)]
    return region.size > 0 and float(np.mean(region)) >= fraction


class NLF:
    """Neural Localizer Fields (Sarandi et al., NeurIPS 2024), NLF-S: on
    the 8 annotated views PA-MPJPE 26-27 mm (Multi-HMR 672_S 49, 672_B 38)
    at 0.23-0.30 s on the CPU with a person box handed in
    (docs/baseline/body-regressors.json). Its TorchScript needs a newer
    torch than the project's, so it runs in `tryon/body/nlf_worker.py` under
    `models/pretrained/nlf/venv`; this class is the same interface as
    `MultiHMR` (`predict` -> global_orient, body_pose, betas, joints_2d,
    image_size). Code MIT, models non-commercial research use."""

    # torch threads in the worker. Alone on the M1: 2 / 4 / 6 / 8 threads
    # 87 / 92 / 117 / 126 ms a call; in the live loop, beside the parsers and
    # the render, 2 threads left more of the machine to them (regressor
    # 182 -> 152-165 ms, the whole frame 425-451 -> 374-420 ms).
    THREADS = 2
    # where the worker runs the image backbone: "mps" (Apple GPU, the rest
    # stays on the CPU; nlf_worker's docstring) or "cpu"
    BACKBONE = "mps"
    # NLF's test-time augmentation: crops averaged per call (nlf_worker.NUM_AUG).
    # Garment IoU vs the real shirt, 00170_upper / 00190_upper / 00152_outer:
    # 1 crop 0.787 / 0.726 / 0.655, 2 crops 0.788 / 0.731 / 0.657, 3 crops
    # 0.791 / 0.729 / 0.661. NLF 73 / 91 / 120 ms a call; it runs beside the
    # render (prefetch), and 80 live frames took 28.6-30.2 s with 1, 27.5-32.1
    # with 2 (no measurable cost) and 33.8-38.1 with 3 (15-25 % fewer fps).
    NUM_AUG = 2

    def __init__(self, weights=NLF_WEIGHTS, fov_degrees=60.0, threads=None):
        import subprocess

        threads = self.THREADS if threads is None else threads

        self.fov = float(fov_degrees)
        self._process = subprocess.Popen(
            [str(NLF_PYTHON), str(Path(__file__).with_name("nlf_worker.py")), str(weights), str(threads), self.BACKBONE, str(self.NUM_AUG)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        )
        # not waited for here: the worker loads and warms up (a few seconds)
        # while the caller loads everything else; the first predict waits
        self._ready = False
        self._pending = collections.deque()  # (frame, box, K) sent by `prefetch`, answers not yet read

    def _wait_ready(self):
        if not self._ready:
            if self._receive() != "ready":
                raise RuntimeError("the NLF worker did not start")
            self._ready = True

    def _send(self, value):
        import pickle
        import struct

        data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        self._process.stdin.write(struct.pack("<Q", len(data)) + data)
        self._process.stdin.flush()

    def _receive(self):
        import pickle
        import struct

        header = self._process.stdout.read(8)
        if len(header) < 8:
            raise RuntimeError("the NLF worker exited")
        return pickle.loads(self._process.stdout.read(struct.unpack("<Q", header)[0]))

    def close(self):
        if self._process.poll() is None:
            self._process.stdin.close()
            self._process.wait(timeout=10)

    def _request(self, frame_bgr, intrinsics, keypoints, box):
        height, width = frame_bgr.shape[:2]
        if intrinsics is None:
            focal = max(height, width) / (2.0 * np.tan(np.radians(self.fov) / 2.0))
            intrinsics = np.array([[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]], np.float32)
        return (box if box is not None else box_from_landmarks(keypoints, frame_bgr.shape)), np.asarray(intrinsics, np.float32)

    def prefetch(self, frame_bgr, intrinsics=None, keypoints=None, box=None):
        """Send the request for `frame_bgr` now; the `predict` call with the
        same frame object and box reads its answer. The worker runs it while
        the caller finishes the previous frame. A prefetched answer nobody
        asks for is read and dropped by the next `predict`."""
        box, k = self._request(frame_bgr, intrinsics, keypoints, box)
        self._wait_ready()
        self._send({"frame": np.ascontiguousarray(frame_bgr), "box": box, "K": k})
        self._pending.append((frame_bgr, box, k))

    def predict(self, frame_bgr, intrinsics=None, keypoints=None, box=None):
        """As `MultiHMR.predict`; `keypoints`: MediaPipe's (33, >=3)
        landmarks in frame pixels for the person box, or `box` [x, y, w, h]
        directly (else NLF's own detector, ~1.5 s)."""
        from scipy.spatial.transform import Rotation

        height, width = frame_bgr.shape[:2]
        box, k = self._request(frame_bgr, intrinsics, keypoints, box)
        self._wait_ready()
        answer, found = None, False
        while self._pending and not found:
            sent_frame, sent_box, sent_k = self._pending.popleft()
            answer = self._receive()
            found = (sent_frame is frame_bgr and np.array_equal(sent_k, k)
                     and (sent_box is None) == (box is None) and (box is None or np.array_equal(sent_box, box)))
        if not found:
            self._send({"frame": np.ascontiguousarray(frame_bgr), "box": box, "K": k})
            answer = self._receive()
        if answer is None:
            return None
        rotvec = answer["rotvec"]
        joints = np.asarray(answer["joints3d"], np.float64)[:22]
        projected = (k @ joints.T).T
        joints_2d = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-6)
        return {
            "joints_2d": joints_2d,
            "image_size": (height, width),
            "global_orient": Rotation.from_matrix(_FLIP @ Rotation.from_rotvec(rotvec[0]).as_matrix()).as_rotvec(),
            "body_pose": rotvec[1:22].reshape(-1),
            "betas": answer["betas"],
            "score": 1.0,
        }


# ---------------------------------------------------------------- arm refinement
# SMPL-X joints (collar, shoulder, elbow, wrist) of each arm; the elbow is a hinge about its local y,
# flexion = -y on the left, +y on the right (NLF's elbow axis-angles, 10004_outer).
ARM_JOINTS = {"left": (13, 16, 18, 20, -1.0), "right": (14, 17, 19, 21, 1.0)}
MAX_FLEXION = 2.7  # rad, a little past the elbow's range
MIN_FLEXION = -0.15


def _rotation(vectors):
    """3x3 rotation of an axis-angle vector (Rodrigues)."""
    angle = float(np.linalg.norm(vectors))
    if angle < 1e-9:
        return np.eye(3)
    x, y, z = vectors / angle
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def refine_arm(frames, targets, length, camera, sign, shoulder_prior=0.3, twist_prior=1.0, flexion_prior=0.1, scale_loss=0.1, steps=5):
    """Rotations of one arm's shoulder and elbow that put its elbow and wrist on 2D targets.

    `frames`: dict of the current chain, all in the world the camera projects: `R_collar` (3, 3) the
    collar's global rotation, `shoulder` (3,) the shoulder joint, `a_shoulder`, `a_elbow` (3,) the
    current local axis-angles, `o_elbow` (3,) the elbow in the shoulder's frame and `o_wrist` (3,) the
    wrist in the elbow's frame. `targets`: {"elbow": (x, y) or None, "wrist": ...} pixels. `length`: the
    torso length in px, the unit of the residuals. `camera`: (scale, translation (2,)): y flips, as
    `BodyFitter._project`. The prior holds the shoulder (`shoulder_prior` per rad) and the elbow's
    non-hinge components (`twist_prior`) near the regressor's pose and leaves the flexion free inside
    its range. Returns (a_shoulder, a_elbow, cost before, cost after) with the cost in torso lengths.
    """
    from scipy.optimize import least_squares

    scale, translation = camera
    a0 = np.concatenate([frames["a_shoulder"], frames["a_elbow"]])
    wanted = [(name, np.asarray(xy, np.float64)) for name, xy in targets.items() if xy is not None]

    def joints(a):
        r_sh = frames["R_collar"] @ _rotation(a[:3])
        elbow = frames["shoulder"] + r_sh @ frames["o_elbow"]
        wrist = elbow + (r_sh @ _rotation(a[3:])) @ frames["o_wrist"]
        return {"elbow": elbow, "wrist": wrist}

    def residual(a):
        posed = joints(a)
        out = []
        for name, xy in wanted:
            out.extend((posed[name][:2] * (1.0, -1.0) * scale + translation - xy) / length)
        delta = a - a0
        flexion = sign * a[4]
        out.extend(shoulder_prior * delta[:3])
        out.extend((twist_prior * delta[3], flexion_prior * delta[4], twist_prior * delta[5]))
        out.append(twist_prior * (max(flexion - MAX_FLEXION, 0.0) + max(MIN_FLEXION - flexion, 0.0)))
        return np.array(out)

    before = float(np.linalg.norm(residual(a0)[: 2 * len(wanted)]))
    solution = least_squares(residual, a0, loss="soft_l1", f_scale=scale_loss, max_nfev=steps, diff_step=1e-4)
    after = float(np.linalg.norm(residual(solution.x)[: 2 * len(wanted)]))
    return solution.x[:3], solution.x[3:], before, after
