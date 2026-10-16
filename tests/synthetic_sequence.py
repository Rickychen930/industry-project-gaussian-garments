"""A body raising its arms, with exact 3D ground truth.

A real video cannot give true joint positions, so a SMPL-X body is posed
per frame, projected through a known camera into 640x480 and jittered like
a keypoint detector's noise.
"""

import numpy as np
import torch

from tryon.body.fitter import BodyFitter
from tryon.body.landmarks import JOINT_MAP, landmarks_to_targets
from tryon.body.smoothing import acceleration_error


def make_sequence(frames=40, noise_px=2.0, seed=0):
    """A SMPL-X body raising its arms, placed like a person in a 640x480 frame."""
    truth = BodyFitter()
    with torch.no_grad():
        truth.scale.fill_(180.0)
        truth.translation[:] = torch.tensor([[320.0, 260.0]])
    LSZ, RSZ = 15 * 3 + 2, 16 * 3 + 2
    joints = []
    obs = []
    for f in range(frames):
        a = 1.2 * f / (frames - 1)
        with torch.no_grad():
            truth.body_pose.zero_()
            truth.body_pose[0, LSZ] = -a
            truth.body_pose[0, RSZ] = a
            j = truth._forward().joints[0, truth._smplx_indices]
            joints.append(j.numpy().copy())
            px = truth._project(j).detach().numpy()
        pts = np.zeros((33, 3), np.float32)
        pts[:, 2] = 1.0
        rng = np.random.default_rng(seed * 1000 + f)
        for k, idx in enumerate(sorted(JOINT_MAP)):
            pts[idx, :2] = px[k] + rng.normal(0, noise_px, 2)
        obs.append(pts)
    return np.stack(joints), obs


def run(obs, truth_joints, track_shape, fps=30.0, iters=3):
    fitter = BodyFitter()
    fitter.fit(obs[0], iterations=150)
    out = []
    for pts in obs:
        t, w = landmarks_to_targets(pts)
        fitter._run(t, w, iters, 0.02, shape=track_shape)
        with torch.no_grad():
            out.append(fitter._forward().joints[0, fitter._smplx_indices].numpy().copy())
    out = np.stack(out)
    return np.linalg.norm(out - truth_joints, axis=-1).mean() * 1000, acceleration_error(out, truth_joints, fps)


if __name__ == "__main__":
    tj, obs = make_sequence()
    print("truth joints in frame? first-frame pixel x range:", end=" ")
    t, _ = landmarks_to_targets(obs[0])
    print(t[:, 0].min().round(), "..", t[:, 0].max().round())
    for name, ts in (("shape FREE each frame", True), ("shape FROZEN (VIBE)", False)):
        m, a = run(obs, tj, ts)
        print(f"{name:24} MPJPE {m:6.1f} mm   accel {a:8.1f} mm/s^2")
