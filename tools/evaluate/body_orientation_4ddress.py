"""Facing direction and root-relative error of the live body fit on 4D-DRESS.

`body_accuracy_4ddress` removes the body's rotation with a Procrustes step, so it cannot see a body
that faces the wrong way. This tool keeps the rotation. For each view it puts the capture's SMPL-X
joints into the camera frame (`extrinsics`) and then the fitter's own frame (y up, z toward the camera), fits the body the live path would fit
(MediaPipe -> NLF-S -> `BodyFitter.fit_from_regression`, the camera's field of view as live) and reports

- the facing error: the signed angle, about the camera's vertical axis, between the fitted and the true
  line across the shoulders and hips; a view is a *flip* when the error exceeds 90 degrees;
- the root-relative error: joint distance after moving both hip centres to the origin, with no rotation
  and no scale change.

    .venv/bin/python -m tools.evaluate.body_orientation_4ddress [--frames 3] [--limit N]
"""

from __future__ import annotations

import argparse

import numpy as np

from tools.evaluate._common import write_json
from tryon.paths import BASELINE

OUTPUT = BASELINE / "body-orientation-4ddress.json"
FLIP_DEGREES = 90.0


def camera_to_model(points: np.ndarray) -> np.ndarray:
    """OpenCV camera frame (x right, y down, z away) to the fitter's frame (x right, y up, z toward the camera)."""
    return np.asarray(points, dtype=float) * np.array([1.0, -1.0, -1.0])


def facing_angle(left: np.ndarray, right: np.ndarray) -> float:
    """Angle in degrees of the line from the right joint to the left joint, in the camera's x-z plane.

    0 when the left joint lies to the camera's right (facing the camera), 180 when it lies to the left."""
    across = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    return float(np.degrees(np.arctan2(across[2], across[0])))


def wrapped_difference(predicted: float, truth: float) -> float:
    """Signed difference of two angles in degrees, wrapped into [-180, 180)."""
    return float((predicted - truth + 180.0) % 360.0 - 180.0)


def body_facing(joints: np.ndarray, left: tuple[int, int], right: tuple[int, int]) -> float:
    """Facing angle from the mean of the shoulder and hip pairs; `left` and `right` index joints."""
    return facing_angle(joints[list(left)].mean(axis=0), joints[list(right)].mean(axis=0))


def root_relative_mpjpe(predicted: np.ndarray, truth: np.ndarray, hips: tuple[int, int]) -> float:
    """Mean joint distance (same unit as the inputs) with both hip centres moved to the origin."""
    p = predicted - predicted[list(hips)].mean(axis=0)
    t = truth - truth[list(hips)].mean(axis=0)
    return float(np.linalg.norm(p - t, axis=1).mean())


def score(data_root, records: list[dict]) -> list[dict]:
    import cv2
    import torch

    from tools.evaluate import real_fit
    from tools.evaluate.fit_audit import stop_regressor
    from tryon.body.fitter import BodyFitter
    from tryon.body.landmarks import JOINT_MAP
    from tryon.body.regressor import NLF

    real_fit.DATA = data_root
    order = sorted(JOINT_MAP)
    place = {landmark: order.index(landmark) for landmark in (11, 12, 23, 24)}
    left = (place[11], place[23])
    right = (place[12], place[24])
    hips = (place[23], place[24])
    nlf = NLF()
    rows = []
    for record in records:
        take, cam, frame = record["take"], record["cam"], record["frame"]
        image = cv2.imread(str(data_root / take / cam / "rgb_images" / f"{frame}.png"))
        row = {**record, "view": f"{take}/{cam}/{frame}"}
        points = None if image is None else real_fit.landmarks_from_mediapipe(image)
        if points is None:
            rows.append({**row, "detected": False})
            continue
        _, extrinsics = real_fit.load_camera(take, cam)
        truth_world = real_fit.ground_truth_joints(take, frame)[[JOINT_MAP[i] for i in order]]
        truth = camera_to_model(truth_world @ extrinsics[:, :3].T + extrinsics[:, 3])
        prediction = nlf.predict(image, intrinsics=None, keypoints=points)
        fitter = BodyFitter()
        fitter.fit_from_regression(points, prediction)
        with torch.no_grad():
            fitted = fitter.joints().numpy()
        error = wrapped_difference(body_facing(fitted, left, right), body_facing(truth, left, right))
        row.update(detected=True, facing_error_degrees=error,
                   root_relative_mm=1000 * root_relative_mpjpe(fitted, truth, hips))
        rows.append(row)
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    stop_regressor(nlf)
    return rows


def summarise(rows: list[dict]) -> dict:
    found = [r for r in rows if r["detected"]]
    out = {"views": len(rows), "detected": len(found)}
    if not found:
        return out
    errors = np.abs([r["facing_error_degrees"] for r in found])
    out.update(
        facing_error_median_degrees=float(np.median(errors)),
        facing_error_p90_degrees=float(np.percentile(errors, 90)),
        flips=int((errors > FLIP_DEGREES).sum()),
        flip_share=float((errors > FLIP_DEGREES).mean()),
        root_relative_mm_mean=float(np.mean([r["root_relative_mm"] for r in found])),
        root_relative_mm_median=float(np.median([r["root_relative_mm"] for r in found])),
    )
    return out


def main() -> None:
    from tools.evaluate import body_accuracy_4ddress as base

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=3, help="frames per take")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N views")
    args = parser.parse_args()
    records = base.build_view(args.frames, args.limit)
    rows = score(base.VIEW_ROOT, records)
    summary = summarise(rows)
    write_json(OUTPUT, {"protocol": "MediaPipe -> NLF-S (live field of view) -> fit_from_regression; "
                                    "facing error about the camera's vertical axis; root-relative joint error "
                                    "without rotation or scale", "summary": summary, "rows": rows})
    print(summary)


if __name__ == "__main__":
    main()
