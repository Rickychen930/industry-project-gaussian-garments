"""Does a learned residual correction improve the live body path, on real ground truth?

(Not the "close and truncated views" corrector sketched in docs/LOG.md, which needs synthetic
cameras and is not built: this one corrects the fit of a studio-captured full body.)

The live body is MediaPipe keypoints -> NLF-S regression -> `BodyFitter.fit_from_regression`.
`tools/evaluate/body_accuracy_4ddress.py` measured it at 42 mm PA-MPJPE on 439 views of two
4D-DRESS subjects (24.6 mm on the 16-view sample). This trains a small network that maps the
fitted joints (plus MediaPipe's visibilities) to the offset that would bring them onto the
capture's own SMPL-X, and measures it the only way that says anything about generalisation:
trained on one subject, scored on the other.

Supervision: the fitted 12 JOINT_MAP joints are put in a canonical torso frame (origin pelvis,
y shoulder-ward, scale torso length); the ground truth is similarity-aligned onto them (the
metric's own alignment) and the offset in that frame is the regression target. A correction
therefore improves PA-MPJPE exactly when it predicts that offset.

    .venv/bin/python -m tools.train.body_corrector extract [--stride 4] [--limit-takes N]
    .venv/bin/python -m tools.train.body_corrector train

`extract` writes one .npz per take under `~/tryon-data/4d-dress/_features/parts/` (resumable).
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import cv2
import numpy as np

from tools.evaluate._common import write_json
from tryon.paths import BASELINE, REPO_ROOT

SOURCE = Path.home() / "tryon-data" / "4d-dress"
PARTS = SOURCE / "_features" / "parts_v2"  # v2 adds the per-view reliability signals (see extract)
CHECKPOINT = REPO_ROOT / "models" / "body_corrector.pt"
OUTPUT = BASELINE / "body-corrector.json"
CAMERAS = ("0004", "0028", "0052", "0076")
N_JOINTS = 12


def _joint_slots():
    """Positions of the torso joints inside `sorted(JOINT_MAP)`, by name."""
    from tryon.body.landmarks import JOINT_MAP, POSE

    order = sorted(JOINT_MAP)
    ids = POSE.PoseLandmark
    slot = {name: order.index(getattr(ids, name).value)
            for name in ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP")}
    return order, slot


# ------------------------------------------------------------------ canonical frame

def canonical_frame(joints: np.ndarray, slot: dict) -> tuple[np.ndarray, float, np.ndarray]:
    """Origin, scale and rotation (rows = side, up, front) of the torso of `joints` (12, 3)."""
    ls, rs = joints[slot["LEFT_SHOULDER"]], joints[slot["RIGHT_SHOULDER"]]
    lh, rh = joints[slot["LEFT_HIP"]], joints[slot["RIGHT_HIP"]]
    origin = (lh + rh) / 2
    up = (ls + rs) / 2 - origin
    scale = float(np.linalg.norm(up)) + 1e-9
    up = up / scale
    side = (lh - rh) + (ls - rs)
    side = side - up * float(side @ up)
    side = side / (np.linalg.norm(side) + 1e-9)
    return origin, scale, np.stack([side, up, np.cross(side, up)])


def to_canonical(joints, frame):
    origin, scale, rotation = frame
    return ((joints - origin) @ rotation.T) / scale


def from_canonical(coords, frame):
    origin, scale, rotation = frame
    return (coords * scale) @ rotation + origin


def make_example(pred, truth, vis, slot):
    """(features, target) of one view; `truth` is aligned onto `pred` first (the metric's own step)."""
    from tools.evaluate.real_fit import procrustes_align

    frame = canonical_frame(pred, slot)
    aligned_truth = procrustes_align(truth, pred)
    features = np.concatenate([to_canonical(pred, frame).ravel(), vis])
    target = (to_canonical(aligned_truth, frame) - to_canonical(pred, frame)).ravel()
    return features.astype(np.float32), target.astype(np.float32)


def pa_error_mm(pred, truth):
    from tools.evaluate.real_fit import procrustes_align

    return float(np.linalg.norm(procrustes_align(pred, truth) - truth, axis=1).mean() * 1000)


# ------------------------------------------------------------------ extract

def _frames(take_dir: Path, stride: int) -> list[str]:
    ids = []
    for pkl in sorted((take_dir / "SMPLX").glob("mesh-f*_smplx.pkl")):
        number = pkl.stem.split("-f")[1].split("_")[0]
        if all((take_dir / "Capture" / cam / "images" / f"capture-f{number}.png").exists() for cam in CAMERAS):
            ids.append(number)
    return ids[::stride]


def extract(stride: int, limit_takes: int) -> None:
    import mediapipe as mp
    import smplx
    import torch

    from tools.evaluate.fit_audit import stop_regressor
    from tryon.body.fitter import SMPLX_ROOT, BodyFitter
    from tryon.body.landmarks import JOINT_MAP
    from tryon.body.regressor import NLF

    order, slot = _joint_slots()
    smplx_index = [JOINT_MAP[i] for i in order]
    body = smplx.create(str(SMPLX_ROOT), model_type="smplx", gender="neutral", use_pca=True,
                        num_pca_comps=12, num_betas=10)
    nlf = NLF()
    PARTS.mkdir(parents=True, exist_ok=True)
    takes = [t for s in sorted(SOURCE.iterdir()) if s.is_dir() and not s.name.startswith("_")
             for layer in sorted(p for p in s.iterdir() if p.is_dir())
             for t in sorted(layer.glob("Take*"), key=lambda p: int(p.name[4:]))]
    if limit_takes:
        takes = takes[:limit_takes]
    started = time.time()
    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        for take_dir in takes:
            subject, layer = take_dir.parents[1].name, take_dir.parent.name
            name = f"{subject}_{layer}_{take_dir.name}"
            target = PARTS / f"{name}.npz"
            if target.exists():
                continue
            rows = {"p": [], "g": [], "vis": [], "cam": [], "frame": [],
                    "repro": [], "gap_mean": [], "gap_max": [], "torso_px": []}
            for number in _frames(take_dir, stride):
                with open(take_dir / "SMPLX" / f"mesh-f{number}_smplx.pkl", "rb") as f:
                    params = pickle.load(f)
                with torch.no_grad():
                    out = body(**{k: torch.tensor(params[k], dtype=torch.float32)[None] for k in (
                        "global_orient", "body_pose", "betas", "transl", "left_hand_pose",
                        "right_hand_pose", "expression")})
                truth = out.joints[0].numpy()[smplx_index]
                for cam in CAMERAS:
                    image = cv2.imread(str(take_dir / "Capture" / cam / "images" / f"capture-f{number}.png"))
                    result = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    if result.pose_landmarks is None:
                        continue
                    h, w = image.shape[:2]
                    points = np.array([[lm.x * w, lm.y * h, lm.visibility] for lm in result.pose_landmarks.landmark],
                                      np.float32)
                    prediction = nlf.predict(image, intrinsics=None, keypoints=points)
                    if prediction is None:
                        continue
                    fitter = BodyFitter()
                    fitter.fit_from_regression(points, prediction)
                    with torch.no_grad():
                        fitted = fitter.joints().numpy()
                    # Signals a live system also has, none of which needs the ground truth: how far the
                    # fit lands from MediaPipe's own keypoints, and how far NLF's 2D joints disagree with
                    # MediaPipe's, both in torso lengths (so they do not depend on the person's distance).
                    detected = points[order][:, :2]
                    torso_px = float(np.linalg.norm(
                        (detected[slot["LEFT_SHOULDER"]] + detected[slot["RIGHT_SHOULDER"]]) / 2
                        - (detected[slot["LEFT_HIP"]] + detected[slot["RIGHT_HIP"]]) / 2)) + 1e-6
                    gap = np.linalg.norm(np.asarray(prediction["joints_2d"])[smplx_index] - detected, axis=1) / torso_px
                    rows["p"].append(fitted)
                    rows["g"].append(truth)
                    rows["vis"].append(points[order][:, 2])
                    rows["cam"].append(cam)
                    rows["frame"].append(number)
                    rows["repro"].append(float(fitter.reprojection_error(points)) / torso_px)
                    rows["gap_mean"].append(float(gap.mean()))
                    rows["gap_max"].append(float(gap.max()))
                    rows["torso_px"].append(torso_px)
            np.savez(target, subject=subject, layer=layer, take=name, p=np.array(rows["p"], np.float32),
                     g=np.array(rows["g"], np.float32), vis=np.array(rows["vis"], np.float32),
                     cam=np.array(rows["cam"]), frame=np.array(rows["frame"]),
                     repro=np.array(rows["repro"], np.float32), gap_mean=np.array(rows["gap_mean"], np.float32),
                     gap_max=np.array(rows["gap_max"], np.float32), torso_px=np.array(rows["torso_px"], np.float32))
            print(f"{name}: {len(rows['p'])} views  ({time.time() - started:.0f} s elapsed)", flush=True)
    stop_regressor(nlf)


# ------------------------------------------------------------------ train

def load_parts() -> dict:
    parts = [np.load(p, allow_pickle=False) for p in sorted(PARTS.glob("*.npz"))]
    parts = [p for p in parts if len(p["p"])]
    if not parts:
        raise SystemExit(f"no features under {PARTS}; run `extract` first")

    def cat(key):
        return np.concatenate([part[key] for part in parts])

    def repeated(key):
        return np.concatenate([np.full(len(part["p"]), str(part[key])) for part in parts])

    return {"p": cat("p"), "g": cat("g"), "vis": cat("vis"),
            "subject": repeated("subject"), "layer": repeated("layer"), "take": repeated("take")}


def fit_ridge(x, y, alpha):
    mean, std = x.mean(0), x.std(0) + 1e-6
    z = (x - mean) / std
    bias = y.mean(0)
    weights = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - bias))
    return lambda q: ((q - mean) / std) @ weights + bias


def fit_mlp(x, y, seed=0, epochs=400, hidden=128, weight_decay=1e-3):
    import torch

    torch.manual_seed(seed)
    mean, std = x.mean(0), x.std(0) + 1e-6
    xt = torch.as_tensor((x - mean) / std)
    yt = torch.as_tensor(y)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, y.shape[1]))
    optimiser = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=weight_decay)
    for _ in range(epochs):
        optimiser.zero_grad()
        torch.nn.functional.smooth_l1_loss(net(xt), yt, beta=0.05).backward()
        optimiser.step()
    net.eval()

    def predict(q):
        with torch.no_grad():
            return net(torch.as_tensor((q - mean) / std)).numpy()

    return predict, net, (mean, std)


def corrected_errors(model, data, rows, slot):
    errors = []
    for i in rows:
        pred, truth = data["p"][i].astype(np.float64), data["g"][i].astype(np.float64)
        frame = canonical_frame(pred, slot)
        features = np.concatenate([to_canonical(pred, frame).ravel(), data["vis"][i]]).astype(np.float32)[None]
        offset = model(features)[0].reshape(N_JOINTS, 3)
        errors.append(pa_error_mm(from_canonical(to_canonical(pred, frame) + offset, frame), truth))
    return np.array(errors)


def summary(values):
    values = np.asarray(values)
    return {"n": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)), "over_60mm": int((values > 60).sum())}


def train() -> None:
    import torch

    _, slot = _joint_slots()
    data = load_parts()
    examples = [make_example(data["p"][i].astype(np.float64), data["g"][i].astype(np.float64), data["vis"][i], slot)
                for i in range(len(data["p"]))]
    x = np.stack([e[0] for e in examples])
    y = np.stack([e[1] for e in examples])
    subjects = sorted(set(data["subject"]))
    result: dict = {"views": int(len(x)), "subjects": subjects, "splits": {}}
    baseline = np.array([pa_error_mm(data["p"][i].astype(np.float64), data["g"][i].astype(np.float64))
                         for i in range(len(x))])
    result["baseline_all"] = summary(baseline)
    print("baseline (no correction), all views:", result["baseline_all"], flush=True)

    best = None
    for held_out in subjects:
        train_rows = np.flatnonzero(data["subject"] != held_out)
        test_rows = np.flatnonzero(data["subject"] == held_out)
        if len(train_rows) == 0 or len(test_rows) == 0:
            continue
        # alpha chosen by leave-one-take-out on the TRAINING subject only
        takes = sorted(set(data["take"][train_rows]))
        scores = {}
        for alpha in (1.0, 10.0, 100.0, 1000.0):
            fold = []
            for take in takes:
                inner_train = train_rows[data["take"][train_rows] != take]
                inner_test = train_rows[data["take"][train_rows] == take]
                model = fit_ridge(x[inner_train], y[inner_train], alpha)
                fold.append(corrected_errors(model, data, inner_test, slot).mean())
            scores[alpha] = float(np.mean(fold))
        alpha = min(scores, key=scores.get)
        ridge = fit_ridge(x[train_rows], y[train_rows], alpha)
        mlp, net, norm = fit_mlp(x[train_rows], y[train_rows])
        entry = {
            "train_subjects": sorted(set(data["subject"][train_rows])), "test_subject": held_out,
            "train_views": int(len(train_rows)), "test_views": int(len(test_rows)),
            "ridge_alpha": alpha, "ridge_alpha_cv_mm": {str(k): v for k, v in scores.items()},
            "baseline": summary(baseline[test_rows]),
            "ridge": summary(corrected_errors(ridge, data, test_rows, slot)),
            "mlp": summary(corrected_errors(mlp, data, test_rows, slot)),
        }
        for layer in sorted(set(data["layer"][test_rows])):
            rows = test_rows[data["layer"][test_rows] == layer]
            entry[f"mlp_layer_{layer}"] = {"baseline": summary(baseline[rows]),
                                           "mlp": summary(corrected_errors(mlp, data, rows, slot))}
        result["splits"][f"test_{held_out}"] = entry
        print(f"\nheld-out subject {held_out}: baseline {entry['baseline']['mean']:.1f} mm  "
              f"ridge {entry['ridge']['mean']:.1f}  mlp {entry['mlp']['mean']:.1f}  "
              f"(median {entry['baseline']['median']:.1f} -> {entry['mlp']['median']:.1f}, "
              f"p90 {entry['baseline']['p90']:.1f} -> {entry['mlp']['p90']:.1f})", flush=True)
        if best is None or entry["mlp"]["mean"] < best[0]:
            best = (entry["mlp"]["mean"], net, norm, held_out)

    # leave-one-take-out WITHIN subject: optimistic (same person, same clothing), reported as such
    within = []
    for take in sorted(set(data["take"])):
        rows_test = np.flatnonzero(data["take"] == take)
        rows_train = np.flatnonzero((data["subject"] == data["subject"][rows_test[0]]) & (data["take"] != take))
        model, _, _ = fit_mlp(x[rows_train], y[rows_train])
        within.append(corrected_errors(model, data, rows_test, slot))
    result["within_subject_leave_take_out_mlp"] = summary(np.concatenate(within))
    result["within_subject_baseline"] = summary(baseline)
    print("\nwithin-subject leave-take-out (optimistic):", result["within_subject_leave_take_out_mlp"], flush=True)

    result["protocol"] = ("features = 12 fitted joints in canonical torso frame + MediaPipe visibility; "
                          "target = similarity-aligned ground-truth offset in that frame; metric PA-MPJPE (mm)")
    write_json(OUTPUT, result)
    if best is not None:
        CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": best[1].state_dict(), "mean": best[2][0], "std": best[2][1],
                    "trained_without_subject": best[3], "note": "research artefact, not wired into the live path"},
                   CHECKPOINT)
    print(f"wrote {OUTPUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--stride", type=int, default=4, help="every Nth ground-truth frame of a take")
    e.add_argument("--limit-takes", type=int, default=0)
    sub.add_parser("train")
    args = parser.parse_args()
    if args.command == "extract":
        extract(args.stride, args.limit_takes)
    else:
        train()


if __name__ == "__main__":
    main()
