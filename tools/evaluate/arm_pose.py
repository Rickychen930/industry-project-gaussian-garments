"""NLF's arm pose against MediaPipe's elbows and wrists, and the arm refinement that follows from it.

    record    one release video through MediaPipe (as vs_registration) and NLF every `--every`-th
              frame; saves the filtered landmarks and NLF's prediction (one process, one video)
    diagnose  from such a recording (no NLF): per frame the 2D distance, in torso lengths, between the
              fitted body's elbows / wrists and MediaPipe's visible ones, with and without
              `BodyFitter.ARM_REFINE`, the left / right swap test, and a contact sheet of the worst frames

    nice -n 15 .venv/bin/python -m tools.evaluate.arm_pose record 10004_outer --out $T/10004_outer.npz
    nice -n 15 .venv/bin/python -m tools.evaluate.arm_pose diagnose $T/10004_outer.npz --sheet docs/figures/arm-pose.jpg
"""

from __future__ import annotations

import argparse
import json
import pickle

import cv2
import numpy as np

from tools.evaluate._common import write_json

ARMS = {"left": (18, 20, 13, 15), "right": (19, 21, 14, 16)}  # SMPL-X elbow, wrist; MediaPipe elbow, wrist
VISIBLE = 0.5
FAR = 0.15  # torso lengths: "disagree"


def record(name, every, out):
    import mediapipe as mp

    from tools.evaluate.reference_frame import find_panel
    from tools.evaluate.vs_registration import HEIGHT, POSE_COMPLEXITY
    from tryon.app.person import LANDMARK_FILTER, landmarks_from
    from tryon.body.regressor import NLF
    from tryon.body.smoothing import OneEuroFilter
    from tryon.paths import RELEASE_VIDEOS

    capture = cv2.VideoCapture(str(RELEASE_VIDEOS / f"{name}.mp4"))
    filt, nlf, box, rows = OneEuroFilter(**LANDMARK_FILTER), NLF(), None, []
    with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=POSE_COMPLEXITY) as pose:
        t = -1
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            t += 1
            if box is None:
                box = find_panel(frame, "left")
            y0, y1, x0, x1 = box
            scale = HEIGHT / (y1 - y0)
            panel = cv2.resize(frame[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            raw = landmarks_from(pose.process(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)), panel.shape[1], panel.shape[0])
            points = None if raw is None else filt(raw, t / 25.0)
            if t % every or raw is None:
                continue
            rows.append({"frame": t, "raw": raw, "points": points, "prediction": nlf.predict(panel, keypoints=raw), "panel_shape": panel.shape})
    with open(out, "wb") as f:
        pickle.dump({"video": name, "rows": rows}, f)
    try:
        nlf.close()
    except Exception:  # a worker slow to exit is killed; the recording is already saved
        nlf._process.kill()
    print(name, len(rows), "frames recorded", flush=True)


def torso_px(points):
    return 0.5 * float(np.linalg.norm(points[11, :2] - points[23, :2]) + np.linalg.norm(points[12, :2] - points[24, :2]))


def arm_errors(joints_2d, points):
    """{arm: (elbow, wrist)} distance to MediaPipe in torso lengths (NaN: landmark below VISIBLE)."""
    length = torso_px(points)
    out = {}
    for arm, (je, jw, me, mw) in ARMS.items():
        out[arm] = tuple(float(np.linalg.norm(joints_2d[j] - points[m, :2]) / length) if points[m, 2] >= VISIBLE else np.nan
                         for j, m in ((je, me), (jw, mw)))
    return out


def fitted_joints(row, refine):
    import torch

    from tryon.body.fitter import BodyFitter

    fitter = BodyFitter()
    BodyFitter.ARM_REFINE = refine
    fitter.fit_from_regression(row["points"], row["prediction"])
    with torch.no_grad():
        return fitter._project(fitter.joints(list(range(22)))).numpy().astype(np.float64), fitter


def swapped(joints_2d, points):
    """True when swapping NLF's left and right wrists would sit nearer MediaPipe's (both visible)."""
    if min(points[15, 2], points[16, 2]) < VISIBLE:
        return False
    keep = np.linalg.norm(joints_2d[20] - points[15, :2]) + np.linalg.norm(joints_2d[21] - points[16, :2])
    flip = np.linalg.norm(joints_2d[21] - points[15, :2]) + np.linalg.norm(joints_2d[20] - points[16, :2])
    return bool(flip < 0.5 * keep)


def diagnose(path, sheet, worst, refine, out):
    from tools.evaluate.reference_frame import find_panel
    from tools.evaluate.vs_registration import HEIGHT
    from tryon.paths import RELEASE_VIDEOS

    data = pickle.load(open(path, "rb"))
    records = []
    for row in data["rows"]:
        if row["points"] is None or row["prediction"] is None or min(row["points"][i, 2] for i in (11, 12, 23, 24)) < 0.4:
            continue
        joints, _ = fitted_joints(row, False)
        before = arm_errors(joints, row["points"])
        record_ = {"frame": row["frame"], "nlf": before, "swap": swapped(joints, row["points"])}
        if refine:
            joints_after, _ = fitted_joints(row, True)
            record_["refined"] = arm_errors(joints_after, row["points"])
        records.append(record_)
    flat = lambda key: np.array([v for r in records for arm in r[key].values() for v in arm])  # noqa: E731
    summary = {"video": data["video"], "frames": len(records), "swaps": sum(r["swap"] for r in records)}
    for key in ("nlf", "refined"):
        if key in records[0]:
            values = flat(key)
            seen = values[np.isfinite(values)]
            summary[key] = {"mean": float(seen.mean()), "share_over_far": float((seen > FAR).mean()),
                            "frames_with_far": sum(any(np.nanmax(arm) > FAR if np.isfinite(arm).any() else False for arm in r[key].values()) for r in records)}
    print(summary, flush=True)
    if out:
        write_json(out, {"summary": summary, "frames": records})
    if sheet:
        score = lambda r: max([np.nanmax(a) for a in r["nlf"].values() if np.isfinite(a).any()] or [0.0])  # noqa: E731
        chosen = sorted(records, key=score)[-worst:]
        capture = cv2.VideoCapture(str(RELEASE_VIDEOS / f"{data['video']}.mp4"))
        panels, box, t = {}, None, -1
        want = {r["frame"] for r in chosen}
        while len(panels) < len(want):
            ok, frame = capture.read()
            if not ok:
                break
            t += 1
            box = box or find_panel(frame, "left")
            if t in want:
                y0, y1, x0, x1 = box
                s = HEIGHT / (y1 - y0)
                panels[t] = cv2.resize(frame[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        rows_by = {r["frame"]: r for r in data["rows"]}
        tiles = []
        for r in chosen:
            row, image = rows_by[r["frame"]], panels[r["frame"]].copy()
            joints, _ = fitted_joints(row, False)
            shown = [(joints, (255, 0, 255))]
            if refine:
                shown.append((fitted_joints(row, True)[0], (0, 255, 0)))
            for js, colour in shown:
                for _, (je, jw, _, _) in ARMS.items():
                    shoulder = {18: 16, 19: 17}[je]
                    pts = [tuple(int(v) for v in js[j]) for j in (shoulder, je, jw)]
                    cv2.polylines(image, [np.array(pts)], False, colour, 3)
            for arm, (_, _, me, mw) in ARMS.items():
                for m in (me, mw):
                    if row["points"][m, 2] >= VISIBLE:
                        cv2.circle(image, tuple(int(v) for v in row["points"][m, :2]), 7, (255, 255, 0), -1)
            cv2.putText(image, f"{data['video'][:5]} f{r['frame']} {score(r):.2f}", (8, 30), 0, 1.0, (255, 255, 255), 2)
            x, y, w, h = cv2.boundingRect(np.array([tuple(int(v) for v in p) for p in joints[[16, 17, 18, 19, 20, 21]]]))
            cx, cy = x + w // 2, y + h // 2
            half = int(max(w, h) * 0.9) + 40
            crop = image[max(cy - half, 0): cy + half, max(cx - half, 0): cx + half]
            tiles.append(cv2.resize(crop, (330, 330)))
        cv2.imwrite(sheet, np.hstack(tiles) if len(tiles) <= 4 else np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:8])]), [cv2.IMWRITE_JPEG_QUALITY, 85])


def annotated_record(out):
    """The 16 annotated views through MediaPipe and NLF (live field of view and true intrinsics) with their ground truth."""
    from tools.evaluate import real_fit
    from tools.evaluate.fit_audit import annotated_root, stop_regressor
    from tryon.body.landmarks import JOINT_MAP
    from tryon.body.regressor import NLF

    real_fit.DATA = annotated_root()
    nlf, order, rows = NLF(), sorted(JOINT_MAP), []
    for take in ("Take1", "Take10"):
        for cam in real_fit.CAMERAS:
            for frame in ("00001", "00002"):
                image = cv2.imread(str(real_fit.DATA / take / cam / "rgb_images" / f"{frame}.png"))
                points = real_fit.landmarks_from_mediapipe(image)
                if points is None:
                    continue
                intrinsics, _ = real_fit.load_camera(take, cam)
                rows.append({"view": f"{take}/{cam}/{frame}", "points": points, "truth": real_fit.ground_truth_joints(take, frame)[[JOINT_MAP[i] for i in order]],
                             "predictions": {"live_fov": nlf.predict(image, keypoints=points), "true_intrinsics": nlf.predict(image, intrinsics=intrinsics.astype(np.float32), keypoints=points)}})
    with open(out, "wb") as f:
        pickle.dump(rows, f)
    stop_regressor(nlf)
    print(len(rows), "views recorded", flush=True)


def annotated_eval(path, settings):
    """PA-MPJPE (mm) of the 12 joints and of the 4 elbow / wrist joints, for each `settings` dict of BodyFitter attributes."""
    import torch

    from tools.evaluate.fit_audit import procrustes_mpjpe
    from tools.evaluate.real_fit import procrustes_align
    from tryon.body.fitter import BodyFitter

    rows = pickle.load(open(path, "rb"))
    result = {}
    for name, attrs in settings.items():
        for key, value in attrs.items():
            setattr(BodyFitter, key, value)
        totals = {"live_fov": [[], []], "true_intrinsics": [[], []]}
        for row in rows:
            for label, prediction in row["predictions"].items():
                fitter = BodyFitter()
                fitter.fit_from_regression(row["points"], prediction)
                with torch.no_grad():
                    joints = fitter.joints().numpy()
                totals[label][0].append(1000 * procrustes_mpjpe(joints, row["truth"]))
                aligned = procrustes_align(joints, row["truth"])
                totals[label][1].append(1000 * float(np.linalg.norm(aligned - row["truth"], axis=1)[2:6].mean()))
        result[name] = {label: {"pa_mpjpe_mm": float(np.mean(v[0])), "arm_mm": float(np.mean(v[1]))} for label, v in totals.items()}
        print(name, result[name], flush=True)
    return result


def iou_run(video, settings, every, out, png=None):
    """Garment IoU against the real shirt through the live NLF path (vs_registration) with BodyFitter attributes set."""
    from tools.evaluate import vs_registration
    from tryon.body.fitter import BodyFitter
    from tryon.garment.renderer import GarmentRenderer

    GarmentRenderer.GPU_RASTER = True
    for key, value in settings.items():
        setattr(BodyFitter, key, value)
    vs_registration.REGRESSOR = "nlf"
    vs_registration.SHOW = None if png is None else [80, 85, 195, 45]
    result = vs_registration.evaluate(video, every, None, png)
    frames = {row["frame"]: row["ours"] for row in result["frames"]}
    print(video, settings, "IoU", result["summary"]["ours"], flush=True)
    if out:
        write_json(out, {"video": video, "settings": settings, "iou": result["summary"]["ours"], "frames": frames,
                           "rows": [{k: row[k] for k in ("frame", "ours", "ours_area_ratio", "ours_missed", "ours_extra_background", "ours_extra_skin", "ours_extra_other")} for row in result["frames"]]})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["record", "diagnose", "annotated-record", "annotated-eval", "iou"])
    parser.add_argument("target")
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--out", default=None)
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--worst", type=int, default=8)
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--settings", default=None, help="annotated-eval: JSON {name: {BodyFitter attribute: value}}")
    args = parser.parse_args()
    if args.mode == "record":
        record(args.target, args.every, args.out)
    elif args.mode == "iou":
        iou_run(args.target, json.loads(args.settings or "{}"), args.every, args.out, args.sheet)
    elif args.mode == "annotated-record":
        annotated_record(args.target)
    elif args.mode == "annotated-eval":
        sweep = json.loads(args.settings) if args.settings else {"off": {"ARM_REFINE": False}, "on": {"ARM_REFINE": True}}
        result = annotated_eval(args.target, sweep)
        if args.out:
            write_json(args.out, result)
    else:
        diagnose(args.target, args.sheet, args.worst, args.refine, args.out)


if __name__ == "__main__":
    main()
