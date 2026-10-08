"""Where a wrong fitted body comes from: NLF's prediction, its box, or the tracking.

Runs `vs_registration.evaluate` on one release video (the live NLF path) with
probes on the side, and for every scored frame records

  iou            the garment's IoU against the real shirt (the evaluator's number)
  body_person    IoU of our fitted body's silhouette with the person mask (SCHP ATR, any non-background label)
  body_in        share of the body silhouette that is person; person_in: share of the person the body covers
  nlf_torso      NLF's raw projected shoulders / hips against MediaPipe's, mean px over the torso length
  body_torso     the same for the fitted body (after the camera placement)
  yaw, mp_yaw    fitted body's yaw and MediaPipe's world-landmark yaw (degrees)
  box            the box NLF was given, and the share of the person mask outside it
  nlf_ok         whether NLF answered, and how many of its 22 joints fall in the image

and writes a contact sheet of the worst and best frames (the body mesh contour over the frame).

    .venv/bin/python -m tools.evaluate.body_failures 00176_outer --every 5 --out a.json --sheet a.npz
    .venv/bin/python -m tools.evaluate.body_failures --merge a.npz,b.npz --image docs/figures/body-failures.jpg

`--box landmarks|mask|union|detector` swaps the box NLF gets (mask: the person
mask's bounding box, union: both, detector: NLF's own YOLO), `--num-aug N` its
test-time crops, `--set module.NAME=value` any module constant (a candidate
gate's switch).
"""

from __future__ import annotations

import argparse
import importlib
import time

import cv2
import numpy as np

from tools.evaluate._common import iou, write_json

TORSO_PAIRS = ((16, 11), (17, 12), (1, 23), (2, 24))  # (SMPL-X joint, MediaPipe landmark)
BONE_PAIRS = TORSO_PAIRS + ((18, 13), (19, 14), (20, 15), (21, 16), (4, 25), (5, 26))
VISIBLE = 0.4
WORST, BEST = 8, 4


def pair_errors(joints_2d, landmarks, pairs):
    """Per-pair pixel distance between projected SMPL-X joints and visible MediaPipe landmarks (NaN: not visible)."""
    out = []
    for joint, landmark in pairs:
        if landmarks is None or landmarks[landmark, 2] < VISIBLE or not np.all(np.isfinite(joints_2d[joint])):
            out.append(np.nan)
        else:
            out.append(float(np.linalg.norm(joints_2d[joint] - landmarks[landmark, :2])))
    return np.array(out)


def torso_length(landmarks):
    """Mean shoulder-to-hip distance in px (None without the four torso landmarks)."""
    if landmarks is None or min(landmarks[i, 2] for i in (11, 12, 23, 24)) < VISIBLE:
        return None
    return 0.5 * float(np.linalg.norm(landmarks[11, :2] - landmarks[23, :2]) + np.linalg.norm(landmarks[12, :2] - landmarks[24, :2]))


def mask_box(mask, margin=0.1):
    """[x, y, w, h] around a boolean mask, grown by `margin` of its larger side (None when empty)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    grow = margin * max(xs.max() - xs.min(), ys.max() - ys.min())
    x0, y0 = max(xs.min() - grow, 0), max(ys.min() - grow, 0)
    x1, y1 = min(xs.max() + grow, mask.shape[1] - 1), min(ys.max() + grow, mask.shape[0] - 1)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]


def union_box(a, b):
    if a is None or b is None:
        return a if b is None else b
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return [x0, y0, x1 - x0, y1 - y0]


def outside_box(mask, box):
    """Share of `mask`'s pixels outside `box` [x, y, w, h]."""
    if box is None or not mask.any():
        return float("nan")
    x, y, w, h = (int(round(v)) for v in box)
    inside = mask[max(y, 0): y + h, max(x, 0): x + w].sum()
    return float(1.0 - inside / mask.sum())


class Probe:
    """Everything the patches record; frame index `t` is counted at MediaPipe's `process`."""

    def __init__(self, every, box_mode):
        self.t, self.every, self.box_mode = -1, every, box_mode
        self.panel = None
        self.labels = {}
        self.nlf = {}
        self.scored = {}


def install(probe, parser_size=512):
    import mediapipe as mp

    from tryon.body import regressor
    from tryon.garment.renderer import GarmentRenderer
    from tryon.removal.parsing_schp import SCHPParser

    process = mp.solutions.pose.Pose.process

    def counted(self, image, *args, **kwargs):
        probe.t += 1
        probe.panel = image
        return process(self, image, *args, **kwargs)

    mp.solutions.pose.Pose.process = counted

    labels_of = SCHPParser.labels

    def labelled(self, image, *args, **kwargs):
        out = labels_of(self, image, *args, **kwargs)
        if self.size == parser_size and probe.t % probe.every == 0:
            probe.labels[probe.t] = out != 0
        return out

    SCHPParser.labels = labelled

    predict = regressor.NLF.predict
    own = {"parser": None}

    def watched(self, frame_bgr, intrinsics=None, keypoints=None, box=None):
        landmarks_box = regressor.box_from_landmarks(keypoints, frame_bgr.shape)
        if probe.box_mode != "landmarks":
            if probe.box_mode == "detector":
                box = None
            else:
                from tryon.removal.parsing_schp import ATR_CHECKPOINT

                own["parser"] = own["parser"] or SCHPParser(ATR_CHECKPOINT)
                from_mask = mask_box(own["parser"].labels(frame_bgr) != 0)
                box = from_mask if probe.box_mode == "mask" else union_box(from_mask, landmarks_box)
        else:
            box = landmarks_box if box is None else box
        started = time.perf_counter()
        prediction = predict(self, frame_bgr, intrinsics=intrinsics, keypoints=keypoints if box is None else None, box=box)
        probe.nlf[probe.t] = {"prediction": prediction, "box": box, "landmarks": None if keypoints is None else np.array(keypoints),
                              "ms": 1000 * (time.perf_counter() - started)}
        return prediction

    regressor.NLF.predict = watched

    render = GarmentRenderer.render

    def rendered(self, fitter, size, *args, **kwargs):
        out = render(self, fitter, size, *args, **kwargs)
        if probe.t % probe.every == 0 and fitter is not None:
            from tryon.removal.background import body_silhouette

            with np.errstate(all="ignore"):
                import torch

                with torch.no_grad():
                    joints = fitter._project(fitter.joints(list(range(22)))).numpy().astype(np.float64)
            probe.scored[probe.t] = {"body": body_silhouette(fitter, size, grow=0.0), "joints": joints,
                                     "yaw": float(np.degrees(fitter.global_orient.detach()[0, 1])), "panel": probe.panel.copy()}
        return out

    GarmentRenderer.render = rendered


def frame_record(probe, row):
    """The numbers of one scored frame: `row` is vs_registration's, the probe holds the rest."""
    t = row["frame"]
    record = {"frame": t, "iou": row["ours"]}
    shot, nlf, person = probe.scored.get(t), probe.nlf.get(t), probe.labels.get(t)
    landmarks = None if nlf is None else nlf["landmarks"]
    length = torso_length(landmarks)
    record["torso_px"] = length
    if shot is not None and person is not None:
        body = shot["body"]
        record["body_person"] = iou(body, person, empty=0.0)
        record["body_in"] = float((body & person).sum() / max(body.sum(), 1))
        record["person_in"] = float((body & person).sum() / max(person.sum(), 1))
        record["yaw"] = shot["yaw"]
        if length:
            record["body_torso"] = float(np.nanmean(pair_errors(shot["joints"], landmarks, TORSO_PAIRS)) / length)
            record["body_bones"] = float(np.nanmean(pair_errors(shot["joints"], landmarks, BONE_PAIRS)) / length)
        if nlf is not None:
            record["outside_box"] = outside_box(person, nlf["box"])
    prediction = None if nlf is None else nlf["prediction"]
    record["nlf_ok"] = prediction is not None
    if prediction is not None:
        joints = np.asarray(prediction["joints_2d"], np.float64)
        height, width = prediction["image_size"]
        record["nlf_inside"] = int(((joints[:, 0] >= 0) & (joints[:, 0] < width) & (joints[:, 1] >= 0) & (joints[:, 1] < height)).sum())
        if length:
            record["nlf_torso"] = float(np.nanmean(pair_errors(joints, landmarks, TORSO_PAIRS)) / length)
            record["nlf_bones"] = float(np.nanmean(pair_errors(joints, landmarks, BONE_PAIRS)) / length)
    if landmarks is not None:
        from tryon.body.landmarks import landmark_yaw

        yaw = landmark_yaw(landmarks)
        record["mp_yaw"] = None if yaw is None else float(np.degrees(yaw))
    return record


def tile(shot, box, label, height=420):
    image = shot["panel"][..., ::-1].copy() if False else cv2.cvtColor(shot["panel"], cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(shot["body"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    if box is not None:
        x, y, w, h = (int(v) for v in box)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 200, 255), 1)
    for x, y in shot["joints"][[16, 17, 1, 2]]:
        if np.isfinite(x) and np.isfinite(y):
            cv2.circle(image, (int(x), int(y)), 4, (255, 0, 255), -1)
    out = cv2.resize(image, (int(image.shape[1] * height / image.shape[0]), height))
    cv2.putText(out, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3)
    cv2.putText(out, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return out


def sheet_tiles(name, probe, records):
    """npz-able dict of the worst and best frames' tiles for `name`."""
    ranked = sorted((r for r in records if r["frame"] in probe.scored), key=lambda r: r["iou"])
    chosen = [("worst", r) for r in ranked[:WORST]] + [("best", r) for r in ranked[-BEST:]]
    tiles = [tile(probe.scored[r["frame"]], probe.nlf.get(r["frame"], {}).get("box"), f"{name[:5]} {kind} f{r['frame']} IoU {r['iou']:.2f}") for kind, r in chosen]
    return {"tiles": np.stack([cv2.resize(t, (260, 420)) for t in tiles]), "kinds": np.array([k for k, _ in chosen])}


def summarize(records):
    keys = sorted({k for r in records for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)} - {"frame"})
    return {k: float(np.nanmean([r[k] for r in records if r.get(k) is not None])) for k in keys}


def merge(paths, image):
    rows = []
    for path in paths:
        data = np.load(path)
        tiles, kinds = data["tiles"], data["kinds"]
        rows.append(np.hstack(list(tiles[kinds == "worst"][:WORST])))
        rows.append(np.hstack(list(tiles[kinds == "best"][:BEST]) + [np.zeros((420, 260 * (WORST - BEST), 3), np.uint8)]))
    cv2.imwrite(image, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", nargs="?")
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--box", choices=["landmarks", "mask", "union", "detector"], default="landmarks")
    parser.add_argument("--num-aug", type=int, default=None)
    parser.add_argument("--regressor", default="nlf", help="'none': the keypoint-only fit")
    parser.add_argument("--set", action="append", default=[], help="module.NAME=value, e.g. tryon.body.fitter.REGRESSION_GATE=True")
    parser.add_argument("--out", default=None)
    parser.add_argument("--sheet", default=None, help="npz for this clip's contact-sheet tiles")
    parser.add_argument("--merge", default=None, help="comma list of --sheet npz files to join into --image")
    parser.add_argument("--image", default=None)
    args = parser.parse_args()
    if args.merge:
        merge(args.merge.split(","), args.image)
        return
    from tryon.body import regressor
    from tryon.garment.renderer import GarmentRenderer
    from tools.evaluate import vs_registration

    GarmentRenderer.GPU_RASTER = True
    if args.num_aug:
        regressor.NLF.NUM_AUG = args.num_aug
    for spec in args.set:
        target, value = spec.split("=")
        module, name = target.rsplit(".", 1)
        setattr(importlib.import_module(module), name, eval(value))
    vs_registration.REGRESSOR = None if args.regressor == "none" else args.regressor
    probe = Probe(args.every, args.box)
    install(probe)
    started = time.perf_counter()
    result = vs_registration.evaluate(args.video, args.every, args.frames)
    records = [frame_record(probe, row) for row in result["frames"]]
    summary = summarize(records)
    summary["wall_s"] = time.perf_counter() - started
    print(args.video, summary, flush=True)
    if args.out:
        write_json(args.out, {"video": args.video, "args": vars(args), "summary": summary, "frames": records})
    if args.sheet:
        np.savez(args.sheet, **sheet_tiles(args.video, probe, records))


if __name__ == "__main__":
    main()
