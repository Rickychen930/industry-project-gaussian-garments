"""Our live pipeline against the authors' own registration, frame by frame.

Each Gaussian-Garments release video shows the captured person ("Ground
Truth", left) beside the authors' registration of the garment ("Ours -
Registration", right: their body in grey, their garment colour-coded) -
a reference that demonstrably works. This scores both against the real
shirt on every `every`-th frame of the whole clip, not only frame 0:

  truth     the parser's upper clothing (SCHP ATR 4 / 7) on the left panel
  theirs    the registration's saturated pixels (the colour-coded garment;
            its body is grey), mapped into the left panel by one similarity
            found on the first frame by matching the two person silhouettes
  ours      the garment `GarmentRenderer` draws on the left panel, driven by
            the live body logic (MediaPipe, One Euro, fit / `track_step`),
            cloth solver on

and reports silhouette IoU per frame and on average. Theirs is a 4-camera
photometric registration; ours is one camera, live. Where the two curves
part is where we lose.

    .venv/bin/python -m tools.evaluate.vs_registration [00152_outer] [--every 5] [--out j.json] [--png sheet.jpg]
"""

from __future__ import annotations

import argparse
import collections
import time

import cv2
import numpy as np

from tools.evaluate._common import iou, write_json
from tools.evaluate.cloth_trajectory import REFERENCE_FPS

from tryon.paths import GARMENTS, RELEASE_VIDEOS
HEIGHT = 870
SHOW = None  # frames to put in the overlay sheet (None: every 4th scored frame)
POSE_COMPLEXITY = 1  # the heavier MediaPipe model: the recorded baselines (docs/baseline/vs-registration-*.json) were measured with it; live uses 0 (tryon/app/pipeline.py, IoU within 0.001)
REGRESSOR = None  # "multiHMR_672_S" or "nlf": body from tryon.body.regressor
REGRESS_EVERY = 1  # run it every n-th frame, `track_step` in between
REGRESS_LAG = 0  # frames before its answer is used (live: it runs on a thread)
CAPTURE = None  # a list: every scored frame's (t, raw landmarks, our garment mask, real shirt mask) is appended


def registration_masks(region, white=235, saturation=60):
    """(person, garment) masks of the registration panel region."""
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    person = region.astype(np.int32).sum(axis=2) < 3 * white
    person = cv2.morphologyEx(person.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    garment = person & (hsv[..., 1] > saturation)
    return person, garment


def fit_similarity(source, target):
    """Scale + offset mapping mask `source` onto mask `target`: bounding
    boxes first, then a small search maximising IoU."""
    ys, xs = np.nonzero(source)
    yt, xt = np.nonzero(target)
    scale = (yt.max() - yt.min()) / max(ys.max() - ys.min(), 1)
    ox = xt.mean() - scale * xs.mean()
    oy = yt.min() - scale * ys.min()
    best = None
    for ds in np.linspace(-0.04, 0.04, 9):
        for dx in range(-24, 25, 6):
            for dy in range(-24, 25, 6):
                s, x, y = scale * (1 + ds), ox + dx, oy + dy
                warped = warp(source, s, x, y, target.shape)
                score = iou(warped, target)
                if best is None or score > best[0]:
                    best = (score, s, x, y)
    return best


def warp(mask, s, ox, oy, shape):
    m = np.array([[s, 0, ox], [0, s, oy]], np.float32)
    return cv2.warpAffine(mask.astype(np.uint8), m, (shape[1], shape[0]), flags=cv2.INTER_NEAREST).astype(bool)


def evaluate(name, every=5, frames=None, png=None, dynamics=True, export=None, cloth=None, subsample=None, sweep=()):
    import mediapipe as mp

    from tools.evaluate.reference_frame import find_panel
    from tryon.body.fitter import BodyFitter
    from tryon.app.person import LANDMARK_FILTER, landmarks_from, torso_visible, track_step
    from tryon.removal.parsing_schp import ATR_CHECKPOINT, SCHPParser
    from tryon.garment.renderer import GarmentRenderer
    from tryon.body.smoothing import OneEuroFilter

    capture = cv2.VideoCapture(str(RELEASE_VIDEOS / f"{name}.mp4"))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = min(frames or total, total)
    parser = SCHPParser(ATR_CHECKPOINT)
    from tryon.removal.parsing_schp import LIP_CHECKPOINT

    lip = SCHPParser(LIP_CHECKPOINT, size=473)
    renderer = GarmentRenderer(GARMENTS / name, dynamics=dynamics, subsample=subsample)
    # `sweep`: more fractions to score on the very same fitted bodies, each
    # "F" (uniform subset) or "Fi" (importance subset); see GarmentRenderer.subsample
    variants = {}
    for spec in sweep:
        GarmentRenderer.SUBSAMPLE_BY_IMPORTANCE = spec.endswith("i")
        fraction = float(spec.rstrip("i"))
        variants[spec] = GarmentRenderer(GARMENTS / name, dynamics=dynamics, subsample=fraction if fraction < 1.0 else None)
    # `export`: save every frame's fitted body (model space) and the frame-0
    # garment for an offline simulator (none in the repo now); `cloth`: an
    # npz of such a simulator's garment per frame (reference 7972-vertex
    # order) to render instead of our own skinning / cloth solver.
    exported_bodies, exported_cloth0 = [], None
    replay = None if cloth is None else np.load(cloth)["cloth"]
    from tools.evaluate.cloth_trajectory import reference_vertex_map

    to_ref, ref_count = reference_vertex_map(renderer)
    regressor = None
    if REGRESSOR:
        from tryon.body import regressor as body_regressor

        if REGRESSOR == "nlf":
            regressor = body_regressor.NLF()
        else:
            regressor = body_regressor.MultiHMR(REGRESSOR, input_size=body_regressor.INPUT_SIZE)
    filt = OneEuroFilter(**LANDMARK_FILTER)
    pending = None
    fitter, errors, box, similarity = None, collections.deque(maxlen=10), None, None
    previous_person = previous_skin = None
    rows, tiles = [], []
    with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=POSE_COMPLEXITY) as pose:
        for t in range(frames):
            ok, frame = capture.read()
            if not ok:
                break
            if box is None:
                box = find_panel(frame, "left")
            y0, y1, x0, x1 = box
            scale = HEIGHT / (y1 - y0)
            panel = cv2.resize(frame[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            region = cv2.resize(frame[y0:y1, x1:], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            raw = landmarks_from(pose.process(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)), panel.shape[1], panel.shape[0])
            points = None if raw is None else filt(raw, t / 25.0)
            import tryon.body.fitter as _bf

            if _bf.SILHOUETTE_WEIGHT > 0 and fitter is not None:
                # live has the PREVIOUS frame's parse when it tracks (the
                # mask runs on its own thread after), so that is what is used
                fitter.set_silhouette(previous_person)
                fitter.set_skin(previous_skin)
            if points is not None and torso_visible(points, tracking=fitter is not None):
                prediction = None
                if regressor is not None and (fitter is None or t % REGRESS_EVERY == 0) and pending is None:
                    pending = (t + (0 if fitter is None else REGRESS_LAG), regressor.predict(panel, keypoints=raw) if REGRESSOR == "nlf" else regressor.predict(panel))
                if pending is not None and t >= pending[0]:
                    prediction, pending = pending[1], None
                if prediction is not None:
                    fitter = fitter or BodyFitter()
                    fitter.fit_from_regression(points, prediction)
                elif fitter is None:
                    fitter = BodyFitter()
                    fitter.fit(points)
                else:
                    if track_step(fitter, points, errors, HEIGHT, raw_points=raw) == "refit":
                        filt.reset()
                        renderer.reset_motion(fitter)
                mesh = renderer.posed_mesh(fitter, dt=1.0 / 25.0)
                if export is not None:
                    exported_bodies.append(np.asarray(fitter.vertices(), np.float32))
                    if exported_cloth0 is None:
                        summed = np.zeros((ref_count, 3))
                        np.add.at(summed, to_ref, mesh)
                        exported_cloth0 = (summed / np.bincount(to_ref, minlength=ref_count)[:, None]).astype(np.float32)
            else:
                fitter = None
            if _bf.SILHOUETTE_WEIGHT > 0:
                labels = parser.labels(panel)
                previous_person = labels != 0
                previous_skin = np.isin(labels, (11, 12, 13, 14, 15))
            if t % every:
                continue
            if _bf.SILHOUETTE_WEIGHT <= 0:
                labels = parser.labels(panel)
            person, shirt = labels != 0, np.isin(labels, (4, 7))
            # The OUTER garment only: ATR calls an undershirt "upper clothes"
            # too (00152_outer's yellow top under the open shirt), which
            # credited a garment drawn over it and penalised the
            # registration for leaving it bare. Same rule as the removal
            # (ClothingRemover.KEEP_INNER_LAYER): with a LIP coat present,
            # LIP's upper-clothes / jumpsuit pixels are the inner layer.
            lip_labels = lip.labels(panel)
            coat, inner = lip_labels == 7, np.isin(lip_labels, (5, 10))
            if coat.sum() >= 0.5 * max(int((coat | inner).sum()), 1):
                shirt &= ~inner
            reg_person, reg_garment = registration_masks(region)
            if similarity is None:
                similarity = fit_similarity(reg_person, person)
            theirs = warp(reg_garment, *similarity[1:], shirt.shape)
            ours = np.zeros_like(shirt)
            ours_person = np.zeros_like(shirt)
            if fitter is not None:
                from tryon.removal.background import body_silhouette

                if replay is not None and t < len(replay):
                    frame_mesh = np.asarray(replay[t], np.float64)[to_ref]
                    renderer.posed_mesh = lambda _f, dt=None, _m=frame_mesh: _m
                ours = renderer.render(fitter, panel.shape[:2], dt=every / REFERENCE_FPS)[..., 3] > 25
                ours_person = ours | body_silhouette(fitter, panel.shape, grow=0.0)
            theirs_person = warp(reg_person, *similarity[1:], person.shape)
            swept = {}
            if fitter is not None:
                for spec, variant in variants.items():
                    started = time.perf_counter()
                    drawn = variant.render(fitter, panel.shape[:2], dt=every / REFERENCE_FPS)[..., 3] > 25
                    swept[f"render_ms@{spec}"] = 1000 * (time.perf_counter() - started)
                    swept[f"ours@{spec}"] = iou(drawn, shirt)
            if CAPTURE is not None:
                CAPTURE.append((t, None if raw is None else np.array(raw), ours.copy(), shirt.copy()))
            rows.append({"frame": t, "theirs": iou(theirs, shirt), "ours": iou(ours, shirt),
                         "theirs_person_alignment": iou(theirs_person, person),
                         "ours_person": iou(ours_person, person),
                         # garment area over the real shirt's: > 1 means drawn too big
                         "ours_area_ratio": float(ours.sum() / max(shirt.sum(), 1)),
                         "theirs_area_ratio": float(theirs.sum() / max(shirt.sum(), 1)),
                         # where our extra pixels fall, as fractions of the real shirt's area:
                         # outside the person (outline too big) or on the person's
                         # other parts, by ATR class group
                         "ours_extra_background": float((ours & ~shirt & ~person).sum() / max(shirt.sum(), 1)),
                         "ours_extra_skin": float((ours & ~shirt & np.isin(labels, (11, 14, 15, 12, 13))).sum() / max(shirt.sum(), 1)),
                         "ours_extra_lower": float((ours & ~shirt & np.isin(labels, (5, 6))).sum() / max(shirt.sum(), 1)),
                         "ours_extra_other": float((ours & ~shirt & person & ~np.isin(labels, (11, 14, 15, 12, 13, 5, 6))).sum() / max(shirt.sum(), 1)),
                         "ours_missed": float((shirt & ~ours).sum() / max(shirt.sum(), 1)), **swept})
            print(f"{name} frame {t}: IoU vs real shirt - theirs {rows[-1]['theirs']:.3f}, ours {rows[-1]['ours']:.3f}", flush=True)
            if png is not None and len(tiles) < 8 and (t % (every * 4) == 0 if not SHOW else t in SHOW):
                vis = panel.copy()
                vis[shirt & ~theirs & ~ours] = (0.5 * vis[shirt & ~theirs & ~ours]).astype(np.uint8)
                overlay = vis.copy()
                overlay[theirs] = (0.5 * overlay[theirs] + [120, 0, 0]).astype(np.uint8)
                overlay[ours] = (0.5 * overlay[ours] + [0, 0, 120]).astype(np.uint8)
                cv2.putText(overlay, str(t), (8, 30), 0, 1.0, (255, 255, 255), 2)
                tiles.append(cv2.resize(overlay, (int(overlay.shape[1] * 400 / overlay.shape[0]), 400)))
    if export is not None:
        np.savez(export, bodies=np.stack(exported_bodies), cloth0=exported_cloth0, body_faces=np.asarray(BodyFitter().faces))
    if png is not None and tiles:
        cv2.imwrite(png, np.hstack(tiles))
    summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "frame"}
    return {"video": name, "similarity": similarity[1:], "alignment_iou_frame0": similarity[0], "frames": rows, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="?", default="00152_outer")
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--png", default=None)
    parser.add_argument("--rigid", action="store_true", help="rigid skinning, no cloth solver")
    parser.add_argument("--export", default=None, help="npz: every frame's fitted body + frame-0 garment (one video only)")
    parser.add_argument("--regressor", default=None, help="Multi-HMR model name, e.g. multiHMR_672_S")
    parser.add_argument("--regress-every", type=int, default=1)
    parser.add_argument("--regressor-size", type=int, default=None, help="input px (default body_regressor.INPUT_SIZE)")
    parser.add_argument("--regress-lag", type=int, default=0)
    parser.add_argument("--regression-camera", choices=["regressor", "keypoints"], default=None)
    parser.add_argument("--cloth", default=None, help="npz with 'cloth' (T, V_ref, 3): render this garment sequence instead")
    parser.add_argument("--subsample", type=float, default=None, help="draw only this fraction of the garment's Gaussians (live: --garment-subsample)")
    parser.add_argument("--importance", action="store_true", help="with --subsample: keep the splats with the largest opacity * footprint, not a uniform subset")
    parser.add_argument("--gpu-render", action="store_true", help="the live path's OpenGL splat renderer (the launcher's --gpu-render)")
    parser.add_argument("--size-mode", default=None, help="GarmentRenderer.SIZE_MODE: how the wearer's size ratios scale the garment (ScanPoseBinder.SIZE_MODES)")
    parser.add_argument("--classic-cloth", action="store_true", help="LightCloth instead of SmoothCloth (live: --classic-cloth)")
    parser.add_argument("--relaxed-rest", action="store_true", help="GarmentRenderer.RELAXED_REST='mean': bind to the garment's relaxed rest shape (stage1/relaxed_rest.npz, tools/evaluate/relaxed_rest.py)")
    parser.add_argument("--relaxed-scale", type=float, default=None, help="GarmentRenderer.RELAXED_SCALE: fraction of the relaxed-rest offset applied")
    parser.add_argument("--sweep", default="", help="comma list of fractions scored on the same bodies: 0.5 uniform, 0.5i importance (1.0 = all)")
    args = parser.parse_args()
    from tryon.garment.cloth import LightCloth
    from tryon.garment.renderer import GarmentRenderer

    if args.classic_cloth:
        GarmentRenderer.CLOTH = LightCloth
    GarmentRenderer.SUBSAMPLE_BY_IMPORTANCE = args.importance
    if args.relaxed_rest:
        GarmentRenderer.RELAXED_REST = "mean"
    if args.relaxed_scale is not None:
        GarmentRenderer.RELAXED_SCALE = args.relaxed_scale
    if args.size_mode:
        GarmentRenderer.SIZE_MODE = args.size_mode
    GarmentRenderer.GPU_RASTER = GarmentRenderer.GPU_RASTER or args.gpu_render
    global REGRESSOR, REGRESS_EVERY
    global REGRESS_LAG
    REGRESSOR, REGRESS_EVERY, REGRESS_LAG = args.regressor, args.regress_every, args.regress_lag
    if args.regressor_size:
        import tryon.body.regressor as body_regressor

        body_regressor.INPUT_SIZE = args.regressor_size
    if args.regression_camera:
        import tryon.body.fitter as bodyfit

        bodyfit.REGRESSION_CAMERA = args.regression_camera
    results = []
    for name in args.videos.split(","):
        png = None if args.png is None else args.png.replace(".jpg", f"_{name}.jpg")
        result = evaluate(name, args.every, args.frames, png, dynamics=not args.rigid, export=args.export, cloth=args.cloth, subsample=args.subsample, sweep=[x for x in args.sweep.split(',') if x])
        print(name, result["summary"], flush=True)
        results.append(result)
    if args.out:
        write_json(args.out, results)


if __name__ == "__main__":
    main()
