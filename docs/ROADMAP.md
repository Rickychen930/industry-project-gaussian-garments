# Roadmap

Where the project stands, what is next, and what was tried and dropped.
The dated record behind every number is [`LOG.md`](LOG.md); how to run
things is [`USAGE.md`](USAGE.md).

**Working rules.** One part of the system at a time. Every change is
measured against ground truth (the authors' registration, annotated views,
the teacher network) before it is kept; if it does not measure better it is
removed and recorded below. Pretrained models are used wherever they measure
better than the alternative.

## Current state (8 Oct 2026)

| part | now | measured by |
|---|---|---|
| body | NLF-S (backbone on the GPU), refined toward MediaPipe's keypoints; 24 mm PA-MPJPE on 8 annotated views (keypoint fit alone: 87-94 mm) | `docs/baseline/body-regressors.json` |
| garment | skinned from the scan pose; `LightCloth` springs, body collision, strain limit | `tools/evaluate/cloth_trajectory.py` |
| shading | per-garment student (1.55M parameters) with a stored per-Gaussian mean, trained on the garment's own motion and every other simulation the authors released: 29.1 dB on unseen motions (one-motion student: 26.9) | `tools/evaluate/shading_students.py` |
| removal | SCHP (ATR + LIP) on CoreML, learned background plate, Telea fill | `tools/evaluate/background_fill.py` |
| accuracy | garment IoU vs the real shirt **0.730** (5 release videos; the authors' 4-camera registration: 0.827) | `tools/evaluate/vs_registration.py` |
| speed (M1, sync) | 0.21-0.23 s a frame, 4.4-4.8 fps on a quiet machine; the body regressor works on the next frame while this one is drawn; smooth display at camera rate (~25 fps). The 8 Oct removal of repeated work cut a loaded-machine frame from 300-388 to 187-217 ms (clean-machine figure still to be taken) | `python -m tryon.app --profile` |

## Targets

| target | goal | now | |
|---|---|---|---|
| garment on the real shirt (IoU, 5 release videos) | >= 0.80 | 0.730 | open |
| body accuracy (PA-MPJPE, 8 annotated views) | <= 30 mm | 24 mm | done |
| display frame rate | camera rate (>= 20 fps) | ~25 fps (smooth display) | done |
| pipeline time a frame (M1) | <= 0.5 s | 0.21-0.23 s | done |
| garment present while a person is in view | >= 95 % of frames | 100 % on 15 release videos, 99.5 % on 14 webcam scenarios | done |
| smooth display vs exact (webcam-speed motion) | >= 90 % of the exact IoU | 93 % at 1/8 of dancer speed | done |

## Done

Body tracking (MediaPipe -> SMPL-X, then NLF-S from the image) · garment
follows the body (skinning from the scan pose) · Gaussian-splat rendering
(CPU, then GPU: 143 -> 37 ms) · clothing removal (SCHP, learned background
fill) · cloth (springs, collision, strain limit) · 360-degree turns (facing
from MediaPipe's 3D landmarks and the regressor) · live system with a
fluid display process · robustness to desk framing, close-ups, dim light,
blur, objects in front, a person leaving · pipeline under 0.5 s a frame ·
pose-dependent shading from a distilled student · code audit and refactor
([`AUDIT.md`](AUDIT.md)).

## Next

1. **Garment motion closer to the authors' simulation** (IoU 0.730 -> 0.80):
   the gap is where the cloth, not the body, differs
   (`tools/evaluate/reference_frame.py` splits the loss level by level).
2. **Scene lighting on the garment**: light estimated from the face; the
   garments carry studio light (`--harmonize` is the stop-gap).
3. **Mask for the next frame from `prefetch`**: the clothing-mask thread idles while the previous frame is removed, composited and shown; starting it one frame early could save up to ~60 ms (needs the mask job to hand back its labels).
4. **Shading refresh rate**: the student updates every 0.7 s on a worker
   thread; test 0.35 s with the GPU free and keep it only if fps holds.
5. **AnyWear / Decart comparison**: possible only with the user's own API
   key and their approval to upload the release videos.
6. **4D-DRESS**: its access request was archived (the data cannot be
   redistributed, so no copies are used); see [`DATASETS.md`](DATASETS.md).

## Tried and dropped

Each row was measured; the entry in [`LOG.md`](LOG.md) has the numbers.

| idea | why not |
|---|---|
| NLF with 3 test-time crops | +0.0014 IoU, 15-25 % fewer fps (2 crops are kept) |
| NLF on every other frame | mean IoU 0.723 -> 0.694: the tracker in between falls behind on fast turns |
| body placement refined to the person's silhouette | IoU 0.7243 / 0.7163 against 0.7253 |
| clothing parse on every other frame | the mask moves too much (IoU 0.11 at worst) |
| garment rendered at 0.75 / 0.5 scale | render time is per splat, not per pixel |
| FASHN parser on CoreML, parser on the Neural Engine | still 4-5x SCHP; kept as the opt-in quality parser |
| pruning transparent Gaussians | nothing to prune (0-2 % below 0.05 opacity) |
| ambient occlusion as an input of the student | measured, not kept (the student reads normals and body distance) |
| the project's own U-Net clothing parser | SCHP measured better on every source; removed 8 Oct (tag `archive/unet-segmentation-2026-10-08`) |
| position-based cloth solver, learned pose correctives, 3D re-skin display | no net gain; removed 1 Oct (tag `archive/pre-cleanup-2026-10-01`) |
| SAM 2 for the clothing mask, RTMPose, SMPLer-X, Multi-HMR 672_B | compared, not adopted (numbers in `LOG.md`) |
| numba `parallel=True` kernel on the mask thread | aborts the process (numba's workqueue layer is not thread-safe) and serially is no faster than numpy |
| diffusion try-on (ViViD, CatVTON, FASHN, VTON360, FitMe360) | tens of seconds to 30 minutes a frame on the M1 ([`REFERENCE_RUNS.md`](REFERENCE_RUNS.md)) |
| ContourCraft cloth | 2.2-2.6 s a step on the M1; driven by our tracked bodies it billows (IoU 0.476 against 0.600) |
