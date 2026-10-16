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
| body | NLF-S (backbone on the GPU), refined toward MediaPipe's keypoints; 24.6 mm PA-MPJPE on 16 annotated views (keypoint fit alone: 87-94 mm) | `docs/baseline/body-regressors.json` |
| garment | skinned from the scan pose; `LightCloth` springs, body collision, strain limit | `tools/evaluate/cloth_trajectory.py` |
| shading | per-garment student (1.55M parameters) with a stored per-Gaussian mean, trained on the garment's own motion and every other simulation the authors released: 29.1 dB on unseen motions (one-motion student: 26.9) | `tools/evaluate/shading_students.py` |
| removal | SCHP (ATR + LIP) on CoreML, learned background plate, Telea fill | `tools/evaluate/background_fill.py` |
| accuracy | garment IoU vs the real shirt **0.730** (5 release videos; the authors' 4-camera registration: 0.827) | `tools/evaluate/vs_registration.py` |
| speed (M1, sync) | 0.1-0.2 s of work a frame (0.085-0.10 s on a quiet machine; HUD 7.3-7.5 fps on a video file, whose decode adds ~40 ms a frame; a webcam figure is still to be taken); 0.21-0.23 s (4.4-4.8 fps) before 8 Oct; the body regressor works on the next frame while this one is drawn; smooth display at camera rate (~25 fps). The 8 Oct removal of repeated work cut a loaded-machine frame from 300-388 to 187-217 ms (clean-machine figure still to be taken) | `python -m tryon.app --profile` |

## Targets

| target | goal | now | |
|---|---|---|---|
| garment on the real shirt (IoU, 5 release videos) | >= 0.80 | 0.730 | open |
| body accuracy (PA-MPJPE, 16 annotated views) | <= 30 mm | 24.6 mm | done |
| display frame rate | camera rate (>= 20 fps) | ~25 fps (smooth display) | done |
| pipeline time a frame (M1) | <= 0.5 s | 0.1-0.2 s (0.085-0.10 s quiet) | done |
| garment present while a person is in view | >= 95 % of frames | 100 % on 15 release videos, 99.5 % on 14 webcam scenarios | done |
| smooth display vs exact (webcam-speed motion) | >= 90 % of the exact IoU | 93 % at 1/8 of dancer speed | done |

## Where the IoU gap is (measured 8 Oct)

Garment IoU against the real shirt is 0.728 over the 5 release videos; the
authors' registration reaches 0.827. Four measurements say where the 0.10 is not:

| not the cause | evidence |
|---|---|
| cloth stepping | with the cloth stepped at the clip's own dt the IoU is 0.7281, without it 0.7218, with `SmoothCloth` 0.7283; replaying 5 garments, `LightCloth` is 5.467 cm from the authors' simulation and plain skinning 5.454 (`tools/evaluate/cloth_smoothness.py`). The authors' cloth differs from skinning by 7.7 cm, of which 4.5 cm is a static template offset and the rest pose-dependent drape, not inertia (velocity regression explains < 0.2 % of the variance) |
| placement / calibration | one constant (scale, shift, rotation) per video fitted on half the frames and scored on the other half: +0.005; even an oracle per-frame 2D alignment only reaches 0.767 (`tools/evaluate/placement_gap.py`) |
| garment size scaling | 7 variants (no resize, shrink only, damped, frozen, height only, widen to the wearer, current): none passes +0.005 and none beats the current one, which is also the one that keeps the garment widening with a wider body; widening to the wearer costs 0.03 (`docs/baseline/size-ablation.json`) |
| regressed shape jitter | freezing the garment's scale from the first 30 frames changes nothing (0.7278) |
| splat footprint / opacity / floaters / alpha threshold | ~20 % of the extra pixels are fringe, ~80 % solid; every setting held out: -0.00025 (`tools/evaluate/silhouette_study.py`) |
| distance to the authors' simulation as a guide | a relaxed rest shape learned from it is 17 % closer on unseen motions and 0.095 worse in IoU: the simulation is not the real garment |

What is left is the garment's SHAPE: after the best alignment our silhouette
still draws too much (extra pixels 0.178 of the real shirt area, missed 0.107),
most on the loose outer shirts and the sleeves (00152_outer: torso 14 % too
wide, hem 4.5 % of its height too low). The next leads are the garment geometry
itself (the skinning of loose hems and sleeves from the scan pose, the splat
footprint and opacity at the silhouette threshold), not the cloth, the size or
the camera. Without registered multi-view data (4D-DRESS, archived) a fit to
the real shirt cannot be validated further, so 0.80 stays open.

## Does it fit? The audit (8 Oct)

`tools/evaluate/fit_audit.py`, sheets in `docs/figures/fit-audit-*.jpg`, numbers in
`docs/baseline/fit-audit.json` (details in `LOG.md`). Body **good** (24.6 mm; the
profile camera is the weak one, 28-38 mm); motion **good** (no dropouts, no lag,
as steady as the real shirt: centroid jitter 0.66x, area jitter 0.98x of the real
shirt's); edges **good**; garment against the body **acceptable** (1.0 % of the
vertices inside the body, 10.6 mm mean penetration, equal to the reference's);
silhouette **acceptable** for closed garments (IoU 0.77-0.88) and **weak** for
loose and open outer shirts (0.50-0.68); texture **acceptable** on plain garments
and **weak** on 00136_outer (dark by 19-21 levels) and on plaid and knit patterns, which do not register pixel for pixel. What looks worst on the sheets is the
erased clothing (grey patches, a flat pale skin fill on the arms), not the garment.

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
3a. **A recorded webcam session** (the arm reaching toward the camera is fixed, 9 Oct, ninth part of `LOG.md`: `BodyFitter.ARM_REFINE` on at a gate of 0.4 torso lengths, IoU unchanged on five release videos) (`./live-tryon record`, 9 Oct): blind frames limited to 4 in a row and the collar below the chin blended into the neck are fixed (`LOG.md`); a sleeve cuff drawn as a ring in front of a hand reaching to the camera, and the pale background patches either side of the neck (the fitted body is narrower than the real neck at a low camera angle), are not; the cuff ring (the arm's depth toward the camera, NLF) and the pale shards in the neckline (the wearer's own collar stripes, labelled face and restored over the garment by `keep_in_front`) are both fixed (`LOG.md`, 9 Oct, ninth and tenth parts).
3. **Hem and the closed jacket** are done as far as measured (9 Oct, `LOG.md`): the hem hangs low because the sleeves carry the real shirt up at the armhole, fixed by a sleeve-anchored long-range attachment plus a gentle narrowing (loose-garment mean IoU +0.015); the closed-worn 00136_outer is a per-garment flag (`assets/closed_front/`, +0.038). What is left: the lower front of loose open shirts hangs open about 30 % of the width in the real video and is nearly closed in ours (the 'torso 13-16 % too wide' of earlier entries was a hole in the real mask, not an excess of width: `LOG.md` 9 Oct), and fully erased arms that still read as sticks (a learned limb inpainting, not shading).
4. **The erased clothing** (the arm study of 9 Oct says the arm fill is seen on 7-20 % of its area on most garments, 48 % on a T-shirt, and that no shading, texture or general inpainter makes it read as an arm: a person-aware learned inpainter is the only real fix, not downloaded) (the weakest thing on the fit-audit sheets after the 8 Oct fill: arm tubes read as sausages, a flat torso skin, a pale halo on 00122_outer) and **loose shirts** (the long hem and wide torso of 00152_outer; closed-worn jackets against open templates). The open-front 'floating panel' was a sleeve on a wrong scan pose, fixed 8 Oct (`LOG.md`).
5. **Render and cloth on the critical path** (profile after the 8 Oct work: cloth + posed mesh ~54 ms of which `push_out_of_body` 22; `render_splats_gl` ~59 ms of which `_instances` 26; MediaPipe for the next frame could run beside the render): each needs to be bit-identical and >= 3 ms.
6. **Shading refresh rate**: the student updates every 0.7 s on a worker
   thread; test 0.35 s with the GPU free and keep it only if fps holds.
7. **AnyWear / Decart comparison**: possible only with the user's own API
   key and their approval to upload the release videos.
8. **4D-DRESS**: its access request was archived (the data cannot be
   redistributed, so no copies are used); see [`DATASETS.md`](DATASETS.md).

## Tried and dropped

Switches that measured no gain were removed from the code on 10 Oct (an unpublished local tag has them): the arm shading model, skin texture transfer and body-outline tube of the fill; importance subsampling; the relaxed rest; the front opening; ride-up; neighbour coupling, wrap, target filter, velocity inheritance and the vertical attachment of the cloth; the fitter's skin term; and the studies that only measured them.

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
| the project's own U-Net clothing parser | SCHP measured better on every source; removed 8 Oct (an unpublished local tag) |
| position-based cloth solver, learned pose correctives, 3D re-skin display | no net gain; removed 1 Oct (an unpublished local tag) |
| SAM 2 for the clothing mask, RTMPose, SMPLer-X, Multi-HMR 672_B | compared, not adopted (numbers in `LOG.md`) |
| numba `parallel=True` kernel on the mask thread | aborts the process (numba's workqueue layer is not thread-safe) and serially is no faster than numpy |
| diffusion try-on (ViViD, CatVTON, FASHN, VTON360, FitMe360) | tens of seconds to 30 minutes a frame on the M1 ([`REFERENCE_RUNS.md`](REFERENCE_RUNS.md)) |
| neighbour coupling of the cloth, velocity inheritance, One Euro filter on the cloth target, long-range-attachment clamp, low-dimensional pose residual | each measured against the authors' simulation: no distance gain (the cloth is not what separates us from the registration); the One Euro filter lags (IoU 0.7277 vs 0.7282) |
| garment size variants, per-video calibration of the placement | see "Where the IoU gap is" |
| a relaxed rest shape learned from the authors' simulation (`GarmentRenderer.RELAXED_REST`, removed 10 Oct) | 17 % closer to the simulation on unseen motions, but IoU against the real shirt 0.7283 -> 0.6328 (0.7198 even at a quarter of the offset): the simulation's garment is shorter and tighter than the real one |
| splat footprint, opacity sharpening, floater removal, alpha threshold | the over-drawing is ~80 % solid geometry; held-out gain -0.00025 (`docs/baseline/silhouette-study.json`) |
| ContourCraft cloth | 2.2-2.6 s a step on the M1; driven by our tracked bodies it billows (IoU 0.476 against 0.600) |
