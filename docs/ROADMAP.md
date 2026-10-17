# Roadmap

Where the project stands, what is next, and what was tried and dropped.
Every number is in docs/baseline; how to run
things is [`USAGE.md`](USAGE.md).

**Working rules.** One part of the system at a time. Every change is
measured against ground truth (the real garment, the authors' fit, annotated views,
the teacher network) before it is kept; if it does not measure better it is
removed and recorded below. Pretrained models are used wherever they measure
better than the alternative.

## State at the final report (17 Oct 2026)

The final report (32040 Industry Project) grades the six objectives and the six registered learning
outcomes as follows. The thresholds are the report's.

| objective (threshold) | result | verdict |
|---|---|---|
| O1 body placed accurately, garment drawn in almost every frame (<= 30 mm on the 16 development views and on 4,208 unselected views; >= 95 % in each set and each recording) | body 24.6 mm on the 16 views, 42.5 mm on the 4,208 views of two 4D-DRESS subjects; garment in 100 % of frames on 15 release videos, 99.5 % on 14 webcam scenarios, 98.6 % over six sessions, but 90.1 % on session 1 and 89.5 % on the 17 Oct recording (100 % on frames with a person in view) | partly met |
| O2 garment turns with the wearer through 360 degrees (synthetic: under 5 cm at every step; real: no threshold) | `tests/test_goals.py` full turn passes for the skinning; two real recordings reach side-on, garment against wearer width 0.98 head-on and 1.11 side-on (`tools/evaluate/turn_profile.py`, 970 frames); no ground truth | partly met |
| O3 the wearer's own top hidden (median uncovered share <= 0.10 in each session, set after measuring) | 3 of 6 sessions at or under 0.10 (medians 0.07-0.29); erased arms look like smooth tubes | partly met |
| O4 garment covers the real garment's pixels (IoU >= 0.80) | 0.730 on five release videos (authors' fit 0.827) | missed |
| O5 keeps up with a live wearer ((a) median work <= 0.5 s in each run; (b) display >= 20 fps) | (a) 106-411 ms in 16 of 19 runs; three above 0.5 s, all on a loaded machine: 670 and 602 ms (two 10 Oct replays before that day's fixes) and 1,034 ms (the turn replay); (b) default display 7.3-7.5 fps on a video file; smooth mode's 25 fps is the clip's rate, never logged on the webcam | partly met (R4 partly met, R5 missed) |
| O6 three current systems scored on the same frame with five measures | FASHN VTON 1.5 and CatVTON scored on all five; VTON 360 run but judged by eye ([`REFERENCE_RUNS.md`](REFERENCE_RUNS.md)) | partly met |

| learning outcome | verdict |
|---|---|
| LO1 camera motion and 3D from 2D frames | partly met (camera motion from SIFT matches failed; 3D points need the capture's calibration) |
| LO2 complete classical pipeline, frames to surface | met with limits (four calibrated studio cameras, no smartphone video) |
| LO3 limits of classical algorithms on garments | met with limits (one studio capture, nine takes) |
| LO4 experiments and quantitative evaluation | met |
| LO5 compare alternatives, justify with evidence | met |
| LO6 method, documentation, collaboration, validation, presentation | met with limits (one member wrote all the code) |

The registered classical route was run on the four calibrated 4D-DRESS cameras (17 Oct): visual hull of the
upper garment F-score 0.50 at 2 cm, camera motion from SIFT matches failed, COLMAP registered all four cameras
in 2 of 30 frames. Tests: 512 on 17 Oct (503 passed, 9 skipped), 513 after the intrinsics fix of the
classical tool. Evaluation: 28 tools in `tools/evaluate/`.

## Current state (8 Oct 2026) (later numbers in the section above)

| part | now | measured by |
|---|---|---|
| body | NLF-S (backbone on the GPU), refined toward MediaPipe's keypoints; 24.6 mm PA-MPJPE on the 16 development views (keypoint fit alone: 87-94 mm) | `docs/baseline/body-regressors.json` |
| garment | skinned from the scan pose; `LightCloth` springs, body collision, strain limit | `tools/evaluate/cloth_trajectory.py` |
| shading | per-garment student (1.55M parameters) with a stored per-Gaussian mean, trained on the garment's own motion and every other simulation the authors released: 29.1 dB on unseen motions on 7 Oct (one-motion student: 26.9), 31.1 dB (mean of 15 students) after the rounds of 10-12 Oct | `tools/evaluate/shading_students.py` |
| removal | SCHP (ATR + LIP) on CoreML, learned background plate, Telea fill | `tools/evaluate/background_fill.py` |
| accuracy | garment IoU vs the real garment **0.730** (5 release videos; the authors' fit, four cameras: 0.827) | `tools/evaluate/vs_registration.py` |
| speed (M1, default display) | 0.1-0.2 s of work a frame (0.085-0.10 s on a quiet machine; HUD 7.3-7.5 fps on a video file, whose decode adds ~40 ms a frame; 129 ms of work a frame on the live camera on 17 Oct, on a busy machine); 0.21-0.23 s (4.4-4.8 fps) before 8 Oct; the body regressor works on the next frame while this one is drawn; smooth mode shows a video file at its clip rate (~25 fps), and its rate on the webcam was never logged. The 8 Oct removal of repeated work cut a loaded-machine frame from 300-388 to 187-217 ms (clean-machine figure still to be taken) | `python -m tryon.app --profile` |

## Targets (at the final report, 17 Oct)

| target | goal | now | |
|---|---|---|---|
| garment on the real garment (IoU, 5 release videos) | >= 0.80 | 0.730 | missed (O4) |
| body accuracy (PA-MPJPE) | <= 30 mm on the 16 development views and on 4,208 unselected views | 24.6 mm and 42.5 mm | met on the 16 views only (O1 partly met) |
| display frame rate | >= 20 fps | 7.3-7.5 fps in the default display on a video file; smooth mode's ~25 fps is the clip's rate and was never logged on the webcam | missed (R5; O5 partly met) |
| pipeline time a frame (M1) | median <= 0.5 s in each run | 106-411 ms in 16 of 19 timed runs; 670, 602 and 1,034 ms in three runs on a loaded machine | partly met (R4, O5) |
| garment present | >= 95 % of frames in each set and each recording | 100 % on 15 release videos, 99.5 % on 14 webcam scenarios, 98.6 % over six sessions; 90.1 % and 89.5 % in two recordings | not in every recording (O1 partly met) |
| smooth mode vs exact (webcam-speed motion) | >= 90 % of the exact IoU | 93 % at 1/8 of dancer speed | done |

## Where the IoU gap is (measured 8 Oct)

Garment IoU against the real garment is 0.730 over the 5 release videos (0.728 in the fit audit's own run); the
authors' fit reaches 0.827. Four measurements say where the 0.10 is not:

| not the cause | evidence |
|---|---|
| cloth stepping | with the cloth stepped at the clip's own dt the IoU is 0.7281, without it 0.7218, with `SmoothCloth` 0.7283; replaying 5 garments, `LightCloth` is 5.467 cm from the authors' simulation and plain skinning 5.454 (`tools/evaluate/cloth_smoothness.py`). The authors' cloth differs from skinning by 7.7 cm, of which 4.5 cm is a static template offset and the rest pose-dependent drape, not inertia (velocity regression explains < 0.2 % of the variance) |
| placement / calibration | one constant (scale, shift, rotation) per video fitted on half the frames and scored on the other half: +0.005; even an oracle per-frame 2D alignment only reaches 0.767 (`tools/evaluate/placement_gap.py`) |
| garment size scaling | 7 variants (no resize, shrink only, damped, frozen, height only, widen to the wearer, current): none passes +0.005 and none beats the current one, which is also the one that keeps the garment widening with a wider body; widening to the wearer costs 0.03 (`docs/baseline/size-ablation.json`) |
| regressed shape jitter | freezing the garment's scale from the first 30 frames changes nothing (0.7278) |
| splat footprint / opacity / floaters / alpha threshold | ~20 % of the extra pixels are fringe, ~80 % solid; every setting held out: -0.00025 (`tools/evaluate/silhouette_study.py`) |
| distance to the authors' simulation as a guide | a relaxed rest shape learned from it is 17 % closer on unseen motions and 0.095 worse in IoU: the simulation is not the real garment |

What is left is the garment's SHAPE: after the best alignment our silhouette
still draws too much (extra pixels 0.178 of the real garment's area, missed 0.107),
most on the loose outer shirts and the sleeves (00152_outer: torso 14 % too
wide, hem 4.5 % of its height too low). The next leads are the garment geometry
itself (the skinning of loose hems and sleeves from the scan pose, the splat
footprint and opacity at the silhouette threshold), not the cloth, the size or
the camera. Without registered multi-view data (4D-DRESS arrived too late to test a fix) a fit to
the real garment cannot be validated further, and 0.80 is missed (O4).

## Does it fit? The audit (8 Oct)

`tools/evaluate/fit_audit.py`, sheets in `docs/figures/fit-audit-*.jpg`, numbers in
`docs/baseline/fit-audit.json`. Body **good** (24.6 mm; the
profile camera is the weak one, 28-38 mm); motion **good** (no dropouts, no lag,
as steady as the real garment: centroid jitter 0.66x, area jitter 0.98x of the real
garment's); edges **good**; garment against the body **acceptable** (1.0 % of the
vertices inside the body, 10.6 mm mean penetration, equal to the reference's);
silhouette **acceptable** for closed garments (IoU 0.74-0.82) and **weak** for
loose and open outer shirts (0.50-0.68); texture **acceptable** on plain garments
and **weak** on 00136_outer (dark by 19-21 levels) and on plaid and knit patterns, which do not register pixel for pixel. What looks worst on the sheets is the
erased clothing (grey patches, a flat pale skin fill on the arms), not the garment.

## Done

Body tracking (MediaPipe -> SMPL-X, then NLF-S from the image) · garment
follows the body (skinning from the scan pose) · Gaussian-splat rendering
(CPU, then GPU: 143 -> 37 ms) · clothing removal (SCHP, learned background
fill) · cloth (springs, collision, strain limit) · 360-degree turns (facing
from MediaPipe's 3D landmarks and the regressor) · live system with a
separate display process (smooth mode) · robustness to desk framing, close-ups, dim light,
blur, objects in front, a person leaving · pipeline under 0.5 s a frame in 16 of 19 timed runs ·
pose-dependent shading from a distilled student · code audit and refactor.

## Next

1. **Garment shape closer to the real garment** (IoU 0.730 -> 0.80):
   the gap is the garment's shape, not placement, cloth or size ("Where the
   IoU gap is" above; `tools/evaluate/reference_frame.py` splits the loss level by level).
2. **Scene lighting on the garment**: light estimated from the face; the
   garments carry studio light (`--harmonize` is the stop-gap).
3a. **A recorded webcam session** (the arm reaching toward the camera is fixed, 9 Oct: `BodyFitter.ARM_REFINE` on at a gate of 0.4 torso lengths, IoU unchanged on five release videos) (`./live-tryon record`, 9 Oct): blind frames limited to 4 in a row and the collar below the chin blended into the neck are fixed; a sleeve cuff drawn as a ring in front of a hand reaching to the camera, and the pale background patches either side of the neck (the fitted body is narrower than the real neck at a low camera angle), are not; the cuff ring (the arm's depth toward the camera, NLF) and the pale shards in the neckline (the wearer's own collar stripes, labelled face and restored over the garment by `keep_in_front`) are both fixed.
3. **Hem and the closed jacket** are done as far as measured (9 Oct): the hem hangs low because the sleeves carry the real shirt up at the armhole, fixed by a sleeve-anchored long-range attachment plus a gentle narrowing (loose-garment mean IoU +0.015); the closed-worn 00136_outer is a per-garment flag (`assets/closed_front/`, +0.038). What is left: the lower front of loose open shirts hangs open about 30 % of the width in the real video and is nearly closed in ours (the 'torso 13-16 % too wide' of earlier entries was a hole in the real mask, not an excess of width), and fully erased arms that still read as sticks (a learned limb inpainting, not shading).
4. **The erased clothing** (the arm study of 9 Oct says the arm fill is seen on 7-20 % of its area on most garments, 48 % on a T-shirt, and that no shading, texture or general inpainter makes it read as an arm: a person-aware learned inpainter is the only real fix, not downloaded) (the weakest thing on the fit-audit sheets after the 8 Oct fill: arm tubes read as sausages, a flat torso skin, a pale halo on 00122_outer) and **loose shirts** (the long hem and wide torso of 00152_outer; closed-worn jackets against open templates). The open-front 'floating panel' was a sleeve on a wrong scan pose, fixed 8 Oct.
5. **Render and cloth on the critical path** (profile after the 8 Oct work: cloth + posed mesh ~54 ms of which `push_out_of_body` 22; `render_splats_gl` ~59 ms of which `_instances` 26; MediaPipe for the next frame could run beside the render): each needs to be bit-identical and >= 3 ms.
6. **Shading refresh rate**: the student updates every 0.7 s on a worker
   thread; test 0.35 s with the GPU free and keep it only if fps holds.
7. **AnyWear / Decart comparison**: possible only with a paid API
   key and their approval to upload the release videos.
8. **4D-DRESS**: access was granted late (manual approval; the data cannot be
   redistributed, so no copies are in the repository) and used only for the body test; see [`DATASETS.md`](DATASETS.md).

## Tried and dropped

Switches that measured no gain were removed from the code on 10 Oct: the arm shading model, skin texture transfer and body-outline tube of the fill; importance subsampling; the relaxed rest; the front opening; ride-up; neighbour coupling, wrap, target filter, velocity inheritance and the vertical attachment of the cloth; the fitter's skin term; and the studies that only measured them.

Each row was measured..

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
| the project's own U-Net clothing parser | SCHP measured better on every source; removed 8 Oct |
| position-based cloth solver, learned pose correctives, 3D re-skin display | no net gain; removed 1 Oct |
| SAM 2 for the clothing mask, RTMPose, SMPLer-X, Multi-HMR 672_B | compared, not adopted |
| numba `parallel=True` kernel on the mask thread | aborts the process (numba's workqueue layer is not thread-safe) and serially is no faster than numpy |
| diffusion try-on (ViViD, CatVTON, FASHN VTON 1.5, VTON 360, FitMe360) | tens of seconds to 30 minutes a frame on the M1 ([`REFERENCE_RUNS.md`](REFERENCE_RUNS.md)) |
| neighbour coupling of the cloth, velocity inheritance, One Euro filter on the cloth target, long-range-attachment clamp, low-dimensional pose residual | each measured against the authors' simulation: no distance gain (the cloth is not what separates us from the authors' fit); the One Euro filter lags (IoU 0.7277 vs 0.7282) |
| garment size variants, per-video calibration of the placement | see "Where the IoU gap is" |
| a relaxed rest shape learned from the authors' simulation (`GarmentRenderer.RELAXED_REST`, removed 10 Oct) | 17 % closer to the simulation on unseen motions, but IoU against the real garment 0.7283 -> 0.6328 (0.7198 even at a quarter of the offset): the simulation's garment is shorter and tighter than the real one |
| splat footprint, opacity sharpening, floater removal, alpha threshold | the over-drawing is ~80 % solid geometry; held-out gain -0.00025 (`docs/baseline/silhouette-study.json`) |
| ContourCraft cloth | 2.2-2.6 s a step on the M1; driven by our tracked bodies it billows (IoU 0.476 against 0.600) |
