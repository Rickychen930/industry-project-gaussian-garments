# Using the live try-on

This guide covers the current system (branch `main`): how to
start it, what each option does, how it works inside, how fast and how
accurate it is (measured, with where each number comes from), its
limits, and how to fix common problems. Branches and comparisons are in
[`BRANCHES.md`](BRANCHES.md); what is planned is in [`ROADMAP.md`](ROADMAP.md)
and the dated record of every measurement in [`LOG.md`](LOG.md).

## 1. Quick start

```bash
./live-tryon
```

It lists the garments, asks for a number (Enter picks the striped
sweater), and opens the webcam window:

```
  Available garments  (* closed = cleanest result)
   9  00170_upper  sweater vest + white shirt  * closed
  12  00176_upper  white T-shirt  * closed
  13  00190_upper  striped knit sweater  * closed
  ...
  In the video window: n = next garment, p = previous, q = quit
```

The first start takes about 10-20 s while the models load. Until the first
garment is ready the window shows the camera with "starting ...".

**Keys in the window:** `n` next garment, `p` previous garment, `q` or
`Esc` quit.

**For the best result:**
- Stand about 2-3 m from the camera so you are visible at least down to
  the knees. The body fit needs the shoulders and preferably the hips.
- Choose a garment marked `* closed` (9, 12, 13). An open-front jacket
  shows the erased shirt underneath through its opening.
- Move at a normal pace. Fast moves make the garment trail (see 4).

## 2. Commands

| command | what it does |
|---|---|
| `./live-tryon` | Menu, then the live try-on from the webcam. |
| `./live-tryon --smooth` | Camera at full rate; the garment is carried with the torso between pipeline frames (fluid, but the garment trails on moves). |
| `./live-tryon --delayed` | `--smooth` shown ~1 s late, with the garment from the nearest pipeline frame (fluid and closer, but delayed). |
| `./live-tryon --harmonize` | Also recolour the garment to the room's light (Harmonizer; needs its clone, see 7). |
| `./live-tryon --source video.mp4` | Run on a recorded video instead of the camera. |
| `./live-tryon run [N or name] [flags]` | The same without the menu, e.g. `./live-tryon run 13`. |
| `./live-tryon bench [N\|name] [frames] [source]` | Time the pipeline on the camera (source 0) or a video; prints the median ms of each stage, saves the output and the camera frames as read (`captures/bench-<time>[-input].mp4`). Run it in your own terminal (the camera permission) with nothing else busy. |
| `./live-tryon record [seconds] [basic\|extended]` | Film yourself while it says which pose to take (basic: 12 poses, 5 s each; extended: 31 poses, `record 8 extended` is ~4 min); replay with `./live-tryon run NAME --source FILE`. |
| `python -m tools.evaluate.session_benchmark FILE --garment NAME --out docs/baseline/session-X.json` | One recording through the live pipeline, with numbers (dressed share, reprojection, shoulder ratio, uncovered clothing, stage times): the same recording before and after a change says whether it helped. |
| `./live-tryon garments` | List the garments by number and open `docs/figures/garments.jpg` (a picture of each). |
| `./live-tryon compare <branch> [video] [frames]` | Measure any branch or tag on a release video (see `BRANCHES.md`). |
| `./live-tryon test` | Run the test suite (`unittest`). |
| `./live-tryon doctor` | Check what this machine has installed. |
| `./live-tryon pipeline [flags]` | `tryon.app` with every flag spelled out (for development). |

### Flags of the live pipeline

`./live-tryon run` turns on the recommended set for you: `--remove-clothing
--orientation-regressor models/pose_regressor.pt --gpu-render
--appearance --async-appearance --student-shading`, with the synchronous
display (each shown frame carries the garment drawn for it).

| flag | effect |
|---|---|
| `--garment DIR` | Garment checkpoint folder (`~/tryon-data/garments/<name>`). |
| `--remove-clothing` | Erase the wearer's own upper clothing before drawing the garment. |
| `--smooth-display` | Show every camera frame; the pipeline runs behind it (see 3). |
| `--gpu-render` | Composite the splats with OpenGL instead of the CPU loop. |
| `--rigid` | No cloth motion: the garment follows the body by skinning only (for comparison). |
| `--delayed-sync` | With `--smooth-display`: show the video about one pipeline period (~1 s) late and take the garment from the pipeline frame nearest in time, before or after (default in the launchers). |
| `--dump-alpha out.npz` | With `--smooth-display`: record the garment shown on every frame (for `tools/evaluate/live_display.py`). |
| `--appearance --async-appearance` | The garment's own shading network (folds, shadows), updated on a worker thread. |
| `--harmonize` | Match the garment's colours to the scene. |
| `--classic-cloth` | The first cloth model (`LightCloth`) instead of the default `SmoothCloth` (same distance to the authors' simulation and the same IoU, about a third of the jitter). |
| `--garment-subsample F` | Draw only a fraction F of the garment's Gaussians. Measured: F = 0.75 (a uniform random subset) loses < 0.003 IoU; the default stays 1.0 (the speed gain was not cleanly measured). |
| `--max-width N` | Process frames at most N pixels wide (default 960; `0` = as captured). `720` is a speed mode: removal and compositing ~17 ms a frame cheaper, garment accuracy unchanged (0.7262 vs 0.7253 IoU on three release videos), a smaller picture. |
| `--source 0 / 1 / file` | Camera index or a video file. |
| `--headless --output-video out.mp4 --max-frames N` | Run without a window and save the result. |
| `--profile` | Print the median time of each pipeline stage at the end (not with `--smooth-display`). |
| `--parser fashn --fill lama --appearance` (without `--async-appearance`) | Quality mode for recorded videos: ~4-5 s a frame (FASHN parser, LaMa fill, the full 55M appearance network). |

## 3. How it works

```
 camera ──► DISPLAY PROCESS (camera rate) ─────────────────────────► window
              │  MediaPipe lite + person mask (every frame)        ▲
              │  newest garment layer, moved with the torso ───────┘
              │  (this frame's face and hands stay in front)
              │
              └─ newest frame ─► PIPELINE PROCESS (~0.2-0.25 s a frame)
                                  1. MediaPipe landmarks
                                  2. SMPL-X body from NLF-S (image regressor)
                                  3. garment skinned to the body + cloth springs
                                  4. Gaussian splats placed on the garment,
                                     shaded by the shading student
                                     (worker thread, every 0.7 s)
                                  5. splats composited on the GPU (OpenGL)
                                  6. wearer's clothing erased (SCHP parser)
                                     and filled (skin / learned background)
                                  7. garment composited, face and hair kept
                                  ──► finished layer back to the display
```

- **The body** comes from NLF-S, an image-to-SMPL-X regressor, run on the
  same frame the garment is drawn for (it starts on that frame while the
  previous one is still being drawn). A pose applied to a later frame
  was measured worse, so the 3D pose is never reused across frames; only
  the finished 2D layer is.
- **The garment** is a Gaussian Garments checkpoint: about 93k 3D Gaussians
  bound to an ~8k-vertex template mesh. The mesh is skinned by the body's
  joints; damped springs give it some swing and a collision step keeps it
  out of the body (`tryon/garment/cloth.py`).
- **The shading** comes from a 1.55M-parameter student distilled from the
  checkpoint's own StyleUNet (55M parameters, which needs an AO bake and
  ~0.75 s a frame). The student stores the teacher's average output per
  Gaussian and predicts the pose-dependent part (folds, shadows) from
  cheap maps of the posed garment (`tryon/garment/shading_student.py`).
- **Clothing removal** parses the frame with SCHP, erases the upper
  clothing near the body (not shirts hanging behind you), and fills the
  hole with skin colour or the room learned from earlier frames.
- **The smooth display** runs in its own process, so Python's interpreter
  lock cannot stall it. Between pipeline results it moves the newest layer
  with a clamped torso similarity (shoulders and hips). That warp is
  limited to scale 0.85-1.18 and 20 degrees of rotation, so a side-on body
  cannot blow it up.

Main code (a walk-through is in docs/ARCHITECTURE.md): `tryon/app/pipeline.py`
(loop), `tryon/app/smooth_display.py` (fluid display), `tryon/body/fitter.py`
and `tryon/body/regressor.py` (body), `tryon/garment/skinning.py` and
`tryon/garment/renderer.py` (garment), `tryon/garment/cloth.py` (cloth),
`tryon/garment/appearance.py` (shading), `tryon/rendering/splat_cpu.py` and `tryon/rendering/splat_gl.py`
(rasterisers), `tryon/removal/remover.py` and `tryon/removal/parsing_schp.py`
(removal), `tryon/app/layer_warp.py` (display warp).

## 4. Speed and accuracy (Apple M1, measured)

| | value | source |
|---|---|---|
| Display frame rate (smooth mode) | 25 fps | `smooth_display`, dancer clip |
| Pipeline time per frame | 0.085-0.10 s of work, HUD 7.3-7.5 fps on a video file (decode adds ~40 ms), sync mode; 0.21-0.23 s before 8 Oct | `--profile`; the pipeline's HUD |
| Stages (960-wide frame, medians) | MediaPipe 13, fit 8, render 56-66, erase beside the render (a worker thread; 0-23 ms of waiting), composite 5 ms; the body regressor and the clothing mask of the NEXT frame run during this frame | `--profile` |
| Garment IoU vs the real shirt, per frame (`--sync`) | 0.728 on 5 release videos with the cloth stepped at the clip's own dt (0.722 without cloth; authors' 4-camera registration: 0.827) | `tools/evaluate/vs_registration.py` |
| Smooth mode, share of the per-frame accuracy | 77 % at the dancers' speed, 86-93 % at a quarter to an eighth of it | `tools/evaluate/async_display.py` (simulated) |
| Smooth mode run in real time, garment IoU on every shown frame | 0.575 live (torso warp), **0.625 delayed ~1 s** (`--delayed-sync`) mean of 00152 / 00170 / 00134 | `tools/evaluate/live_display.py` |
| Shading | the student against the 55M teacher, every garment | `tools/evaluate/shading_students.py`, `docs/baseline/shading-students.json` |

**Sync (default) vs smooth vs delayed.** The default shows only finished
pipeline frames: the garment always matches the frame it is on (nothing
floats), at the pipeline's rate (7.3-7.5 fps on the M1, video file). `--smooth` and
`--delayed` trade that for a fluid video. With `--delayed` the video is shown about one
pipeline period (~1 s) late: every shown frame then has a pipeline result
both before and after it, and the garment comes from the nearer one, so it
is never carried more than half a period (garment IoU 0.625 vs 0.575 live,
measured end to end). Blending the two results instead scored 0.638 but
showed both garments half-transparent. With `--smooth` the video is live
and the garment trails on moves (it looks like it floats). With `--sync` the garment fits the shown frame
exactly, but that frame is ~0.2 s old and stays frozen until the next
one. With the smooth display the video is live and fluid, and the
garment trails a little on fast moves.

## 5. Limitations

- **Garments.** Only the 15 captured Gaussian Garments checkpoints. Holes
  that the studio capture never saw are in the checkpoint itself
  (`docs/figures/garments.jpg`: e.g. 00134_upper, 00148_upper). The lower
  garments (6, 8, 10) are drawn over your trousers without erasing them.
- **Fast motion.** In smooth mode the garment trails; the real jacket can
  show at the edges for a moment.
- **Extreme poses** (arms overhead, deep bends) stretch the cloth at the
  armpits and sleeves.
- **Lighting.** Garments carry studio light; `--harmonize` helps in
  warm or dim rooms.
- **Photographic realism** at the level of diffusion try-on (FASHN,
  CatVTON) is not possible live on this machine: 78 s to 30 min a frame.
  See `docs/REFERENCE_RUNS.md`.

## 6. Troubleshooting

| symptom | fix |
|---|---|
| `no such file or directory: .venv/bin/python` | Run from the repository folder; create the venv with `./setup.sh`. |
| Window never opens / black | Allow camera access for Terminal (System Settings > Privacy > Camera) and start again. |
| Wrong camera (e.g. the phone via Continuity) | `./live-tryon --source 1`. |
| Garment misplaced, HUD `reproj` in the hundreds | Step back so shoulders and hips are visible; face the camera for a second so the body can lock on. |
| Grey patches on the background | Fixed: removal is limited to near the body. Report it if it comes back. |
| Very slow | Check nothing else heavy is running; `--max-width 720` trades picture size for speed (~17 ms a frame). |

## 7. What must be installed

| what | where | notes |
|---|---|---|
| Python venv | `.venv/` | `./setup.sh` (`requirements.txt`) |
| SMPL-X body models | `~/tryon-data/body_models/smplx/` | register at smpl-x.is.tue.mpg.de |
| Garment checkpoints | `~/tryon-data/garments/<name>/` | Gaussian Garments models of 4D-DRESS subjects |
| SCHP / LaMa weights | `models/pretrained/` | `tools/data/download_pretrained.py` |
| NLF-S body regressor | `models/pretrained/nlf/` + its own venv | see README; non-commercial research use |
| Orientation regressor | `models/pose_regressor.pt` | `tools/train/orientation.py` |
| Harmonizer (optional) | `~/tryon-data/references/Harmonizer` | only for `--harmonize` |

External systems run for comparison (VTON360, FASHN, CatVTON, RTV, SAM2)
keep their clones and venvs under `~/Downloads/tryon_refs/`, outside the
data root (a virtualenv cannot be moved); see `docs/REFERENCE_RUNS.md`.
`TRYON_DATA` moves the data root.

### GPU setup

**Apple Silicon (M1/M2/M3; what this project is built and measured on).**
There is no separate GPU install: `./setup.sh` gives every piece its
accelerator, and `./live-tryon doctor` shows which one each uses.

| stage | runs on | how it is selected | measured (M1) |
|---|---|---|---|
| MediaPipe pose | CPU (XNNPACK) | fixed | ~18 ms |
| NLF-S body | backbone on the GPU (MPS), the rest on the CPU, fp32 | its own venv (`models/pretrained/nlf/venv`); the whole model on MPS was ~0.9 s | 60-90 ms, overlapped with the previous frame |
| SMPL-X, skinning, cloth | CPU (numba) | fixed; SMPL-X on MPS was 38.8 vs 6 ms | - |
| splat compositing | GPU: OpenGL 4.1 on Metal (moderngl) | `--gpu-render` (on in `./live-tryon run`) | 143 -> 37 ms |
| shading student | GPU: MPS, fp16 | automatic when `torch.backends.mps.is_available()` | ~0.1 s every 0.7 s, on a worker thread (the 55M teacher, without a student: ~0.75 s) |
| SCHP clothing parser | Neural Engine / GPU: ONNX Runtime CoreML | automatic when the CoreML provider is present (the macOS `onnxruntime` wheel has it) | 304 ms vs 2.1 s CPU |
| Multi-HMR (fallback body) | GPU: MPS at 518 px | automatic; `PYTORCH_ENABLE_MPS_FALLBACK=1` is set by `tryon/app/pipeline.py` | ~0.2-0.3 s |

Steps:

```bash
./setup.sh                      # venv + requirements (torch 2.1 with MPS, onnxruntime with CoreML, moderngl)
./live-tryon doctor             # "Compute": MPS available, SCHP on CoreML, OpenGL Apple M1 4.1 Metal, NLF installed
# NLF-S (the body): its own venv, see README "body regressor"
uv venv models/pretrained/nlf/venv --python 3.12
VIRTUAL_ENV=models/pretrained/nlf/venv uv pip install torch==2.14.0 torchvision==0.29.0 "numpy<2" scipy "opencv-python-headless<4.11"
./live-tryon              # the first run caches NLF's fp32 graph (~20 s) and compiles numba (~10 s)
```

Close other GPU-heavy apps (browsers with video, training jobs): the
appearance net, SCHP and the splats share the one GPU, and a training run
on MPS was measured to double every stage.

**NVIDIA (Linux / Windows) - not tested with this project.** What changes
and what is already in the code:

1. torch with CUDA in the project venv, same version as the pin:
   `pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121`
   (replace the `torch` line of `requirements.txt`).
2. `pip uninstall onnxruntime && pip install onnxruntime-gpu==1.23.2`:
   `tryon/removal/parsing_schp.py` picks `CUDAExecutionProvider` when present.
3. NLF: install a CUDA torch 2.14 in its venv
   (`uv pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126`);
   `tryon/body/nlf_worker.py` then loads the released fp16 model on the GPU
   as its authors run it. With the GPU, its own person detector is cheap,
   so a missing MediaPipe box costs little.
4. `--gpu-render` needs an OpenGL 3.3 context: a desktop session, or EGL on
   a headless server (moderngl `create_standalone_context(backend="egl")`
   is not wired in yet - drop `--gpu-render` there; the numba CPU loop
   gives identical output).
5. Not yet CUDA-aware: the appearance net (`tryon/garment/appearance.py` picks MPS
   or CPU) and Multi-HMR in `tryon/app/pipeline.py` - on NVIDIA they run on the CPU until
   `"cuda"` is added to their device choice.
6. `./live-tryon doctor` reports CUDA and the SCHP provider.

The repository layout and the documentation map are in the [README](../README.md).
