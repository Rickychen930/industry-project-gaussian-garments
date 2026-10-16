# Live 3D virtual try-on

Put a captured 3D garment on a person in a webcam: it follows their motion,
turns with them, and replaces what they are wearing. The garments are
[Gaussian Garments](https://github.com/eth-ait/Gaussian-Garments) (captured
3D Gaussian-splat clothing), worn on an SMPL-X body regressed from the
camera. Runs on an Apple M1 (no CUDA). Every stage names its method and its
measured number; what is planned and what was tried and dropped is in
[`docs/ROADMAP.md`](docs/ROADMAP.md).

## Start here

```bash
./live-tryon                # pick a garment, then the live webcam try-on
./live-tryon help           # every command: run, garments, doctor, test, compare
./live-tryon doctor         # what this machine has, and which GPU each stage uses
```

## Documentation

| read | for |
|---|---|
| [docs/USAGE.md](docs/USAGE.md) | running it: every option, the display modes, speed and accuracy, GPU setup, limitations, troubleshooting |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | the code: one frame through the modules, what each folder holds, a reading order |
| [docs/DATASETS.md](docs/DATASETS.md) | every dataset and model file: where it lives, where it was downloaded from, what was dropped |
| [docs/ROADMAP.md](docs/ROADMAP.md) | where the project stands, the targets, what is next, what was tried and dropped |
| [docs/LOG.md](docs/LOG.md) | the dated engineering log: every measurement and decision, in order |
| [docs/REFERENCE_RUNS.md](docs/REFERENCE_RUNS.md) | other try-on systems run on this machine for comparison |
| [docs/AUDIT.md](docs/AUDIT.md) | the code audit: every module's size, complexity, findings and what was done |
| [docs/BRANCHES.md](docs/BRANCHES.md) | the branch, the archive tags, and comparing any two versions |

## Project layout

    live-tryon          the launcher: menu, run, garments, doctor, test, compare
    setup.sh, doctor.py first-time setup; a check of this machine
    tryon/              the system, one package by responsibility
      app/              the live loop (pipeline.py), display modes, command line (cli.py)
      body/             the person's 3D body: NLF regressor, SMPL-X fitter, smoothing, orientation
      garment/          the garment on that body: skinning, cloth, collision, renderer, shading
      rendering/        Gaussian-splat rasterisers (CPU reference, OpenGL)
      removal/          erasing the person's own clothing: parsers, fill, background plate
      paths.py          where every file lives (one place; TRYON_DATA moves the data root)
    tools/
      evaluate/         one tool per measurement; results go to docs/baseline/
      train/            training the project's own networks (shading students, orientation)
      data/             downloads, pose fits
    tests/              test_<folder>_<module>.py for each module
    docs/               the documents above; baseline/ (results), figures/
    models/, data/      weights and test data (git-ignored; ./setup.sh fills models/)

    ~/tryon-data/       outside the repository (TRYON_DATA moves it; docs/DATASETS.md):
      garments/ trajectories/ release_videos/ body_models/smplx/ references/ captures/ training_logs/

Run everything from the repository root: `python -m tryon.app` for the live
system, `python -m tools.evaluate.<tool>` for a measurement.

## The pipeline (one live frame)

    webcam frame (at most 960 px wide)
      -> 2D keypoints        MediaPipe Pose (BlazePose)                  tryon/app/person.py
      -> SMPL-X body         NLF-S image regressor                       tryon/body/regressor.py, nlf_worker.py
                             (fallbacks: Multi-HMR, then the keypoint fit tryon/body/fitter.py)
      -> garment on body     linear blend skinning from the scan pose    tryon/garment/skinning.py
      -> cloth               springs toward the skinned pose + collision tryon/garment/cloth.py, collision.py
      -> appearance          distilled shading network (worker thread)   tryon/garment/shading_student.py
      -> render              Gaussian splats on the GPU (OpenGL)         tryon/rendering/splat_gl.py
                             (splat_cpu.py: the numba reference)
      -> clothing removal    SCHP parser near the body; fill             tryon/removal/remover.py
      -> composite           face, hair and hands stay in front          tryon/app/compositing.py

The loop that runs it is `tryon/app/pipeline.py::run`. **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**
walks through one frame in the code and suggests a reading order.

By default every shown frame carries the garment drawn for it, at the
pipeline's rate (7.3-7.5 fps on the M1 on a video file, 4.4-4.8 before the 8 Oct speed work; the body regressor works on the
next frame while this one is drawn). `--smooth` shows
every camera frame and carries the newest garment with the torso;
`--delayed` shows that about a second late, with the garment from the
nearest pipeline frame.

## Where it stands (measured)

| | number | source |
|---|---|---|
| garment IoU vs the real shirt, 5 release videos | **0.730** (the authors' 4-camera registration: 0.827) | `tools/evaluate/vs_registration.py` |
| body accuracy, 16 annotated views (PA-MPJPE) | **24.6 mm** (keypoint fit 87-94 mm) | `docs/baseline/body-regressors.json` |
| pipeline time a frame, M1 | 0.1-0.2 s of work (0.085-0.10 s on a quiet machine, 125 ms in the 13 Oct profiled run; 0.21-0.23 s before the 8 Oct speed work); smooth display at camera rate | `python -m tryon.app --profile` |

## Setup, step by step

Tested on macOS with Apple Silicon (M1, 16 GB). Tick each box in order;
`./live-tryon doctor` checks most of them for you.

- [ ] **1. Tools**: Python 3.10, git, and [uv](https://docs.astral.sh/uv/)
      (`brew install python@3.10 uv`).
- [ ] **2. SMPL-X body models** (licence, manual): register at
      https://smpl-x.is.tue.mpg.de, download SMPL-X v1.1, put the `.npz`
      files in `~/tryon-data/body_models/smplx/`.
- [ ] **3. Garment checkpoints** (manual): Gaussian Garments models of
      4D-DRESS subjects in `~/tryon-data/garments/<name>/` (each with
      `stage1/`), and the authors' trajectory files in
      `~/tryon-data/trajectories/` (`docs/DATASETS.md`).
- [ ] **4. Everything else**: `./setup.sh` - the Python environment, the
      SCHP parsers, the NLF-S body regressor and its environment, and each
      garment's scan pose. The 15 scan poses ship in `assets/scan_poses/`
      (four of them hand-refined, `refined=1`, because the authors'
      trajectory does not hold their template's body); setup copies them
      into the checkpoints that lack one, never over an existing file, and
      fits (and, when the trajectory frame is not the template, refines) a
      pose only for a garment with no asset. `./live-tryon doctor` lists
      where each pose comes from. Re-running it only does what is missing.
- [ ] **5. Check**: `./live-tryon doctor` ends with `Ready.` and lists MPS,
      SCHP on CoreML, OpenGL on Metal and NLF-S as available.
- [ ] **6. Tests**: `./live-tryon test` passes.
- [ ] **7. Try it**: `./live-tryon`, pick a garment, stand 2-3 m from
      the camera. macOS asks once for camera access for the terminal.

Optional: Multi-HMR as a second body regressor (clone
https://github.com/naver/multi-hmr into `models/pretrained/multi-hmr`, put
`multiHMR_672_S.pt` in its `models/multiHMR/`, symlink its `models/smplx`
to the SMPL-X models, `pip install --no-deps roma`); Big-LaMa for the
offline quality fill (`tools/data/download_pretrained.py lama`). GPU notes,
including NVIDIA, are in docs/USAGE.md, "GPU setup".

## Methods and references

| stage | method | reference |
|---|---|---|
| 2D keypoints | BlazePose via MediaPipe Pose | Bazarevsky et al., *BlazePose: On-device Real-time Body Pose Tracking*, CVPR-W 2020 |
| body model | SMPL-X | Pavlakos et al., *Expressive Body Capture: 3D Hands, Face, and Body from a Single Image*, CVPR 2019 |
| 3D body from the image | NLF-S (default) | Sárándi & Pons-Moll, *Neural Localizer Fields for Continuous 3D Human Pose and Shape Estimation*, NeurIPS 2024 - [github.com/isarandi/nlf](https://github.com/isarandi/nlf) |
| | Multi-HMR (fallback) | Baradel et al., *Multi-HMR: Multi-Person Whole-Body Human Mesh Recovery in a Single Shot*, ECCV 2024 |
| | keypoint fit (fallback) | SMPLify-style optimisation: Bogo et al., *Keep it SMPL*, ECCV 2016 |
| keypoint smoothing | One Euro filter | Casiez et al., CHI 2012 |
| garment representation | Gaussian Garments (mesh-bound 3D Gaussians + appearance network) | Rong et al., *Gaussian Garments: Reconstructing Simulation-Ready Clothing with Photorealistic Appearance from Multi-View Video*, 3DV 2025 - [github.com/eth-ait/Gaussian-Garments](https://github.com/eth-ait/Gaussian-Garments) |
| capture data | 4D-DRESS | Wang et al., *4D-DRESS: A 4D Dataset of Real-World Human Clothing with Semantic Annotations*, CVPR 2024 |
| garment follows body | linear blend skinning, inverse-skinned from the scan pose | Loper et al., *SMPL*, SIGGRAPH Asia 2015 |
| cloth | damped springs toward the skinned pose (`LightCloth`), body collision, strain limiting | Müller et al., *Position Based Dynamics*, 2007 (collision projection); Provot, *Deformation Constraints in a Mass-Spring Model*, 1995 |
| rendering | 3D Gaussian splatting (EWA splats), CPU (numba) and OpenGL | Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, SIGGRAPH 2023; Zwicker et al., *EWA Splatting*, 2002 |
| clothing parser | SCHP (ATR and LIP label sets) | Li et al., *Self-Correction for Human Parsing*, TPAMI 2020 |
| fill | learned background plate, Telea inpainting; Big-LaMa (offline) | Telea, 2004; Suvorov et al., *LaMa*, WACV 2022 |
| evaluation reference | the authors' own registration in their release videos | Gaussian Garments release videos |

Compared and not adopted (docs/ROADMAP.md, "Tried and dropped"; numbers in docs/LOG.md): SAM 2 for the
clothing mask, FASHN parser (more accurate, 8x slower), RTMPose,
ContourCraft, learned pose correctives, SMPLer-X, Multi-HMR 672_B,
video diffusion try-on (ViViD, CatVTON - tens of seconds a frame on the M1),
FitMe360 and VTON360 (reproduced as references).

## Targets and checkpoints

The targets (garment IoU 0.80, body 30 mm, camera-rate display, under 0.5 s a
frame, ...), which are met, and what is next are in [docs/ROADMAP.md](docs/ROADMAP.md).

## Measuring

Each number above comes from a tool in `tools/evaluate/`, which writes its
result as JSON under `docs/baseline/`: `vs_registration.py` (the garment
against the real shirt over whole release clips), `dropouts.py` (how often
the garment disappears), `async_display.py` and `live_display.py` (the
smooth display), `tracking_jitter.py` and `real_fit.py` (the body),
`cloth_trajectory.py` and `resize_trajectories.py` (the cloth against the
authors' simulation), `reference_frame.py` (where quality is lost, level by
level), `background_fill.py`, `shading_students.py` (the shading networks).
