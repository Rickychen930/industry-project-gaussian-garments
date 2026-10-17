# How the code is organised

A guide for reading and changing this project. It follows one live frame
through the code, lists what each folder holds, and suggests an order to
read it in. What is planned is in [`ROADMAP.md`](ROADMAP.md), the history of
every measurement is in docs/baseline; how to run things is in
[`USAGE.md`](USAGE.md).

## The idea in one paragraph

A garment was captured by the Gaussian Garments authors as a mesh with about
100,000 small 3D Gaussians ("splats") stuck to its triangles. To dress a
person in a webcam frame we (1) find their body in 3D, (2) pose the garment
mesh like that body, (3) re-place every splat on the posed mesh and draw
them into an image, (4) erase the person's own clothing where the new
garment does not cover it, and (5) paste the drawn garment on top, keeping
the face, hair and hands in front. Every step is a module below.

## One frame, step by step

```
camera frame
  │
  ├─ app/person.py        landmarks_from()      MediaPipe: 33 keypoints (pixels, visibility, depth)
  │                       dressable()           is anyone there to dress?
  │
  ├─ removal/remover.py   clothing_mask()       started on a worker thread now, read later
  │
  ├─ body/regressor.py    NLF.predict()         a 3D SMPL-X body from the image (own process: nlf_worker.py)
  ├─ body/fitter.py       fit_from_regression() place that body in the image (scale, translation)
  │                       (no regressor: fit()/track() fit the body to the keypoints instead,
  │                        body/landmarks.py turning the keypoints into fitting targets)
  │
  ├─ garment/renderer.py  GarmentRenderer.render()
  │     ├─ bind()               once per person: garment/scan_pose.py poses a body like the scan,
  │     │                       garment/skinning.py takes the skin weights on it
  │     │                       (the garment itself was read by garment/checkpoint.py)
  │     ├─ posed_mesh()         skinning.apply() -> garment/cloth.py springs -> garment/collision.py
  │     ├─ splats()             every Gaussian re-placed on the posed triangles (numba kernel)
  │     ├─ _shaded()            pose-dependent colours: garment/shading_student.py (worker thread);
  │     │                       the 55M network it was distilled from: garment/appearance.py
  │     │                       (styleunet.py its architecture, texels.py its texture space)
  │     └─ rendering/splat_gl.py   draw the splats on the GPU, hiding those behind the body
  │
  ├─ removal/remover.py   remove()              erase + fill (removal/background.py learns the room)
  ├─ app/compositing.py   composite_garment()   garment over the frame
  │                       keep_in_front()       face, hair, hands back on top
  └─ shown (app/pipeline.py, default display), or handed to app/smooth_display.py (smooth mode)
```

`app/pipeline.py` holds it: `LiveSession` keeps what lives from frame to frame (the
models, the tracked body) and `process()` is one frame, in small named steps
(`_update_body`, `_handle_loss`, `_dress`, `_start_erase` / `_erase`); `run()` is the
camera loop around it, between a `FrameSource` (camera, file, or smooth mode's
channel) and a `FrameOutput` (window, video, keys). `app/smooth_display.py` is smooth
mode (`--smooth`, a separate display process): `_live_loop` / `_delayed_loop` with a shared `DisplayOutput`.
The loop reads one frame ahead: frame t+1's landmarks are found, it is sent
to the NLF worker and its clothing parse is started (`LiveSession.prefetch`,
`ClothingRemover.parse_clothing`, chained `MaskJob`s) before frame t is drawn,
so the regressor and the parser work while the renderer does; the erase of
frame t (plate update, mask limits, fill) runs on a worker thread beside its
render (`pipeline.ERASE_BESIDE_RENDER`).

## The folders

| folder | what it answers | start with |
|---|---|---|
| `tryon/body/` | Where is the person and how do they stand? | `fitter.py` (`BodyFitter`), then `regressor.py` |
| `tryon/garment/` | What does the garment look like on that body? | `renderer.py` (`GarmentRenderer`) |
| `tryon/rendering/` | How are Gaussian splats drawn into pixels? | `splat_cpu.py` (readable reference), then `splat_gl.py` |
| `tryon/removal/` | How is the person's own clothing erased? | `remover.py` (`ClothingRemover`) |
| `tryon/app/` | The live application: loop, display modes, command line, camera, the window's garment bar | `pipeline.py` (`run`), `cli.py`; `camera.py`, `ui.py` |
| `tools/evaluate/` | How good is it? Each tool scores one part against ground truth | `vs_registration.py` |
| `tools/train/` | Training the project's own networks | `shading_student.py` |
| `tools/data/` | Downloads, dataset preparation, pose fits, the demo video | `download_pretrained.py` |
| `tools/record_session.py` | The guided camera recording (`./live-tryon record`) | |
| `tests/` | `test_<folder>_<module>.py` for each module | `test_goals.py` (what the system must do) |
| `docs/baseline/` | Every measured result as JSON, written by `tools/evaluate/` | |

## Where files live

`tryon/paths.py` is the only place a path is written down: `MODELS`
(`models/`, weights), `DATA` (`data/`, test sets), `BASELINE`
(`docs/baseline/`, results), and outside the repository the garment
checkpoints (`GARMENTS`), the authors' trajectories and release videos and
the SMPL-X models, all under one data root (`~/tryon-data`, or
`TRYON_DATA`). Every module, tool and test imports them from there.
[`DATASETS.md`](DATASETS.md) says where each was downloaded from.

## Running the tools

From the repository root, as modules: `python -m tryon.app` for the live
system, `python -m tools.evaluate.<tool>` for a measurement (each writes
its JSON to `docs/baseline/`), `python -m tools.train.<tool>` to train.
Each prints its own `--help`. The shading students for all garments:
`tools/train/all_shading_students.sh` (trains each, keeps it if it is the
newest, logs to `~/tryon-data/training_logs`); scored together by
`python -m tools.evaluate.shading_students`.

## Suggested reading order

1. `tryon/__init__.py` and each folder's `__init__.py`: one paragraph each.
2. `tryon/app/pipeline.py`: `LiveSession.process` (one frame) and `run` (the loop).
3. `tryon/body/fitter.py`: `BodyFitter`, `fit`, `track`, `fit_from_regression`.
4. `tryon/garment/skinning.py`: `SkinnedGarment.build_in_pose` and `apply`.
5. `tryon/garment/renderer.py`: `bind`, `posed_mesh`, `splats`, `render`; then
   `scan_pose.py` (`ScanPoseBinder.bind`) and `checkpoint.py`.
6. `tryon/rendering/splat_cpu.py`: `_prepare` and the tile compositing; then
   `splat_gl.py`, which does the same on the GPU.
7. `tryon/removal/remover.py`: `clothing_mask` and `remove`.
8. `tests/test_goals.py`: the behaviour everything above must keep.

## Rules the code follows

- **Every number in a comment was measured**, with where and how. If a
  constant has no measurement next to it, it is a guess and says so.
- **Optimised code keeps its original as an oracle.** `splat_cpu.py` for
  `splat_gl.py`, `GarmentRenderer._splats_numpy` for the numba placement kernel,
  `_texture_binding_python` for the compiled binding, and so on; the tests
  require them to agree.
- **A model or method is kept only where it measured better** than the
  alternative on the same data; what lost is removed and listed in
  `ROADMAP.md` ("Tried and dropped").
- **numba `parallel=True` kernels run on one thread only** (the main thread: the renderer and the splat rasteriser). numba's only available threading layer here, `workqueue`, aborts the process when two threads enter parallel regions; worker threads (the mask, the shading student) use serial kernels, numpy, OpenCV, ONNX or torch.
- **Switches are class attributes** (`GarmentRenderer.GPU_RASTER`,
  `ClothingRemover.INPAINT_SCALE`, ...) so an evaluation tool can flip one
  and measure it without new flags everywhere.
