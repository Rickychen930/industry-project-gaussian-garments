# Datasets

What the project uses and where it was downloaded from, the request that
was archived, and what was dropped and why. Updated 8 Oct 2026; the history is in `LOG.md`.

Everything outside the repository lives under one data root, `~/tryon-data/`
(`TRYON_DATA` moves it; `tryon/paths.py` is the one place the folders are
named): `garments/`, `trajectories/`, `release_videos/`, `body_models/smplx/`,
`references/`, `captures/`, `training_logs/`. `tools/data/migrate_layout.sh`
moves the older layout (loose folders in `~/Downloads`) into it.

## In use (on disk, no registration)

| data | where | used by |
|---|---|---|
| Gaussian Garments checkpoints (15 garments, 4D-DRESS subjects) | `~/tryon-data/garments/` | everything |
| The authors' release videos (ground-truth panel + their registration) | `~/tryon-data/release_videos/` | `tools/evaluate/vs_registration.py`, `tracking_jitter.py`, `dropouts.py`, live profiling |
| The authors' cloth trajectories (`individual-garments/`, `resize/`, ...) | `~/tryon-data/trajectories/` | scan poses (`tools/data/fit_scan_pose.py`), shading-student poses, `cloth_trajectory.py`, `resize_trajectories.py` |
| SMPL-X body models | `~/tryon-data/body_models/smplx/` | the body |

## Where each was downloaded from

| data | from |
|---|---|
| Garment checkpoints | https://drive.google.com/file/d/1EnmIQQ3BIN9nqykhEAOv18R5mH1eq5fx/view (README of https://github.com/eth-ait/Gaussian-Garments) |
| Cloth trajectories | https://drive.google.com/file/d/1VoCmCfL-YmWL4BsgcdN8gxcEt3djJp-v/view (same README) |
| 4D-DRESS demo sample (`data/annotated/`) | https://drive.google.com/file/d/1gtlAiL1_uXGIvV6YFWHp4N9EJk8Vt977/view (the repo's `DataPreparation.md`) |
| Release videos | a Google Drive folder from the authors, downloaded 29 Aug 2026; the link was not recorded |
| SMPL-X | https://smpl-x.is.tue.mpg.de (free account) |
| NLF-S | https://github.com/isarandi/nlf/releases/download/v0.2.2/nlf_s_multi_0.2.2.torchscript |
| SCHP ATR / LIP (ONNX) | https://huggingface.co/yisol/IDM-VTON/tree/main/humanparsing |
| Big-LaMa (optional) | https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt |

The pretrained models are fetched by `./setup.sh` (`tools/data/download_pretrained.py`).

## Archived: 4D-DRESS (not requested)

4D-DRESS (ETH Zurich, CVPR 2024, https://4d-dress.ait.ethz.ch) is the
dataset the 15 garments were captured from. It is only available through a
registration form with an institutional e-mail and a manual approval of
days to weeks; there is no download in the official repository
(github.com/eth-ait/4d-dress, code only) or on the project page, and the
licence forbids redistribution, so no copy elsewhere may be used. With no
time left to wait for approval, it was not requested (7 Oct 2026).

The project does not need it to run or to be evaluated: a small official
sample is in `data/annotated/` (from the Gaussian Garments
`DataPreparation.md` demo, 2 takes x 4 cameras with ground-truth SMPL-X).
What full access would have added, recorded as future work: per-frame
ground-truth body shape (body-shape error on video), 30 fps scans of the
real garments (cloth motion against real cloth instead of the authors'
simulation), and 4-camera masks (a multi-view test set).

## Dropped

| dataset | why not any more |
|---|---|
| Human-parsing sets (`mattmdjaga/human_parsing_dataset`, `SaffalPoosh/VITON-HD-test`, `YashBhamare123/human_parsing_dataset_plus_neck`, Fashionpedia) | They trained and scored the project's own U-Net, which SCHP replaced on 25 Sep and which was removed on 8 Oct (tag `archive/unet-segmentation-2026-10-08`). `data/segmentation/` can be deleted. |
| CLOTH3D (ChaLearn) | Was for training a TailorNet-style garment model. Garment motion is now `LightCloth`, measured as accurate as a full solver against the authors' simulation. |
| LIP (ChaLearn) | Was for training our own parser. SCHP (pretrained on LIP and ATR) is used and measured better. |
| HBW / SHAPY (MPI) | Was for body-shape error. The NLF body regressor replaced the keypoint shape fit, and 4D-DRESS would measure the same thing on the garments' own subjects. |
| MV-Fashion | Gated, 68 cameras; no use the release videos and 4D-DRESS do not already cover. |
| ActorsHQ | Requested before the project used Gaussian Garments checkpoints; gives no garment assets. |
| DeepFashion2, VITON-HD train | Frontal 2D photos: no cameras, no 3D. |
| VTON360 renders | Downloaded (`~/Downloads/vton360/`, outside the data root with the other reference systems) for the 360-degree comparison, not used by the live system. |
