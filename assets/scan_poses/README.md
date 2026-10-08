# Scan poses

`<garment>.npz` (about 2 KB each, 15 files) is the SMPL-X pose and shape a
garment was scanned in: keys `betas`, `body_pose`, `global_orient`, `offset`
(plus fit diagnostics). `ScanPoseBinder` (tryon/garment/scan_pose.py) binds the
garment's skin weights on a body posed like this, so the files are the starting
point of every bind. When one exists here it is used in preference to the
checkpoint's own `stage1/scan_pose.npz`, so every machine binds identically;
`./setup.sh` also copies it into the checkpoint if that has none.

How they were made: `tools/data/fit_scan_pose.py` fits SMPL-X to the body of
the first frame of the authors' cloth trajectory (`obstacle[0]`). That body is
the template's only when the trajectory's first garment frame is the template
(median vertex distance 0.0-0.7 mm for 11 garments). For 00122_outer,
00176_outer, 10004_outer and 00148_upper it is not (33, 46, 43 and 218 mm), so
`tools/data/refine_scan_pose.py` moved pose, orientation and offset until the
template sits around the body. Those four carry `refined = 1`, which makes the
binder also refit the arms to the sleeves (`REFINE_ARMS`). The refinement is
deterministic (re-running it on 00122_outer reproduced the file bit for bit).

Licence: these are our own fits of the SMPL-X body (parameters only, no mesh,
no model weights) derived from the Gaussian Garments checkpoints and
trajectories. They contain no capture data, images or garment geometry.
