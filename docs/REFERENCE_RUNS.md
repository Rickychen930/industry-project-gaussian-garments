# Reference systems, actually run on this machine

The reading notes on these repositories are in `docs/LOG.md`. This file
records what happened when each was **installed and run by its README** on
this Apple M1 (16 GB, no CUDA), and how its output compares with ours on the
same frame. Everything lives outside the project environment:
`~/Downloads/tryon_refs/` (clones, one virtualenv per system, runner
scripts), weights and data under `~/Downloads/`.

## VTON360 (He et al., CVPR 2025) - multi-view try-on, run 27 Sep 2026

**What it is.** An offline per-subject editor: a textured 3D scan
(THuman2.1) is rendered from 80 views; a multi-view-consistent diffusion
model (SD-1.5-sized UNet + ReferenceNet, pose from SMPL-X normal maps,
cross-view attention, camera-aware CLIP garment embeddings) redraws N of
those views wearing a new garment given as a front and a back image; a
customised Splatfacto (nerfstudio 3DGS) then lifts the edited views back to
a static 3D avatar. Nothing is live or animated.

**Setup, following the README.**
- Demo data: the README's Google Drive link is a single FILE (not a
  folder): `gdown "https://drive.google.com/uc?id=1K0RfX79kMZ9rC-kJWC6mLg_a5VKgucNL"`
  gives `demo_data.zip` (158 MB): THuman2.0 subjects 0009 / 0024 / 0228
  already rendered (images, normals, agnostic, parse, cameras), two
  MVHumanNet subjects and the Splatfacto demo.
- Weights (HuggingFace mirrors named in the README): `nielsr/VTON360-THuman2.0`
  checkpoint-30000 (UNet 3.3 GB + ReferenceNet 3.3 GB fp32 + pose guider),
  `stabilityai/sd-vae-ft-mse`, `openai/clip-vit-base-patch32`
  (`pytorch_model.bin`; the repo has no safetensors), SD-1.5's scheduler
  config.
- README step D (`get_cloth_thuman.py`, paths edited): garment front/back
  images extracted for all three subjects from the rendered views' ATR
  parse (label 4).
- Stage B (`infer_tryon_multi.py`) person 0009 wearing garment 0024.

**What it took to run without CUDA** (all in our clone, none in the project):
1. `pipelines/pipeline_tryon_multi.py` imports `data.dataset`, which the
   repository does not ship; the imported names are unused - commented out
   (release bug).
2. `models/mv_attn_processor.py` calls `xformers.ops.memory_efficient_attention`
   unconditionally; replaced by `torch.nn.functional.scaled_dot_product_attention`
   with the same inputs and scale (the same attention).
3. `infer_mps.py`: the README's script with `cuda` -> `mps`, the RNG on the
   CPU, no `enable_xformers_memory_efficient_attention`, no data-loader
   workers, batch 1.
4. Views per pass made configurable (`VTON360_VIEWS`, default 16): 16 views
   at 512 x 384 thrashed 13 GB of swap and never finished step 1 (every view
   attends to every view's tokens); 8 views ran.
- The authors' own config notes `sample_size [512,384]` "for 40G" GPUs and
  uses 768 x 576 on larger ones; 512 x 384 was used here.

**Result** (`docs/figures/vton360_0009_in_0024.jpg`, inputs in
`docs/figures/vton360_conditioning.jpg`): person 0009 (black sweatshirt)
wears 0024's pink "MEOW SCHOOL" sweatshirt consistently over all 8 views -
front, profiles and back - with the sleeves on the arms and the pose kept.
The garment's identity (colour, cut, print placement) carries across views,
which is the paper's claim. Weak points visible at this resolution: the face
is regenerated inside the agnostic region and comes out distorted, and the
print's text is illegible.

**Cost:** 7,229 s wall (2 h 0 min) for 8 views x 30 DDIM steps on MPS fp16,
about 4 min per step with heavy swapping; the 3D lifting stage (Splatfacto,
CUDA-only gsplat) was not run.

**What it means for this project.** VTON360 answers a different question -
"what would this scanned person look like in this garment, from every side"
- in two hours per subject, offline. It confirms the value of two ideas we
already use (SMPL-X-derived pose conditioning; a clothing-agnostic region
that erases the old garment including the arms) and one we do not have
(generating the garment's appearance with a diffusion prior instead of a
captured asset). It cannot serve a live webcam: every frame would be a new
multi-view edit plus a 3DGS fit.

## FASHN VTON 1.5 (FASHN AI, Apache-2.0) - maskless 2D try-on, run 27 Sep 2026

**What it is.** A pixel-space MMDiT diffusion model (1.9 GB, bf16 weights)
that takes a person photo and a garment photo (worn by a model or flat-lay)
and redraws the person wearing the garment, with DWPose and the FASHN human
parser as conditioning. No segmentation mask is needed. One image at a time;
no 3D, no video.

**Setup, following the README.** `pip install -e .` with the CPU
`onnxruntime` (the README's own instruction for non-CUDA machines),
`scripts/download_weights.py`. The README example reproduced exactly on the
CPU (1,514 s, fp32). On MPS two things fail in torch 2.2 - bfloat16
weights cannot be placed on MPS, and its RoPE is computed in float64 -
worked around in our runner (`~/Downloads/tryon_refs/fashn_run.py`: load on
the CPU and cast to fp32, RoPE on the CPU); MPS then ran out of unified
memory (53 min without finishing one image while swapping 14 GB), so every
run below is on the CPU at the README defaults (30 steps, CFG 1.5, seed 42):
**27-33 min per image**.

**Scored on our reference frame** (script since removed; result in
`docs/baseline/compare-tryon-systems-00152_outer.json`): every system gets
the same person - release video 00152_outer frame 0 with the shirt erased -
and is judged against the real frame (shirt IoU from the SCHP parser on
both images; PSNR over the union of the shirts; SSIM / DISTS / LPIPS on the
real shirt's box at 256 x 341, TryOffDiff's protocol; lower DISTS/LPIPS is
better). FASHN's garment image is the same shirt photographed at frame 60
or 120 of the same video.

| system | shirt IoU | PSNR | SSIM | DISTS | LPIPS | time / frame |
|---|---|---|---|---|---|---|
| erased photo only | 0.190 | 10.62 | 0.558 | 0.299 | 0.576 | - |
| ours, their body + cloth (L1) | 0.809 | 14.38 | 0.563 | 0.106 | 0.184 | - |
| **ours, full live pipeline (L3)** | **0.750** | **14.03** | **0.541** | **0.108** | **0.244** | **0.1-0.2 s** (~0.4 s when this table was made) |
| FASHN VTON 1.5, garment photo f60 | 0.670 | 13.56 | 0.511 | 0.169 | 0.338 | 33 min |
| FASHN VTON 1.5, garment photo f120 | 0.759 | 13.94 | 0.500 | 0.170 | 0.303 | 30 min |

Figures: `docs/figures/compare_ref_ours_vs_fashn.jpg`,
`docs/figures/compare_clips_ours_vs_fashn.jpg`.

**Reading.** FASHN's pictures are the most photographic of anything tested -
clean hands, no arm tubes, lighting that matches - and its silhouette is as
good as ours with a well-chosen garment photo (0.759 vs 0.750). But the
garment it draws is a *plausible* shirt, not *this* shirt: perceptually
(DISTS 0.17 vs 0.11, LPIPS 0.30-0.34 vs 0.24) ours is closer to the real
frame on every measure, because ours renders the captured garment. On the
clips it also rewrites what it should keep: the dancer's and the turning
wearer's white T-shirts come out brown / yellow (it copies the inner layer
of the garment photo's model), and it invents a ripped knee in the jeans.
It is 4,000x slower than our live path on this machine.

**What to take from it.** The gap that remains between us and a diffusion
try-on is the *integration* - hands, arm skin, lighting - not the garment.
That is exactly the part a diffusion model is good at, so the promising
combination is offline refinement of our composite (our garment and body
kept, a diffusion pass only on the seams, hands and skin), not replacing the
pipeline.

## Harmonizer (Ke et al., ECCV 2022, CC BY-NC-SA 4.0) - adopted as `--harmonize`

Found by a survey of live try-on work (see the end of this file). Its demo
ran by its README after one fix in our clone: `torch.load` without
`map_location` cannot load the CUDA-saved weights on a CPU-only machine,
although the demo itself falls back to the CPU.

| composite scored against the real frame | PSNR | SSIM | DISTS | LPIPS |
|---|---|---|---|---|
| ours (L3), same studio light as the capture | 14.03 | 0.541 | 0.108 | 0.244 |
| + Harmonizer | 13.39 | 0.536 | 0.114 | 0.251 |
| scene recoloured warm and dim, garment still in studio colours | 14.88 | 0.559 | 0.122 | 0.237 |
| + Harmonizer | 15.37 | 0.560 | 0.112 | 0.239 |
| ideal: garment lit by the same recolouring | 15.50 | 0.562 | 0.116 | 0.237 |

With nothing to correct it only adds error; with a lighting mismatch - the
webcam case, since every garment was captured in the 4D-DRESS studio - it
recovers ~80% of the PSNR gap and all of the DISTS gap (the recoloured
scene is a controlled stand-in for a real webcam, whose ground truth we do
not have). Figure: `docs/figures/harmonizer_cast_test.jpg`. Cost at 640 x
870: argument prediction 31 ms on MPS, application 57 ms on the CPU, so
`tryon/app/harmonize.py` predicts every 10 frames (EMA 0.5) and applies every
frame. Off by default (`live.py --harmonize`); loaded from its own clone at
`~/Downloads/tryon_refs/survey/Harmonizer`, not vendored.

## Survey of live try-on work (27 Sep 2026)

Nothing released runs a full diffusion try-on live on this machine:
Mobile-VTON (CVPR 2026) needs ~80 s per image on a phone NPU; CatV2TON,
DreamVVT, MagicTryOn are offline DiT video models; LiveVVT has no code.
The shortlist for *our* pipeline, in order:
1. A pose-space corrective for the garment baked from HOOD / ContourCraft
   rollouts or the 4D-DRESS registrations (armpit tearing, swing; runs in
   milliseconds). ContourCraft ships a `GaussianGarments.ipynb` converter
   for exactly these checkpoints; its collision module is CUDA-only, HOOD
   has no CUDA extension.
2. Harmonizer (done, above).
3. MediaPipe's selfie multiclass segmenter (6 classes incl. body skin and
   clothes) and MI-GAN (~10x cheaper than LaMa) for the per-frame parser
   and inpainter.
Also: RobustVideoMatting (GPL-3.0, CoreML model released) for a stable
person alpha; DM-VTON (real-time 2D GAN, needs its cupy correlation
rewritten) only as a baseline.

## MediaPipe selfie multiclass segmenter - measured, rejected (27 Sep 2026)

Survey item 3: replace the SCHP parser on the per-frame path with
MediaPipe's `selfie_multiclass_256x256` (Apache-2.0, 15.6 MB, Tasks API).
Its single "clothes" class covers trousers too, so the upper garment is
taken above a cut 0.35 torso lengths below the hips from MediaPipe Pose
(`tryon3d/parsing_mp_multiclass.py`, since removed). Scored by `tools/evaluate/parsers.py` (since removed with the
U-Net) on the same labelled test images (150 per source, SCHP
one pass):

| source | SCHP IoU / BF@2px | MediaPipe multiclass IoU / BF@2px | ms/image SCHP / MediaPipe |
|---|---|---|---|
| lip_style | 0.884 / 0.797 | 0.644 / 0.602 | 220 / 216 |
| vitonhd | 0.930 / 0.768 | 0.654 / 0.603 | 112 / 144 |

(Timings taken while CatVTON ran on the GPU, so both are inflated; the
ranking does not depend on it.) A quarter of the mask accuracy for no
saving: SCHP stays.

## CatVTON (Chong et al., ICLR 2025) - mask-based 2D try-on, run 27 Sep 2026

**What it is.** Stable Diffusion 1.5 inpainting with the person and the
garment concatenated side by side in one latent; only the self-attention
layers are trained (49 M parameters). The garment region is erased by
AutoMasker (DensePose + SCHP ATR/LIP) for a chosen cloth type.

**Setup, following the README** (own venv: torch 2.4, diffusers 0.31,
accelerate 1.0.1; weights `zhengchong/CatVTON` "mix" checkpoint and
`booksforcharlie/stable-diffusion-inpainting`). Our runner
`~/Downloads/tryon_refs/catvton_run.py` repeats `submit_function` in its
`app.py` step by step. Changes needed on the M1:
1. `model/DensePose/__init__.py` never sets `cfg.MODEL.DEVICE`, so detectron2
   builds DensePose for CUDA; the masker's own `device` is now passed through.
2. The pipeline's `__call__` defaults to 1024 x 768 regardless of the input
   size; the runner passes `height`/`width` (at 1024 x 768 one attention
   buffer alone is 18 GB).
3. 768 x 576 used 15 GB and swapped, so every run below is at **512 x 384,
   25 steps** (app: 1024 x 768, 50 steps), fp16 on MPS, CFG 2.5, seed 42 -
   a real departure from the authors' setting, which favours them less.
**74-115 s per image.** For scoring, the 3:4 centre crop it works on is
pasted back into the erased frame.

| system | shirt IoU | PSNR | SSIM | DISTS | LPIPS | time / frame |
|---|---|---|---|---|---|---|
| ours, full live pipeline (L3) | 0.750 | 14.03 | 0.541 | 0.108 | 0.244 | 0.1-0.2 s (~0.4 s when this table was made) |
| FASHN VTON 1.5, garment photo f120 | 0.759 | 13.94 | 0.500 | 0.170 | 0.303 | 30 min |
| CatVTON, garment photo f60 | 0.478 | 12.18 | 0.485 | 0.237 | 0.526 | 78 s |
| CatVTON, garment photo f120 | 0.478 | 11.89 | 0.493 | 0.258 | 0.552 | 92 s |

Figures: `docs/figures/compare_ref_ours_fashn_catvton.jpg`,
`docs/figures/compare_clips_ours_fashn_catvton.jpg` (columns: input, ours,
FASHN, CatVTON).

**Reading.** CatVTON turns the knee-length overshirt into a waist-length
jacket on every input. Its mask ends at the hips, and inpainting cannot draw
outside the mask. The plaid breaks up, with blotches on the sleeves and
pockets. It does keep what lies outside the mask: the dancer's white T-shirt
stays white, which FASHN failed to do. It is the weakest of the three on
every measure. Its one advantage is speed (about 20x faster than FASHN on
this machine), and it is still about 200x slower than live.
