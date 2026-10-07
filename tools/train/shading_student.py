"""Distil a garment's appearance network into `tryon.garment.shading_student`.

Online distillation, nothing written to disk but checkpoints (the teacher's
target for one pose is 93k x 51 values): a pool of POOL (student input,
teacher output) pairs is kept in memory; each cycle makes one new pair and
takes TRAIN_STEPS gradient steps on random batches from the pool.

Poses: the garment's own trajectory fits (docs/baseline/trajectory-fits-
<name>.npz; the last HOLDOUT of the frames are never trained on), each
turned by a random yaw of up to +-YAW degrees and jittered by Gaussian
pose noise, so the student sees the body from the sides too. The teacher
is `AppearanceNet` (fp16 on MPS) with its AO baked at 8 rays (42.6 dB from
the 32-ray bake, see AsyncAppearance).

Measured every EVAL_EVERY cycles on fixed held-out poses: the render of
the garment with the student's shading against the render with the
teacher's (PSNR, 640 x 870, GPU compositor), next to the static (no
network) render against the teacher, and the student's time a frame.

    .venv/bin/python -m tools.train.shading_student 00190_upper [--hours 4]

`--mean-poses K` (default 32) first averages the teacher's output per
Gaussian over K training poses and stores it with the student, which then
predicts only the residual around it (see `shading_student`); 0 trains the
original, mean-free student.

`--resume` continues from the garment's saved student (a second cosine
schedule, usually at a lower `--lr`) and only overwrites it when the
held-out score beats the one it was saved with; the held-out poses are
fixed, so the two scores are comparable. The log then goes to
`shading-student-<name>-resumed.json`.
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np
import torch

from tools.data.fit_motions import MOTION_FITS
from tools.evaluate._common import write_json
from tryon.paths import BASELINE, GARMENTS
SIZE = (870, 640)
POOL, TRAIN_STEPS, BATCH = 80, 3, 4  # pool ~0.8 GB: a 160 pool with other jobs running was killed under memory pressure
HOLDOUT, YAW, POSE_NOISE = 0.2, 60.0, 0.06
EVAL_EVERY = 100
# Motions from the multi-garment simulations (tools/data/fit_motions.py):
# `--with-motions` trains on all of them but these, which stay unseen for
# tools/evaluate/shading_students.py. On 00148_upper the student trained
# on its own motion alone scored 37.9 dB on its own held-out poses but
# 32.5 / 31.0 on these; with the other 13 motions added, 37.5 / 36.1 / 35.8.
TEST_MOTIONS = ("02_02", "02_04")


def frame_camera(fitter, mesh):
    """Scale and place the fitter's weak-perspective camera so the garment
    fills the frame (the camera `GarmentRenderer.camera` derives from it)."""
    height = float(mesh[:, 1].max() - mesh[:, 1].min())
    scale = 0.8 * SIZE[0] / max(height, 1e-3)
    centre = mesh.mean(0)
    with torch.no_grad():
        fitter.scale.fill_(scale)
        fitter.translation.copy_(torch.tensor([SIZE[1] / 2 - scale * centre[0], SIZE[0] / 2 + scale * centre[1]]).reshape(fitter.translation.shape))


class Poser:
    def __init__(self, name, renderer):
        from tools.evaluate.cloth_trajectory import set_fitter

        from tryon.body.fitter import BodyFitter

        data = np.load(BASELINE / f"trajectory-fits-{name}.npz")
        self.fits = [{k: data[k][i] for k in ("betas", "body_pose", "global_orient", "offset")} for i in range(len(data["frames"]))]
        split = int(len(self.fits) * (1 - HOLDOUT))
        self.train, self.held = self.fits[:split], self.fits[split:]
        self.set_fitter = set_fitter
        self.fitter = BodyFitter()
        self.renderer = renderer
        set_fitter(self.fitter, self.fits[0])
        renderer.bind(self.fitter)

    def pose(self, fit, yaw_deg, noise, rng):
        from scipy.spatial.transform import Rotation

        fit = dict(fit)
        orient = Rotation.from_rotvec(np.asarray(fit["global_orient"], np.float64).reshape(3))
        fit["global_orient"] = (Rotation.from_euler("y", yaw_deg, degrees=True) * orient).as_rotvec().astype(np.float32)
        fit["body_pose"] = (np.asarray(fit["body_pose"], np.float64) + rng.normal(0, noise, np.shape(fit["body_pose"]))).astype(np.float32)
        self.set_fitter(self.fitter, fit)
        # Skinned AND pushed out of the body, as the live loop draws it (each
        # pose a fresh start for the cloth: the poses are unrelated). Skinning
        # alone left 26 % of 00134_upper's texels inside the body; the AO bake
        # (the teacher's input) went black there (0.05) and the teacher
        # painted green / red / magenta patches the students then learned.
        self.renderer.reset_motion(self.fitter)
        mesh = self.renderer.posed_mesh(self.fitter)
        frame_camera(self.fitter, mesh)
        view, intrinsics, _ = self.renderer.camera(self.fitter, SIZE)
        body = self.fitter.vertices()
        return mesh, view, intrinsics, body


def teacher_target(teacher, mesh, view, body, body_faces):
    ambient, normal = teacher.bake(mesh, [(body, body_faces)], rays=8)
    result = teacher.forward(mesh, view, ambient=ambient, normal=normal)
    target = np.concatenate([result["offset"], result["shadow_shs"].reshape(len(result["offset"]), -1)], 1)
    return target.astype(np.float32), result


def main():
    from tryon.rendering.splat_gl import render_splats_gl
    from tryon.garment.appearance import AppearanceNet, merge
    from tryon.garment.renderer import GarmentRenderer
    from tryon.garment.shading_student import FILE_NAME, StudentInputs, StudentUNet

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("garment", nargs="?", default="00190_upper")
    parser.add_argument("--hours", type=float, default=4.0)
    parser.add_argument("--cycles", type=int, default=None,
                        help="train this many cycles instead of for --hours: a loaded machine then trains slower, not less "
                             "(at 0.35 h a queue beside other jobs stopped some students under 100 cycles)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-poses", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="continue from the saved student")
    parser.add_argument("--mean-poses", type=int, default=32, help="poses the stored per-Gaussian mean is averaged over (0: none)")
    parser.add_argument("--out-name", default=None, help="save under stage3/<this> instead of the live student's file")
    parser.add_argument("--with-motions", action="store_true",
                        help="also train on the other garments' motions and the fitted simulations in docs/baseline/motion-fits, "
                             "except TEST_MOTIONS")
    parser.add_argument("--own-share", type=float, default=0.5,
                        help="with extra poses: the share of samples drawn from the garment's own motion")
    parser.add_argument("--extra-poses", nargs="*", default=[],
                        help="npz files of SMPL-X fits (tools.evaluate.cloth_trajectory.fit_frames) from other motions, "
                             "added to the training poses; the held-out poses stay the garment's own")
    parser.add_argument("--promote", action="store_true",
                        help="with --out-name: afterwards make it the live student (the first one it replaces is kept as "
                             "shading_student_nomean.pt)")
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    random.seed(0)
    torch.manual_seed(0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = GARMENTS / args.garment
    renderer = GarmentRenderer(checkpoint, dynamics=True)  # for its collision (see Poser.pose)
    poser = Poser(args.garment, renderer)
    if args.with_motions:
        args.extra_poses += [str(p) for p in sorted(MOTION_FITS.glob("*.npz")) if p.stem.split("-")[-1] not in TEST_MOTIONS]
        args.extra_poses += [str(p) for p in sorted(BASELINE.glob("trajectory-fits-*.npz"))
                             if p.stem != f"trajectory-fits-{args.garment}"]
    extra = []
    for path in args.extra_poses:
        data = np.load(path)
        extra += [{k: data[k][i] for k in ("betas", "body_pose", "global_orient", "offset")} for i in range(len(data["frames"]))]
    if extra:
        print(f"training poses: {len(poser.train)} own + {len(extra)} from {len(args.extra_poses)} other takes "
              f"(own share {args.own_share:.0%})", flush=True)
    body_faces = np.asarray(poser.fitter.model.faces)
    teacher = AppearanceNet(checkpoint, faces=renderer.faces, half=True)
    inputs = StudentInputs(teacher.binding, renderer.faces)
    model = StudentUNet().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"student: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters on {device}", flush=True)

    def sample(fits, yaw_range, noise):
        if extra and rng.random() >= args.own_share:
            fits = extra
        fit = fits[rng.integers(len(fits))]
        mesh, view, intrinsics, body = poser.pose(fit, rng.uniform(-yaw_range, yaw_range), noise, rng)
        target, result = teacher_target(teacher, mesh, view, body, body_faces)
        x = inputs.maps(mesh, view, body)
        return x, target, (mesh, view, intrinsics, result)

    # fixed held-out poses: trajectory frames never trained on, yaw spread over the range
    held = []
    eval_rng = np.random.default_rng(123)
    for i in range(args.eval_poses):
        fit = poser.held[(i * len(poser.held)) // args.eval_poses]
        mesh, view, intrinsics, body = poser.pose(fit, -YAW + 2 * YAW * i / max(args.eval_poses - 1, 1), 0.0, eval_rng)
        target, result = teacher_target(teacher, mesh, view, body, body_faces)
        splats = renderer.splats(mesh)
        kept = renderer._kept_indices.copy()  # which Gaussians THIS pose's splats hold (torn faces differ per pose)
        teacher_image, _ = render_splats_gl(merge(splats, result, kept), view, intrinsics, SIZE)
        static_image, _ = render_splats_gl(splats, view, intrinsics, SIZE)
        held.append((inputs.maps(mesh, view, body), mesh, view, intrinsics, splats, kept, teacher_image, static_image))
    def psnr(a, b):
        return float(10 * np.log10(1 / max(((a - b) ** 2).mean(), 1e-12)))
    static_psnr = float(np.mean([psnr(h[7], h[6]) for h in held]))
    print(f"held-out: static (no shading) vs teacher {static_psnr:.2f} dB", flush=True)

    # The teacher's output averaged per Gaussian over training poses: most of
    # its fine detail is fixed to the garment (a texture-like pattern a
    # network that sees no surface coordinates cannot place), and the mean
    # carries it; the student predicts the pose-dependent residual. The
    # targets double as the pool's first entries.
    pool, mean = [], None
    if args.mean_poses > 0 and not args.resume:
        targets = []
        for _ in range(args.mean_poses):
            x, target, _ = sample(poser.train, YAW, POSE_NOISE)
            targets.append(target)
            pool.append((torch.as_tensor(x, dtype=torch.float16), torch.as_tensor(target, dtype=torch.float16)))
        mean = np.mean(targets, axis=0).astype(np.float32)
        del targets
        with torch.no_grad():  # start exactly at the mean
            model.head.weight.zero_()
            model.head.bias.zero_()
    mean_t = None

    def predict(out):
        """Per-Gaussian (N, 51) from the network's output map(s)."""
        per = inputs.gather(out)
        return per if mean_t is None else per + mean_t

    def evaluate():
        model.eval()
        scores, times = [], []
        with torch.no_grad():
            for x, mesh, view, intrinsics, splats, kept, teacher_image, _ in held:
                t0 = time.perf_counter()
                out = model(torch.as_tensor(x)[None].to(device))
                per = predict(out).float().cpu().numpy().astype(np.float64)
                if device.type == "mps":
                    torch.mps.synchronize()
                times.append(time.perf_counter() - t0)
                shs = teacher.static_features() + per[:, 3:].reshape(-1, 16, 3)
                result = {"feature_dc": shs[:, :1], "feature_rest": shs[:, 1:], "xyz": teacher.world_xyz(mesh, teacher.xyz + per[:, :3])}
                image, _ = render_splats_gl(merge(splats, result, kept), view, intrinsics, SIZE)
                scores.append(psnr(image, teacher_image))
        model.train()
        return float(np.mean(scores)), float(np.median(times[1:] if len(times) > 1 else times) * 1000)

    log = {"garment": args.garment, "static_vs_teacher_db": static_psnr, "mean_poses": args.mean_poses, "evals": []}
    out_path = checkpoint / "stage3" / (args.out_name or FILE_NAME)
    started, cycle, best = time.perf_counter(), 0, -1.0
    log_name = f"shading-student-{args.garment}.json"
    if mean is not None:
        mean_t = torch.as_tensor(mean, device=device)
    if args.resume:
        saved = torch.load(out_path, map_location="cpu")
        model.load_state_dict(saved["model"])
        mean = saved.get("mean")
        mean = None if mean is None else np.asarray(mean, np.float32)
        mean_t = None if mean is None else torch.as_tensor(mean, device=device)
        best, start_score = saved["held_out_db"], evaluate()[0]
        log.update(resumed_from_db=best, resumed_score_db=start_score)
        log_name = f"shading-student-{args.garment}-resumed.json"
        print(f"resumed from {out_path.name}: saved {best:.2f} dB, re-measured {start_score:.2f} dB", flush=True)
    deadline = started + args.hours * 3600
    while (cycle < args.cycles) if args.cycles else (time.perf_counter() < deadline):
        x, target, _ = sample(poser.train, YAW, POSE_NOISE)
        pool.append((torch.as_tensor(x, dtype=torch.float16), torch.as_tensor(target, dtype=torch.float16)))
        if len(pool) > POOL:
            pool.pop(random.randrange(len(pool)))
        fraction = cycle / args.cycles if args.cycles else (time.perf_counter() - started) / (deadline - started)
        for group in optimiser.param_groups:
            group["lr"] = args.lr * 0.5 * (1 + np.cos(np.pi * min(fraction, 1.0)))
        for _ in range(TRAIN_STEPS):
            batch = random.sample(pool, min(BATCH, len(pool)))
            xb = torch.stack([b[0] for b in batch]).float().to(device)
            yb = torch.stack([b[1] for b in batch]).float().to(device)
            out = model(xb)
            pred = torch.stack([predict(out[i:i + 1]) for i in range(len(batch))])
            # Charbonnier (a smooth L1): L1's gradient (sign) falls back to
            # the CPU on MPS in torch 2.1, 1270 vs 997 ms a step
            loss = torch.sqrt((pred - yb) ** 2 + 1e-6).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        cycle += 1
        if cycle % EVAL_EVERY == 0:
            score, ms = evaluate()
            elapsed = (time.perf_counter() - started) / 3600
            log["evals"].append({"cycle": cycle, "hours": elapsed, "loss": float(loss), "student_vs_teacher_db": score, "student_ms": ms})
            print(f"cycle {cycle} ({elapsed:.2f} h): loss {float(loss):.4f} | held-out student vs teacher {score:.2f} dB "
                  f"(static {static_psnr:.2f}) | student {ms:.0f} ms", flush=True)
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "held_out_db": score, "static_db": static_psnr, "cycle": cycle,
                            "mean": None if mean is None else mean.astype(np.float16)}, out_path)
            write_json(BASELINE / log_name, log)
    print(f"done: best held-out {best:.2f} dB -> {out_path}", flush=True)
    if args.promote and out_path.name != FILE_NAME and out_path.exists():
        live, backup = out_path.with_name(FILE_NAME), out_path.with_name("shading_student_nomean.pt")
        if live.exists():
            if backup.exists():
                live.unlink()
            else:
                live.rename(backup)
        out_path.rename(live)
        print(f"promoted -> {live}", flush=True)


if __name__ == "__main__":
    main()
