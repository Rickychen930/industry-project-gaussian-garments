"""Static SH against the checkpoint's own appearance network, side by side.

Renders the rest (template) mesh of a garment checkpoint through our
rasteriser four ways, so each change is isolated:

  1. what the pipeline draws today: static SH, splats anchored at the
     face CENTRE (`GarmentRenderer.splats`);
  2. static SH, splats anchored at their texel's barycentric point - the
     anchor the stage-3 model was trained with (`get_barycentric_3d`);
  3. `shadow_net` colours, at the face-centre anchor (colours only);
  4. the reference's full output: `shadow_net` colours AND its per-texel
     position offsets about the barycentric anchor (`get_final_xyz`).

The camera is the capture rig's frontal camera (`cameras.json`, id
0004) at half resolution, so the picture can be put next to the release
video's "Ours - Registration" panel. Prints the network's forward time on
MPS (float32 and float16) and on CPU, and the AO/normal bake time.

    .venv/bin/python -m tools.evaluate.appearance_port [--out /tmp/appearance_ab.png]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from tryon.paths import GARMENTS as MODELS_ROOT
from tryon.rendering.splat_cpu import render_splats
from tryon.garment.appearance import AppearanceNet, merge
from tryon.garment.renderer import GarmentRenderer
from tryon.paths import RELEASE_VIDEOS, SCRATCH


def scan_camera(checkpoint_dir, camera_id, scale):
    cams = json.load(open(checkpoint_dir / "stage1" / "cameras.json"))[camera_id]
    view = np.eye(4)
    view[:3] = np.asarray(cams["extrinsics"], np.float64)
    intrinsics = np.asarray(cams["intrinsics"], np.float64)
    intrinsics[:2] *= scale
    return view, intrinsics


def render(splats, view, intrinsics, size, background=0.5):
    """BGR uint8 over a grey background, plus the alpha for cropping."""
    image, alpha = render_splats(splats, view, intrinsics, size)
    composite = image + background * (1 - alpha)[..., None]
    return (np.clip(composite, 0, 1) * 255).astype(np.uint8), alpha


def crop_box(alphas, margin=12):
    """One box round everything any panel drew, so the panels stay aligned."""
    rows, cols = np.where(np.max(alphas, axis=0) > 0.01)
    height, width = alphas.shape[1:]
    return (max(rows.min() - margin, 0), min(rows.max() + margin, height),
            max(cols.min() - margin, 0), min(cols.max() + margin, width))


def label(image, text):
    out = image.copy()
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


RELEASE_VIDEO = [p for p in [RELEASE_VIDEOS / "00152_outer.mp4"] if p.exists()]
# The garment in the "Ground Truth" half of that video (2614x1888 frames).
GROUND_TRUTH_CROP = (470, 1300, 430, 900)  # rows, then columns


def ground_truth_panel(video, height):
    """A crop of the release video's ground-truth half, scaled to `height`.
    The pose differs from the rest mesh; this is the colour reference."""
    if video is None or not Path(video).exists():
        return None
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) // 3)
    ok, frame = capture.read()
    if not ok:
        return None
    r0, r1, c0, c1 = GROUND_TRUTH_CROP
    crop = frame[r0:r1, c0:c1]
    scale = height / crop.shape[0]
    return cv2.resize(crop, (int(round(crop.shape[1] * scale)), height), interpolation=cv2.INTER_AREA)


def time_forward(net, vertices, view, ambient, normal, repeats):
    net.forward(vertices, view, ambient=ambient, normal=normal)  # warm-up (compiles MPS kernels)
    samples = []
    for _ in range(repeats):
        net.forward(vertices, view, ambient=ambient, normal=normal)
        samples.append(net.last_timing["network"])
    return statistics.median(samples), min(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=MODELS_ROOT / "00152_outer")
    parser.add_argument("--out", type=Path, default=SCRATCH / "appearance_ab.png")
    parser.add_argument("--camera", default="0004", help="camera id in cameras.json")
    parser.add_argument("--scale", type=float, default=0.5, help="fraction of the 940x1280 capture resolution")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-cpu", action="store_true", help="skip the CPU timing (tens of seconds)")
    parser.add_argument("--ground-truth", type=Path, default=RELEASE_VIDEO[-1] if RELEASE_VIDEO else None,
                        help="release video whose ground-truth half is appended as a colour reference")
    args = parser.parse_args()

    size = (int(round(1280 * args.scale)), int(round(940 * args.scale)))
    view, intrinsics = scan_camera(args.checkpoint, args.camera, args.scale)

    t0 = time.perf_counter()
    renderer = GarmentRenderer(args.checkpoint)
    vertices = renderer.rest_vertices
    print(f"renderer loaded in {time.perf_counter() - t0:.1f} s ({len(renderer.local_xyz)} splats)")

    t0 = time.perf_counter()
    net = AppearanceNet(args.checkpoint, faces=renderer.faces)
    print(f"appearance net loaded in {time.perf_counter() - t0:.1f} s on {net.device}")

    t0 = time.perf_counter()
    ambient, normal = net.bake(vertices)
    cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    net.bake(vertices)
    print(f"AO + normal bake: {cold:.2f} s cold (open3d/scipy first use), {time.perf_counter() - t0:.2f} s warm "
          f"(AO mean over covered texels {ambient[net.binding.gaussian_mask].mean():.3f})")

    result = net.forward(vertices, view, ambient=ambient, normal=normal)
    print(f"offset |mean| {np.abs(result['offset']).mean():.4f} face-scale units, "
          f"shadow SH |mean| {np.abs(result['shadow_shs']).mean():.4f}, "
          f"DC shift mean {result['shadow_shs'][:, 0].mean(0).round(3)}")

    # 1. today's pipeline
    current = renderer.splats(vertices)
    assert len(current["xyz"]) == net.num_gs, "rest mesh should keep every splat"
    # 2. static SH at the barycentric anchor
    static_anchored = dict(current, xyz=net.static_xyz(vertices))
    # 3. shadow colours only
    shaded = merge(current, result)
    shaded["xyz"] = current["xyz"]
    # 4. the reference's full output
    reference = merge(current, result)

    panels = [
        ("1 static SH, face-centre anchor (today)", current),
        ("2 static SH, barycentric anchor", static_anchored),
        ("3 shadow_net colours, face-centre anchor", shaded),
        ("4 shadow_net colours + offsets, bary anchor", reference),
    ]
    images, alphas = [], []
    for text, splats in panels:
        t0 = time.perf_counter()
        image, alpha = render(splats, view, intrinsics, size)
        images.append(image)
        alphas.append(alpha)
        print(f"rendered '{text}' in {time.perf_counter() - t0:.2f} s "
              f"(covers {int((alpha > 0.01).sum())} px)")
    r0, r1, c0, c1 = crop_box(np.stack(alphas))
    labelled = [label(image[r0:r1, c0:c1], text.split(" ", 1)[0]) for image, (text, _) in zip(images, panels)]
    truth = ground_truth_panel(args.ground_truth, r1 - r0)
    if truth is not None:
        labelled.append(label(truth, "GT"))
        panels.append(("GT: a frame of the release video (colour reference only; different pose)", None))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.concatenate(labelled, axis=1))
    print(f"wrote {args.out} (panels, left to right):")
    for text, _ in panels:
        print(f"  {text}")

    # Timings, warm.
    median, best = time_forward(net, vertices, view, ambient, normal, args.repeats)
    print(f"shadow_net forward on {net.device} float32: median {median * 1000:.0f} ms, best {best * 1000:.0f} ms")
    if net.device.type == "mps":
        half = AppearanceNet(args.checkpoint, faces=renderer.faces, half=True)
        median, best = time_forward(half, vertices, view, ambient, normal, args.repeats)
        drift = np.abs(half.forward(vertices, view, ambient=ambient, normal=normal)["shadow_shs"] - result["shadow_shs"])
        print(f"shadow_net forward on mps float16: median {median * 1000:.0f} ms, best {best * 1000:.0f} ms "
              f"(vs float32: max {drift.max():.2e}, mean {drift.mean():.2e})")
    if not args.no_cpu:
        cpu = AppearanceNet(args.checkpoint, device="cpu", faces=renderer.faces)
        t0 = time.perf_counter()
        cpu_result = cpu.forward(vertices, view, ambient=ambient, normal=normal)
        print(f"shadow_net forward on cpu float32: {cpu.last_timing['network'] * 1000:.0f} ms "
              f"(whole forward {time.perf_counter() - t0:.1f} s); "
              f"mps vs cpu max diff {np.abs(cpu_result['shadow_shs'] - result['shadow_shs']).max():.2e}")
    print(f"torch {torch.__version__}, mps available: {torch.backends.mps.is_available()}")


if __name__ == "__main__":
    main()
