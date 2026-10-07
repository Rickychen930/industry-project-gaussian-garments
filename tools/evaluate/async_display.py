"""Would an asynchronous display keep the garment on the body?

Live, the full pipeline takes ~0.65 s a frame; an asynchronous display
shows every camera frame and draws the newest FINISHED garment layer on it.
That layer belongs to an earlier frame. This scores, on the release
videos against the real shirt (the same truth as
`tools/evaluate/vs_registration.py`, whose per-frame garment it reuses),
what the viewer would see at every frame t:

  sync   the garment rendered for frame t itself (the upper bound: what a
         pipeline as fast as the camera would draw)
  hold   the newest finished layer as it is
  warp   the same layer carried to frame t with the torso's similarity
         (`tryon.app.layer_warp.torso_transform`)

with keyframes every `period` frames whose layers finish `period` frames
later (a pipeline taking `period` frames per frame; the release videos run
at 25 fps, so 16 frames is the live pipeline's ~0.65 s).

    .venv/bin/python -m tools.evaluate.async_display 00152_outer,00170_upper [--frames 150] [--periods 8,16] [--regressor nlf]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools.evaluate._common import iou, write_json


def simulate(captured, period):
    from tryon.app.layer_warp import torso_layer

    by_t = {t: (kp, ours, shirt) for t, kp, ours, shirt in captured}
    rows = []
    for t in sorted(by_t):
        # the newest keyframe whose layer has finished by frame t
        k = ((t - period) // period) * period
        if k < 0 or k not in by_t:
            continue
        kp_t, ours_t, shirt_t = by_t[t]
        kp_k, ours_k, _ = by_t[k]
        warped = torso_layer(ours_k.astype(np.uint8) * 255, kp_k, kp_t) > 127
        rows.append({"t": t, "sync": iou(ours_t, shirt_t), "hold": iou(ours_k, shirt_t), "warp": iou(warped, shirt_t)})
    return {k: float(np.mean([r[k] for r in rows])) for k in ("sync", "hold", "warp")} | {"frames": len(rows)}


def main():
    import tools.evaluate.vs_registration as evr

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", default="00152_outer")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--periods", default="8,16")
    parser.add_argument("--regressor", default="nlf")
    parser.add_argument("--out", default=None)
    parser.add_argument("--save", default=None, help="directory: save each video's captured frames (npz) for inspection")
    args = parser.parse_args()
    evr.REGRESSOR = args.regressor or None
    report = {}
    for name in args.videos.split(","):
        evr.CAPTURE = []
        evr.evaluate(name, 1, args.frames, None)
        if args.save:
            Path(args.save).mkdir(parents=True, exist_ok=True)
            t, kp, ours, shirt = zip(*evr.CAPTURE)
            np.savez_compressed(Path(args.save) / f"{name}.npz", t=np.array(t), kp=np.array([k if k is not None else np.zeros((33, 3)) for k in kp]),
                                ours=np.array(ours), shirt=np.array(shirt))
        report[name] = {str(p): simulate(evr.CAPTURE, int(p)) for p in args.periods.split(",")}
        for p, r in report[name].items():
            print(f"{name} period {p}: sync {r['sync']:.3f} | hold {r['hold']:.3f} | warp {r['warp']:.3f}  ({r['frames']} frames)", flush=True)
    if args.out:
        write_json(args.out, report)


if __name__ == "__main__":
    main()
