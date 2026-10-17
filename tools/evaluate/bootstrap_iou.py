"""Bootstrap intervals for the garment IoU of the live pipeline on the five release videos.

Reads the per-frame results of `vs_registration` (`docs/baseline/vs-registration-nlf.json`) and resamples
them two ways: frames within each video (frame noise only) and whole videos (which also counts the spread
between garments, so it is the wider and more honest interval for five videos). For the mean IoU of ours,
of the authors' registration and of their difference it writes the 95 percent interval.

    python -m tools.evaluate.bootstrap_iou     # writes docs/baseline/bootstrap-iou.json
"""
from __future__ import annotations

import json

import numpy as np

from tools.evaluate._common import write_json
from tryon import paths

SOURCE = paths.BASELINE / "vs-registration-nlf.json"
OUTPUT = paths.BASELINE / "bootstrap-iou.json"
RESAMPLES = 10000
SEED = 0


def _interval(samples: np.ndarray) -> list[float]:
    return [float(v) for v in np.percentile(samples, [2.5, 97.5])]


def bootstrap(videos: list[dict], resamples: int = RESAMPLES, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    ours = [np.array([f["ours"] for f in v["frames"]]) for v in videos]
    theirs = [np.array([f["theirs"] for f in v["frames"]]) for v in videos]
    means = {"ours": float(np.mean([o.mean() for o in ours])), "theirs": float(np.mean([t.mean() for t in theirs]))}
    frame_draws = {"ours": [], "theirs": []}
    for _ in range(resamples):
        picks = [rng.integers(0, len(o), len(o)) for o in ours]
        frame_draws["ours"].append(np.mean([o[p].mean() for o, p in zip(ours, picks)]))
        frame_draws["theirs"].append(np.mean([t[p].mean() for t, p in zip(theirs, picks)]))
    video_means = {"ours": np.array([o.mean() for o in ours]), "theirs": np.array([t.mean() for t in theirs])}
    video_draws = {"ours": [], "theirs": []}
    for _ in range(resamples):
        chosen = rng.integers(0, len(videos), len(videos))
        video_draws["ours"].append(video_means["ours"][chosen].mean())
        video_draws["theirs"].append(video_means["theirs"][chosen].mean())
    out = {"videos": len(videos), "frames": int(sum(len(o) for o in ours)), "resamples": resamples, "mean": means}
    for name, draws in (("frame_bootstrap", frame_draws), ("video_bootstrap", video_draws)):
        ours_draws, theirs_draws = np.array(draws["ours"]), np.array(draws["theirs"])
        out[name] = {"ours": _interval(ours_draws), "theirs": _interval(theirs_draws),
                     "gap": _interval(theirs_draws - ours_draws)}
    return out


def main():
    with open(SOURCE) as handle:
        videos = json.load(handle)
    result = bootstrap(videos)
    write_json(OUTPUT, result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
