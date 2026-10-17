"""Bootstrap intervals for the garment IoU of the live pipeline on the five release videos.

Reads the per-frame results of `vs_registration` (`docs/baseline/vs-registration-nlf.json`) and resamples
them two ways: frames within each video (frame noise only) and whole videos (which also counts the spread
between garments, so it is the wider and more honest interval for five videos). For the mean IoU of ours,
of the authors' registration and of their difference it writes the 95 percent interval.

    python -m tools.evaluate.bootstrap_iou     # writes docs/baseline/bootstrap-iou.json

With `--paired` it instead bootstraps paired per-frame differences on the same frames: the authors' fit
minus ours, and NLF-S minus each other body source. Besides frames and whole videos it resamples blocks of
consecutive scored frames within each video, since neighbouring frames are alike.

    python -m tools.evaluate.bootstrap_iou --paired   # writes docs/baseline/paired-bootstrap-iou.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

from tools.evaluate._common import write_json
from tryon import paths

SOURCE = paths.BASELINE / "vs-registration-nlf.json"
OUTPUT = paths.BASELINE / "bootstrap-iou.json"
PAIRED_OUTPUT = paths.BASELINE / "paired-bootstrap-iou.json"
OTHER_BODY_SOURCES = {
    "multihmr_672S": "vs-registration-multihmr.json",
    "multihmr_672B": "vs-registration-multihmr-672B.json",
    "keypoint_fit": "vs-registration.json",
}
BLOCK = 5
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


def _frames_by_video(videos: list[dict]) -> dict:
    return {v["video"]: {f["frame"]: f for f in v["frames"]} for v in videos}


def paired_differences(first: dict, second: dict, key_first: str, key_second: str) -> list[np.ndarray]:
    """Per video, `first[key_first] - second[key_second]` on the frames both runs scored, in frame order."""
    diffs = []
    for video, frames in first.items():
        if video not in second:
            continue
        shared = sorted(set(frames) & set(second[video]))
        diffs.append(np.array([frames[f][key_first] - second[video][f][key_second] for f in shared]))
    return diffs


def _block_indices(rng, n: int, block: int) -> np.ndarray:
    block = min(block, n)
    starts = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def paired_bootstrap(diffs: list[np.ndarray], resamples: int = RESAMPLES, seed: int = SEED,
                     block: int = BLOCK) -> dict:
    """Mean of the per-video mean difference with 95 percent intervals over frames, blocks of `block`
    consecutive scored frames, and whole videos."""
    rng = np.random.default_rng(seed)
    video_means = np.array([d.mean() for d in diffs])
    frame_draws, block_draws, video_draws = [], [], []
    for _ in range(resamples):
        frame_draws.append(np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in diffs]))
        block_draws.append(np.mean([d[_block_indices(rng, len(d), block)].mean() for d in diffs]))
        video_draws.append(video_means[rng.integers(0, len(diffs), len(diffs))].mean())
    return {"frames": int(sum(len(d) for d in diffs)), "mean_difference": float(video_means.mean()),
            "per_video": [float(m) for m in video_means], "frame_bootstrap": _interval(np.array(frame_draws)),
            "block_bootstrap": _interval(np.array(block_draws)), "video_bootstrap": _interval(np.array(video_draws))}


def paired_main():
    with open(SOURCE) as handle:
        nlf = _frames_by_video(json.load(handle))
    result = {"resamples": RESAMPLES, "block_frames": BLOCK, "source": SOURCE.name,
              "authors_minus_ours": paired_bootstrap(paired_differences(nlf, nlf, "theirs", "ours"))}
    for name, filename in OTHER_BODY_SOURCES.items():
        with open(paths.BASELINE / filename) as handle:
            other = _frames_by_video(json.load(handle))
        result[f"nlf_s_minus_{name}"] = paired_bootstrap(paired_differences(nlf, other, "ours", "ours"))
    write_json(PAIRED_OUTPUT, result)
    print(json.dumps(result, indent=1))


def main():
    if "--paired" in sys.argv[1:]:
        paired_main()
        return
    with open(SOURCE) as handle:
        videos = json.load(handle)
    result = bootstrap(videos)
    write_json(OUTPUT, result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
