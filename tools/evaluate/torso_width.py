"""Where the loose-shirt torso is too wide: the width profile by height band and pose, and a correction for it.

`record` is `open_front.record` (one live NLF pass: the fitted bodies and the real shirt masks) plus what the
width study needs from the same pass: the person mask (SCHP), the fitted body's silhouette and the MediaPipe
landmarks of every scored frame. `profile` replays the recorded bodies under any class-attribute variant and
reports, per normalised height band of the REAL shirt (collar row .. hem row) the width of our mask over the
real one, torso columns only, split by pose class. `fit` / `loo` fit and validate a correction.

    .venv/bin/python -m tools.evaluate.torso_width record 00152_outer
    .venv/bin/python -m tools.evaluate.torso_width profile 00152_outer [--set Class.ATTR=v ...] [--out j.json]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools.evaluate import open_front
from tools.evaluate._common import write_json

BANDS = 12
CACHE = open_front.CACHE
LOOSE = ("00152_outer", "10006_outer", "00176_outer", "00136_outer", "00122_outer")
CLOSED = ("00170_upper", "00134_upper", "00176_upper")
SHOULDER, WRIST, HIP = (11, 12), (15, 16), (23, 24)
RAISED_MARGIN = 0.0  # a wrist above the shoulder line


class _Tee(list):
    """The CAPTURE list `open_front.record` made, plus the extras of the same frame."""

    def __init__(self, inner, state, extras):
        super().__init__()
        self.inner, self.state, self.extras = inner, state, extras

    def append(self, item):
        self.inner.append(item)
        self.extras.append((item[1], self.state.get("labels"), self.state.get("body")))


def record(name, every=5, cache=CACHE):
    from tools.evaluate import vs_registration as vr
    from tryon.removal import background
    from tryon.removal.parsing_schp import SCHPParser

    state, extras = {}, []
    original_evaluate, original_labels, original_sil = vr.evaluate, SCHPParser.labels, background.body_silhouette

    def labels(self, image, *args, **kwargs):
        out = original_labels(self, image, *args, **kwargs)
        if image.shape[0] == 870:
            state["labels"] = (np.asarray(out) != 0).copy()
        return out

    def silhouette(*args, **kwargs):
        out = original_sil(*args, **kwargs)
        state["body"] = np.asarray(out).astype(bool).copy()
        return out

    def evaluate(*args, **kwargs):
        vr.CAPTURE = _Tee(vr.CAPTURE, state, extras)
        return original_evaluate(*args, **kwargs)

    vr.evaluate, SCHPParser.labels, background.body_silhouette = evaluate, labels, silhouette
    try:
        open_front.record(name, every, cache)
    finally:
        vr.evaluate, SCHPParser.labels, background.body_silhouette = original_evaluate, original_labels, original_sil
    person = np.stack([e[1] if e[1] is not None else np.zeros_like(extras[0][1]) for e in extras])
    body = np.stack([e[2] if e[2] is not None else np.zeros_like(person[0]) for e in extras])
    raw = np.stack([np.zeros((33, 4)) if e[0] is None else np.asarray(e[0])[:33, :4] if np.asarray(e[0]).ndim == 2 else np.zeros((33, 4)) for e in extras])
    np.savez_compressed(Path(cache) / f"{name}_extra.npz", person=np.packbits(person, axis=-1), body=np.packbits(body, axis=-1),
                        width=person.shape[-1], raw=raw)
    print(name, "extras", person.shape, raw.shape, flush=True)


# ------------------------------------------------------------------ profile

WINDOW = 0.65  # torso columns: within this many shoulder widths of the shoulder midpoint


def load_extra(name, cache=CACHE):
    data = np.load(Path(cache) / f"{name}_extra.npz")
    width = int(data["width"])
    return {"person": np.unpackbits(data["person"], axis=-1)[..., :width].astype(bool),
            "body": np.unpackbits(data["body"], axis=-1)[..., :width].astype(bool), "raw": data["raw"]}


def pose_class(raw):
    """"down": both wrists below the hip line; "up": a wrist above the shoulder line; else "mid". Pixel y grows downward."""
    if raw is None or not np.any(raw):
        return "none"
    shoulder, hip = raw[list(SHOULDER), 1].mean(), raw[list(HIP), 1].mean()
    wrist = raw[list(WRIST), 1]
    if wrist.min() < shoulder - RAISED_MARGIN:
        return "up"
    return "down" if wrist.min() > hip else "mid"


def torso_window(raw, shape):
    """Column range [lo, hi) of the torso: the shoulder midpoint +- WINDOW shoulder widths."""
    centre = raw[list(SHOULDER), 0].mean()
    half = WINDOW * abs(raw[SHOULDER[0], 0] - raw[SHOULDER[1], 0])
    return max(int(centre - half), 0), min(int(centre + half), shape[1])


def band_widths(masks, truth, window, bands=BANDS):
    """Per band of the real shirt (collar row .. hem row), for each mask, inside the window: the mean row EXTENT
    (outermost pixel to outermost pixel, px; robust to the open front, which the SCHP shirt mask leaves empty) and
    the mean row pixel count. Returns (extent, count), each (masks, bands)."""
    rows = np.nonzero(truth.any(axis=1))[0]
    if len(rows) == 0:
        return None
    edges = np.linspace(rows.min(), rows.max() + 1, bands + 1).astype(int)
    lo, hi = window
    extent, count = np.full((len(masks), bands), np.nan), np.full((len(masks), bands), np.nan)
    for k, mask in enumerate(masks):
        inside = mask[:, lo:hi]
        any_row = inside.any(axis=1)
        first, last = inside.argmax(axis=1), inside.shape[1] - 1 - inside[:, ::-1].argmax(axis=1)
        span = np.where(any_row, last - first + 1, 0)
        total = inside.sum(axis=1)
        for b in range(bands):
            if edges[b + 1] > edges[b]:
                extent[k, b], count[k, b] = span[edges[b]:edges[b + 1]].mean(), total[edges[b]:edges[b + 1]].mean()
    return extent, count


def profile(name, assignments=(), cache=CACHE):
    """Per scored frame: pose class and the band widths of ours, the real shirt, the person and the fitted body."""
    data = open_front.load_record(name, cache)
    extra = load_extra(name, cache)
    frames = [int(t) for t in data["t"]]
    kept = dict.fromkeys(frames)
    result = open_front.replay(name, 5, False, assignments, cache, keep_rgba=kept)
    rows = []
    for k, frame in enumerate(frames):
        rgba = kept[frame]
        raw = extra["raw"][k]
        if rgba is None or not np.any(raw):
            continue
        ours = rgba[..., 3] > open_front.ALPHA_MASK
        measured = band_widths([ours, data["truth"][k], extra["person"][k], extra["body"][k]], data["truth"][k], torso_window(raw, ours.shape))
        if measured is not None:
            (extent, count) = measured
            rows.append({"frame": frame, "pose": pose_class(raw), "ours": extent[0], "truth": extent[1], "person": extent[2], "body": extent[3],
                         "ours_n": count[0], "truth_n": count[1]})
    return {"video": name, "rows": rows, "summary": result["summary"], "frames": result["frames"]}


def ratio_table(rows, key="ours", over="truth", pose=None):
    """Per band sum(key) / sum(over) over the rows of the pose class."""
    chosen = [r for r in rows if pose is None or r["pose"] == pose]
    if not chosen:
        return np.full(BANDS, np.nan)
    return np.nansum([r[key] for r in chosen], axis=0) / np.maximum(np.nansum([r[over] for r in chosen], axis=0), 1e-9)


def table(tags, directory="/tmp/gg"):
    """Rows of the `profile --out directory/tw-<tag>-<video>.json` results: IoU, extra / missed, torso width
    ratio (pixel count), inside-body share, and the band extent ratio of the lower four bands."""
    import json

    lines = ["tag | video | iou | extra | missed | width_ratio(count) | inside | extent ratio bands 9-12 (all / up)"]
    for tag in tags:
        for path in sorted(Path(directory).glob(f"tw-{tag}-*.json")):
            data = json.loads(path.read_text())
            summary = data["summary"]
            rows = [{**r, **{k: np.asarray(r[k]) for k in ("ours", "truth")}} for r in data["rows"]]
            low = ratio_table(rows)[8:]
            low_up = ratio_table(rows, pose="up")[8:]
            lines.append(f"{tag} | {data['video']} | {summary['iou']:.4f} | {summary.get('split_extra', float('nan')):.3f} | {summary.get('split_missed', float('nan')):.3f} | "
                         f"{summary.get('split_torso_width_ratio', float('nan')):.3f} | {summary.get('inside_share', float('nan')):.4f} | "
                         f"{' '.join(f'{x:.2f}' for x in low)} / {' '.join(f'{x:.2f}' for x in low_up)}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["record", "profile", "table"])
    parser.add_argument("video")
    parser.add_argument("--every", type=int, default=5)
    parser.add_argument("--out", default=None)
    parser.add_argument("--set", dest="assignments", action="append", default=[])
    args = parser.parse_args()
    if args.command == "record":
        record(args.video, args.every)
        return
    if args.command == "table":
        print(table(args.video.split(",")))
        return
    result = profile(args.video, args.assignments)
    print(args.video, "IoU", round(result["summary"]["iou"], 4), "torso width", round(result["summary"].get("split_torso_width_ratio", float("nan")), 3))
    for pose in (None, "down", "mid", "up"):
        count = sum(1 for r in result["rows"] if pose is None or r["pose"] == pose)
        print(f"{pose or 'all':5s} n={count:2d} ours/real", " ".join(f"{x:.2f}" for x in ratio_table(result["rows"], pose=pose)),
              "| body/real", " ".join(f"{x:.2f}" for x in ratio_table(result["rows"], "body", pose=pose)),
              "| count ours/real", " ".join(f"{x:.2f}" for x in ratio_table(result["rows"], "ours_n", "truth_n", pose=pose)))
    if args.out:
        write_json(args.out, {"video": args.video, "assignments": args.assignments, "summary": result["summary"],
                              "rows": [{**r, **{k: np.asarray(r[k]).tolist() for k in ("ours", "truth", "person", "body", "ours_n", "truth_n")}} for r in result["rows"]]})


if __name__ == "__main__":
    main()
