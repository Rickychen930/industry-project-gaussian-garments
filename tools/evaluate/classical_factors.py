"""Limits of classical matching on garments: texture and shape, light, blur, noise and viewpoint.

The registered project asked for an analysis of the limits of classical algorithms on garments with
different textures, shapes, light, backgrounds and views. On the four 4D-DRESS cameras this tool measures how
many SIFT matches survive RANSAC (verified matches) between two views of the person when

- the garment changes: three subject and layer combinations (00134 inner, 00136 inner, 00136 outer), each
  with its own texture and shape;
- the light changes: both images darkened or brightened by a gamma curve;
- the image degrades: Gaussian blur or noise;
- the background changes: whole image against person only (the 4D-DRESS mask);
- the view changes: same camera five frames later, cameras 90 degrees apart, cameras 180 degrees apart.

    python -m tools.evaluate.classical_factors     # writes docs/baseline/classical-factors.json
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from tools.evaluate._common import write_json
from tools.evaluate.classical_matching import CAMERAS, _frame_path, _match
from tryon import paths

OUTPUT = paths.BASELINE / "classical-factors.json"
ADJACENT = {("0004", "0028"), ("0028", "0052"), ("0052", "0076"), ("0004", "0076")}
GROUPS = (("00134", "Inner"), ("00136", "Inner"), ("00136", "Outer"))
CONTROL_STEP = 5
SAME_CAMERA = "same camera, 5 frames later"


def _gamma(gray: np.ndarray, value: float) -> np.ndarray:
    table = (255 * (np.arange(256) / 255) ** value).astype(np.uint8)
    return cv2.LUT(gray, table)


def _noise(gray: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.clip(gray + rng.normal(0, 10, gray.shape), 0, 255).astype(np.uint8)


PERTURBATIONS = {
    "none": lambda g: g,
    "dark (gamma 2.2)": lambda g: _gamma(g, 2.2),
    "bright (gamma 0.45)": lambda g: _gamma(g, 0.45),
    "blur (sigma 3)": lambda g: cv2.GaussianBlur(g, (0, 0), 3),
    "noise (sigma 10)": _noise,
}


def _features(capture: Path, camera: str, index: int, perturb, masked: bool):
    gray = cv2.imread(str(_frame_path(capture, camera, index, "images")), cv2.IMREAD_GRAYSCALE)
    mask = None
    if masked:
        mask = (cv2.imread(str(_frame_path(capture, camera, index, "masks")), cv2.IMREAD_GRAYSCALE) > 0)
        mask = mask.astype(np.uint8) * 255
    return cv2.SIFT_create().detectAndCompute(perturb(gray), mask)


def evaluate(takes_per_group: int, index: int) -> list[dict]:
    rows = []
    for subject, layer in GROUPS:
        takes = sorted((paths.DATA_ROOT / "4d-dress" / subject / layer).glob("Take*"))[:takes_per_group]
        for take in takes:
            capture = take / "Capture"
            base = {"group": f"{subject} {layer.lower()}", "take": take.name}
            for name, perturb in PERTURBATIONS.items():
                for masked in (True, False):
                    region = "person" if masked else "whole image"
                    feats = {c: _features(capture, c, index, perturb, masked) for c in CAMERAS}
                    for a, b in combinations(CAMERAS, 2):
                        view = "adjacent" if (a, b) in ADJACENT else "opposite"
                        rows.append({**base, "factor": name, "region": region, "view": view,
                                     "inliers": _match(feats[a], feats[b])[3]})
            for masked in (True, False):
                region = "person" if masked else "whole image"
                for cam in CAMERAS:
                    now = _features(capture, cam, index, PERTURBATIONS["none"], masked)
                    later = _features(capture, cam, index + CONTROL_STEP, PERTURBATIONS["none"], masked)
                    rows.append({**base, "factor": "none", "region": region, "view": SAME_CAMERA,
                                 "inliers": _match(now, later)[3]})
    return rows


def _median(rows: list[dict], select) -> dict | None:
    values = [r["inliers"] for r in rows if select(r)]
    return {"n": len(values), "median_inliers": float(np.median(values))} if values else None


def summarise(rows: list[dict]) -> dict:
    wide = [r for r in rows if r["view"] != SAME_CAMERA]
    plain = [r for r in wide if r["factor"] == "none"]
    return {
        "by_garment": {g: _median(plain, lambda r, g=g: r["group"] == g and r["region"] == "person")
                       for g in sorted({r["group"] for r in rows})},
        "by_light_and_degradation": {f: _median(wide, lambda r, f=f: r["factor"] == f and r["region"] == "person")
                                     for f in PERTURBATIONS},
        "by_region": {reg: _median(plain, lambda r, reg=reg: r["region"] == reg) for reg in ("person", "whole image")},
        "by_view": {v: _median([r for r in rows if r["factor"] == "none"],
                               lambda r, v=v: r["view"] == v and r["region"] == "person")
                    for v in (SAME_CAMERA, "adjacent", "opposite")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--takes", type=int, default=3, help="takes per subject and layer")
    parser.add_argument("--frame-index", type=int, default=40)
    parser.add_argument("--out", default=str(OUTPUT))
    args = parser.parse_args()
    rows = evaluate(args.takes, args.frame_index)
    result = {"cameras": list(CAMERAS), "frame_index": args.frame_index, "rows": rows, "summary": summarise(rows)}
    write_json(Path(args.out), result)
    print(json.dumps(result["summary"], indent=1))


if __name__ == "__main__":
    main()
