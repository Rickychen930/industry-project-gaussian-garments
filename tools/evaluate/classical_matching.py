"""Classical multi-view matching on the 4D-DRESS capture: SIFT with a ratio test and RANSAC,
then COLMAP on the four cameras of one frame.

The registered project planned a classical pipeline (feature matching, camera motion,
triangulation). This tool measures its first steps on the only multi-view data we have: four
studio cameras about 90 degrees apart. For each take it reports, per camera pair, the SIFT
features, the ratio-test matches and the RANSAC inliers of a fundamental matrix, and the
same numbers for the same camera a few frames later as a control. COLMAP is then run on the
four images of one frame and the number of registered images is counted.

    python -m tools.evaluate.classical_matching     # writes docs/baseline/classical-matching.json
"""
import argparse
import json
import subprocess
import tempfile
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from tryon import paths

CAMERAS = ("0004", "0028", "0052", "0076")
RATIO = 0.8
CONTROL_STEP = 5  # frames between the two control images


def _frame_path(capture: Path, camera: str, index: int, kind: str) -> Path:
    return sorted((capture / camera / kind).glob("*.png"))[index]


def _sift(image_path: Path, mask_path: Path | None):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    mask = None
    if mask_path is not None:
        mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8) * 255
    return cv2.SIFT_create().detectAndCompute(image, mask)


def _match(a, b):
    """(features in a, features in b, ratio-test matches, RANSAC inliers)."""
    (kp_a, des_a), (kp_b, des_b) = a, b
    if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return len(kp_a), len(kp_b), 0, 0
    pairs = cv2.BFMatcher().knnMatch(des_a, des_b, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    if len(good) < 8:
        return len(kp_a), len(kp_b), len(good), 0
    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])
    _, inliers = cv2.findFundamentalMat(pts_a, pts_b, cv2.USAC_MAGSAC, 1.0, 0.999)
    return len(kp_a), len(kp_b), len(good), int(inliers.sum()) if inliers is not None else 0


def _colmap_registered(images: list[Path]) -> int:
    """Images COLMAP registers in one reconstruction of `images` (0 if it builds none)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "images").mkdir()
        for path in images:
            (root / "images" / f"{path.parent.parent.name}.png").write_bytes(path.read_bytes())
        db, sparse = root / "db.db", root / "sparse"
        sparse.mkdir()

        def run(*args):
            return subprocess.run(args, capture_output=True, text=True, check=False)

        run("colmap", "feature_extractor", "--database_path", str(db), "--image_path", str(root / "images"),
            "--FeatureExtraction.use_gpu", "0")
        run("colmap", "exhaustive_matcher", "--database_path", str(db), "--FeatureMatching.use_gpu", "0")
        run("colmap", "mapper", "--database_path", str(db), "--image_path", str(root / "images"),
            "--output_path", str(sparse))
        models = sorted(p for p in sparse.iterdir() if p.is_dir())
        if not models:
            return 0
        out = run("colmap", "model_analyzer", "--path", str(models[0]))
        for line in (out.stdout + out.stderr).splitlines():
            if "Registered images" in line:
                return int(line.split(":")[-1].strip())
        return 0


def evaluate(takes: list[Path], frame_index: int) -> dict:
    rows = []
    for take in takes:
        capture = take / "Capture"
        name = f"{take.parent.parent.name}/{take.parent.name}/{take.name}"
        for masked in (False, True):
            kind_mask = "masks" if masked else None

            def features(cam, index):
                mask = _frame_path(capture, cam, index, kind_mask) if masked else None
                return _sift(_frame_path(capture, cam, index, "images"), mask)

            now = {cam: features(cam, frame_index) for cam in CAMERAS}
            for a, b in combinations(CAMERAS, 2):
                n_a, n_b, good, inl = _match(now[a], now[b])
                rows.append({"take": name, "pair": f"{a}-{b}", "masked": masked, "kind": "views",
                             "features": [n_a, n_b], "ratio_matches": good, "inliers": inl})
            for cam in CAMERAS:
                n_a, n_b, good, inl = _match(now[cam], features(cam, frame_index + CONTROL_STEP))
                rows.append({"take": name, "pair": cam, "masked": masked, "kind": "control",
                             "features": [n_a, n_b], "ratio_matches": good, "inliers": inl})
        registered = _colmap_registered([_frame_path(capture, cam, frame_index, "images") for cam in CAMERAS])
        rows.append({"take": name, "kind": "colmap", "registered_images_of_4": registered})
    return {"cameras": list(CAMERAS), "ratio": RATIO, "frame_index": frame_index,
            "control_step": CONTROL_STEP, "rows": rows}


def summarise(result: dict) -> dict:
    out = {}
    for kind in ("views", "control"):
        for masked in (False, True):
            sel = [r for r in result["rows"] if r["kind"] == kind and r.get("masked") == masked]
            if sel:
                out[f"{kind}_{'masked' if masked else 'full'}"] = {
                    "pairs": len(sel),
                    "median_features": float(np.median([min(r["features"]) for r in sel])),
                    "median_ratio_matches": float(np.median([r["ratio_matches"] for r in sel])),
                    "median_inliers": float(np.median([r["inliers"] for r in sel])),
                }
    out["colmap_registered_of_4"] = [r["registered_images_of_4"] for r in result["rows"] if r["kind"] == "colmap"]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--takes", type=int, default=3, help="takes per subject")
    parser.add_argument("--frame-index", type=int, default=10)
    parser.add_argument("--out", default=str(paths.BASELINE / "classical-matching.json"))
    args = parser.parse_args()
    takes = []
    for subject in ("00134", "00136"):
        takes += sorted((paths.DATA_ROOT / "4d-dress" / subject / "Inner").glob("Take*"))[: args.takes]
    result = evaluate(takes, args.frame_index)
    result["summary"] = summarise(result)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result["summary"], indent=1))


if __name__ == "__main__":
    main()
