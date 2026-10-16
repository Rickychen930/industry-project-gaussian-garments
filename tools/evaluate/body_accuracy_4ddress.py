"""Body accuracy of the live path on the full 4D-DRESS captures.

Same protocol as `fit_audit.body_accuracy` (MediaPipe -> NLF-S ->
`BodyFitter.fit_from_regression`, PA-MPJPE of the 12 JOINT_MAP joints against
the capture's SMPL-X), over a few frames of every take under
`~/tryon-data/4d-dress/<subject>/<Inner|Outer>/<Take>/`, all four cameras,
reported per subject and layer. The raw capture layout is exposed to
`real_fit` through a directory of symlinks outside the repository.

    .venv/bin/python -m tools.evaluate.body_accuracy_4ddress [--frames 3] [--limit N] [--control DIR]

`--control DIR` also scores the annotated sample in the same run for comparison.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np

from tools.evaluate._common import write_json
from tryon.paths import BASELINE, DATA_ROOT

SOURCE = DATA_ROOT / "4d-dress"
VIEW_ROOT = SOURCE / "_view"
OUTPUT = BASELINE / "body-accuracy-4ddress.json"
CAMERAS = ("0004", "0028", "0052", "0076")


def _frame_ids(take_dir: Path) -> list[str]:
    """Frames that have both a ground-truth SMPL-X file and an image from every camera."""
    ids = []
    for pkl in sorted((take_dir / "SMPLX").glob("mesh-f*_smplx.pkl")):
        number = pkl.stem.split("-f")[1].split("_")[0]
        if all((take_dir / "Capture" / cam / "images" / f"capture-f{number}.png").exists() for cam in CAMERAS):
            ids.append(number)
    return ids


def _spread(items: list[str], count: int) -> list[str]:
    """`count` items at the centres of equal slices of the list."""
    if len(items) <= count:
        return items
    return [items[int((i + 0.5) * len(items) / count)] for i in range(count)]


def build_view(frames_per_take: int, limit: int = 0) -> list[dict]:
    """Symlink the chosen frames of every extracted take into `VIEW_ROOT`, annotated-sample layout.

    Returns one record per (take, camera, frame). The view directory is rebuilt from scratch."""
    if VIEW_ROOT.exists():
        shutil.rmtree(VIEW_ROOT)
    records = []
    for subject in sorted(p for p in SOURCE.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for layer in sorted(p for p in subject.iterdir() if p.is_dir()):
            for take_dir in sorted(layer.glob("Take*"), key=lambda p: int(p.name[4:])):
                if not (take_dir / "Capture" / "cameras.pkl").exists():
                    continue
                name = f"{subject.name}_{layer.name}_{take_dir.name}"
                view = VIEW_ROOT / name
                (view / "smplx").mkdir(parents=True)
                with open(take_dir / "Capture" / "cameras.pkl", "rb") as f:
                    cameras = pickle.load(f)
                (view / "cameras.json").write_text(json.dumps(
                    {cam: {"intrinsics": np.asarray(c["intrinsics"]).tolist(),
                           "extrinsics": np.asarray(c["extrinsics"]).tolist()} for cam, c in cameras.items()}))
                for number in _spread(_frame_ids(take_dir), frames_per_take):
                    (view / "smplx" / f"{number}.pkl").symlink_to(take_dir / "SMPLX" / f"mesh-f{number}_smplx.pkl")
                    for cam in CAMERAS:
                        folder = view / cam / "rgb_images"
                        folder.mkdir(parents=True, exist_ok=True)
                        (folder / f"{number}.png").symlink_to(take_dir / "Capture" / cam / "images" / f"capture-f{number}.png")
                        records.append({"take": name, "subject": subject.name, "layer": layer.name,
                                        "cam": cam, "frame": number})
    return records[:limit] if limit else records


def score(data_root: Path, records: list[dict]) -> list[dict]:
    """PA-MPJPE (mm) of every view, `live_fov` (as live) and `true_intrinsics` (calibrated camera)."""
    import torch

    from tools.evaluate import real_fit
    from tools.evaluate.fit_audit import procrustes_mpjpe, stop_regressor
    from tryon.body.fitter import BodyFitter
    from tryon.body.landmarks import JOINT_MAP
    from tryon.body.regressor import NLF

    real_fit.DATA = data_root
    order = sorted(JOINT_MAP)
    nlf = NLF()
    rows = []
    for record in records:
        take, cam, frame = record["take"], record["cam"], record["frame"]
        image = cv2.imread(str(data_root / take / cam / "rgb_images" / f"{frame}.png"))
        row = {**record, "view": f"{take}/{cam}/{frame}"}
        if image is None:
            rows.append({**row, "detected": False, "reason": "image missing"})
            continue
        points = real_fit.landmarks_from_mediapipe(image)
        if points is None:
            rows.append({**row, "detected": False, "reason": "no person detected"})
            continue
        truth = real_fit.ground_truth_joints(take, frame)[[JOINT_MAP[i] for i in order]]
        intrinsics, _ = real_fit.load_camera(take, cam)
        for label, k in (("live_fov", None), ("true_intrinsics", intrinsics.astype(np.float32))):
            prediction = nlf.predict(image, intrinsics=k, keypoints=points)
            fitter = BodyFitter()
            fitter.fit_from_regression(points, prediction)
            with torch.no_grad():
                row[f"mpjpe_mm_{label}"] = 1000 * procrustes_mpjpe(fitter.joints().numpy(), truth)
        rows.append({**row, "detected": True})
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    stop_regressor(nlf)
    return rows


def summarise(rows: list[dict]) -> dict:
    found = [r for r in rows if r["detected"]]

    def mean_of(selection, label):
        values = [r[f"mpjpe_mm_{label}"] for r in selection]
        return float(np.mean(values)) if values else None

    out = {"views": len(rows), "detected": len(found)}
    for label in ("live_fov", "true_intrinsics"):
        out[f"pa_mpjpe_mm_{label}"] = mean_of(found, label)
        out[f"by_layer_mm_{label}"] = {
            layer: mean_of([r for r in found if r.get("layer") == layer], label)
            for layer in sorted({r.get("layer") for r in found if r.get("layer")})}
        out[f"by_subject_mm_{label}"] = {
            subject: mean_of([r for r in found if r.get("subject") == subject], label)
            for subject in sorted({r.get("subject") for r in found if r.get("subject")})}
        out[f"by_camera_mm_{label}"] = {
            cam: mean_of([r for r in found if r["cam"] == cam], label) for cam in CAMERAS}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=3, help="frames per take (evenly spaced)")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N views (smoke test)")
    parser.add_argument("--control", type=Path, help="annotated-sample directory to score with the same code")
    args = parser.parse_args()

    result: dict = {"protocol": "MediaPipe(static, complexity 2) -> NLF-S (box from MediaPipe) -> "
                                "fit_from_regression -> PA-MPJPE of the 12 JOINT_MAP joints (fit_audit A1)"}
    if args.control:
        control = [{"take": take, "subject": "sample", "layer": "annotated", "cam": cam, "frame": frame}
                   for take in ("Take1", "Take10") for cam in CAMERAS for frame in ("00001", "00002")]
        rows = score(args.control, control)
        result["control"] = {"source": str(args.control), **summarise(rows), "rows": rows}
        print("CONTROL", {k: v for k, v in result["control"].items() if k != "rows"}, flush=True)

    records = build_view(args.frames, args.limit)
    rows = score(VIEW_ROOT, records)
    result["new"] = {**summarise(rows), "frames_per_take": args.frames, "rows": rows}
    write_json(OUTPUT, result)
    print("NEW", {k: v for k, v in result["new"].items() if k != "rows"})
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
