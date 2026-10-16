"""Every garment's shading students against the teacher, on the training tool's
fixed held-out poses (unseen trajectory frames, turned -60..60 degrees).

Per garment: PSNR against the render with the teacher's shading, for no shading,
the pre-7-Oct student (`stage3/shading_student_nomean.pt`, if kept) and the live
student, on its own held-out poses and on the two multi-garment motions no
student trains on (`shading_student.TEST_MOTIONS`).

    .venv/bin/python -m tools.evaluate.shading_students [00134_upper ...] [--poses 10] [--out docs/baseline/shading-students.json]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools.evaluate._common import write_json
from tryon.paths import BASELINE, GARMENTS



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("garments", nargs="*")
    parser.add_argument("--poses", type=int, default=10)
    parser.add_argument("--compare", nargs="*", default=["shading_student_nomean.pt"],
                        help="other student files in stage3/ to score beside the live one (those present)")
    parser.add_argument("--out", type=Path, default=BASELINE / "shading-students.json")
    args = parser.parse_args()

    import tools.train.shading_student as train
    from tryon.garment.appearance import AppearanceNet, merge
    from tryon.garment.renderer import GarmentRenderer
    from tryon.garment.shading_student import ShadingStudent
    from tryon.rendering.splat_gl import render_splats_gl

    def psnr(a, b):
        return float(10 * np.log10(1 / max(((a - b) ** 2).mean(), 1e-12)))

    garments = args.garments or sorted(p.name for p in GARMENTS.iterdir() if (p / "stage3" / "shading_student.pt").exists())
    unseen = []
    for path in sorted(train.MOTION_FITS.glob("*.npz")):
        if path.stem.split("-")[-1] in train.TEST_MOTIONS:  # every body shape the test motions were simulated on
            data = np.load(path)
            unseen += [{k: data[k][i] for k in ("betas", "body_pose", "global_orient", "offset")} for i in range(len(data["frames"]))]
    rows = {}
    for name in garments:
        checkpoint = GARMENTS / name
        renderer = GarmentRenderer(checkpoint, dynamics=True)
        poser = train.Poser(name, renderer)
        body_faces = np.asarray(poser.fitter.model.faces)
        teacher = AppearanceNet(checkpoint, faces=renderer.faces, half=True)
        students = {"live": ShadingStudent(checkpoint, faces=renderer.faces)}
        for other in args.compare:
            if (checkpoint / "stage3" / other).exists():
                students[Path(other).stem.replace("shading_student_", "")] = ShadingStudent(checkpoint, faces=renderer.faces, weights=checkpoint / "stage3" / other)

        def score(fits):
            """Mean PSNR against the teacher of no shading and each student,
            the poses turned by -YAW..YAW in order."""
            scores = {"static": [], **{k: [] for k in students}}
            rng = np.random.default_rng(123)
            for i, fit in enumerate(fits):
                yaw = -train.YAW + 2 * train.YAW * i / max(len(fits) - 1, 1)
                mesh, view, intrinsics, body = poser.pose(fit, yaw, 0.0, rng)
                _, result = train.teacher_target(teacher, mesh, view, body, body_faces)
                splats = renderer.splats(mesh)
                kept = renderer._kept_indices.copy()
                truth, _ = render_splats_gl(merge(splats, result, kept), view, intrinsics, train.SIZE)
                scores["static"].append(psnr(render_splats_gl(splats, view, intrinsics, train.SIZE)[0], truth))
                for key, student in students.items():
                    shaded = merge(splats, student.forward(mesh, view, body), kept)
                    scores[key].append(psnr(render_splats_gl(shaded, view, intrinsics, train.SIZE)[0], truth))
            return {k: float(np.mean(v)) for k, v in scores.items()}

        held_out = [poser.held[(i * len(poser.held)) // args.poses] for i in range(args.poses)]
        rows[name] = {"held_out": score(held_out), "unseen_motions": score(unseen)}
        print(name, " | ".join(f"{part}: " + "  ".join(f"{k} {v:.2f}" for k, v in r.items()) for part, r in rows[name].items()), flush=True)
        write_json(args.out, rows)

if __name__ == "__main__":
    main()
