"""Live 3D virtual try-on: a captured Gaussian-splat garment worn by the
person in a webcam video, on a laptop.

One frame flows through the sub-packages in this order (see
docs/ARCHITECTURE.md for the full walk-through):

    body/       who is in the frame and how they stand: 2D keypoints,
                a 3D SMPL-X body (NLF regressor or keypoint fit), facing,
                landmark smoothing
    garment/    the garment on that body: skinning from the scan pose,
                cloth springs and body collision, the splats placed on the
                posed mesh, their shading
    rendering/  the splats drawn into an image (OpenGL; a CPU reference)
    removal/    the wearer's own clothing erased: clothing parser, fill,
                learned background
    app/        the live loop that runs all of it per frame, the display
                modes, the command line

Run it with `python -m tryon.app` (or the `./live-tryon` / `./live-tryon`
launchers).
"""

import os

# numpy's OpenBLAS threads and torch's own pool contended for the M1's
# cores: a (121k, 3) @ (3, 3) projection took 15-18 ms with torch loaded
# and 1.7 ms with one BLAS thread; an SMPL-X forward plus that projection
# 37 -> 7 ms. Set before numpy is first imported (`python -m tryon.app`
# imports this package first); an explicit setting still wins.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
