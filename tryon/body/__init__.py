"""Who is in the frame and how they stand.

    fitter.py          SMPL-X body fitted to 2D keypoints, tracked frame to frame
    regressor.py       a 3D body straight from the image (NLF; Multi-HMR as fallback)
    nlf_worker.py      NLF in its own process (it needs a newer PyTorch)
    orientation.py     which way the person faces, from the keypoints
    orientation_data.py  synthetic training data for it
    smoothing.py       One Euro filter on the landmarks, jitter metrics
"""
