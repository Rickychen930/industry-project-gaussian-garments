"""Can the live body path tell, per frame, when it is about to be badly wrong?

A residual correction does not help (docs/baseline/body-corrector.json: R^2 < 0
on the held-out subject) and ~15 % of views carry half the squared error. If
those were recognisable without ground truth, the live system could hold the last
good pose instead.

Label: view PA-MPJPE > `THRESHOLD_MM`. Features are live-only: MediaPipe joint
visibilities, the fit's reprojection error and NLF-vs-MediaPipe 2D gap (torso
lengths), and the fitted joints in the canonical torso frame.

Trained on one subject, applied to the other: reports the AUC and the mean error
of the views kept after dropping the riskiest fraction, against random dropping
and the oracle ceiling.

    .venv/bin/python -m tools.train.failure_detector
"""

from __future__ import annotations

import numpy as np

from tools.evaluate._common import write_json
from tools.train.body_corrector import (
    PARTS,
    _joint_slots,
    canonical_frame,
    pa_error_mm,
    to_canonical,
)
from tryon.paths import BASELINE

OUTPUT = BASELINE / "failure-detector.json"
THRESHOLD_MM = 60.0
KEEP_FRACTIONS = (0.95, 0.9, 0.85, 0.8, 0.7)
SCALARS = ("repro", "gap_mean", "gap_max")


def load() -> dict:
    parts = [np.load(p, allow_pickle=False) for p in sorted(PARTS.glob("*.npz"))]
    parts = [p for p in parts if len(p["p"]) and "repro" in p.files]
    if not parts:
        raise SystemExit(f"no v2 features under {PARTS}; run `body_corrector extract` first")

    def cat(key):
        return np.concatenate([part[key] for part in parts])

    def repeated(key):
        return np.concatenate([np.full(len(part["p"]), str(part[key])) for part in parts])

    return {"p": cat("p"), "g": cat("g"), "vis": cat("vis"), **{k: cat(k) for k in SCALARS},
            "subject": repeated("subject"), "layer": repeated("layer"), "take": repeated("take")}


def features(data: dict) -> tuple[np.ndarray, list[str]]:
    _, slot = _joint_slots()
    canonical = np.stack([to_canonical(p.astype(np.float64), canonical_frame(p.astype(np.float64), slot)).ravel()
                          for p in data["p"]])
    scalars = np.stack([data[k] for k in SCALARS], axis=1)
    vis = data["vis"]
    summary = np.stack([vis.min(1), vis.mean(1)], axis=1)
    names = list(SCALARS) + ["vis_min", "vis_mean"] + [f"vis_{i}" for i in range(vis.shape[1])] + \
        [f"canon_{i}" for i in range(canonical.shape[1])]
    return np.concatenate([scalars, summary, vis, canonical], axis=1).astype(np.float32), names


def selective_curve(risk: np.ndarray, error: np.ndarray) -> dict:
    """Mean error of the views kept when the riskiest (1 - keep) are dropped, three ways."""
    rng = np.random.default_rng(0)
    out = {}
    for keep in KEEP_FRACTIONS:
        n = int(round(keep * len(error)))
        by_risk = error[np.argsort(risk)[:n]].mean()
        oracle = np.sort(error)[:n].mean()
        random = float(np.mean([error[rng.permutation(len(error))[:n]].mean() for _ in range(20)]))
        out[f"keep_{keep:.2f}"] = {"detector": float(by_risk), "random": random, "oracle_ceiling": float(oracle)}
    return out


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    data = load()
    x, names = features(data)
    error = np.array([pa_error_mm(p.astype(np.float64), g.astype(np.float64)) for p, g in zip(data["p"], data["g"])])
    label = error > THRESHOLD_MM
    subjects = sorted(set(data["subject"]))
    result: dict = {"views": int(len(x)), "subjects": subjects, "threshold_mm": THRESHOLD_MM,
                    "positive_rate": float(label.mean()), "overall_mean_mm": float(error.mean()), "splits": {}}
    print(f"{len(x)} views, {label.mean():.1%} above {THRESHOLD_MM:.0f} mm, overall mean {error.mean():.1f} mm", flush=True)

    for held_out in subjects:
        train = np.flatnonzero(data["subject"] != held_out)
        test = np.flatnonzero(data["subject"] == held_out)
        if len(train) == 0 or len(test) == 0 or label[train].min() == label[train].max():
            continue
        entry: dict = {"train_views": int(len(train)), "test_views": int(len(test)),
                       "test_positive_rate": float(label[test].mean()), "test_mean_mm": float(error[test].mean())}
        # each live signal alone (AUC; 0.5 = nothing)
        single = {}
        for i, name in enumerate(names[:5]):
            auc = roc_auc_score(label[test], x[test, i])
            single[name] = float(max(auc, 1 - auc))
        entry["single_feature_auc"] = single
        for model_name in ("gradient_boosting", "logistic"):
            if model_name == "gradient_boosting":
                model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                       l2_regularization=1.0, random_state=0)
                model.fit(x[train], label[train])
                risk = model.predict_proba(x[test])[:, 1]
            else:
                scaler = StandardScaler().fit(x[train])
                model = LogisticRegression(C=0.1, max_iter=2000).fit(scaler.transform(x[train]), label[train])
                risk = model.predict_proba(scaler.transform(x[test]))[:, 1]
            entry[model_name] = {"auc": float(roc_auc_score(label[test], risk)),
                                 "selective_mean_mm": selective_curve(risk, error[test])}
        result["splits"][f"test_{held_out}"] = entry
        gb = entry["gradient_boosting"]
        print(f"\nheld-out {held_out}: AUC gradient boosting {gb['auc']:.3f}, logistic "
              f"{entry['logistic']['auc']:.3f}; best single signal "
              f"{max(single, key=single.get)} {max(single.values()):.3f}", flush=True)
        for key, v in gb["selective_mean_mm"].items():
            print(f"  {key}: detector {v['detector']:.1f} mm | random {v['random']:.1f} | "
                  f"ceiling {v['oracle_ceiling']:.1f}   (all views {error[test].mean():.1f})", flush=True)

    result["protocol"] = ("label = PA-MPJPE > threshold; features = live signals only (no camera id, no ground "
                          "truth); trained on one subject, tested on the other")
    write_json(OUTPUT, result)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
