"""The measurements behind tools/evaluate/parser_study.py."""

from __future__ import annotations

import os
import time

import numpy as np

from tools.evaluate.parser_study import load_average, percentile_row, top_process
from tryon.removal.parsing_schp import ATR_CHECKPOINT, SCHPParser

CACHE_ROOT = os.environ.get("PARSER_STUDY_CACHE", "/tmp/gg/coreml_cache")


def make_parser(units, fmt="NeuralNetwork", extra=None, checkpoint=ATR_CHECKPOINT, size=512):
    options = {"ModelFormat": fmt, "ModelCacheDirectory": f"{CACHE_ROOT}/{units}-{fmt}-{sorted((extra or {}).items())}".replace(" ", "")}
    if units != "ALL":
        options["MLComputeUnits"] = units
    options.update(extra or {})
    os.makedirs(options["ModelCacheDirectory"], exist_ok=True)
    return SCHPParser(checkpoint, size=size, providers=[("CoreMLExecutionProvider", options), "CPUExecutionProvider"])


def units(frames, clips, argv):
    configs = {
        "ALL": ("ALL", "NeuralNetwork", None),
        "CPUAndGPU": ("CPUAndGPU", "NeuralNetwork", None),
        "CPUAndNeuralEngine": ("CPUAndNeuralEngine", "NeuralNetwork", None),
        "CPUOnly": ("CPUOnly", "NeuralNetwork", None),
        "ALL+static": ("ALL", "NeuralNetwork", {"RequireStaticInputShapes": "1"}),
        "ALL+subgraphs": ("ALL", "NeuralNetwork", {"EnableOnSubgraphs": "1"}),
        "MLProgram": ("ALL", "MLProgram", None),
    }
    wanted = argv[0].split(",") if argv else list(configs)
    rounds = int(argv[1]) if len(argv) > 1 else 3
    per = int(argv[2]) if len(argv) > 2 else 20
    parsers = {}
    for name in wanted:
        t0 = time.time()
        try:
            parsers[name] = make_parser(*configs[name])
            parsers[name]._square_logits(frames[0])
            print(f"{name}: loaded in {time.time() - t0:.1f} s", flush=True)
        except Exception as error:  # noqa: BLE001 - a study reports every failure
            print(f"{name}: FAILED {str(error)[:150]}", flush=True)
    times = {name: [] for name in parsers}
    for r in range(rounds):
        for name, parser in parsers.items():
            for _ in range(3):
                parser._square_logits(frames[0])
            load, top = load_average(), top_process()
            batch = []
            for i in range(per):
                f = frames[(r * per + i) % len(frames)]
                t0 = time.perf_counter()
                parser._square_logits(f)
                batch.append(time.perf_counter() - t0)
            times[name] += batch
            print(f"round {r} {name:20s} {percentile_row(batch)} | load {load:.1f} top {top}", flush=True)
    print("--- all rounds")
    for name, t in times.items():
        print(f"{name:20s} {percentile_row(t)}")


def post(frames, clips, argv):
    """`fast_clothing` against its reference on the same network output."""
    parser = make_parser("ALL")
    picks = list(range(0, len(frames), max(len(frames) // 12, 1)))[:12]
    cached = {i: parser._square_logits(frames[i]) for i in picks}
    equal_prob = equal_labels = equal_mask = 0
    for i in picks:
        parser._square_logits = lambda _f, i=i: cached[i]
        p0, l0 = parser.fast_clothing_reference(frames[i], (4, 7))
        p1, l1 = parser.fast_clothing(frames[i], (4, 7))
        equal_prob += int(np.abs(p0 - p1).max() < 1e-6)
        equal_labels += int(np.array_equal(l0, l1))
        equal_mask += int(np.array_equal(p0 > 0.5, p1 > 0.5))
    total = same_labels = same_mask = same_fast_labels = 0
    for frame in frames:  # every frame: one network pass, both implementations on it
        logits = parser._square_logits(frame)
        parser._square_logits = lambda _f, logits=logits: logits
        p0, l0 = parser.fast_clothing_reference(frame, (4, 7))
        p1, l1 = parser.fast_clothing(frame, (4, 7))
        total += 1
        same_labels += int(np.array_equal(l0, l1))
        same_mask += int(np.array_equal(p0 > 0.5, p1 > 0.5))
        same_fast_labels += int(np.array_equal(parser.fast_labels_reference(frame), parser.fast_labels(frame)))
        del parser._square_logits
    print(f"all {total} frames: labels equal {same_labels}, thresholded mask equal {same_mask}, fast_labels equal {same_fast_labels}")
    print(f"{len(picks)} frames: probability within 1e-6 {equal_prob}, labels {equal_labels}, mask {equal_mask}")
    times = {"reference": [], "new": []}
    for n in range(30):
        i = picks[n % len(picks)]
        parser._square_logits = lambda _f, i=i: cached[i]
        for name, fn in (("reference", parser.fast_clothing_reference), ("new", parser.fast_clothing)):
            t0 = time.perf_counter()
            fn(frames[i], (4, 7))
            times[name].append(time.perf_counter() - t0)
    print(f"load {load_average():.1f}")
    for name, t in times.items():
        print(f"{name:10s} {percentile_row(t)}")


def boundary_f(a, b, tolerance=2):
    """F-score of the boundary pixels of masks `a` and `b` within `tolerance` px."""
    import cv2

    def edge(m):
        m = m.astype(np.uint8)
        return (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool)

    ea, eb = edge(a), edge(b)
    if not ea.any() and not eb.any():
        return 1.0
    if not ea.any() or not eb.any():
        return 0.0
    near_b = cv2.distanceTransform((~eb).astype(np.uint8), cv2.DIST_L2, 3) <= tolerance
    near_a = cv2.distanceTransform((~ea).astype(np.uint8), cv2.DIST_L2, 3) <= tolerance
    precision, recall = near_b[ea].mean(), near_a[eb].mean()
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def iou(a, b):
    union = (a | b).sum()
    return 1.0 if union == 0 else (a & b).sum() / union


def size(frames, clips, argv):
    sizes = [int(s) for s in (argv[0] if argv else "512,448,384,320").split(",")]
    parsers = {}
    for n in sizes:
        try:
            parsers[n] = make_parser("ALL", size=n)
            print(n, "input shape", parsers[n].session.get_inputs()[0].shape, flush=True)
            parsers[n]._square_logits(frames[0])
        except Exception as error:  # noqa: BLE001
            print(n, "FAILED", str(error)[:200], flush=True)
    base = parsers[512]
    reference = [base.fast_clothing(f, (4, 7)) for f in frames]
    scores = {n: {"iou": [], "label": [], "bf": [], "time": []} for n in parsers}
    for k, frame in enumerate(frames):
        p0, l0 = reference[k]
        for n, parser in parsers.items():
            t0 = time.perf_counter()
            p, lab = parser.fast_clothing(frame, (4, 7))
            scores[n]["time"].append(time.perf_counter() - t0)
            m0, m1 = p0 > 0.5, p > 0.5
            scores[n]["iou"].append(iou(m0, m1))
            scores[n]["label"].append((l0 == lab).mean())
            scores[n]["bf"].append(boundary_f(m0, m1))
    print(f"load {load_average():.1f} top {top_process()}")
    for n, s in scores.items():
        by_clip = {c: np.mean([v for v, cc in zip(s["iou"], clips) if cc == c]) for c in sorted(set(clips))}
        print(f"{n}: {percentile_row(s['time'])} | mask IoU {np.mean(s['iou']):.4f} (min {np.min(s['iou']):.3f}) label {np.mean(s['label']):.4f} boundaryF2 {np.mean(s['bf']):.4f} | {' '.join(f'{c[:5]}={v:.3f}' for c, v in by_clip.items())}")


def lip(frames, clips, argv):
    """The LIP coat / inner-layer decision at 473 and at smaller sizes."""
    from tryon.removal.parsing_schp import LIP_CHECKPOINT

    sizes = [int(s) for s in (argv[0] if argv else "473,384,320").split(",")]
    parsers = {n: make_parser("ALL", checkpoint=LIP_CHECKPOINT, size=n) for n in sizes}
    for p in parsers.values():
        p.fast_labels(frames[0])
    rows = {n: [] for n in sizes}
    times = {n: [] for n in sizes}
    for frame in frames:
        for n, parser in parsers.items():
            t0 = time.perf_counter()
            labels = parser.fast_labels(frame)
            times[n].append(time.perf_counter() - t0)
            coat, inner = labels == 7, np.isin(labels, (5, 10))
            has_coat = bool(coat.sum() >= 0.5 * max(int((coat | inner).sum()), 1))
            rows[n].append((has_coat, inner))
    print(f"load {load_average():.1f} top {top_process()}")
    for n in sizes:
        agree = np.mean([a[0] == b[0] for a, b in zip(rows[473], rows[n])])
        inner_iou = [iou(a[1], b[1]) for a, b in zip(rows[473], rows[n]) if a[0] and b[0]]
        print(f"LIP {n}: {percentile_row(times[n])} | decision agrees with 473 on {agree:.3f} of frames | kept-inner IoU {np.mean(inner_iou) if inner_iou else float('nan'):.3f}")
    for c in sorted(set(clips)):
        idx = [i for i, cc in enumerate(clips) if cc == c]
        print(c, "coat decision at 473 over its frames:", "".join("C" if rows[473][i][0] else "." for i in idx))


def contend(frames, clips, argv):
    """Synthetic Neural Engine contention: predict in a loop for `seconds`."""
    seconds = float(argv[0]) if argv else 30.0
    parser = make_parser("CPUAndNeuralEngine")
    parser._square_logits(frames[0])
    end, count = time.time() + seconds, 0
    while time.time() < end:
        parser._square_logits(frames[count % len(frames)])
        count += 1
    print(f"contender: {count} passes in {seconds:.0f} s")
