"""Train the orientation (facing) classifier on synthetic data.

Cross-entropy against the yaw octant each synthetic sample was generated from
(`tryon/body/orientation_data.py`, which explains why 48 real photos are too few).
Model: a plain MLP (`tryon.body.orientation.OrientationClassifier`), Adam. The
best-val-accuracy state is saved to models/pose_regressor.pt.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from tryon.body.orientation import OrientationClassifier
from tryon.body.orientation_data import N_BINS, make_dataset

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = ROOT / "models"


def device(force_cpu=False):
    if not force_cpu and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_loader(features, labels, batch_size, shuffle):
    dataset = TensorDataset(torch.as_tensor(features), torch.as_tensor(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def run_epoch(model, loader, optimiser, dev, train):
    model.train(train)
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0
    for features, labels in loader:
        features = features.to(dev)
        labels = labels.to(dev)
        if train:
            optimiser.zero_grad()
        logits = model(features)
        # label_smoothing=0.1 raised held-out validation accuracy from ~0.83 to 0.844
        # (docs/baseline/pose-regressor-training.json): a yaw near a bin boundary
        # can belong to either bin.
        loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=0.1)
        if train:
            loss.backward()
            optimiser.step()
        total_loss += float(loss.detach())
        n_batches += 1
        with torch.no_grad():
            predicted = logits.argmax(dim=1)
            correct += int((predicted == labels).sum())
            total += labels.shape[0]
    mean_loss = total_loss / max(n_batches, 1)
    accuracy = correct / max(total, 1)
    return mean_loss, accuracy


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--val-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-px", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    dev = device(force_cpu=args.cpu)
    print(f"device: {dev}")

    # Disjoint RNG seeds per split: the data is generated, not a finite pool.
    print("generating synthetic training data...")
    t0 = time.perf_counter()
    train_features, _, train_labels = make_dataset(args.train_samples, seed=args.seed, noise_px=args.noise_px)
    val_features, _, val_labels = make_dataset(args.val_samples, seed=args.seed + 1, noise_px=args.noise_px)
    test_features, _, test_labels = make_dataset(args.test_samples, seed=args.seed + 2, noise_px=args.noise_px)
    print(f"generated {args.train_samples + args.val_samples + args.test_samples} samples in {time.perf_counter()-t0:.1f}s")

    train_loader = _to_loader(train_features, train_labels, args.batch_size, shuffle=True)
    val_loader = _to_loader(val_features, val_labels, args.batch_size, shuffle=False)
    test_loader = _to_loader(test_features, test_labels, args.batch_size, shuffle=False)

    model = OrientationClassifier(n_bins=N_BINS).to(dev)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "pose_regressor.pt"
    history = []
    best_val_accuracy = -1.0
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        train_loss, train_accuracy = run_epoch(model, train_loader, optimiser, dev, train=True)
        val_loss, val_accuracy = run_epoch(model, val_loader, optimiser, dev, train=False)
        dt = time.perf_counter() - t0
        print(
            f"epoch {epoch+1}/{args.epochs}  train loss {train_loss:.3f} acc {train_accuracy:.3f}  "
            f"val loss {val_loss:.3f} acc {val_accuracy:.3f}  ({dt:.1f}s)"
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {"model_state": model.state_dict(), "n_bins": N_BINS, "val_accuracy": val_accuracy},
                checkpoint_path,
            )
            print(f"  new best val accuracy {val_accuracy:.3f}, saved to {checkpoint_path}")

    best = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    model.load_state_dict(best["model_state"])
    test_loss, test_accuracy = run_epoch(model, test_loader, optimiser, dev, train=False)
    print(f"\nFINAL (best-val checkpoint) TEST: loss {test_loss:.3f}  accuracy {test_accuracy:.3f}")

    report = {
        "n_bins": N_BINS,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "test_samples": args.test_samples,
        "history": history,
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }
    report_path = ROOT / "docs" / "baseline" / "pose-regressor-training.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
