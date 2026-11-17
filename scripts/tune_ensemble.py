#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from catboost import CatBoostClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_detection.metrics import confusion_matrix, f1_from_confusion
from traffic_detection.model import PacketCNN


def cnn_probabilities(
    model: PacketCNN,
    packets: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    result = np.empty((len(packets), model.classifier[-1].out_features), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(packets), batch_size):
            stop = min(start + batch_size, len(packets))
            x = torch.as_tensor(
                np.asarray(packets[start:stop], dtype=np.int16).copy(), device=device
            )
            result[start:stop] = torch.softmax(model(x), dim=1).cpu().numpy()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune CatBoost/CNN blend on shared validation rows")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--catboost-dir", type=Path, required=True)
    parser.add_argument("--cnn-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=25_000)
    parser.add_argument("--cnn-batch-size", type=int, default=8192)
    args = parser.parse_args()

    cat_indices = np.load(args.catboost_dir / "val_indices.npy")
    cnn_indices = np.load(args.cnn_dir / "val_indices.npy")
    if not np.array_equal(cat_indices, cnn_indices):
        raise ValueError("CatBoost and CNN validation indices differ; train both with the same seed/val-size")
    indices = cat_indices
    features = np.load(args.data_dir / "train_features.npy", mmap_mode="r")
    packets = np.load(args.data_dir / "train_packets.npy", mmap_mode="r")
    y = np.load(args.data_dir / "train_labels.npy", mmap_mode="r")
    labels = json.loads((args.data_dir / "classes.json").read_text(encoding="utf-8"))
    n_classes = len(labels)

    cat_model = CatBoostClassifier()
    cat_model.load_model(args.catboost_dir / "model.cbm")
    checkpoint = torch.load(args.cnn_dir / "best.pt", map_location="cpu", weights_only=False)
    cnn_model = PacketCNN(num_classes=checkpoint["num_classes"])
    cnn_model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model.to(device).eval()

    weights = np.linspace(0.0, 1.0, 11)
    matrices = [np.zeros((n_classes, n_classes), dtype=np.int64) for _ in weights]
    for start in range(0, len(indices), args.chunk_size):
        batch_indices = indices[start : start + args.chunk_size]
        cat_prob = np.asarray(cat_model.predict_proba(features[batch_indices]), dtype=np.float32)
        cnn_prob = cnn_probabilities(
            cnn_model, packets[batch_indices], device, args.cnn_batch_size
        )
        target = np.asarray(y[batch_indices], dtype=np.int32)
        for position, weight in enumerate(weights):
            pred = np.argmax(weight * cnn_prob + (1.0 - weight) * cat_prob, axis=1)
            matrices[position] += confusion_matrix(target, pred, n_classes)
        print(f"ensemble validation: {min(start + len(batch_indices), len(indices)):,}/{len(indices):,}")

    rows = []
    for weight, matrix in zip(weights, matrices):
        metrics = f1_from_confusion(matrix)
        rows.append(
            {
                "cnn_weight": float(weight),
                "macro_f1": float(metrics["macro_f1"]),
                "weighted_f1": float(metrics["weighted_f1"]),
                "accuracy": float(metrics["accuracy"]),
            }
        )
    best = max(rows, key=lambda row: row["macro_f1"])
    payload = {"best": best, "grid": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

