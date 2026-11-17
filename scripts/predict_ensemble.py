#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from catboost import CatBoostClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_detection.model import PacketCNN


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an online CatBoost/CNN ensemble submission")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--catboost-model", type=Path, required=True)
    parser.add_argument("--cnn-checkpoint", type=Path, required=True)
    parser.add_argument("--ensemble-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=25_000)
    parser.add_argument("--cnn-batch-size", type=int, default=8192)
    args = parser.parse_args()

    features = np.load(args.data_dir / "test_features.npy", mmap_mode="r")
    packets = np.load(args.data_dir / "test_packets.npy", mmap_mode="r")
    ids = np.load(args.data_dir / "test_ids.npy", mmap_mode="r")
    labels = json.loads((args.data_dir / "classes.json").read_text(encoding="utf-8"))
    config = json.loads(args.ensemble_config.read_text(encoding="utf-8"))
    cnn_weight = float(config["best"]["cnn_weight"])

    cat_model = CatBoostClassifier()
    cat_model.load_model(args.catboost_model)
    checkpoint = torch.load(args.cnn_checkpoint, map_location="cpu", weights_only=False)
    cnn_model = PacketCNN(num_classes=checkpoint["num_classes"])
    cnn_model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model.to(device).eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "app_service"])
        for start in range(0, len(packets), args.chunk_size):
            stop = min(start + args.chunk_size, len(packets))
            cat_prob = np.asarray(cat_model.predict_proba(features[start:stop]), dtype=np.float32)
            cnn_prob = np.empty_like(cat_prob)
            with torch.inference_mode():
                for inner_start in range(start, stop, args.cnn_batch_size):
                    inner_stop = min(inner_start + args.cnn_batch_size, stop)
                    x = torch.as_tensor(
                        np.asarray(
                            packets[inner_start:inner_stop], dtype=np.int16
                        ).copy(),
                        device=device,
                    )
                    cnn_prob[inner_start - start : inner_stop - start] = torch.softmax(
                        cnn_model(x), dim=1
                    ).cpu().numpy()
            pred = np.argmax(cnn_weight * cnn_prob + (1.0 - cnn_weight) * cat_prob, axis=1)
            writer.writerows(
                (int(row_id), labels[int(label)])
                for row_id, label in zip(ids[start:stop], pred)
            )
            print(f"ensemble prediction: {stop:,}/{len(packets):,}")


if __name__ == "__main__":
    main()

