#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_detection.dataset import MemmapPacketDataset
from traffic_detection.model import PacketCNN


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Kaggle submission from CNN")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "submission_cnn.csv")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    labels = checkpoint["labels"]
    model = PacketCNN(num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    dataset = MemmapPacketDataset(args.data_dir / "test_packets.npy")
    ids = np.load(args.data_dir / "test_ids.npy", mmap_mode="r")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "app_service"])
        with torch.inference_mode():
            for x in tqdm(loader, desc="prediction"):
                pred = model(x.to(device, non_blocking=True)).argmax(dim=1).cpu().numpy()
                batch_ids = ids[offset : offset + len(pred)]
                writer.writerows((int(row_id), labels[int(label)]) for row_id, label in zip(batch_ids, pred))
                offset += len(pred)


if __name__ == "__main__":
    main()

