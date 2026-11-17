#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_detection.dataset import MemmapPacketDataset
from traffic_detection.metrics import confusion_matrix, f1_from_confusion
from traffic_detection.model import PacketCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the packet-sequence CNN")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "cnn")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight-alpha", type=float, default=0.25)
    parser.add_argument("--max-class-weight", type=float, default=5.0)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=None)
    return parser.parse_args()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
) -> dict[str, float | np.ndarray]:
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    with torch.inference_mode():
        for x, y in tqdm(loader, desc="validation", leave=False):
            pred = model(x.to(device, non_blocking=True)).argmax(dim=1).cpu().numpy()
            cm += confusion_matrix(y.numpy(), pred, n_classes)
    return f1_from_confusion(cm)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x_path = args.data_dir / "train_packets.npy"
    y_path = args.data_dir / "train_labels.npy"
    labels_path = args.data_dir / "classes.json"
    y = np.load(y_path, mmap_mode="r")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    n_classes = len(labels)

    all_indices = np.arange(len(y), dtype=np.int32)
    if args.max_train_rows and args.max_train_rows < len(all_indices):
        selected, _ = train_test_split(
            all_indices,
            train_size=args.max_train_rows,
            random_state=args.seed,
            stratify=np.asarray(y),
        )
        all_indices = selected.astype(np.int32, copy=False)
    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=args.val_size,
        random_state=args.seed,
        stratify=np.asarray(y)[all_indices],
    )
    train_idx = train_idx.astype(np.int32, copy=False)
    val_idx = val_idx.astype(np.int32, copy=False)
    np.save(args.output_dir / "val_indices.npy", val_idx)

    train_dataset = MemmapPacketDataset(x_path, y_path, train_idx)
    val_dataset = MemmapPacketDataset(x_path, y_path, val_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    counts = np.bincount(np.asarray(y)[train_idx], minlength=n_classes).astype(np.float64)
    class_weights = (counts.max() / np.maximum(counts, 1.0)) ** args.class_weight_alpha
    class_weights = np.minimum(class_weights, args.max_class_weight)
    class_weights /= class_weights.mean()

    model = PacketCNN(num_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=0.02,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_macro_f1 = -1.0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for x, target in progress:
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            batch_size = len(x)
            loss_sum += float(loss.detach()) * batch_size
            seen += batch_size
            progress.set_postfix(loss=f"{loss_sum / seen:.4f}")

        metrics = evaluate(model, val_loader, device, n_classes)
        row = {
            "epoch": float(epoch),
            "train_loss": loss_sum / max(seen, 1),
            "macro_f1": float(metrics["macro_f1"]),
            "weighted_f1": float(metrics["weighted_f1"]),
            "accuracy": float(metrics["accuracy"]),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        (args.output_dir / "history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if row["macro_f1"] > best_macro_f1:
            best_macro_f1 = row["macro_f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_classes": n_classes,
                    "labels": labels,
                    "epoch": epoch,
                    "metrics": row,
                },
                args.output_dir / "best.pt",
            )


if __name__ == "__main__":
    main()

