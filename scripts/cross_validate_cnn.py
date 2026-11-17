#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_detection.dataset import MemmapPacketDataset
from traffic_detection.cross_validation import balanced_sample_indices
from traffic_detection.metrics import confusion_matrix, f1_from_confusion
from traffic_detection.model import PacketCNN


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
) -> tuple[dict[str, float | np.ndarray], np.ndarray]:
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for x, target in tqdm(loader, desc="validation", leave=False):
            prediction = model(x.to(device, non_blocking=True)).argmax(dim=1).cpu().numpy()
            target_numpy = target.numpy()
            predictions.append(prediction.astype(np.int16, copy=False))
            cm += confusion_matrix(target_numpy, prediction, n_classes)
    return f1_from_confusion(cm), np.concatenate(predictions)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stratified K-fold cross-validation for the packet CNN"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--cap-per-class", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight-alpha", type=float, default=0.25)
    parser.add_argument("--max-class-weight", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x_path = args.data_dir / "train_packets.npy"
    y_path = args.data_dir / "train_labels.npy"
    labels_path = args.data_dir / "classes.json"
    y = np.load(y_path, mmap_mode="r")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    n_classes = len(labels)
    sample_indices = balanced_sample_indices(y, args.cap_per_class, args.seed)
    sample_labels = np.asarray(y[sample_indices], dtype=np.int32)
    counts = np.bincount(sample_labels, minlength=n_classes)
    if counts.min() < args.folds:
        raise ValueError(
            f"Every class needs at least {args.folds} rows; minimum is {counts.min()}"
        )

    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    oof_prediction = np.full(len(sample_indices), -1, dtype=np.int16)
    fold_rows: list[dict[str, object]] = []
    print(
        f"CNN CV: rows={len(sample_indices):,}, classes={n_classes}, "
        f"folds={args.folds}, epochs={args.epochs}, device={device}"
    )

    for fold, (fit_local, val_local) in enumerate(
        splitter.split(np.zeros(len(sample_labels), dtype=np.uint8), sample_labels),
        start=1,
    ):
        fold_seed = args.seed + fold
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)
        fit_indices = sample_indices[fit_local]
        val_indices = sample_indices[val_local]
        train_loader = DataLoader(
            MemmapPacketDataset(x_path, y_path, fit_indices),
            shuffle=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            MemmapPacketDataset(x_path, y_path, val_indices),
            shuffle=False,
            **loader_kwargs,
        )

        fold_counts = np.bincount(sample_labels[fit_local], minlength=n_classes).astype(
            np.float64
        )
        class_weights = (fold_counts.max() / np.maximum(fold_counts, 1.0)) ** args.class_weight_alpha
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
        best_prediction: np.ndarray | None = None
        best_epoch = 0
        history: list[dict[str, float]] = []

        print(
            f"fold {fold}/{args.folds}: fit={len(fit_indices):,}, "
            f"validation={len(val_indices):,}"
        )
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0
            seen = 0
            progress = tqdm(
                train_loader,
                desc=f"fold {fold}/{args.folds}, epoch {epoch}/{args.epochs}",
            )
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
                batch_rows = len(x)
                loss_sum += float(loss.detach()) * batch_rows
                seen += batch_rows
                progress.set_postfix(loss=f"{loss_sum / seen:.4f}")

            metrics, prediction = evaluate(model, val_loader, device, n_classes)
            row = {
                "epoch": float(epoch),
                "train_loss": loss_sum / max(seen, 1),
                "macro_f1": float(metrics["macro_f1"]),
                "weighted_f1": float(metrics["weighted_f1"]),
                "accuracy": float(metrics["accuracy"]),
            }
            history.append(row)
            print(json.dumps(row, ensure_ascii=False))
            if row["macro_f1"] > best_macro_f1:
                best_macro_f1 = row["macro_f1"]
                best_prediction = prediction.copy()
                best_epoch = epoch
                if args.save_models:
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "num_classes": n_classes,
                            "labels": labels,
                            "fold": fold,
                            "epoch": epoch,
                            "metrics": row,
                        },
                        args.output_dir / f"fold_{fold}_best.pt",
                    )

        if best_prediction is None:
            raise RuntimeError(f"Fold {fold} produced no validation prediction")
        oof_prediction[val_local] = best_prediction
        best_row = history[best_epoch - 1]
        fold_rows.append(
            {
                "fold": fold,
                "fit_rows": int(len(fit_indices)),
                "validation_rows": int(len(val_indices)),
                "best_epoch": best_epoch,
                "best_metrics": best_row,
                "history": history,
            }
        )
        del model, optimizer, scheduler, scaler, train_loader, val_loader, best_prediction
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if np.any(oof_prediction < 0):
        raise RuntimeError("Some rows did not receive an out-of-fold prediction")
    overall_cm = confusion_matrix(sample_labels, oof_prediction, n_classes)
    overall = f1_from_confusion(overall_cm)
    summary: dict[str, dict[str, float]] = {}
    for name in ("macro_f1", "weighted_f1", "accuracy"):
        values = np.asarray(
            [float(row["best_metrics"][name]) for row in fold_rows], dtype=np.float64
        )
        summary[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    payload = {
        "model": "PacketCNN",
        "folds": args.folds,
        "epochs": args.epochs,
        "seed": args.seed,
        "cap_per_class": args.cap_per_class,
        "sample_rows": int(len(sample_indices)),
        "classes": n_classes,
        "device": str(device),
        "fold_results": fold_rows,
        "fold_summary": summary,
        "oof_metrics": {
            "macro_f1": float(overall["macro_f1"]),
            "weighted_f1": float(overall["weighted_f1"]),
            "accuracy": float(overall["accuracy"]),
        },
    }
    (args.output_dir / "cnn_cv.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(args.output_dir / "sample_indices.npy", sample_indices)
    np.save(args.output_dir / "oof_predictions.npy", oof_prediction)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

