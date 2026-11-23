#!/usr/bin/env python3
"""Generate multiclass ROC/PR and error diagnostics from saved model weights.

The script reconstructs the untouched validation split used by the final models.
It can read prepared memmaps or extract a stratified validation sample directly
from the original train.csv.  No model is trained.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_detection.features import extract_features
from traffic_detection.model import PacketCNN


COLORS = ["#275D8C", "#D17A22", "#37805B", "#B84A4A", "#7656A8", "#2A8C9D"]
PREFERRED_CLASSES = [
    "Telegram",
    "WhatsApp",
    "YouTube",
    "yandex_video",
    "Twitch",
    "Spotify",
    "zoom",
    "Teams",
    "google_drive",
    "DoH_DoT",
]


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def expected_calibration_error(
    target: np.ndarray, probability: np.ndarray, bins: int = 15
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    prediction = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    correct = prediction == target
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.minimum(np.digitize(confidence, edges[1:-1]), bins - 1)
    mean_confidence = np.full(bins, np.nan, dtype=np.float64)
    mean_accuracy = np.full(bins, np.nan, dtype=np.float64)
    counts = np.zeros(bins, dtype=np.int64)
    error = 0.0
    for bin_index in range(bins):
        mask = bin_ids == bin_index
        counts[bin_index] = int(mask.sum())
        if counts[bin_index] == 0:
            continue
        mean_confidence[bin_index] = float(confidence[mask].mean())
        mean_accuracy[bin_index] = float(correct[mask].mean())
        error += counts[bin_index] / len(target) * abs(
            mean_accuracy[bin_index] - mean_confidence[bin_index]
        )
    return float(error), mean_confidence, mean_accuracy, counts


def stratified_validation_indices(
    labels: np.ndarray, val_size: float, max_rows: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    all_indices = np.arange(len(labels), dtype=np.int32)
    _, validation = train_test_split(
        all_indices,
        test_size=val_size,
        random_state=seed,
        stratify=labels,
    )
    validation = validation.astype(np.int32, copy=False)
    validation_counts = np.bincount(labels[validation])
    if max_rows and max_rows < len(validation):
        sample, _ = train_test_split(
            validation,
            train_size=max_rows,
            random_state=seed,
            stratify=labels[validation],
        )
        validation = sample.astype(np.int32, copy=False)
    return np.sort(validation), validation_counts


def read_raw_labels(
    train_csv: Path, class_to_index: dict[str, int], chunk_size: int
) -> np.ndarray:
    parts: list[np.ndarray] = []
    processed = 0
    for chunk in pd.read_csv(
        train_csv,
        usecols=["app_service"],
        dtype={"app_service": "string"},
        chunksize=chunk_size,
    ):
        mapped = chunk["app_service"].astype(str).map(class_to_index)
        if mapped.isna().any():
            unknown = sorted(set(chunk.loc[mapped.isna(), "app_service"].astype(str)))
            raise ValueError(f"Unknown labels in train.csv: {unknown[:10]}")
        parts.append(mapped.to_numpy(dtype=np.int16))
        processed += len(chunk)
        print(f"reading labels: {processed:,}", flush=True)
    return np.concatenate(parts)


def read_raw_packet_sample(
    train_csv: Path, sample_indices: np.ndarray, chunk_size: int
) -> np.ndarray:
    packets = np.empty((len(sample_indices), 30), dtype=np.int16)
    written = 0
    row_start = 0
    for chunk in pd.read_csv(
        train_csv,
        chunksize=chunk_size,
        dtype={"app_service": "string"},
    ):
        row_stop = row_start + len(chunk)
        left = int(np.searchsorted(sample_indices, row_start, side="left"))
        right = int(np.searchsorted(sample_indices, row_stop, side="left"))
        if right > left:
            local = sample_indices[left:right] - row_start
            packets[left:right] = (
                chunk.iloc[local, 1:31].fillna(0).to_numpy(dtype=np.int16, copy=False)
            )
            written += right - left
        row_start = row_stop
        print(f"reading packets: {row_start:,}, selected={written:,}", flush=True)
    if written != len(sample_indices):
        raise RuntimeError(f"Read {written} selected rows, expected {len(sample_indices)}")
    return packets


def load_validation_sample(
    data_dir: Path | None,
    train_csv: Path | None,
    classes: list[str],
    val_size: float,
    max_rows: int,
    seed: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if data_dir is not None:
        labels = np.load(data_dir / "train_labels.npy", mmap_mode="r")
        sample_indices, validation_counts = stratified_validation_indices(
            np.asarray(labels), val_size, max_rows, seed
        )
        packets_memmap = np.load(data_dir / "train_packets.npy", mmap_mode="r")
        packets = np.asarray(packets_memmap[sample_indices], dtype=np.int16)
        target = np.asarray(labels[sample_indices], dtype=np.int32)
        return packets, target, sample_indices, validation_counts

    if train_csv is None:
        raise ValueError("Either data_dir or train_csv is required")
    class_to_index = {label: index for index, label in enumerate(classes)}
    labels = read_raw_labels(train_csv, class_to_index, chunk_size)
    sample_indices, validation_counts = stratified_validation_indices(
        labels, val_size, max_rows, seed
    )
    packets = read_raw_packet_sample(train_csv, sample_indices, chunk_size)
    target = labels[sample_indices].astype(np.int32, copy=False)
    return packets, target, sample_indices, validation_counts


def cnn_probabilities(
    checkpoint: dict[str, object], packets: np.ndarray, batch_size: int
) -> tuple[np.ndarray, str]:
    model = PacketCNN(num_classes=int(checkpoint["num_classes"]))
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    result = np.empty((len(packets), int(checkpoint["num_classes"])), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(packets), batch_size):
            stop = min(start + batch_size, len(packets))
            batch = torch.as_tensor(packets[start:stop].copy(), device=device)
            result[start:stop] = torch.softmax(model(batch), dim=1).cpu().numpy()
            print(f"CNN probabilities: {stop:,}/{len(packets):,}", flush=True)
    return result, str(device)


def model_metrics(
    target: np.ndarray, probability: np.ndarray, n_classes: int
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    prediction = probability.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        target,
        prediction,
        labels=np.arange(n_classes),
        zero_division=0,
    )
    binary = label_binarize(target, classes=np.arange(n_classes))
    per_class_roc = np.zeros(n_classes, dtype=np.float64)
    per_class_ap = np.zeros(n_classes, dtype=np.float64)
    for index in range(n_classes):
        per_class_roc[index] = roc_auc_score(binary[:, index], probability[:, index])
        per_class_ap[index] = average_precision_score(binary[:, index], probability[:, index])
    ece, _, _, _ = expected_calibration_error(target, probability)
    metrics = {
        "accuracy": float(accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(target, prediction, average="macro")),
        "weighted_f1": float(f1_score(target, prediction, average="weighted")),
        "top_3_accuracy": float(
            top_k_accuracy_score(target, probability, k=3, labels=np.arange(n_classes))
        ),
        "top_5_accuracy": float(
            top_k_accuracy_score(target, probability, k=5, labels=np.arange(n_classes))
        ),
        "log_loss": float(log_loss(target, probability, labels=np.arange(n_classes))),
        "roc_auc_ovr_macro": float(per_class_roc.mean()),
        "roc_auc_ovr_weighted": float(np.average(per_class_roc, weights=support)),
        "average_precision_macro": float(per_class_ap.mean()),
        "average_precision_weighted": float(np.average(per_class_ap, weights=support)),
        "expected_calibration_error": ece,
    }
    per_class = pd.DataFrame(
        {
            "class_index": np.arange(n_classes),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(np.int64),
            "roc_auc_ovr": per_class_roc,
            "average_precision": per_class_ap,
        }
    )
    return metrics, per_class, confusion_matrix(target, prediction, labels=np.arange(n_classes))


def plot_auc_comparison(metrics: dict[str, dict[str, float]], output_dir: Path) -> None:
    names = list(metrics)
    measures = ["roc_auc_ovr_macro", "average_precision_macro"]
    labels = ["ROC-AUC OvR macro", "PR-AUC macro"]
    x = np.arange(len(names))
    width = 0.34
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    for index, (measure, label) in enumerate(zip(measures, labels)):
        values = [metrics[name][measure] for name in names]
        bars = axis.bar(
            x + (index - 0.5) * width,
            values,
            width=width,
            label=label,
            color=COLORS[index],
        )
        axis.bar_label(bars, labels=[f"{value:.4f}" for value in values], padding=3, fontsize=8)
    axis.set_xticks(x, labels=[name.upper() for name in names])
    axis.set_ylim(0, 1.06)
    axis.set_ylabel("Значение метрики")
    axis.set_title("Вероятностные метрики моделей")
    axis.legend(frameon=False)
    save_figure(figure, output_dir / "auc_model_comparison.png")


def plot_selected_curves(
    target: np.ndarray,
    probability: np.ndarray,
    classes: list[str],
    output_dir: Path,
) -> None:
    available = {label: index for index, label in enumerate(classes)}
    selected = [label for label in PREFERRED_CLASSES if label in available][:8]

    figure, axis = plt.subplots(figsize=(8.0, 6.0))
    for position, label in enumerate(selected):
        index = available[label]
        binary = target == index
        fpr, tpr, _ = roc_curve(binary, probability[:, index])
        auc_value = roc_auc_score(binary, probability[:, index])
        axis.plot(fpr, tpr, linewidth=1.7, color=COLORS[position % len(COLORS)], label=f"{label}: {auc_value:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#6B7280", linewidth=1)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("One-vs-Rest ROC-кривые выбранных классов (ансамбль)")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    save_figure(figure, output_dir / "roc_curves_selected_classes.png")

    figure, axis = plt.subplots(figsize=(8.0, 6.0))
    for position, label in enumerate(selected):
        index = available[label]
        binary = target == index
        precision, recall, _ = precision_recall_curve(binary, probability[:, index])
        ap = average_precision_score(binary, probability[:, index])
        axis.plot(recall, precision, linewidth=1.7, color=COLORS[position % len(COLORS)], label=f"{label}: {ap:.3f}")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("One-vs-Rest Precision–Recall выбранных классов (ансамбль)")
    axis.legend(frameon=False, fontsize=8, ncol=2)
    save_figure(figure, output_dir / "precision_recall_selected_classes.png")


def plot_confusion_diagnostics(
    matrix: np.ndarray,
    per_class: pd.DataFrame,
    classes: list[str],
    validation_counts: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    n_classes = len(classes)
    top = np.argsort(validation_counts)[-30:]
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    )
    figure, axis = plt.subplots(figsize=(11.0, 9.5))
    image = axis.imshow(normalized[np.ix_(top, top)], cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(len(top)), labels=[classes[i] for i in top], rotation=90, fontsize=7)
    axis.set_yticks(np.arange(len(top)), labels=[classes[i] for i in top], fontsize=7)
    axis.set_xlabel("Предсказанный класс")
    axis.set_ylabel("Истинный класс")
    axis.set_title("Нормализованная confusion matrix: 30 крупнейших классов")
    figure.colorbar(image, ax=axis, pad=0.02, label="Доля объектов истинного класса")
    axis.grid(False)
    save_figure(figure, output_dir / "confusion_matrix_top30.png")

    worst = per_class.nsmallest(20, "f1").sort_values("f1")
    figure, axis = plt.subplots(figsize=(8.8, 6.0))
    bars = axis.barh([classes[i] for i in worst["class_index"]], worst["f1"], color=COLORS[3])
    axis.bar_label(bars, labels=[f"{value:.3f}" for value in worst["f1"]], padding=3, fontsize=8)
    axis.set_xlim(0, 1.05)
    axis.set_xlabel("F1")
    axis.set_title("Двадцать классов с минимальным F1 (ансамбль)")
    save_figure(figure, output_dir / "lowest_class_f1.png")

    rows: list[dict[str, object]] = []
    for true_index in range(n_classes):
        total = int(matrix[true_index].sum())
        if total == 0:
            continue
        for predicted_index in np.flatnonzero(matrix[true_index]):
            if predicted_index == true_index:
                continue
            count = int(matrix[true_index, predicted_index])
            rows.append(
                {
                    "true_class": classes[true_index],
                    "predicted_class": classes[predicted_index],
                    "count": count,
                    "share_of_true_class": count / total,
                }
            )
    pairs = pd.DataFrame(rows).sort_values(["count", "share_of_true_class"], ascending=False)
    top_pairs = pairs.head(20).sort_values("count")
    labels = [f"{row.true_class} → {row.predicted_class}" for row in top_pairs.itertuples()]
    figure, axis = plt.subplots(figsize=(9.5, 7.2))
    bars = axis.barh(labels, top_pairs["count"], color=COLORS[1])
    axis.bar_label(bars, padding=3, fontsize=8)
    axis.set_xlabel("Число ошибок в диагностической выборке")
    axis.set_title("Наиболее частые пары ошибок ансамбля")
    save_figure(figure, output_dir / "top_confusion_pairs.png")
    return pairs


def plot_calibration(
    target: np.ndarray, probabilities: dict[str, np.ndarray], output_dir: Path
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 6.0))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#6B7280", linewidth=1, label="идеальная калибровка")
    for position, (name, probability) in enumerate(probabilities.items()):
        ece, confidence, accuracy, counts = expected_calibration_error(target, probability)
        valid = counts > 0
        axis.plot(confidence[valid], accuracy[valid], marker="o", linewidth=1.7, color=COLORS[position], label=f"{name.upper()}, ECE={ece:.3f}")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Средняя уверенность")
    axis.set_ylabel("Фактическая accuracy")
    axis.set_title("Калибровка максимальной вероятности")
    axis.legend(frameon=False)
    save_figure(figure, output_dir / "calibration_curves.png")


def plot_top_k(metrics: dict[str, dict[str, float]], output_dir: Path) -> None:
    names = list(metrics)
    measures = ["accuracy", "top_3_accuracy", "top_5_accuracy"]
    labels = ["Top-1", "Top-3", "Top-5"]
    x = np.arange(len(names))
    width = 0.22
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    for index, (measure, label) in enumerate(zip(measures, labels)):
        values = [metrics[name][measure] for name in names]
        axis.bar(x + (index - 1) * width, values, width=width, label=label, color=COLORS[index])
    axis.set_xticks(x, labels=[name.upper() for name in names])
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Accuracy")
    axis.set_title("Top-k accuracy")
    axis.legend(frameon=False)
    save_figure(figure, output_dir / "top_k_accuracy.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-dir", type=Path)
    source.add_argument("--train-csv", type=Path)
    parser.add_argument("--cnn-checkpoint", type=Path, required=True)
    parser.add_argument("--catboost-model", type=Path)
    parser.add_argument("--ensemble-config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-size", type=float, default=0.05)
    parser.add_argument("--max-rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--csv-chunk-size", type=int, default=250_000)
    parser.add_argument("--cnn-batch-size", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.val_size < 1:
        raise ValueError("--val-size must be between 0 and 1")
    if args.ensemble_config and not args.catboost_model:
        raise ValueError("--ensemble-config requires --catboost-model")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_plots()

    checkpoint = torch.load(args.cnn_checkpoint, map_location="cpu", weights_only=False)
    classes = [str(label) for label in checkpoint["labels"]]
    n_classes = len(classes)
    packets, target, sample_indices, validation_counts = load_validation_sample(
        args.data_dir,
        args.train_csv,
        classes,
        args.val_size,
        args.max_rows,
        args.seed,
        args.csv_chunk_size,
    )
    if len(np.unique(target)) != n_classes:
        raise ValueError("Diagnostic sample must contain every class")

    probabilities: dict[str, np.ndarray] = {}
    probabilities["cnn"], device = cnn_probabilities(
        checkpoint, packets, args.cnn_batch_size
    )
    if args.catboost_model:
        cat_model = CatBoostClassifier()
        cat_model.load_model(args.catboost_model)
        features = extract_features(packets)
        probabilities["catboost"] = np.asarray(cat_model.predict_proba(features), dtype=np.float32)
    cnn_weight = None
    if args.ensemble_config:
        config = json.loads(args.ensemble_config.read_text(encoding="utf-8"))
        cnn_weight = float(config["best"]["cnn_weight"])
        probabilities["ensemble"] = (
            cnn_weight * probabilities["cnn"]
            + (1.0 - cnn_weight) * probabilities["catboost"]
        )

    metrics: dict[str, dict[str, float]] = {}
    class_frames: list[pd.DataFrame] = []
    matrices: dict[str, np.ndarray] = {}
    for name, probability in probabilities.items():
        metrics[name], per_class, matrices[name] = model_metrics(target, probability, n_classes)
        per_class.insert(0, "model", name)
        per_class.insert(2, "app_service", classes)
        class_frames.append(per_class)

    detailed_name = "ensemble" if "ensemble" in probabilities else "cnn"
    detailed_frame = class_frames[list(probabilities).index(detailed_name)]
    pairs = plot_confusion_diagnostics(
        matrices[detailed_name],
        detailed_frame,
        classes,
        validation_counts,
        args.output_dir,
    )
    plot_auc_comparison(metrics, args.output_dir)
    plot_selected_curves(target, probabilities[detailed_name], classes, args.output_dir)
    plot_calibration(target, probabilities, args.output_dir)
    plot_top_k(metrics, args.output_dir)

    pd.concat(class_frames, ignore_index=True).to_csv(
        args.output_dir / "per_class_metrics.csv", index=False
    )
    pairs.to_csv(args.output_dir / "confusion_pairs.csv", index=False)
    np.save(args.output_dir / "diagnostic_sample_indices.npy", sample_indices)
    payload = {
        "evaluation_type": "stratified_sample_of_untouched_holdout",
        "warning": (
            "ROC/PR use model probabilities on the original seed=42 validation split. "
            "When max_rows is non-zero, the holdout is stratified down to that size."
        ),
        "rows": int(len(target)),
        "full_holdout_rows": int(validation_counts.sum()),
        "classes": n_classes,
        "seed": args.seed,
        "val_size": args.val_size,
        "device": device,
        "cnn_weight": cnn_weight,
        "metrics": metrics,
    }
    (args.output_dir / "evaluation_diagnostics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
