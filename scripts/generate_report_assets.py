#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "blue": "#275D8C",
    "orange": "#D17A22",
    "green": "#37805B",
    "red": "#B84A4A",
    "gray": "#6B7280",
}


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.22,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def weighted_quantile_from_counts(counts: np.ndarray, quantile: float) -> int:
    cumulative = np.cumsum(counts, dtype=np.int64)
    if cumulative[-1] == 0:
        return 0
    threshold = int(np.ceil(quantile * cumulative[-1]))
    return int(np.searchsorted(cumulative, threshold, side="left"))


def analyze_packets(
    data_dir: Path,
    output_dir: Path,
    batch_size: int,
) -> dict[str, object]:
    packets = np.load(data_dir / "train_packets.npy", mmap_mode="r")
    labels = np.load(data_dir / "train_labels.npy", mmap_mode="r")
    classes = json.loads((data_dir / "classes.json").read_text(encoding="utf-8"))
    n_classes = len(classes)
    class_counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(np.int64)
    flow_lengths = np.zeros(31, dtype=np.int64)
    magnitude_counts = np.zeros(32769, dtype=np.int64)
    position_nonzero = np.zeros(30, dtype=np.int64)
    positive_packets = 0
    negative_packets = 0
    padding_values = 0
    non_tail_padding_rows = 0
    min_packet = 0
    max_packet = 0

    for start in range(0, len(packets), batch_size):
        stop = min(start + batch_size, len(packets))
        batch = np.asarray(packets[start:stop], dtype=np.int32)
        valid = batch != 0
        lengths = valid.sum(axis=1)
        flow_lengths += np.bincount(lengths, minlength=31)
        position_nonzero += valid.sum(axis=0)
        positive_packets += int((batch > 0).sum())
        negative_packets += int((batch < 0).sum())
        padding_values += int((batch == 0).sum())
        seen_padding = np.maximum.accumulate(batch == 0, axis=1)
        non_tail_padding_rows += int((seen_padding & valid).any(axis=1).sum())
        nonzero_values = batch[valid]
        if len(nonzero_values):
            min_packet = min(min_packet, int(nonzero_values.min()))
            max_packet = max(max_packet, int(nonzero_values.max()))
            absolute = np.abs(nonzero_values)
            magnitude_counts += np.bincount(absolute, minlength=len(magnitude_counts))[
                : len(magnitude_counts)
            ]
        print(f"dataset analysis: {stop:,}/{len(packets):,}")

    nonzero_packets = positive_packets + negative_packets
    size_ranges = [
        (1, 64, "1–64"),
        (65, 255, "65–255"),
        (256, 511, "256–511"),
        (512, 1199, "512–1199"),
        (1200, 1399, "1200–1399"),
        (1400, 1499, "1400–1499"),
        (1500, 32768, "≥1500"),
    ]
    size_groups = {
        label: int(magnitude_counts[left : right + 1].sum())
        for left, right, label in size_ranges
    }
    test_rows = None
    test_metadata = data_dir / "test_metadata.json"
    if test_metadata.exists():
        test_rows = int(json.loads(test_metadata.read_text(encoding="utf-8"))["rows"])

    order = np.argsort(class_counts)[::-1]
    statistics: dict[str, object] = {
        "train_rows": int(len(packets)),
        "test_rows": test_rows,
        "classes": n_classes,
        "packet_positions": int(packets.shape[1]),
        "packet_dtype": str(packets.dtype),
        "class_count_min": int(class_counts.min()),
        "class_count_median": float(np.median(class_counts)),
        "class_count_max": int(class_counts.max()),
        "class_imbalance_ratio": float(class_counts.max() / class_counts.min()),
        "largest_class": {"label": classes[int(order[0])], "rows": int(class_counts[order[0]])},
        "smallest_class": {
            "label": classes[int(order[-1])],
            "rows": int(class_counts[order[-1]]),
        },
        "mean_flow_length": float(
            np.dot(np.arange(31, dtype=np.int64), flow_lengths) / len(packets)
        ),
        "median_flow_length": weighted_quantile_from_counts(flow_lengths, 0.5),
        "full_length_flows": int(flow_lengths[30]),
        "positive_packets": positive_packets,
        "negative_packets": negative_packets,
        "padding_values": padding_values,
        "client_packet_share": float(positive_packets / max(nonzero_packets, 1)),
        "non_tail_padding_rows": non_tail_padding_rows,
        "minimum_signed_packet": min_packet,
        "maximum_signed_packet": max_packet,
        "packet_magnitude_q25": weighted_quantile_from_counts(magnitude_counts, 0.25),
        "packet_magnitude_median": weighted_quantile_from_counts(magnitude_counts, 0.5),
        "packet_magnitude_q75": weighted_quantile_from_counts(magnitude_counts, 0.75),
        "packet_magnitude_q95": weighted_quantile_from_counts(magnitude_counts, 0.95),
        "class_counts": {classes[index]: int(class_counts[index]) for index in range(n_classes)},
        "flow_length_counts": {str(i): int(value) for i, value in enumerate(flow_lengths)},
        "packet_magnitude_groups": size_groups,
        "position_nonzero_counts": [int(value) for value in position_nonzero],
    }
    (output_dir / "dataset_statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sorted_counts = class_counts[order]
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    axis.plot(np.arange(1, n_classes + 1), sorted_counts, color=COLORS["blue"], linewidth=1.8)
    axis.fill_between(np.arange(1, n_classes + 1), sorted_counts, color=COLORS["blue"], alpha=0.12)
    axis.set_yscale("log")
    axis.set_title("Распределение числа потоков по классам")
    axis.set_xlabel("Класс после сортировки по убыванию")
    axis.set_ylabel("Число потоков логарифмическая шкала")
    save_figure(figure, output_dir / "class_distribution_all.png")

    top = order[:20][::-1]
    figure, axis = plt.subplots(figsize=(8.2, 6.0))
    axis.barh([str(classes[i]) for i in top], class_counts[top], color=COLORS["blue"])
    axis.set_title("Двадцать крупнейших классов")
    axis.set_xlabel("Число потоков")
    axis.grid(axis="x", alpha=0.22)
    axis.grid(axis="y", visible=False)
    save_figure(figure, output_dir / "class_distribution_top20.png")

    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    axis.bar(np.arange(1, 31), flow_lengths[1:], color=COLORS["green"], width=0.82)
    axis.set_title("Распределение длины усечённых TCP потоков")
    axis.set_xlabel("Число непустых пакетов в потоке")
    axis.set_ylabel("Число потоков")
    axis.set_xticks(np.arange(1, 31, 2))
    save_figure(figure, output_dir / "flow_length_distribution.png")

    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    labels_for_plot = list(size_groups)
    values = np.asarray(list(size_groups.values()), dtype=np.float64)
    axis.bar(labels_for_plot, values / max(values.sum(), 1.0) * 100, color=COLORS["orange"])
    axis.set_title("Распределение абсолютных размеров TCP пакетов")
    axis.set_xlabel("Абсолютный размер пакета в байтах")
    axis.set_ylabel("Доля пакетов в процентах")
    axis.tick_params(axis="x", rotation=25)
    save_figure(figure, output_dir / "packet_size_distribution.png")

    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    axis.plot(np.arange(1, 31), position_nonzero / len(packets) * 100, marker="o", markersize=3.5, color=COLORS["red"])
    axis.set_title("Заполненность позиций в последовательности")
    axis.set_xlabel("Номер пакета")
    axis.set_ylabel("Доля потоков с пакетом в позиции в процентах")
    axis.set_xticks(np.arange(1, 31, 2))
    axis.set_ylim(0, 105)
    save_figure(figure, output_dir / "sequence_position_occupancy.png")
    return statistics


def cv_rows(path: Path) -> tuple[str, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    name = str(payload["model"])
    rows = payload["fold_results"]
    if name == "PacketCNN":
        values = [float(row["best_metrics"]["macro_f1"]) for row in rows]
        name = "CNN"
    else:
        values = [float(row["macro_f1"]) for row in rows]
    return name, values


def plot_cross_validation(paths: list[Path], output_dir: Path) -> dict[str, object]:
    series = [cv_rows(path) for path in paths if path and path.exists()]
    if not series:
        return {}
    n_folds = len(series[0][1])
    x = np.arange(1, n_folds + 1)
    width = 0.72 / len(series)
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    summary: dict[str, object] = {}
    for position, (name, values_list) in enumerate(series):
        values = np.asarray(values_list)
        offset = (position - (len(series) - 1) / 2) * width
        axis.bar(x + offset, values, width=width, label=name, color=colors[position])
        summary[name] = {
            "fold_macro_f1": [float(value) for value in values],
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    axis.set_title("Macro F1 по фолдам кросс валидации")
    axis.set_xlabel("Номер фолда")
    axis.set_ylabel("Macro F1")
    axis.set_xticks(x)
    axis.set_ylim(0, 1)
    axis.legend(frameon=False)
    save_figure(figure, output_dir / "cross_validation_folds.png")
    (output_dir / "cross_validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def plot_training_and_models(
    cnn_history_path: Path | None,
    catboost_metrics_path: Path | None,
    ensemble_path: Path | None,
    output_dir: Path,
) -> None:
    history = None
    if cnn_history_path and cnn_history_path.exists():
        history = json.loads(cnn_history_path.read_text(encoding="utf-8"))
        figure, axis = plt.subplots(figsize=(8.2, 4.6))
        epochs = [int(row["epoch"]) for row in history]
        axis.plot(epochs, [row["macro_f1"] for row in history], marker="o", label="Macro F1", color=COLORS["blue"])
        axis.plot(epochs, [row["weighted_f1"] for row in history], marker="o", label="Weighted F1", color=COLORS["green"])
        axis.plot(epochs, [row["accuracy"] for row in history], marker="o", label="Accuracy", color=COLORS["orange"])
        axis.set_title("Качество CNN по эпохам")
        axis.set_xlabel("Эпоха")
        axis.set_ylabel("Значение метрики")
        axis.set_ylim(0, 1)
        axis.set_xticks(epochs)
        axis.legend(frameon=False)
        save_figure(figure, output_dir / "cnn_training_history.png")

    names: list[str] = []
    values: list[float] = []
    if catboost_metrics_path and catboost_metrics_path.exists():
        metrics = json.loads(catboost_metrics_path.read_text(encoding="utf-8"))
        names.append("CatBoost")
        values.append(float(metrics["macro_f1"]))
    if history:
        names.append("CNN")
        values.append(float(max(history, key=lambda row: row["macro_f1"])["macro_f1"]))
    if ensemble_path and ensemble_path.exists():
        ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
        names.append("Ансамбль")
        values.append(float(ensemble["best"]["macro_f1"]))
    if names:
        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        bars = axis.bar(names, values, color=[COLORS["orange"], COLORS["blue"], COLORS["green"]][: len(names)])
        axis.set_title("Сравнение моделей на общей отложенной выборке")
        axis.set_ylabel("Macro F1")
        axis.set_ylim(0, 1)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.4f}", ha="center")
        save_figure(figure, output_dir / "model_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dataset statistics and report figures")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--catboost-cv", type=Path)
    parser.add_argument("--cnn-cv", type=Path)
    parser.add_argument("--catboost-metrics", type=Path)
    parser.add_argument("--cnn-history", type=Path)
    parser.add_argument("--ensemble", type=Path)
    parser.add_argument("--batch-size", type=int, default=100_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_plots()
    analyze_packets(args.data_dir, args.output_dir, args.batch_size)
    plot_cross_validation(
        [path for path in (args.catboost_cv, args.cnn_cv) if path is not None],
        args.output_dir,
    )
    plot_training_and_models(
        args.cnn_history,
        args.catboost_metrics,
        args.ensemble,
        args.output_dir,
    )
    print("Report assets:", args.output_dir)


if __name__ == "__main__":
    main()
