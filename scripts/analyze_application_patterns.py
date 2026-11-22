#!/usr/bin/env python3
"""Build reproducible per-application traffic profiles and visual diagnostics.

The script never reads the full competition dataset into RAM.  It can consume
either the original ``train.csv`` in chunks or the prepared NumPy memmaps made
by ``prepare_data.py``.  All reported values are descriptive statistics; no
model is trained here.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


N_PACKETS = 30
PROFILE_FEATURES = [
    "mean_flow_length",
    "full_length_share",
    "client_packet_share",
    "client_byte_share",
    "mean_abs_packet_size",
    "mean_direction_changes",
    "small_packet_share",
    "large_packet_share",
    "near_mtu_packet_share",
]
FEATURE_LABELS_RU = {
    "mean_flow_length": "средняя длина потока",
    "full_length_share": "доля потоков длины 30",
    "client_packet_share": "доля пакетов клиента",
    "client_byte_share": "доля байтов клиента",
    "mean_abs_packet_size": "средний размер пакета",
    "mean_direction_changes": "смены направления",
    "small_packet_share": "доля пакетов ≤64 Б",
    "large_packet_share": "доля пакетов ≥1200 Б",
    "near_mtu_packet_share": "доля пакетов ≥1400 Б",
}
PREFERRED_LABELS = [
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
    "Instagram",
    "GMail",
]
COLORS = ["#275D8C", "#D17A22", "#37805B", "#B84A4A", "#7656A8", "#2A8C9D"]


@dataclass
class ClassAccumulator:
    flows: int = 0
    packets: int = 0
    client_packets: int = 0
    server_packets: int = 0
    abs_bytes: int = 0
    client_bytes: int = 0
    server_bytes: int = 0
    direction_changes: int = 0
    small_packets: int = 0
    medium_packets: int = 0
    large_packets: int = 0
    near_mtu_packets: int = 0
    first_packet_client: int = 0
    first_packet_abs_sum: int = 0
    prefix_5_abs_sum: int = 0
    flow_length_hist: np.ndarray = field(
        default_factory=lambda: np.zeros(N_PACKETS + 1, dtype=np.int64)
    )
    position_nonzero: np.ndarray = field(
        default_factory=lambda: np.zeros(N_PACKETS, dtype=np.int64)
    )
    position_positive: np.ndarray = field(
        default_factory=lambda: np.zeros(N_PACKETS, dtype=np.int64)
    )
    position_abs_sum: np.ndarray = field(
        default_factory=lambda: np.zeros(N_PACKETS, dtype=np.int64)
    )


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


def _safe_divide(numerator: float | np.ndarray, denominator: float | np.ndarray):
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(np.asarray(numerator), dtype=np.float64),
        where=np.asarray(denominator) != 0,
    )


def _histogram_median(histogram: np.ndarray) -> int:
    total = int(histogram.sum())
    if total == 0:
        return 0
    return int(np.searchsorted(np.cumsum(histogram), (total + 1) // 2))


def iter_csv_batches(path: Path, chunk_size: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for chunk in pd.read_csv(path, chunksize=chunk_size, dtype={"app_service": "string"}):
        labels = chunk.iloc[:, 0].astype(str).to_numpy()
        packets = (
            chunk.iloc[:, 1 : N_PACKETS + 1]
            .fillna(0)
            .to_numpy(dtype=np.int16, copy=False)
        )
        yield labels, packets


def iter_prepared_batches(
    data_dir: Path, batch_size: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    packets = np.load(data_dir / "train_packets.npy", mmap_mode="r")
    encoded = np.load(data_dir / "train_labels.npy", mmap_mode="r")
    classes = np.asarray(
        json.loads((data_dir / "classes.json").read_text(encoding="utf-8")), dtype=object
    )
    for start in range(0, len(packets), batch_size):
        stop = min(start + batch_size, len(packets))
        yield classes[np.asarray(encoded[start:stop], dtype=np.int64)], np.asarray(
            packets[start:stop], dtype=np.int16
        )


def update_accumulators(
    accumulators: dict[str, ClassAccumulator], labels: np.ndarray, packets: np.ndarray
) -> None:
    values = packets.astype(np.int32, copy=False)
    absolute = np.abs(values)
    valid = values != 0
    positive = values > 0
    negative = values < 0
    lengths = valid.sum(axis=1)
    adjacent_valid = valid[:, :-1] & valid[:, 1:]
    direction_changes = (
        (np.sign(values[:, :-1]) != np.sign(values[:, 1:])) & adjacent_valid
    ).sum(axis=1)

    unique_labels, inverse = np.unique(labels.astype(str), return_inverse=True)
    for local_index, label in enumerate(unique_labels):
        mask = inverse == local_index
        rows = values[mask]
        row_abs = absolute[mask]
        row_valid = valid[mask]
        row_positive = positive[mask]
        row_negative = negative[mask]
        row_lengths = lengths[mask]
        accumulator = accumulators[str(label)]

        accumulator.flows += int(mask.sum())
        accumulator.packets += int(row_valid.sum())
        accumulator.client_packets += int(row_positive.sum())
        accumulator.server_packets += int(row_negative.sum())
        accumulator.abs_bytes += int(row_abs.sum(dtype=np.int64))
        accumulator.client_bytes += int(np.where(row_positive, rows, 0).sum(dtype=np.int64))
        accumulator.server_bytes += int(
            np.where(row_negative, -rows, 0).sum(dtype=np.int64)
        )
        accumulator.direction_changes += int(direction_changes[mask].sum())
        accumulator.small_packets += int(((row_abs > 0) & (row_abs <= 64)).sum())
        accumulator.medium_packets += int(((row_abs >= 65) & (row_abs < 1200)).sum())
        accumulator.large_packets += int((row_abs >= 1200).sum())
        accumulator.near_mtu_packets += int((row_abs >= 1400).sum())
        accumulator.first_packet_client += int((rows[:, 0] > 0).sum())
        accumulator.first_packet_abs_sum += int(row_abs[:, 0].sum(dtype=np.int64))
        accumulator.prefix_5_abs_sum += int(row_abs[:, :5].sum(dtype=np.int64))
        accumulator.flow_length_hist += np.bincount(
            row_lengths, minlength=N_PACKETS + 1
        )[: N_PACKETS + 1]
        accumulator.position_nonzero += row_valid.sum(axis=0, dtype=np.int64)
        accumulator.position_positive += row_positive.sum(axis=0, dtype=np.int64)
        accumulator.position_abs_sum += row_abs.sum(axis=0, dtype=np.int64)


def collect_profiles(
    batches: Iterable[tuple[np.ndarray, np.ndarray]], total_rows: int | None = None
) -> dict[str, ClassAccumulator]:
    accumulators: dict[str, ClassAccumulator] = defaultdict(ClassAccumulator)
    processed = 0
    for labels, packets in batches:
        update_accumulators(accumulators, labels, packets)
        processed += len(labels)
        suffix = f"/{total_rows:,}" if total_rows is not None else ""
        print(f"application analysis: {processed:,}{suffix}", flush=True)
    return dict(accumulators)


def build_profile_frame(
    accumulators: dict[str, ClassAccumulator]
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    position_profiles: dict[str, np.ndarray] = {}
    for label, item in sorted(accumulators.items()):
        total_bytes = item.client_bytes + item.server_bytes
        rows.append(
            {
                "app_service": label,
                "flows": item.flows,
                "packets": item.packets,
                "mean_flow_length": item.packets / item.flows,
                "median_flow_length": _histogram_median(item.flow_length_hist),
                "full_length_share": item.flow_length_hist[N_PACKETS] / item.flows,
                "client_packet_share": item.client_packets / item.packets,
                "client_byte_share": item.client_bytes / total_bytes,
                "mean_abs_packet_size": item.abs_bytes / item.packets,
                "mean_direction_changes": item.direction_changes / item.flows,
                "small_packet_share": item.small_packets / item.packets,
                "medium_packet_share": item.medium_packets / item.packets,
                "large_packet_share": item.large_packets / item.packets,
                "near_mtu_packet_share": item.near_mtu_packets / item.packets,
                "first_packet_client_share": item.first_packet_client / item.flows,
                "mean_first_packet_size": item.first_packet_abs_sum / item.flows,
                "mean_prefix_5_bytes": item.prefix_5_abs_sum / item.flows,
            }
        )
        position_profiles[label] = np.vstack(
            (
                _safe_divide(item.position_nonzero, item.flows),
                _safe_divide(item.position_positive, item.position_nonzero),
                _safe_divide(item.position_abs_sum, item.position_nonzero),
            )
        )
    return pd.DataFrame(rows), position_profiles


def add_clusters(profiles: pd.DataFrame, n_clusters: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    standardized = scaler.fit_transform(profiles[PROFILE_FEATURES])
    model = KMeans(n_clusters=n_clusters, n_init=30, random_state=seed)
    profiles["cluster"] = model.fit_predict(standardized)
    profiles["pca_1"], profiles["pca_2"] = PCA(n_components=2, random_state=seed).fit_transform(
        standardized
    ).T
    return standardized, model.cluster_centers_


def build_cluster_summary(
    profiles: pd.DataFrame, standardized: np.ndarray, centers: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster in sorted(profiles["cluster"].unique()):
        mask = profiles["cluster"].to_numpy() == cluster
        center = centers[int(cluster)]
        dominant = np.argsort(np.abs(center))[::-1][:3]
        pattern_parts = [
            ("выше среднего: " if center[index] >= 0 else "ниже среднего: ")
            + FEATURE_LABELS_RU[PROFILE_FEATURES[index]]
            for index in dominant
        ]
        distances = np.linalg.norm(standardized[mask] - center, axis=1)
        members = profiles.loc[mask, "app_service"].to_numpy()
        representatives = members[np.argsort(distances)[:5]]
        row: dict[str, object] = {
            "cluster": int(cluster),
            "applications": int(mask.sum()),
            "flows": int(profiles.loc[mask, "flows"].sum()),
            "dominant_pattern": "; ".join(pattern_parts),
            "representative_applications": ", ".join(map(str, representatives)),
        }
        for feature in PROFILE_FEATURES:
            row[feature] = float(profiles.loc[mask, feature].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def build_nearest_neighbors(profiles: pd.DataFrame, standardized: np.ndarray) -> pd.DataFrame:
    labels = profiles["app_service"].astype(str).to_numpy()
    preferred = [label for label in PREFERRED_LABELS if label in set(labels)]
    rows: list[dict[str, object]] = []
    for label in preferred:
        index = int(np.flatnonzero(labels == label)[0])
        distances = np.linalg.norm(standardized - standardized[index], axis=1)
        for rank, neighbor_index in enumerate(np.argsort(distances)[1:6], start=1):
            rows.append(
                {
                    "app_service": label,
                    "rank": rank,
                    "nearest_application": labels[neighbor_index],
                    "standardized_distance": float(distances[neighbor_index]),
                }
            )
    return pd.DataFrame(rows)


def selected_labels(profiles: pd.DataFrame, maximum: int = 10) -> list[str]:
    available = set(profiles["app_service"].astype(str))
    selected = [label for label in PREFERRED_LABELS if label in available]
    if len(selected) < maximum:
        extremes: list[str] = []
        for feature in PROFILE_FEATURES:
            extremes.extend(
                profiles.nlargest(1, feature)["app_service"].astype(str).tolist()
            )
        selected.extend(label for label in extremes if label not in selected)
    return selected[:maximum]


def plot_class_balance(profiles: pd.DataFrame, output_dir: Path) -> None:
    ordered = profiles.sort_values("flows", ascending=False).reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(1, len(ordered) + 1)
    axis.plot(x, ordered["flows"], color=COLORS[0], linewidth=1.8)
    axis.fill_between(x, ordered["flows"], color=COLORS[0], alpha=0.12)
    axis.set_yscale("log")
    axis.set_title("Число обучающих потоков по классам")
    axis.set_xlabel("Класс после сортировки по убыванию")
    axis.set_ylabel("Число потоков (логарифмическая шкала)")
    save_figure(figure, output_dir / "class_frequency_all.png")

    top = ordered.head(20).sort_values("flows")
    figure, axis = plt.subplots(figsize=(8.8, 6.2))
    axis.barh(top["app_service"], top["flows"], color=COLORS[0])
    axis.set_title("Двадцать крупнейших классов")
    axis.set_xlabel("Число потоков")
    axis.grid(axis="x", alpha=0.2)
    axis.grid(axis="y", visible=False)
    save_figure(figure, output_dir / "class_frequency_top20.png")


def plot_behavior_scatter(profiles: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.0, 6.0))
    sizes = 18 + 95 * np.sqrt(profiles["flows"] / profiles["flows"].max())
    points = axis.scatter(
        profiles["mean_flow_length"],
        profiles["large_packet_share"] * 100,
        c=(1 - profiles["client_byte_share"]) * 100,
        s=sizes,
        cmap="viridis",
        alpha=0.74,
        linewidths=0.25,
        edgecolors="white",
    )
    candidates = pd.concat(
        [
            profiles.nlargest(3, "mean_flow_length"),
            profiles.nsmallest(3, "mean_flow_length"),
            profiles.nlargest(3, "large_packet_share"),
            profiles.nsmallest(3, "large_packet_share"),
        ]
    ).drop_duplicates("app_service")
    for _, row in candidates.iterrows():
        axis.annotate(
            str(row["app_service"]),
            (row["mean_flow_length"], row["large_packet_share"] * 100),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.5,
        )
    axis.set_title("Поведенческие профили приложений")
    axis.set_xlabel("Среднее число пакетов в усечённом потоке")
    axis.set_ylabel("Доля крупных пакетов ≥1200 Б, %")
    colorbar = figure.colorbar(points, ax=axis, pad=0.02)
    colorbar.set_label("Доля байтов от сервера, %")
    save_figure(figure, output_dir / "application_behavior_scatter.png")


def plot_feature_heatmap(profiles: pd.DataFrame, output_dir: Path) -> None:
    labels = selected_labels(profiles)
    selected = profiles.set_index("app_service").loc[labels, PROFILE_FEATURES]
    standardized = (selected - profiles[PROFILE_FEATURES].mean()) / profiles[
        PROFILE_FEATURES
    ].std(ddof=0)
    figure, axis = plt.subplots(figsize=(11.2, 6.0))
    image = axis.imshow(standardized, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    axis.set_yticks(np.arange(len(labels)), labels=labels)
    axis.set_xticks(
        np.arange(len(PROFILE_FEATURES)),
        labels=[FEATURE_LABELS_RU[name] for name in PROFILE_FEATURES],
        rotation=35,
        ha="right",
    )
    axis.set_title("Профили выбранных приложений относительно среднего по классам")
    axis.grid(False)
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Стандартизованное отклонение (z-score)")
    save_figure(figure, output_dir / "selected_application_heatmap.png")


def plot_position_profiles(
    profiles: pd.DataFrame, position_profiles: dict[str, np.ndarray], output_dir: Path
) -> None:
    labels = selected_labels(profiles, maximum=8)
    positions = np.arange(1, N_PACKETS + 1)

    figure, axis = plt.subplots(figsize=(9.4, 5.4))
    for index, label in enumerate(labels):
        axis.plot(
            positions,
            position_profiles[label][2],
            label=label,
            color=COLORS[index % len(COLORS)],
            linewidth=1.7,
            alpha=0.9,
        )
    axis.set_title("Средний абсолютный размер пакета по позиции")
    axis.set_xlabel("Позиция пакета в потоке")
    axis.set_ylabel("Размер, байт (среди непустых позиций)")
    axis.set_xticks(np.arange(1, N_PACKETS + 1, 2))
    axis.legend(ncol=2, frameon=False, fontsize=8)
    save_figure(figure, output_dir / "packet_position_size_profiles.png")

    direction_matrix = np.vstack([2 * position_profiles[label][1] - 1 for label in labels])
    occupancy = np.vstack([position_profiles[label][0] for label in labels])
    direction_matrix = np.where(occupancy > 0, direction_matrix, np.nan)
    figure, axis = plt.subplots(figsize=(11.0, 4.7))
    image = axis.imshow(direction_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axis.set_yticks(np.arange(len(labels)), labels=labels)
    axis.set_xticks(np.arange(N_PACKETS), labels=np.arange(1, N_PACKETS + 1))
    axis.set_title("Преобладающее направление пакетов по позиции")
    axis.set_xlabel("Позиция пакета в потоке")
    axis.grid(False)
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_ticks([-1, 0, 1], labels=["сервер", "баланс", "клиент"])
    save_figure(figure, output_dir / "packet_position_direction_heatmap.png")


def plot_clusters(
    profiles: pd.DataFrame, centers: np.ndarray, output_dir: Path
) -> None:
    figure, axis = plt.subplots(figsize=(9.0, 6.0))
    for cluster in sorted(profiles["cluster"].unique()):
        subset = profiles[profiles["cluster"] == cluster]
        axis.scatter(
            subset["pca_1"],
            subset["pca_2"],
            s=35,
            alpha=0.72,
            label=f"Кластер {cluster}",
            color=COLORS[int(cluster) % len(COLORS)],
        )
    for label in selected_labels(profiles, maximum=10):
        row = profiles[profiles["app_service"] == label].iloc[0]
        axis.annotate(label, (row["pca_1"], row["pca_2"]), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    axis.set_title("Кластеры приложений по агрегированным характеристикам")
    axis.set_xlabel("PCA-компонента 1")
    axis.set_ylabel("PCA-компонента 2")
    axis.legend(frameon=False, ncol=2)
    save_figure(figure, output_dir / "application_clusters_pca.png")

    figure, axis = plt.subplots(figsize=(11.2, 4.6))
    image = axis.imshow(centers, cmap="RdBu_r", vmin=-2.2, vmax=2.2, aspect="auto")
    axis.set_yticks(np.arange(len(centers)), labels=[f"Кластер {i}" for i in range(len(centers))])
    axis.set_xticks(
        np.arange(len(PROFILE_FEATURES)),
        labels=[FEATURE_LABELS_RU[name] for name in PROFILE_FEATURES],
        rotation=35,
        ha="right",
    )
    axis.set_title("Средние стандартизованные признаки кластеров")
    axis.grid(False)
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("z-score центра кластера")
    save_figure(figure, output_dir / "cluster_feature_heatmap.png")


def build_findings(
    profiles: pd.DataFrame, cluster_summary: pd.DataFrame
) -> dict[str, object]:
    extremes: dict[str, object] = {}
    for feature in PROFILE_FEATURES:
        columns = ["app_service", feature]
        extremes[feature] = {
            "highest": profiles.nlargest(5, feature)[columns].to_dict("records"),
            "lowest": profiles.nsmallest(5, feature)[columns].to_dict("records"),
        }
    correlations = profiles[PROFILE_FEATURES].corr()
    pairs: list[dict[str, object]] = []
    for left_index, left in enumerate(PROFILE_FEATURES):
        for right in PROFILE_FEATURES[left_index + 1 :]:
            pairs.append(
                {
                    "feature_1": left,
                    "feature_2": right,
                    "correlation": float(correlations.loc[left, right]),
                }
            )
    strongest = sorted(pairs, key=lambda row: abs(float(row["correlation"])), reverse=True)[:10]
    return {
        "applications": int(len(profiles)),
        "flows": int(profiles["flows"].sum()),
        "profile_features": PROFILE_FEATURES,
        "strongest_profile_correlations": strongest,
        "extremes": extremes,
        "clusters": cluster_summary.to_dict("records"),
        "method_note": (
            "Кластеры построены по стандартизованным средним профилям классов. "
            "Они являются описательным инструментом и не используются как истинные метки."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--train-csv", type=Path)
    source.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=200_000)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_plots()

    if args.train_csv is not None:
        batches = iter_csv_batches(args.train_csv, args.batch_size)
        total_rows = None
    else:
        packet_path = args.data_dir / "train_packets.npy"
        total_rows = len(np.load(packet_path, mmap_mode="r"))
        batches = iter_prepared_batches(args.data_dir, args.batch_size)

    accumulators = collect_profiles(batches, total_rows=total_rows)
    profiles, position_profiles = build_profile_frame(accumulators)
    standardized, centers = add_clusters(profiles, args.clusters, args.seed)
    cluster_summary = build_cluster_summary(profiles, standardized, centers)
    neighbors = build_nearest_neighbors(profiles, standardized)

    profiles.to_csv(args.output_dir / "application_profiles.csv", index=False)
    cluster_summary.to_csv(args.output_dir / "cluster_summary.csv", index=False)
    neighbors.to_csv(args.output_dir / "nearest_applications.csv", index=False)
    profiles.set_index("app_service").loc[
        selected_labels(profiles),
        ["flows", *PROFILE_FEATURES, "first_packet_client_share", "cluster"],
    ].reset_index().to_csv(args.output_dir / "selected_application_profiles.csv", index=False)
    np.savez_compressed(
        args.output_dir / "position_profiles.npz",
        labels=np.asarray(sorted(position_profiles), dtype=str),
        profiles=np.stack([position_profiles[label] for label in sorted(position_profiles)]),
    )

    plot_class_balance(profiles, args.output_dir)
    plot_behavior_scatter(profiles, args.output_dir)
    plot_feature_heatmap(profiles, args.output_dir)
    plot_position_profiles(profiles, position_profiles, args.output_dir)
    plot_clusters(profiles, centers, args.output_dir)
    findings = build_findings(profiles, cluster_summary)
    (args.output_dir / "pattern_findings.json").write_text(
        json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved application analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
