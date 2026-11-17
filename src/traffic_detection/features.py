"""Feature engineering for signed packet-length sequences."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import N_PACKETS, iter_batches


FEATURE_NAMES = (
    "flow_length",
    "client_packets",
    "server_packets",
    "client_bytes",
    "server_bytes",
    "total_bytes",
    "signed_bytes",
    "client_packet_share",
    "packet_direction_balance",
    "byte_direction_balance",
    "mean_abs_size",
    "std_abs_size",
    "min_abs_size",
    "max_abs_size",
    "q25_abs_size",
    "median_abs_size",
    "q75_abs_size",
    "mean_client_size",
    "mean_server_size",
    "direction_changes",
    "direction_runs",
    "max_client_run",
    "max_server_run",
    "packets_le_64",
    "packets_65_1199",
    "packets_ge_1200",
    "packets_ge_1400",
    "unique_abs_sizes",
    "first_signed_size",
    "last_signed_size",
    "prefix_3_abs_bytes",
    "prefix_5_abs_bytes",
    "prefix_10_abs_bytes",
)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(numerator, dtype=np.float32)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def _max_direction_run(signs: np.ndarray, direction: int) -> np.ndarray:
    current = np.zeros(signs.shape[0], dtype=np.int16)
    maximum = np.zeros(signs.shape[0], dtype=np.int16)
    for column in range(signs.shape[1]):
        current = np.where(signs[:, column] == direction, current + 1, 0)
        maximum = np.maximum(maximum, current)
    return maximum


def extract_features(sequences: np.ndarray, *, include_raw: bool = True) -> np.ndarray:
    """Return deterministic ``float32`` features for ``(n, 30)`` sequences.

    The sign encodes direction (positive=client, negative=server), while zero
    is treated exclusively as padding.  When ``include_raw`` is true, the 30
    original ordered values are prepended so tree models can still learn
    position-specific fingerprints.
    """

    values = np.asarray(sequences)
    if values.ndim != 2 or values.shape[1] != N_PACKETS:
        raise ValueError(f"sequences must have shape (n, {N_PACKETS}), got {values.shape}")

    values_i32 = values.astype(np.int32, copy=False)
    absolute = np.abs(values_i32)
    signs = np.sign(values_i32).astype(np.int8)
    valid = values_i32 != 0
    client = values_i32 > 0
    server = values_i32 < 0

    length = valid.sum(axis=1)
    client_packets = client.sum(axis=1)
    server_packets = server.sum(axis=1)
    client_bytes = np.where(client, values_i32, 0).sum(axis=1, dtype=np.int64)
    server_bytes = np.where(server, -values_i32, 0).sum(axis=1, dtype=np.int64)
    total_bytes = client_bytes + server_bytes
    signed_bytes = client_bytes - server_bytes

    abs_sum = absolute.sum(axis=1, dtype=np.int64)
    mean_abs = _safe_divide(abs_sum, length)
    squared_sum = np.square(absolute, dtype=np.float64).sum(axis=1)
    variance = _safe_divide(squared_sum, length) - np.square(mean_abs)
    std_abs = np.sqrt(np.maximum(variance, 0.0), dtype=np.float32)

    masked_min = np.where(valid, absolute, np.iinfo(np.int32).max).min(axis=1)
    min_abs = np.where(length > 0, masked_min, 0)
    max_abs = absolute.max(axis=1)

    quantiles = np.zeros((len(values_i32), 3), dtype=np.float32)
    nonempty = length > 0
    if nonempty.any():
        with_nan = np.where(valid[nonempty], absolute[nonempty], np.nan).astype(np.float32)
        quantiles[nonempty] = np.nanquantile(with_nan, (0.25, 0.5, 0.75), axis=1).T

    adjacent_nonzero = valid[:, :-1] & valid[:, 1:]
    direction_changes = ((signs[:, :-1] != signs[:, 1:]) & adjacent_nonzero).sum(axis=1)
    direction_runs = np.where(length > 0, direction_changes + 1, 0)

    sorted_abs = np.sort(absolute, axis=1)
    unique_abs = (
        (sorted_abs[:, 0] > 0).astype(np.int16)
        + ((sorted_abs[:, 1:] != sorted_abs[:, :-1]) & (sorted_abs[:, 1:] > 0)).sum(axis=1)
    )

    last_index = np.maximum(length - 1, 0)
    last_value = values_i32[np.arange(len(values_i32)), last_index]
    last_value = np.where(nonempty, last_value, 0)

    engineered = np.column_stack(
        (
            length,
            client_packets,
            server_packets,
            client_bytes,
            server_bytes,
            total_bytes,
            signed_bytes,
            _safe_divide(client_packets, length),
            _safe_divide(client_packets - server_packets, length),
            _safe_divide(signed_bytes, total_bytes),
            mean_abs,
            std_abs,
            min_abs,
            max_abs,
            quantiles[:, 0],
            quantiles[:, 1],
            quantiles[:, 2],
            _safe_divide(client_bytes, client_packets),
            _safe_divide(server_bytes, server_packets),
            direction_changes,
            direction_runs,
            _max_direction_run(signs, 1),
            _max_direction_run(signs, -1),
            ((absolute > 0) & (absolute <= 64)).sum(axis=1),
            ((absolute >= 65) & (absolute < 1200)).sum(axis=1),
            (absolute >= 1200).sum(axis=1),
            (absolute >= 1400).sum(axis=1),
            unique_abs,
            values_i32[:, 0],
            last_value,
            absolute[:, :3].sum(axis=1),
            absolute[:, :5].sum(axis=1),
            absolute[:, :10].sum(axis=1),
        )
    ).astype(np.float32, copy=False)
    if include_raw:
        return np.concatenate((values_i32.astype(np.float32), engineered), axis=1)
    return engineered


def merge_large_same_direction(
    sequences: np.ndarray, *, threshold: int = 1200
) -> np.ndarray:
    """Merge consecutive large packets with the same direction.

    Every maximal run whose individual absolute packet sizes are greater than the
    threshold is replaced by its signed sum.  Other packets remain unchanged;
    the shortened sequence is padded with zeros.  ``int32`` output prevents
    overflow when many MTU-sized packets are combined.
    """

    values = np.asarray(sequences)
    if values.ndim != 2 or values.shape[1] != N_PACKETS:
        raise ValueError(f"sequences must have shape (n, {N_PACKETS}), got {values.shape}")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    output = np.zeros(values.shape, dtype=np.int32)
    values_i32 = values.astype(np.int32, copy=False)
    for row_index, row in enumerate(values_i32):
        destination = 0
        source = 0
        while source < N_PACKETS and row[source] != 0:
            current = int(row[source])
            if abs(current) > threshold:
                direction = 1 if current > 0 else -1
                total = current
                source += 1
                while (
                    source < N_PACKETS
                    and row[source] != 0
                    and (1 if row[source] > 0 else -1) == direction
                    and abs(int(row[source])) > threshold
                ):
                    total += int(row[source])
                    source += 1
                output[row_index, destination] = total
            else:
                output[row_index, destination] = current
                source += 1
            destination += 1
    return output


def build_feature_memmap(
    packet_path: str | Path,
    output_path: str | Path,
    *,
    batch_size: int = 100_000,
    include_raw: bool = True,
) -> Path:
    """Compute features in batches and write a memory-mappable ``.npy`` file."""

    packets = np.load(Path(packet_path), mmap_mode="r")
    if packets.ndim != 2 or packets.shape[1] != N_PACKETS:
        raise ValueError(f"Packet array must have shape (n, {N_PACKETS})")
    columns = len(FEATURE_NAMES) + (N_PACKETS if include_raw else 0)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    features = np.lib.format.open_memmap(
        destination, mode="w+", dtype=np.float32, shape=(len(packets), columns)
    )
    for start, stop, batch in iter_batches(packets, batch_size):
        features[start:stop] = extract_features(batch, include_raw=include_raw)
    features.flush()
    del features
    return destination
