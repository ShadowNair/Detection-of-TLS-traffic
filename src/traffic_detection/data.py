"""Streaming CSV reader and NumPy memmap conversion utilities.

The competition files are too large for a typical ``pandas.read_csv`` call on
Colab.  This module deliberately depends only on the Python standard library
and NumPy.  It scans a CSV once to determine its row count and then writes the
30 packet values directly into an on-disk ``.npy`` array.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np


N_PACKETS = 30
PACKET_COLUMNS = tuple(f"tcp_len_{index}" for index in range(1, N_PACKETS + 1))
Split = Literal["train", "test"]


@dataclass(frozen=True)
class ConversionResult:
    """Paths and dimensions produced by :func:`convert_csv_to_memmap`."""

    output_dir: Path
    split: Split
    rows: int
    packet_path: Path
    first_column_path: Path
    metadata_path: Path


def _rows_and_header(csv_path: Path) -> tuple[int, list[str]]:
    """Count logical CSV records without loading the file into memory."""

    with csv_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"CSV is empty: {csv_path}") from error
        rows = sum(1 for row in reader if row)
    return rows, header


def _validate_header(header: list[str], split: Split) -> None:
    expected_first = "app_service" if split == "train" else "id"
    expected = [expected_first, *PACKET_COLUMNS]
    if header != expected:
        raise ValueError(
            "Unexpected CSV columns. "
            f"Expected {expected_first} followed by tcp_len_1..tcp_len_30, got {header!r}"
        )


def _parse_packet(value: str, *, row_number: int, column_name: str) -> int:
    """Parse integral packet sizes such as ``1448`` or ``-1448.0``."""

    if value == "" or value.lower() in {"null", "nan", "none"}:
        return 0
    try:
        numeric = float(value)
    except ValueError as error:
        raise ValueError(
            f"Invalid packet size {value!r} at row {row_number}, column {column_name}"
        ) from error
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(
            f"Packet size must be a finite integer at row {row_number}, "
            f"column {column_name}: {value!r}"
        )
    integer = int(numeric)
    limits = np.iinfo(np.int16)
    if not limits.min <= integer <= limits.max:
        raise ValueError(
            f"Packet size does not fit int16 at row {row_number}, "
            f"column {column_name}: {integer}"
        )
    return integer


def convert_csv_to_memmap(
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    split: Split,
    overwrite: bool = False,
) -> ConversionResult:
    """Convert a competition CSV to compact, memory-mappable ``.npy`` files.

    Train labels are always read as strings and encoded in order of first
    appearance.  ``classes.json`` provides the inverse mapping.  Test IDs are
    stored as ``int64``.  Missing packet values and explicit nulls become the
    dataset's padding value, zero.
    """

    source_path = Path(csv_path)
    destination = Path(output_dir)
    if split not in {"train", "test"}:
        raise ValueError("split must be either 'train' or 'test'")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    rows, header = _rows_and_header(source_path)
    _validate_header(header, split)
    destination.mkdir(parents=True, exist_ok=True)

    packet_path = destination / f"{split}_packets.npy"
    first_path = destination / ("train_labels.npy" if split == "train" else "test_ids.npy")
    metadata_path = destination / f"{split}_metadata.json"
    managed_paths = [packet_path, first_path, metadata_path]
    if split == "train":
        managed_paths.append(destination / "classes.json")
    existing = [path for path in managed_paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing output: " + ", ".join(map(str, existing))
        )

    packets = np.lib.format.open_memmap(
        packet_path, mode="w+", dtype=np.int16, shape=(rows, N_PACKETS)
    )
    first_values = np.lib.format.open_memmap(
        first_path,
        mode="w+",
        dtype=np.int32 if split == "train" else np.int64,
        shape=(rows,),
    )
    label_to_code: dict[str, int] = {}
    classes: list[str] = []

    try:
        with source_path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.reader(source)
            next(reader)
            output_row = 0
            for csv_row_number, row in enumerate(reader, start=2):
                if not row:
                    continue
                if len(row) != N_PACKETS + 1:
                    raise ValueError(
                        f"Expected {N_PACKETS + 1} fields at CSV row {csv_row_number}, "
                        f"got {len(row)}"
                    )
                if split == "train":
                    label = row[0]
                    if label not in label_to_code:
                        label_to_code[label] = len(classes)
                        classes.append(label)
                    first_values[output_row] = label_to_code[label]
                else:
                    try:
                        first_values[output_row] = int(row[0])
                    except ValueError as error:
                        raise ValueError(
                            f"Test id must be an integer at CSV row {csv_row_number}: {row[0]!r}"
                        ) from error

                packets[output_row] = [
                    _parse_packet(value, row_number=csv_row_number, column_name=column)
                    for value, column in zip(row[1:], PACKET_COLUMNS)
                ]
                output_row += 1
    except Exception:
        del packets
        del first_values
        for path in managed_paths:
            path.unlink(missing_ok=True)
        raise

    packets.flush()
    first_values.flush()
    del packets
    del first_values

    metadata = {
        "split": split,
        "rows": rows,
        "packets": N_PACKETS,
        "packet_dtype": "int16",
        "first_column": "app_service" if split == "train" else "id",
        "first_column_dtype": "int32" if split == "train" else "int64",
        "padding_value": 0,
    }
    if split == "train":
        metadata["classes"] = len(classes)
        with (destination / "classes.json").open("w", encoding="utf-8") as target:
            json.dump(classes, target, ensure_ascii=False, indent=2)
    with metadata_path.open("w", encoding="utf-8") as target:
        json.dump(metadata, target, ensure_ascii=False, indent=2)

    return ConversionResult(
        output_dir=destination,
        split=split,
        rows=rows,
        packet_path=packet_path,
        first_column_path=first_path,
        metadata_path=metadata_path,
    )


def load_conversion(
    output_dir: str | Path, split: Split
) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    """Open converted arrays read-only without copying them into RAM."""

    directory = Path(output_dir)
    packets = np.load(directory / f"{split}_packets.npy", mmap_mode="r")
    first_name = "train_labels.npy" if split == "train" else "test_ids.npy"
    first_values = np.load(directory / first_name, mmap_mode="r")
    classes = None
    if split == "train":
        with (directory / "classes.json").open("r", encoding="utf-8") as source:
            classes = json.load(source)
    return packets, first_values, classes


def iter_batches(array: np.ndarray, batch_size: int) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield ``(start, stop, view)`` batches for an array or memmap."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(array), batch_size):
        stop = min(start + batch_size, len(array))
        yield start, stop, array[start:stop]
