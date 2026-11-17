"""Memory-efficient exact and prefix lookup for packet sequences.

The lookup deliberately uses sorted NumPy arrays instead of a Python mapping.
At competition scale, a ``dict`` with millions of byte-string keys costs much
more memory than the packet data itself.  Each retained sequence is represented
by a pair of deterministic 64-bit hashes and is found with binary search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import overload

import numpy as np


_KEY_DTYPE = np.dtype([("first", "<u8"), ("second", "<u8")])


@dataclass(frozen=True)
class PrefixTable:
    """Compact table for one prefix length.

    ``labels`` contains indices into :attr:`PrefixHashLookup.classes_`, not the
    potentially wider original labels.
    """

    prefix_length: int
    keys: np.ndarray
    labels: np.ndarray
    support: np.ndarray
    matched_count: np.ndarray

    @property
    def purity(self) -> np.ndarray:
        return self.matched_count.astype(np.float64) / self.support

    @property
    def nbytes(self) -> int:
        return int(
            self.keys.nbytes
            + self.labels.nbytes
            + self.support.nbytes
            + self.matched_count.nbytes
        )


class PrefixHashLookup:
    """Learn reliable labels for complete sequences and their prefixes.

    Parameters
    ----------
    prefix_lengths:
        Prefixes to learn.  They are always evaluated from longest to shortest,
        irrespective of the order supplied by the caller.
    min_support:
        Minimum number of training rows with the same prefix.
    min_purity:
        Minimum fraction belonging to the majority class.  Impure or
        insufficiently supported prefixes are not stored.
    hash_batch_size:
        Number of rows hashed at once.  This bounds temporary memory during
        fitting and prediction; the sorted table itself remains contiguous.

    Notes
    -----
    Input labels must be integers, but do not need to be contiguous.  They are
    encoded internally to the smallest unsigned dtype and decoded by
    :meth:`predict`.
    """

    def __init__(
        self,
        prefix_lengths: tuple[int, ...] | list[int] = (30, 20, 16, 12, 8, 5),
        *,
        min_support: int = 2,
        min_purity: float = 1.0,
        hash_batch_size: int = 250_000,
    ) -> None:
        lengths = tuple(sorted({int(value) for value in prefix_lengths}, reverse=True))
        if not lengths or any(value <= 0 for value in lengths):
            raise ValueError("prefix_lengths must contain positive integers")
        if min_support < 1:
            raise ValueError("min_support must be at least 1")
        if not 0.0 < min_purity <= 1.0:
            raise ValueError("min_purity must be in (0, 1]")
        if hash_batch_size < 1:
            raise ValueError("hash_batch_size must be at least 1")

        self.prefix_lengths = lengths
        self.min_support = int(min_support)
        self.min_purity = float(min_purity)
        self.hash_batch_size = int(hash_batch_size)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PrefixHashLookup":
        """Fit lookup tables from an ``int16`` packet matrix and integer labels."""

        packets = self._validate_packets(X, fitting=True)
        labels = np.asarray(y)
        if labels.ndim != 1 or labels.shape[0] != packets.shape[0]:
            raise ValueError("y must be one-dimensional and have one label per row")
        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError("y must have an integer dtype")
        if labels.size == 0:
            raise ValueError("cannot fit an empty lookup")
        if self.prefix_lengths[0] > packets.shape[1]:
            raise ValueError(
                "prefix length cannot exceed the number of packet columns"
            )

        self.n_features_in_ = int(packets.shape[1])
        self.classes_ = np.unique(labels)
        code_dtype = np.min_scalar_type(max(len(self.classes_) - 1, 0))
        if not np.issubdtype(code_dtype, np.unsignedinteger):
            code_dtype = np.dtype(np.uint8)

        label_codes = np.empty(labels.shape[0], dtype=code_dtype)
        for start in range(0, labels.shape[0], self.hash_batch_size):
            stop = min(start + self.hash_batch_size, labels.shape[0])
            label_codes[start:stop] = np.searchsorted(
                self.classes_, labels[start:stop]
            ).astype(code_dtype, copy=False)

        self.tables_ = tuple(
            self._build_table(packets, label_codes, prefix_length)
            for prefix_length in self.prefix_lengths
        )
        return self

    @overload
    def predict(
        self,
        X: np.ndarray,
        *,
        default_label: int = -1,
        return_prefix: bool = False,
    ) -> np.ndarray: ...

    @overload
    def predict(
        self,
        X: np.ndarray,
        *,
        default_label: int = -1,
        return_prefix: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def predict(
        self,
        X: np.ndarray,
        *,
        default_label: int = -1,
        return_prefix: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict with longest-prefix backoff.

        Rows not covered by a retained table receive ``default_label``.  When
        ``return_prefix`` is true, a second array reports the matched prefix
        length (zero means no match).
        """

        self._check_is_fitted()
        packets = self._validate_packets(X, fitting=False)
        if packets.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {packets.shape[1]} columns; expected {self.n_features_in_}"
            )

        result_dtype = np.result_type(self.classes_.dtype, np.asarray(default_label).dtype)
        predictions = np.full(packets.shape[0], default_label, dtype=result_dtype)
        matched_prefix = np.zeros(packets.shape[0], dtype=np.uint16)

        for start in range(0, packets.shape[0], self.hash_batch_size):
            stop = min(start + self.hash_batch_size, packets.shape[0])
            block = packets[start:stop]
            unresolved = np.arange(stop - start, dtype=np.int64)

            for table in self.tables_:
                if unresolved.size == 0:
                    break
                if table.keys.size == 0:
                    continue

                first, second = _hash_rows(block[unresolved], table.prefix_length)
                query = _make_keys(first, second)
                positions = np.searchsorted(table.keys, query)
                in_bounds = positions < table.keys.size
                found = np.zeros(positions.size, dtype=bool)
                if np.any(in_bounds):
                    bounded = np.flatnonzero(in_bounds)
                    found[bounded] = table.keys[positions[bounded]] == query[bounded]
                if not np.any(found):
                    continue

                local_rows = unresolved[found]
                codes = table.labels[positions[found]].astype(np.intp, copy=False)
                predictions[start + local_rows] = self.classes_[codes]
                matched_prefix[start + local_rows] = table.prefix_length
                unresolved = unresolved[~found]

        if return_prefix:
            return predictions, matched_prefix
        return predictions

    @property
    def nbytes(self) -> int:
        """Memory occupied by fitted lookup tables, excluding small metadata."""

        self._check_is_fitted()
        return int(sum(table.nbytes for table in self.tables_))

    def _build_table(
        self, packets: np.ndarray, label_codes: np.ndarray, prefix_length: int
    ) -> PrefixTable:
        n_rows = packets.shape[0]
        record_dtype = np.dtype(
            [("first", "<u8"), ("second", "<u8"), ("label", label_codes.dtype)]
        )
        records = np.empty(n_rows, dtype=record_dtype)
        records["label"] = label_codes

        for start in range(0, n_rows, self.hash_batch_size):
            stop = min(start + self.hash_batch_size, n_rows)
            first, second = _hash_rows(packets[start:stop], prefix_length)
            records["first"][start:stop] = first
            records["second"][start:stop] = second

        # In-place quicksort avoids the int64 permutation array produced by
        # np.lexsort (about 64 MB for 8 million rows).
        records.sort(order=("first", "second", "label"), kind="quicksort")

        pair_start = np.empty(n_rows, dtype=bool)
        pair_start[0] = True
        pair_start[1:] = (
            (records["first"][1:] != records["first"][:-1])
            | (records["second"][1:] != records["second"][:-1])
            | (records["label"][1:] != records["label"][:-1])
        )
        pair_indices = np.flatnonzero(pair_start)
        pair_ends = np.empty_like(pair_indices)
        pair_ends[:-1] = pair_indices[1:]
        pair_ends[-1] = n_rows
        pair_counts = (pair_ends - pair_indices).astype(np.uint64, copy=False)

        pair_first = records["first"][pair_indices]
        pair_second = records["second"][pair_indices]
        pair_labels = records["label"][pair_indices]
        group_start = np.empty(pair_indices.size, dtype=bool)
        group_start[0] = True
        group_start[1:] = (
            (pair_first[1:] != pair_first[:-1])
            | (pair_second[1:] != pair_second[:-1])
        )
        group_indices = np.flatnonzero(group_start)

        support = np.add.reduceat(pair_counts, group_indices)
        # Encode count and inverse label rank in one integer.  max-reduce then
        # selects the majority label and deterministically chooses the smallest
        # label code when counts tie.
        base = np.uint64(len(self.classes_) + 1)
        scores = pair_counts * base + (
            np.uint64(len(self.classes_)) - pair_labels.astype(np.uint64)
        )
        best_scores = np.maximum.reduceat(scores, group_indices)
        matched_count = best_scores // base
        best_labels = (
            np.uint64(len(self.classes_)) - (best_scores % base)
        ).astype(label_codes.dtype, copy=False)

        retained = (support >= self.min_support) & (
            matched_count.astype(np.float64)
            >= self.min_purity * support.astype(np.float64)
        )
        key_rows = pair_indices[group_indices[retained]]
        keys = _make_keys(records["first"][key_rows], records["second"][key_rows])
        count_dtype = np.min_scalar_type(n_rows)

        return PrefixTable(
            prefix_length=prefix_length,
            keys=keys,
            labels=best_labels[retained],
            support=support[retained].astype(count_dtype, copy=False),
            matched_count=matched_count[retained].astype(count_dtype, copy=False),
        )

    @staticmethod
    def _validate_packets(X: np.ndarray, *, fitting: bool) -> np.ndarray:
        packets = np.asarray(X)
        if packets.ndim != 2:
            raise ValueError("X must be a two-dimensional packet matrix")
        if packets.dtype != np.int16:
            raise TypeError("X must have dtype int16")
        if fitting and packets.shape[0] == 0:
            raise ValueError("cannot fit an empty lookup")
        return packets

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "tables_"):
            raise RuntimeError("PrefixHashLookup must be fitted before prediction")


def _make_keys(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    keys = np.empty(len(first), dtype=_KEY_DTYPE)
    keys["first"] = first
    keys["second"] = second
    return keys


def _hash_rows(packets: np.ndarray, prefix_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return two independent, deterministic uint64 hashes for every row."""

    n_rows = packets.shape[0]
    first = np.full(
        n_rows,
        np.uint64(0xCBF29CE484222325) ^ np.uint64(prefix_length),
        dtype=np.uint64,
    )
    second = np.full(
        n_rows,
        np.uint64(0x84222325CBF29CE4) ^ np.uint64(prefix_length),
        dtype=np.uint64,
    )
    fnv_prime = np.uint64(0x100000001B3)
    second_prime = np.uint64(0x9E3779B185EBCA87)
    combine_constant = np.uint64(0x9E3779B97F4A7C15)

    for column in range(prefix_length):
        values = packets[:, column].astype(np.uint16, copy=False).astype(
            np.uint64, copy=False
        )
        first ^= values
        first *= fnv_prime
        second ^= values + combine_constant + (second << np.uint64(6)) + (
            second >> np.uint64(2)
        )
        second *= second_prime

    # Avalanche the state so close packet values do not leave correlated high
    # bits.  All arithmetic intentionally wraps modulo 2**64.
    first ^= first >> np.uint64(30)
    first *= np.uint64(0xBF58476D1CE4E5B9)
    first ^= first >> np.uint64(27)
    first *= np.uint64(0x94D049BB133111EB)
    first ^= first >> np.uint64(31)

    second ^= second >> np.uint64(33)
    second *= np.uint64(0xFF51AFD7ED558CCD)
    second ^= second >> np.uint64(33)
    second *= np.uint64(0xC4CEB9FE1A85EC53)
    second ^= second >> np.uint64(33)
    return first, second


__all__ = ["PrefixHashLookup", "PrefixTable"]
