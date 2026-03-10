#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parquet/file loading helpers for episode building."""

from __future__ import annotations

import os
from typing import List, Optional

import pyarrow.dataset as ds
import pyarrow.parquet as pq


def iter_parquet_files(root: str) -> List[str]:
    """Recursively collect and sort parquet files under root."""
    out: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".parquet"):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def open_parquet_file(path: str) -> pq.ParquetFile:
    """Open a parquet file with a small seam for testing/reuse."""
    return pq.ParquetFile(path)


def load_transactions_dataset(tx_root: str, chain: str) -> ds.Dataset:
    """Load transactions dataset from <tx_root>/<chain>/transactions."""
    tx_dir = os.path.join(os.path.expanduser(tx_root), chain, "transactions")
    if not os.path.isdir(tx_dir):
        raise FileNotFoundError(f"[tx] 未找到 transactions 目录: {tx_dir}")
    return ds.dataset(tx_dir, format="parquet")


def load_decoded_events_dataset(events_root: str, chain: str) -> Optional[ds.Dataset]:
    """Load decoded_events dataset from <events_root>/<chain>/decoded_events."""
    events_dir = os.path.join(os.path.expanduser(events_root), chain, "decoded_events")
    if not os.path.isdir(events_dir):
        return None
    return ds.dataset(events_dir, format="parquet")
