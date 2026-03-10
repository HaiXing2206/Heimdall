#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parquet/file loading helpers for episode building."""

from __future__ import annotations

import os
import json
from typing import List, Optional

import pandas as pd

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


def load_time_index_json(path: str) -> Optional[dict]:
    """Load a time-index json file. Return None when missing/unreadable."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def select_files_by_time_index(index_obj: Optional[dict], start: pd.Timestamp, end: pd.Timestamp) -> List[str]:
    """Select parquet files whose [min_ts,max_ts] overlaps [start,end)."""
    if not index_obj:
        return []

    files = index_obj.get("files") or []
    if not isinstance(files, list):
        return []

    s = pd.to_datetime(start, utc=True, errors="coerce")
    e = pd.to_datetime(end, utc=True, errors="coerce")
    if pd.isna(s) or pd.isna(e):
        return []

    selected: List[str] = []
    for item in files:
        if not isinstance(item, dict):
            continue

        fp = item.get("file")
        if not fp:
            continue

        min_ts = pd.to_datetime(item.get("min_ts"), utc=True, errors="coerce")
        max_ts = pd.to_datetime(item.get("max_ts"), utc=True, errors="coerce")
        if pd.isna(min_ts) or pd.isna(max_ts):
            continue

        # overlap when file_max >= start and file_min < end
        if (max_ts >= s) and (min_ts < e):
            selected.append(str(fp))
    return selected
