#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone tool to generate *_time_index.json.

This script is designed to be run manually before episode building.
It is not invoked by other modules.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import sys

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.parquet_io import iter_parquet_files

TIMESTAMP_COL = "block_timestamp"
TARGET_DIRS = ("transactions", "decoded_events", "logs")


def ts_to_iso(x) -> Optional[str]:
    if x is None:
        return None
    ts = pd.to_datetime(x, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.isoformat()


def get_min_max_from_metadata(parquet_path: str, col_name: str) -> Tuple[object, object]:
    """Prefer parquet metadata and row-group statistics."""
    meta = pq.read_metadata(parquet_path)
    col_idx = meta.schema.get_field_index(col_name)
    if col_idx < 0:
        raise ValueError(f"Column not found in schema: {col_name}")

    min_ts = None
    max_ts = None
    for i in range(meta.num_row_groups):
        col = meta.row_group(i).column(col_idx)
        stats = col.statistics
        if stats is None:
            continue
        if stats.min is not None:
            min_ts = stats.min if min_ts is None else min(min_ts, stats.min)
        if stats.max is not None:
            max_ts = stats.max if max_ts is None else max(max_ts, stats.max)

    if min_ts is None or max_ts is None:
        raise ValueError("No statistics available")
    return min_ts, max_ts


def get_min_max_by_reading(parquet_path: str, col_name: str) -> Tuple[object, object]:
    """Fallback: read only timestamp column."""
    table = pq.read_table(parquet_path, columns=[col_name])
    df = table.to_pandas()
    if df.empty:
        raise ValueError("Empty file")
    return df[col_name].min(), df[col_name].max()


def build_index_for_target(base_dir: str, chain: str, target: str, out_dir: Path) -> None:
    target_root = os.path.join(base_dir, chain, target)
    if not os.path.isdir(target_root):
        print(f"[skip] not found: {target_root}")
        return

    files = iter_parquet_files(target_root)
    if not files:
        print(f"[skip] no parquet: {target_root}")
        return

    items = []
    errors = []

    for fp in tqdm(files, desc=f"{chain}/{target}"):
        rel = os.path.relpath(fp, base_dir)
        try:
            try:
                min_ts, max_ts = get_min_max_from_metadata(fp, TIMESTAMP_COL)
                method = "metadata"
            except Exception:
                min_ts, max_ts = get_min_max_by_reading(fp, TIMESTAMP_COL)
                method = "read_column"

            min_iso = ts_to_iso(min_ts)
            max_iso = ts_to_iso(max_ts)
            if min_iso is None or max_iso is None:
                raise ValueError("timestamp parse failed")

            items.append(
                {
                    "file": fp,
                    "rel_file": rel,
                    "chain": chain,
                    "dataset": target,
                    "timestamp_col": TIMESTAMP_COL,
                    "min_ts": min_iso,
                    "max_ts": max_iso,
                    "method": method,
                }
            )
        except Exception as e:
            errors.append(
                {
                    "file": fp,
                    "rel_file": rel,
                    "chain": chain,
                    "dataset": target,
                    "error": str(e),
                }
            )

    items.sort(key=lambda x: (x["min_ts"], x["max_ts"], x["file"]))

    summary = {
        "base_dir": base_dir,
        "chain": chain,
        "dataset": target,
        "timestamp_col": TIMESTAMP_COL,
        "n_files": len(files),
        "n_indexed": len(items),
        "n_errors": len(errors),
        "files": items,
        "errors": errors,
    }

    out_path = out_dir / f"{chain}_{target}_time_index.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[ok] wrote {out_path} indexed={len(items)} errors={len(errors)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=".")
    ap.add_argument("--out_dir", default="time_indexes")
    ap.add_argument("--chains", default="bsc,eth,polygon,arbitrum,optimism,base")
    ap.add_argument("--targets", default=",".join(TARGET_DIRS))
    args = ap.parse_args()

    base_dir = os.path.abspath(os.path.expanduser(args.base_dir))
    out_dir = Path(os.path.expanduser(args.out_dir))
    if not out_dir.is_absolute():
        out_dir = Path(base_dir) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    chains = [c.strip() for c in str(args.chains).split(",") if c.strip()]
    targets = [t.strip() for t in str(args.targets).split(",") if t.strip()]

    for chain in chains:
        for target in targets:
            build_index_for_target(base_dir, chain, target, out_dir)


if __name__ == "__main__":
    main()
