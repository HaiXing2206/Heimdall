#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import pyarrow.parquet as pq


def list_parquets(tx_dir: Path) -> List[Path]:
    return sorted(Path(p) for p in glob.glob(str(tx_dir / "*.parquet")))


def schema_signature(cols: List[str]) -> str:
    # stable signature for grouping similar schemas
    return "|".join(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain_root", default="/home/chain1/zl/chain")
    ap.add_argument("--chains", default="eth,arbitrum,optimism,polygon,base,bsc",
                    help="comma-separated chain dirs under chain_root")
    ap.add_argument("--max_files", type=int, default=50,
                    help="max parquet files to inspect per chain (0=all)")
    ap.add_argument("--out", default="/home/chain1/zl/chain/transactions_schema_report.json")
    args = ap.parse_args()

    chain_root = Path(args.chain_root)
    chains = [c.strip() for c in args.chains.split(",") if c.strip()]

    report: Dict[str, dict] = {}
    global_union = Counter()

    for ch in chains:
        tx_dir = chain_root / ch / "transactions"
        files = list_parquets(tx_dir)
        if not files:
            report[ch] = {"error": f"no parquet files under {tx_dir}"}
            continue

        if args.max_files and args.max_files > 0:
            files = files[: args.max_files]

        col_freq = Counter()
        schema_groups = Counter()
        example_by_sig: Dict[str, str] = {}

        for fp in files:
            try:
                sch = pq.read_schema(fp)
                cols = sch.names
                for c in cols:
                    col_freq[c] += 1
                    global_union[c] += 1
                sig = schema_signature(cols)
                schema_groups[sig] += 1
                example_by_sig.setdefault(sig, str(fp))
            except Exception as e:
                col_freq["_READ_ERROR_"] += 1

        # sort outputs
        cols_sorted = sorted(col_freq.items(), key=lambda x: (-x[1], x[0]))
        groups_sorted = schema_groups.most_common(5)

        report[ch] = {
            "tx_dir": str(tx_dir),
            "files_inspected": len(files),
            "unique_schemas": len(schema_groups),
            "top_schema_groups": [
                {
                    "count": cnt,
                    "example_file": example_by_sig[sig],
                    "num_cols": len(sig.split("|")) if sig else 0,
                    "first_25_cols": sig.split("|")[:25],
                }
                for sig, cnt in groups_sorted
            ],
            "columns": [
                {"name": name, "present_in_files": cnt, "present_ratio": cnt / len(files)}
                for name, cnt in cols_sorted
                if name != "_READ_ERROR_"
            ],
        }

    union_list = [{"name": k, "chains_present": v} for k, v in global_union.most_common()]
    out_obj = {"chains": report, "global_union": union_list}

    out_path = Path(args.out)
    out_path.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")
    print(f"[OK] wrote report: {out_path}")

    # also print a concise summary
    print("\n=== Summary ===")
    for ch in chains:
        info = report.get(ch, {})
        if "error" in info:
            print(f"{ch:10s} ERROR: {info['error']}")
            continue
        print(f"{ch:10s} files={info['files_inspected']} unique_schemas={info['unique_schemas']} cols={len(info['columns'])}")


if __name__ == "__main__":
    main()