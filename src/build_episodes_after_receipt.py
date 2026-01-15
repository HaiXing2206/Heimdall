
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_episodes_after_receipt.py

目的
----
基于“收款地址 + 收款时间(锚点时间) + 目标链”，在目标链 transactions parquet 中抓取
该收款地址在收款后一定时间窗口内发起的交易序列，用于构造“episode”(跨链套利/行为片段)。

输入
----
--crossdata_root 下递归扫描 *.parquet（你的 crossdata / bridge 结果集）。
每一行需要能提供：
  1) 收款地址（destination/recipient address）
  2) 目标链（chain name 或 chain_id）
  3) 一个时间戳（尽量是目标链收款时间；如果只有源链时间也可，用 --start_offset_minutes 做偏移）

输出
----
--out_root 下保持相同相对路径写回 parquet，并新增以下列：
  - EPISODE_TX_COUNT          int32
  - EPISODE_ACTIONS_JSON      string（JSON list, 每个元素是 tx action dict）
  - EPISODE_FIRST_TX_HASH     string
  - EPISODE_LAST_TX_HASH      string
  - EPISODE_LAST_TX_TIMESTAMP string（ISO8601, UTC）
  - EPISODE_DURATION_SEC      int64（last_tx_ts - receipt_ts，秒；无交易则为 null）

使用示例
--------
python /home/chain1/zl/chain/Heimdall/src/build_episodes_after_receipt.py \
  --crossdata_root /home/chain1/zl/chain/crossdata \
  --tx_root        /home/chain1/zl/chain \
  --out_root       /home/chain1/zl/chain/Heimdall/out/crossdata_with_episodes \
  --window_minutes 60 \
  --start_offset_minutes -10 \
  --bucket_minutes 60 \
  --batch_rows 20000 \
  --max_txs_per_episode 50

注意
----
1) 本脚本只用 transactions 表（from_address/to_address/value/input 等）构造 episode。
   如需 token 级别动作（Transfer/Swap），可在下一步把 episode 中 tx_hash 关联 decoded_events。
2) crossdata 字段名可能不一致：脚本会自动探测；如探测不准，用 --recv_col/--chain_col/--ts_col 覆盖。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm
from crossdata_schema import detect_crossdata_columns, canonical_chain, to_utc_ts




TX_REQUIRED_COLS = ["block_timestamp", "transaction_hash", "from_address", "to_address", "value", "input"]


# ----------------------------
# Utils
# ----------------------------


def ts_iso(ts: Optional[pd.Timestamp]) -> Optional[str]:
    if ts is None or pd.isna(ts):
        return None
    # ensure UTC
    if getattr(ts, "tzinfo", None) is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def chunked(seq: Sequence, n: int) -> Iterable[Sequence]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


# ----------------------------
# Tx scanning
# ----------------------------
@dataclass
class TxDataset:
    chain: str
    dataset: ds.Dataset
    ts_field: str = "block_timestamp"
    from_field: str = "from_address"

    @property
    def schema(self) -> pa.Schema:
        return self.dataset.schema

    @property
    def available_cols(self) -> List[str]:
        return list(self.schema.names)

    def _ts_scalar(self, ts: pd.Timestamp) -> pa.Scalar:
        ftype = self.schema.field(self.ts_field).type
        # keep timezone info if the field expects tz
        py_dt = ts.to_pydatetime()
        return pa.scalar(py_dt, type=ftype)

    def scan(
        self,
        addresses: Sequence[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        columns: Sequence[str],
        addr_chunk: int = 2000,
    ) -> pa.Table:
        if not addresses:
            return pa.table({c: pa.array([], type=self.schema.field(c).type) for c in columns if c in self.schema.names})

        # clamp to available columns
        cols = [c for c in columns if c in self.schema.names]
        if not cols:
            # Return empty table when this chain's tx schema doesn't match expected canonical columns.
            return pa.table({})

        ts_start = self._ts_scalar(start)
        ts_end = self._ts_scalar(end)

        tables: List[pa.Table] = []
        for addr_part in chunked(list(addresses), addr_chunk):
            expr = (
                (ds.field(self.ts_field) >= ts_start)
                & (ds.field(self.ts_field) < ts_end)
                & ds.field(self.from_field).isin([a for a in addr_part])
            )
            t = self.dataset.to_table(columns=cols, filter=expr)
            if t.num_rows:
                tables.append(t)

        if not tables:
            # empty table with schema
            empty_arrays = {}
            for c in cols:
                empty_arrays[c] = pa.array([], type=self.schema.field(c).type)
            return pa.table(empty_arrays)

        if len(tables) == 1:
            return tables[0]
        return pa.concat_tables(tables, promote=True)


class TxDatasetCache:
    def __init__(self, tx_root: str):
        self.tx_root = os.path.expanduser(tx_root)
        self._cache: Dict[str, TxDataset] = {}

    def get(self, chain: str) -> TxDataset:
        chain = canonical_chain(chain) or chain
        if chain in self._cache:
            return self._cache[chain]

        tx_dir = os.path.join(self.tx_root, chain, "transactions")
        if not os.path.isdir(tx_dir):
            raise FileNotFoundError(f"[tx] 未找到 transactions 目录: {tx_dir}")

        dset = ds.dataset(tx_dir, format="parquet")
        txds = TxDataset(chain=chain, dataset=dset)
        self._cache[chain] = txds
        return txds


# ----------------------------
# Episode building
# ----------------------------
def build_episodes_for_receipts(
    receipts: pd.DataFrame,
    tx_cache: TxDatasetCache,
    window_minutes: int,
    start_offset_minutes: int,
    bucket_minutes: int,
    max_txs_per_episode: int,
    tx_addr_chunk: int,
) -> pd.DataFrame:
    """
    receipts 必须包含:
      - _idx (原表行号)
      - _chain (canonical chain)
      - _recv  (lowercase address)
      - _ts    (receipt ts, UTC)
    返回一个 DataFrame，index = _idx，包含 episode columns。
    """
    out = pd.DataFrame(index=receipts["_idx"].values)
    out["EPISODE_TX_COUNT"] = 0
    out["EPISODE_ACTIONS_JSON"] = "[]"
    out["EPISODE_FIRST_TX_HASH"] = None
    out["EPISODE_LAST_TX_HASH"] = None
    out["EPISODE_LAST_TX_TIMESTAMP"] = None
    out["EPISODE_DURATION_SEC"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="Int64")

    if receipts.empty:
        return out

    # compute window
    receipts = receipts.copy()
    receipts["_window_start"] = receipts["_ts"] + pd.to_timedelta(start_offset_minutes, unit="m")
    receipts["_window_end"] = receipts["_window_start"] + pd.to_timedelta(window_minutes, unit="m")

    # bucketize by start time
    receipts["_bucket"] = receipts["_window_start"].dt.floor(f"{bucket_minutes}min")

    # process chain by chain
    for chain, df_chain in receipts.groupby("_chain", dropna=False):
        if chain is None or (isinstance(chain, float) and np.isnan(chain)):
            continue

        try:
            txds = tx_cache.get(chain)
        except Exception as e:
            # unknown / missing chain tx dataset
            # leave default empty episodes
            continue

        # iterate buckets
        for bucket, df_b in df_chain.groupby("_bucket"):
            if df_b.empty:
                continue
            addrs = df_b["_recv"].dropna().unique().tolist()
            if not addrs:
                continue

            t_start = df_b["_window_start"].min()
            t_end = df_b["_window_end"].max()

            cols = [c for c in TX_REQUIRED_COLS if c in txds.available_cols]
            t = txds.scan(
                addresses=addrs,
                start=t_start,
                end=t_end,
                columns=cols,
                addr_chunk=tx_addr_chunk,
            )
            if t.num_rows == 0:
                continue

            txdf = t.to_pandas()
            # normalize
            if "from_address" in txdf.columns:
                txdf["from_address"] = txdf["from_address"].astype(str).str.lower()
            if "to_address" in txdf.columns:
                txdf["to_address"] = txdf["to_address"].astype(str).str.lower()
                txdf.loc[txdf["to_address"].isin(["none", "nan", "null"]), "to_address"] = None
            if "block_timestamp" in txdf.columns:
                txdf["block_timestamp"] = pd.to_datetime(txdf["block_timestamp"], utc=True, errors="coerce")
            if "transaction_hash" in txdf.columns:
                txdf["transaction_hash"] = txdf["transaction_hash"].astype(str).str.lower()
            if "input" in txdf.columns:
                txdf["input"] = txdf["input"].astype(str)

            # sort & group by from_address
            txdf = txdf.sort_values(["from_address", "block_timestamp"], kind="mergesort")
            grouped = {addr: g for addr, g in txdf.groupby("from_address", sort=False)}

            # process each receipt in bucket (use name=None to avoid pandas renaming issues for leading-underscore columns)
            for idx, addr, w0, w1, receipt_ts in df_b[["_idx", "_recv", "_window_start", "_window_end", "_ts"]].itertuples(index=False, name=None):

                g = grouped.get(addr)
                if g is None or g.empty:
                    continue

                ts_arr = g["block_timestamp"].to_numpy(dtype="datetime64[ns]")
                # searchsorted expects numpy datetime64
                w0_64 = np.datetime64(pd.Timestamp(w0).to_datetime64())
                w1_64 = np.datetime64(pd.Timestamp(w1).to_datetime64())
                i0 = int(np.searchsorted(ts_arr, w0_64, side="left"))
                i1 = int(np.searchsorted(ts_arr, w1_64, side="left"))

                sub = g.iloc[i0:i1]
                if sub.empty:
                    continue
                if max_txs_per_episode > 0 and len(sub) > max_txs_per_episode:
                    sub = sub.iloc[:max_txs_per_episode]

                actions = []
                for row in sub.itertuples(index=False):
                    ts = getattr(row, "block_timestamp", None)
                    txh = getattr(row, "transaction_hash", None)
                    to = getattr(row, "to_address", None)
                    val = getattr(row, "value", None)
                    inp = getattr(row, "input", None)

                    # method_id
                    method_id = None
                    if isinstance(inp, str) and inp.startswith("0x") and len(inp) >= 10:
                        method_id = inp[:10]

                    # value normalize to int if possible
                    val_out = None
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        try:
                            val_out = int(val)
                        except Exception:
                            # keep as string
                            val_out = str(val)

                    actions.append(
                        {
                            "ts": ts_iso(pd.Timestamp(ts) if ts is not None else None),
                            "tx_hash": txh,
                            "to": to,
                            "value": val_out,
                            "method_id": method_id,
                        }
                    )

                first_tx = actions[0]["tx_hash"] if actions else None
                last_tx = actions[-1]["tx_hash"] if actions else None
                last_ts = actions[-1]["ts"] if actions else None

                out.at[idx, "EPISODE_TX_COUNT"] = len(actions)
                out.at[idx, "EPISODE_ACTIONS_JSON"] = json.dumps(actions, ensure_ascii=False)
                out.at[idx, "EPISODE_FIRST_TX_HASH"] = first_tx
                out.at[idx, "EPISODE_LAST_TX_HASH"] = last_tx
                out.at[idx, "EPISODE_LAST_TX_TIMESTAMP"] = last_ts

                # duration: last tx ts - receipt ts
                try:
                    if last_ts is not None:
                        last_ts_pd = pd.to_datetime(last_ts, utc=True)
                        dur = int((last_ts_pd - receipt_ts).total_seconds())
                        out.at[idx, "EPISODE_DURATION_SEC"] = dur
                except Exception:
                    pass

    return out


# ----------------------------
# File processing
# ----------------------------
def iter_parquet_files(root: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".parquet"):
                out.append(os.path.join(dirpath, fn))
    out.sort()
    return out


def process_one_file(
    in_path: str,
    out_path: str,
    tx_cache: TxDatasetCache,
    recv_col: Optional[str],
    chain_col: Optional[str],
    ts_col: Optional[str],
    id_col: Optional[str],
    window_minutes: int,
    start_offset_minutes: int,
    bucket_minutes: int,
    max_txs_per_episode: int,
    batch_rows: int,
    tx_addr_chunk: int,
    allowed_chains: Optional[set] = None,
    limit_rows: Optional[int] = None,
) -> None:
    pf = pq.ParquetFile(in_path)
    cols = pf.schema.names

    auto_recv, auto_chain, auto_ts, auto_id = detect_crossdata_columns(cols)
    recv_col = recv_col or auto_recv
    chain_col = chain_col or auto_chain
    ts_col = ts_col or auto_ts
    id_col = id_col or auto_id  # optional

    # If we still cannot determine the required columns, we will write passthrough rows with empty episodes.
    missing_required = [name for name, col in [("recv_col", recv_col), ("chain_col", chain_col), ("ts_col", ts_col)] if col is None]
    if missing_required:
        print(f"[WARN] {os.path.basename(in_path)} missing columns {missing_required}; passthrough with empty episodes.", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    writer: Optional[pq.ParquetWriter] = None
    written_rows = 0

    for batch in pf.iter_batches(batch_size=batch_rows):
        tbl_in = pa.Table.from_batches([batch])
        if missing_required:
            # passthrough: append empty episode columns and write out
            n = tbl_in.num_rows
            tx_count = np.zeros(n, dtype=np.int32)
            actions_json = np.array(["[]"] * n, dtype=object)
            first_hash = np.array([None] * n, dtype=object)
            last_hash = np.array([None] * n, dtype=object)
            last_ts = np.array([None] * n, dtype=object)
            duration_sec = np.array([None] * n, dtype=object)

            tbl_out = tbl_in
            tbl_out = tbl_out.append_column("EPISODE_TX_COUNT", pa.array(tx_count, type=pa.int32()))
            tbl_out = tbl_out.append_column("EPISODE_ACTIONS_JSON", pa.array(actions_json, type=pa.string()))
            tbl_out = tbl_out.append_column("EPISODE_FIRST_TX_HASH", pa.array(first_hash, type=pa.string()))
            tbl_out = tbl_out.append_column("EPISODE_LAST_TX_HASH", pa.array(last_hash, type=pa.string()))
            tbl_out = tbl_out.append_column("EPISODE_LAST_TX_TIMESTAMP", pa.array(last_ts, type=pa.string()))
            tbl_out = tbl_out.append_column("EPISODE_DURATION_SEC", pa.array(duration_sec, type=pa.int64()))

            if writer is None:
                writer = pq.ParquetWriter(out_path, tbl_out.schema, compression="snappy")
            writer.write_table(tbl_out)
            written_rows += n
            continue

        # select required cols for computation
        need_cols = [recv_col, chain_col, ts_col]
        if id_col and id_col in tbl_in.column_names:
            need_cols.append(id_col)

        df_need = tbl_in.select(need_cols).to_pandas()
        n = len(df_need)
        if n == 0:
            continue

        # limit_rows for debug
        if limit_rows is not None and written_rows >= limit_rows:
            break
        if limit_rows is not None and written_rows + n > limit_rows:
            df_need = df_need.iloc[: (limit_rows - written_rows)]
            tbl_in = tbl_in.slice(0, len(df_need))
            n = len(df_need)

        # build receipts frame
        receipts = pd.DataFrame(
            {
                "_idx": np.arange(written_rows, written_rows + n, dtype=np.int64),
                "_recv": df_need[recv_col].astype(str).str.lower(),
                "_chain": df_need[chain_col].apply(canonical_chain),
                "_ts": df_need[ts_col].apply(to_utc_ts),
            }
        )
        # drop invalid
        receipts = receipts[receipts["_recv"].str.startswith("0x") & receipts["_ts"].notna()].copy()
        if allowed_chains:
            receipts = receipts[receipts["_chain"].isin(allowed_chains)].copy()

        epi = build_episodes_for_receipts(
            receipts=receipts,
            tx_cache=tx_cache,
            window_minutes=window_minutes,
            start_offset_minutes=start_offset_minutes,
            bucket_minutes=bucket_minutes,
            max_txs_per_episode=max_txs_per_episode,
            tx_addr_chunk=tx_addr_chunk,
        )

        # align episode result back to batch row positions
        # default arrays
        tx_count = np.zeros(n, dtype=np.int32)
        actions_json = np.array(["[]"] * n, dtype=object)
        first_hash = np.array([None] * n, dtype=object)
        last_hash = np.array([None] * n, dtype=object)
        last_ts = np.array([None] * n, dtype=object)
        duration_sec = np.array([None] * n, dtype=object)

        # epi index is global row idx; convert to local pos
        if not epi.empty:
            for global_idx, row in epi.iterrows():
                local_pos = int(global_idx - written_rows)
                if 0 <= local_pos < n:
                    tx_count[local_pos] = int(row["EPISODE_TX_COUNT"])
                    actions_json[local_pos] = row["EPISODE_ACTIONS_JSON"]
                    first_hash[local_pos] = row["EPISODE_FIRST_TX_HASH"]
                    last_hash[local_pos] = row["EPISODE_LAST_TX_HASH"]
                    last_ts[local_pos] = row["EPISODE_LAST_TX_TIMESTAMP"]
                    duration_sec[local_pos] = (int(row["EPISODE_DURATION_SEC"]) if not pd.isna(row["EPISODE_DURATION_SEC"]) else None)

        # append columns with explicit types (避免全空导致 schema 漂移)
        tbl_out = tbl_in
        tbl_out = tbl_out.append_column("EPISODE_TX_COUNT", pa.array(tx_count, type=pa.int32()))
        tbl_out = tbl_out.append_column("EPISODE_ACTIONS_JSON", pa.array(actions_json, type=pa.string()))
        tbl_out = tbl_out.append_column("EPISODE_FIRST_TX_HASH", pa.array(first_hash, type=pa.string()))
        tbl_out = tbl_out.append_column("EPISODE_LAST_TX_HASH", pa.array(last_hash, type=pa.string()))
        tbl_out = tbl_out.append_column("EPISODE_LAST_TX_TIMESTAMP", pa.array(last_ts, type=pa.string()))
        tbl_out = tbl_out.append_column("EPISODE_DURATION_SEC", pa.array(duration_sec, type=pa.int64()))

        if writer is None:
            writer = pq.ParquetWriter(out_path, tbl_out.schema, compression="snappy")
        writer.write_table(tbl_out)

        written_rows += n

    if writer is not None:
        writer.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--crossdata_root",
        default="/home/chain1/zl/chain/crossdata",
        help="crossdata parquet 根目录（递归扫描）",
    )
    ap.add_argument(
        "--tx_root",
        default="/home/chain1/zl/chain",
        help="包含 <chain>/transactions 的根目录",
    )
    ap.add_argument(
        "--out_root",
        default="/home/chain1/zl/chain/Heimdall/out/crossdata_with_episodes",
        help="输出根目录，保持相对路径",
    )
    ap.add_argument(
        "--chains",
        default="eth,arbitrum,optimism",
        help="只处理这些目标链（canonical 名称），逗号分隔，例如 eth,arbitrum,optimism",
    )
    ap.add_argument("--recv_col", default=None, help="crossdata 收款地址列名（覆盖自动探测）")
    ap.add_argument("--chain_col", default=None, help="crossdata 目标链列名（覆盖自动探测）")
    ap.add_argument("--ts_col", default=None, help="crossdata 时间戳列名（覆盖自动探测）")
    ap.add_argument("--id_col", default=None, help="可选：crossdata 主键/tx_hash 列名（目前仅用于保留原列）")

    ap.add_argument("--window_minutes", type=int, default=60, help="episode 窗口长度（分钟）")
    ap.add_argument("--start_offset_minutes", type=int, default=0, help="窗口起点相对收款时间的偏移（分钟，可为负）")
    ap.add_argument("--bucket_minutes", type=int, default=60, help="按时间桶聚合扫描 transactions，越小越精准但扫描次数更多")
    ap.add_argument("--max_txs_per_episode", type=int, default=50, help="每个 episode 最多保留多少笔交易（0=不限）")

    ap.add_argument("--batch_rows", type=int, default=20000, help="每次从 crossdata parquet 读多少行")
    ap.add_argument("--tx_addr_chunk", type=int, default=2000, help="transactions 扫描时，isin 地址集合分块大小")
    ap.add_argument("--limit_files", type=int, default=None, help="调试：只处理前 N 个文件")
    ap.add_argument("--limit_rows", type=int, default=None, help="调试：每个文件最多处理多少行")

    args = ap.parse_args()
    allowed_chains = {c.strip().lower() for c in str(args.chains).split(",") if c.strip()}

    cross_root = os.path.expanduser(args.crossdata_root)
    out_root = os.path.expanduser(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    files = iter_parquet_files(cross_root)
    if args.limit_files is not None:
        files = files[: args.limit_files]

    print(f"[INFO] crossdata files={len(files)} root={cross_root}")
    tx_cache = TxDatasetCache(args.tx_root)

    for in_path in tqdm(files, desc="episode files"):
        rel = os.path.relpath(in_path, cross_root)
        out_path = os.path.join(out_root, rel)
        process_one_file(
            in_path=in_path,
            out_path=out_path,
            tx_cache=tx_cache,
            recv_col=args.recv_col,
            chain_col=args.chain_col,
            ts_col=args.ts_col,
            id_col=args.id_col,
            window_minutes=args.window_minutes,
            start_offset_minutes=args.start_offset_minutes,
            bucket_minutes=args.bucket_minutes,
            max_txs_per_episode=args.max_txs_per_episode,
            batch_rows=args.batch_rows,
            tx_addr_chunk=args.tx_addr_chunk,
            allowed_chains=allowed_chains,
            limit_rows=args.limit_rows,
        )


if __name__ == "__main__":
    main()
