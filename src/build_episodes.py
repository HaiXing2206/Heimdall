#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_episodes.py

相对你当前版本 build_episodes.py 的核心优化
---------------------------------------------------
1) Episode 生成以 transactions 为主：不再先扫 decoded_events 再反查 tx（避免巨量 events 全表扫描）。
2) decoded_events enrichment 变为可选：--enrich_events=0 时不依赖 decoded_events 也能生成 episode。
3) Source-pre window 改为批量扫描：不再 per-row 调 _collect_actions（大幅减少扫描次数）。
4) 事件按 log_index 排序；Transfer/Approval 支持从 args_json/args/topics/data 多路回退解析。

注意
----
- 该脚本与原脚本保持相同 CLI 参数（额外新增少量参数），输出列保持兼容。
- 如果你的项目里存在 crossdata_schema.py，会优先使用；否则使用 fallback（便于单文件测试）。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm

from parquet_io import (
    iter_parquet_files,
    load_decoded_events_dataset,
    load_transactions_dataset,
    open_parquet_file,
)

# ----------------------------
# Optional project imports
# ----------------------------
try:
    from crossdata_schema import detect_crossdata_columns, detect_crossdata_source_columns, canonical_chain, to_utc_ts
except Exception:  # pragma: no cover

    def detect_crossdata_columns(cols: Sequence[str]):
        return None, None, None, None

    def detect_crossdata_source_columns(cols: Sequence[str]):
        return None, None, None

    def canonical_chain(x: object) -> Optional[str]:
        if x is None:
            return None
        s = str(x).strip().lower()
        return s or None

    def to_utc_ts(x: object) -> pd.Timestamp:
        return pd.to_datetime(x, utc=True, errors="coerce")


# ----------------------------
# Canonical cols
# ----------------------------
TX_REQUIRED_COLS = [
    "block_timestamp",
    "transaction_hash",
    "from_address",
    "to_address",
    "value",
    "input",
    "block_number",
    "transaction_index",
]

EVENT_REQUIRED_COLS = [
    "block_timestamp",
    "transaction_hash",
    "log_index",
    "address",
    "event_hash",
    "event_signature",
    "topics",
    "args",
    "args_json",
    "data",
]

# ERC20 topics
TRANSFER_TOPIC0 = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
APPROVAL_TOPIC0 = "0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925"


# ----------------------------
# Helpers
# ----------------------------

def _normalize_addr(x: object) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s in ("", "none", "null", "nan"):
        return None
    return s


def _json_loads_maybe(s: str) -> Optional[object]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _parse_args_any(x: object) -> Optional[object]:
    """Parse args/args_json into python object (list/dict) if possible."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        if hasattr(x, "as_py"):
            x = x.as_py()
    except Exception:
        pass
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        # normalize fancy quotes
        s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            return _json_loads_maybe(s)
    return None


def _as_list_maybe(x: object) -> list:
    obj = _parse_args_any(x)
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # keep stable order? use values only (best-effort)
        return list(obj.values())
    return []


def _topic_addr(topic: object) -> Optional[str]:
    """Extract 0x + last 40 hex chars from a topic (bytes32) string."""
    if topic is None:
        return None
    s = str(topic).strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    if len(s) < 40:
        return None
    return "0x" + s[-40:]


def _topic_addr_padded(addr: str) -> str:
    a = str(addr).strip().lower()
    if a.startswith("0x"):
        a = a[2:]
    a = a.zfill(40)[-40:]
    return "0x" + ("0" * 24) + a


def _parse_u256_hex(data_hex: object) -> Optional[int]:
    if data_hex is None:
        return None
    s = str(data_hex).strip().lower()
    if not s:
        return None
    if s.startswith("0x"):
        s = s[2:]
    if not s:
        return None
    try:
        return int(s, 16)
    except Exception:
        return None


def ts_iso(ts: Optional[pd.Timestamp]) -> Optional[str]:
    if ts is None or pd.isna(ts):
        return None
    if getattr(ts, "tzinfo", None) is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def chunked(seq: Sequence, n: int) -> Iterable[Sequence]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def concat_tables_compat(tables: Sequence[pa.Table]) -> pa.Table:
    """Compatibility wrapper for pyarrow.concat_tables API changes."""
    if len(tables) == 1:
        return tables[0]
    try:
        # pyarrow>=14
        return pa.concat_tables(list(tables), promote_options="default")
    except TypeError:
        # older pyarrow
        return pa.concat_tables(list(tables), promote=True)


def pick_col(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if c.lower() in low:
            return low[c.lower()]
    return None


# ----------------------------
# Tx scanning
# ----------------------------
@dataclass
class TxDataset:
    chain: str
    dataset: ds.Dataset
    ts_field: str = "block_timestamp"

    @property
    def schema(self) -> pa.Schema:
        return self.dataset.schema

    @property
    def available_cols(self) -> List[str]:
        return list(self.schema.names)

    def _ts_scalar(self, ts: pd.Timestamp) -> pa.Scalar:
        ftype = self.schema.field(self.ts_field).type
        py_dt = ts.to_pydatetime()
        return pa.scalar(py_dt, type=ftype)

    def scan(
        self,
        addresses: Sequence[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        columns: Sequence[str],
        addr_field: str = "from_address",
        addr_chunk: int = 2000,
    ) -> pa.Table:
        """Scan txs where addr_field ∈ addresses in [start,end)."""
        if not addresses:
            return pa.table({c: pa.array([], type=self.schema.field(c).type) for c in columns if c in self.schema.names})

        cols = [c for c in columns if c in self.schema.names]
        if not cols:
            return pa.table({})

        if addr_field not in self.schema.names:
            return pa.table({c: pa.array([], type=self.schema.field(c).type) for c in cols})

        ts_start = self._ts_scalar(start)
        ts_end = self._ts_scalar(end)

        tables: List[pa.Table] = []
        for addr_part in chunked(list(addresses), addr_chunk):
            expr = (
                (ds.field(self.ts_field) >= ts_start)
                & (ds.field(self.ts_field) < ts_end)
                & ds.field(addr_field).isin([a for a in addr_part])
            )
            t = self.dataset.to_table(columns=cols, filter=expr)
            if t.num_rows:
                tables.append(t)

        if not tables:
            empty_arrays = {c: pa.array([], type=self.schema.field(c).type) for c in cols}
            return pa.table(empty_arrays)

        return concat_tables_compat(tables)

    def scan_by_hashes(
        self,
        tx_hashes: Sequence[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        columns: Sequence[str],
        hash_chunk: int = 2000,
    ) -> pa.Table:
        cols = [c for c in columns if c in self.schema.names]
        if not cols:
            return pa.table({})
        if not tx_hashes:
            empty_arrays = {c: pa.array([], type=self.schema.field(c).type) for c in cols}
            return pa.table(empty_arrays)

        ts_start = self._ts_scalar(start)
        ts_end = self._ts_scalar(end)

        tables: List[pa.Table] = []
        for part in chunked(list(tx_hashes), hash_chunk):
            expr = (
                (ds.field(self.ts_field) >= ts_start)
                & (ds.field(self.ts_field) < ts_end)
                & ds.field("transaction_hash").isin([h for h in part])
            )
            t = self.dataset.to_table(columns=cols, filter=expr)
            if t.num_rows:
                tables.append(t)

        if not tables:
            empty_arrays = {c: pa.array([], type=self.schema.field(c).type) for c in cols}
            return pa.table(empty_arrays)
        return concat_tables_compat(tables)


class TxDatasetCache:
    def __init__(self, tx_root: str):
        self.tx_root = os.path.expanduser(tx_root)
        self._cache: Dict[str, TxDataset] = {}

    def get(self, chain: str) -> TxDataset:
        chain = canonical_chain(chain) or chain
        if chain in self._cache:
            return self._cache[chain]

        dset = load_transactions_dataset(self.tx_root, chain)
        txds = TxDataset(chain=chain, dataset=dset)
        self._cache[chain] = txds
        return txds


# ----------------------------
# Decoded events scanning
# ----------------------------
@dataclass
class DecodedEventsDataset:
    chain: str
    dataset: ds.Dataset
    ts_field: str = "block_timestamp"
    tx_hash_field: str = "transaction_hash"

    @property
    def schema(self) -> pa.Schema:
        return self.dataset.schema

    @property
    def available_cols(self) -> List[str]:
        return list(self.schema.names)

    def _ts_scalar(self, ts: pd.Timestamp) -> pa.Scalar:
        ftype = self.schema.field(self.ts_field).type
        py_dt = ts.to_pydatetime()
        return pa.scalar(py_dt, type=ftype)

    def scan(
        self,
        tx_hashes: Sequence[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        columns: Sequence[str],
        hash_chunk: int = 2000,
    ) -> pa.Table:
        if not tx_hashes:
            return pa.table({c: pa.array([], type=self.schema.field(c).type) for c in columns if c in self.schema.names})

        cols = [c for c in columns if c in self.schema.names]
        if not cols:
            return pa.table({})

        ts_start = self._ts_scalar(start)
        ts_end = self._ts_scalar(end)

        tables: List[pa.Table] = []
        for h_part in chunked(list(tx_hashes), hash_chunk):
            expr = (
                (ds.field(self.ts_field) >= ts_start)
                & (ds.field(self.ts_field) < ts_end)
                & ds.field(self.tx_hash_field).isin([h for h in h_part])
            )
            t = self.dataset.to_table(columns=cols, filter=expr)
            if t.num_rows:
                tables.append(t)

        if not tables:
            empty_arrays = {c: pa.array([], type=self.schema.field(c).type) for c in cols}
            return pa.table(empty_arrays)

        return concat_tables_compat(tables)


class DecodedEventsDatasetCache:
    def __init__(self, events_root: str):
        self.events_root = os.path.expanduser(events_root)
        self._cache: Dict[str, DecodedEventsDataset] = {}

    def get(self, chain: str) -> Optional[DecodedEventsDataset]:
        chain = canonical_chain(chain) or chain
        if chain in self._cache:
            return self._cache[chain]

        dset = load_decoded_events_dataset(self.events_root, chain)
        if dset is None:
            return None
        evds = DecodedEventsDataset(chain=chain, dataset=dset)
        self._cache[chain] = evds
        return evds


# ----------------------------
# Episode building
# ----------------------------

def _decode_event_row(
    row: pd.Series,
    ts_col: str,
    tx_col: str,
    log_col: Optional[str],
    sig_col: Optional[str],
    topic0_col: Optional[str],
    addr_col: Optional[str],
    topics_col: Optional[str],
    args_col: Optional[str],
    data_col: Optional[str],
    max_args_preview: int = 6,
    max_topics_preview: int = 4,
) -> Optional[Dict[str, object]]:
    txh = row.get(tx_col)
    if not txh:
        return None

    ts = row.get(ts_col)
    ts_out = ts_iso(pd.Timestamp(ts) if ts is not None else None)

    sig = row.get(sig_col) if sig_col else None
    topic0 = row.get(topic0_col) if topic0_col else None
    contract = row.get(addr_col) if addr_col else None
    topics = _as_list_maybe(row.get(topics_col) if topics_col else None)
    args_obj = _parse_args_any(row.get(args_col) if args_col else None)
    args_list = _as_list_maybe(args_obj)

    ev_name = None
    if isinstance(sig, str) and sig:
        ev_name = sig.split("(", 1)[0]

    t0 = str(topic0).lower() if topic0 is not None else (str(topics[0]).lower() if len(topics) else "")
    if not ev_name and t0 == TRANSFER_TOPIC0:
        ev_name = "Transfer"
    if not ev_name and t0 == APPROVAL_TOPIC0:
        ev_name = "Approval"

    event_action: Dict[str, object] = {
        "ts": ts_out,
        "tx_hash": str(txh).lower(),
    }

    if log_col:
        event_action["log_index"] = row.get(log_col)

    if contract:
        event_action["contract"] = _normalize_addr(contract)

    if ev_name:
        event_action["event_name"] = ev_name

    if isinstance(sig, str) and sig and (not ev_name or ev_name not in ("Transfer", "Approval")):
        event_action["event_signature"] = sig

    # previews
    if args_list:
        event_action["args_preview"] = [str(x) for x in args_list[:max_args_preview]]
    if topics:
        event_action["topics_preview"] = [str(x) for x in topics[:max_topics_preview]]

    # structured decode for Transfer/Approval
    if ev_name == "Transfer":
        token = _normalize_addr(contract)
        if token:
            event_action["token"] = token

        from_addr = None
        to_addr = None
        amount = None

        # args: list or dict
        if isinstance(args_obj, list) and len(args_obj) >= 3:
            from_addr = _normalize_addr(args_obj[0])
            to_addr = _normalize_addr(args_obj[1])
            try:
                amount = int(args_obj[2])
            except Exception:
                amount = _parse_u256_hex(args_obj[2]) if isinstance(args_obj[2], str) else None
        elif isinstance(args_obj, dict):
            # common key variants
            from_addr = _normalize_addr(args_obj.get("from") or args_obj.get("src") or args_obj.get("sender"))
            to_addr = _normalize_addr(args_obj.get("to") or args_obj.get("dst") or args_obj.get("recipient"))
            v = args_obj.get("value") if "value" in args_obj else args_obj.get("amount")
            try:
                amount = int(v) if v is not None else None
            except Exception:
                amount = _parse_u256_hex(v) if isinstance(v, str) else None

        # fallback topics
        if (from_addr is None or to_addr is None) and len(topics) >= 3:
            from_addr = from_addr or _topic_addr(topics[1])
            to_addr = to_addr or _topic_addr(topics[2])

        # fallback data
        if amount is None:
            amount = _parse_u256_hex(row.get(data_col) if data_col else None)

        if from_addr is not None:
            event_action["from"] = from_addr
        if to_addr is not None:
            event_action["to"] = to_addr
        if amount is not None:
            event_action["amount"] = amount

    elif ev_name == "Approval":
        token = _normalize_addr(contract)
        if token:
            event_action["token"] = token

        owner = None
        spender = None
        amount = None

        if isinstance(args_obj, list) and len(args_obj) >= 3:
            owner = _normalize_addr(args_obj[0])
            spender = _normalize_addr(args_obj[1])
            try:
                amount = int(args_obj[2])
            except Exception:
                amount = _parse_u256_hex(args_obj[2]) if isinstance(args_obj[2], str) else None
        elif isinstance(args_obj, dict):
            owner = _normalize_addr(args_obj.get("owner"))
            spender = _normalize_addr(args_obj.get("spender"))
            v = args_obj.get("value") if "value" in args_obj else args_obj.get("amount")
            try:
                amount = int(v) if v is not None else None
            except Exception:
                amount = _parse_u256_hex(v) if isinstance(v, str) else None

        if (owner is None or spender is None) and len(topics) >= 3:
            owner = owner or _topic_addr(topics[1])
            spender = spender or _topic_addr(topics[2])

        if amount is None:
            amount = _parse_u256_hex(row.get(data_col) if data_col else None)

        if owner is not None:
            event_action["owner"] = owner
        if spender is not None:
            event_action["spender"] = spender
        if amount is not None:
            event_action["amount"] = amount

    return event_action


def build_episodes_for_receipts(
    receipts: pd.DataFrame,
    tx_cache: TxDatasetCache,
    events_cache: Optional[DecodedEventsDatasetCache],
    window_minutes: int,
    start_offset_minutes: int,
    bucket_minutes: int,
    max_txs_per_episode: int,
    tx_addr_chunk: int,
    event_hash_chunk: int,
    pre_window_minutes: int,
    pre_end_offset_minutes: int,
    enrich_events: bool = True,
    include_to_address: bool = True,
    max_events_per_tx: int = 100,
) -> pd.DataFrame:
    """receipts 必须包含:
      - _idx (原表行号)
      - _chain (canonical chain)
      - _recv  (lowercase address)
      - _ts    (receipt ts, UTC)

    可选:
      - _src_chain, _src_addr

    返回 DataFrame，index = _idx，包含 episode columns。
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

    receipts = receipts.copy()

    # dst post window
    receipts["_dst_start"] = receipts["_ts"] + pd.to_timedelta(start_offset_minutes, unit="m")
    receipts["_dst_end"] = receipts["_dst_start"] + pd.to_timedelta(window_minutes, unit="m")
    receipts["_dst_bucket"] = receipts["_dst_start"].dt.floor(f"{bucket_minutes}min")

    # src pre window (optional)
    has_src = ("_src_chain" in receipts.columns) and (
        ("_src_addr" in receipts.columns) or ("_src_tx_hash" in receipts.columns)
    )
    if has_src:
        receipts["_src_end"] = receipts["_ts"] + pd.to_timedelta(pre_end_offset_minutes, unit="m")
        receipts["_src_start"] = receipts["_src_end"] - pd.to_timedelta(pre_window_minutes, unit="m")
        receipts["_src_bucket"] = receipts["_src_start"].dt.floor(f"{bucket_minutes}min")

    # generic collector
    def _collect(
        chain: str,
        df_b: pd.DataFrame,
        actor_col: str,
        start_col: str,
        end_col: str,
        phase: str,
        bucket_col: str,
    ) -> Dict[int, List[Dict[str, object]]]:
        if df_b.empty:
            return {}
        if actor_col not in df_b.columns:
            return {}

        # actors
        actors = [a for a in df_b[actor_col].dropna().astype(str).str.lower().unique().tolist() if a.startswith("0x")]
        if not actors:
            return {}

        txds = tx_cache.get(chain)

        # time window to scan (union)
        t_start = pd.to_datetime(df_b[start_col].min(), utc=True)
        t_end = pd.to_datetime(df_b[end_col].max(), utc=True)

        # scan txs: from_address in actors, and optionally to_address in actors
        tx_cols = [c for c in TX_REQUIRED_COLS if c in txds.available_cols]
        tx_tbl_from = txds.scan(actors, t_start, t_end, columns=tx_cols, addr_field="from_address", addr_chunk=tx_addr_chunk)
        tx_tables = [tx_tbl_from]
        if include_to_address and ("to_address" in txds.available_cols):
            tx_tbl_to = txds.scan(actors, t_start, t_end, columns=tx_cols, addr_field="to_address", addr_chunk=tx_addr_chunk)
            tx_tables.append(tx_tbl_to)

        tx_tbl = None
        non_empty = [t for t in tx_tables if t is not None and getattr(t, "num_rows", 0) > 0]
        if non_empty:
            tx_tbl = concat_tables_compat(non_empty)
        else:
            tx_tbl = pa.table({c: pa.array([], type=txds.schema.field(c).type) for c in tx_cols})

        txdf = tx_tbl.to_pandas() if tx_tbl.num_rows else pd.DataFrame(columns=tx_cols)
        if not txdf.empty:
            # normalize
            if "transaction_hash" in txdf.columns:
                txdf["transaction_hash"] = txdf["transaction_hash"].astype(str).str.lower()
            if "from_address" in txdf.columns:
                txdf["from_address"] = txdf["from_address"].apply(_normalize_addr)
            if "to_address" in txdf.columns:
                txdf["to_address"] = txdf["to_address"].apply(_normalize_addr)
            if "block_timestamp" in txdf.columns:
                txdf["block_timestamp"] = pd.to_datetime(txdf["block_timestamp"], utc=True, errors="coerce")
            if "input" in txdf.columns:
                txdf["input"] = txdf["input"].astype(str)

            # sort for stable order
            sort_cols = [c for c in ["block_timestamp", "block_number", "transaction_index", "transaction_hash"] if c in txdf.columns]
            if sort_cols:
                txdf = txdf.sort_values(sort_cols)

        # Enrich events (optional)
        event_actions_by_tx: Dict[str, List[Dict[str, object]]] = {}
        if enrich_events and events_cache is not None:
            evds = events_cache.get(chain)
            if evds is not None:
                ev_cols = [c for c in EVENT_REQUIRED_COLS if c in evds.available_cols]
                if ev_cols and (not txdf.empty) and ("transaction_hash" in txdf.columns):
                    tx_hashes = txdf["transaction_hash"].dropna().astype(str).str.lower().unique().tolist()
                    if tx_hashes:
                        ev_tbl = evds.scan(tx_hashes, t_start, t_end, columns=ev_cols, hash_chunk=event_hash_chunk)
                        if ev_tbl.num_rows:
                            evdf = ev_tbl.to_pandas()
                            # normalize
                            if "transaction_hash" in evdf.columns:
                                evdf["transaction_hash"] = evdf["transaction_hash"].astype(str).str.lower()
                            if "block_timestamp" in evdf.columns:
                                evdf["block_timestamp"] = pd.to_datetime(evdf["block_timestamp"], utc=True, errors="coerce")
                            for addr_col in ["address", "contract_address"]:
                                if addr_col in evdf.columns:
                                    evdf[addr_col] = evdf[addr_col].apply(_normalize_addr)

                            ts_col = pick_col(evdf.columns, ["block_timestamp"])
                            tx_col = pick_col(evdf.columns, ["transaction_hash", "tx_hash"])
                            log_col = pick_col(evdf.columns, ["log_index", "event_index", "log_idx"])
                            sig_col = pick_col(evdf.columns, ["event_signature"])
                            topic0_col = pick_col(evdf.columns, ["event_hash"])
                            addr_col = pick_col(evdf.columns, ["address", "contract_address"])
                            topics_col = pick_col(evdf.columns, ["topics"])
                            args_col = pick_col(evdf.columns, ["args_json", "args"])
                            data_col = pick_col(evdf.columns, ["data"])

                            for _, r in evdf.iterrows():
                                ea = _decode_event_row(
                                    r,
                                    ts_col=ts_col,
                                    tx_col=tx_col,
                                    log_col=log_col,
                                    sig_col=sig_col,
                                    topic0_col=topic0_col,
                                    addr_col=addr_col,
                                    topics_col=topics_col,
                                    args_col=args_col,
                                    data_col=data_col,
                                )
                                if ea is None:
                                    continue
                                txh = ea.get("tx_hash")
                                if not txh:
                                    continue
                                event_actions_by_tx.setdefault(str(txh), []).append(ea)

                            # sort by log_index and truncate
                            for txh, evs in list(event_actions_by_tx.items()):
                                evs.sort(key=lambda x: (x.get("log_index") is None, x.get("log_index", 0)))
                                if max_events_per_tx > 0 and len(evs) > max_events_per_tx:
                                    event_actions_by_tx[txh] = evs[:max_events_per_tx]

        # Build quick lookup tables by actor
        by_from: Dict[str, pd.DataFrame] = {}
        by_to: Dict[str, pd.DataFrame] = {}
        if not txdf.empty:
            if "from_address" in txdf.columns:
                by_from = {k: v for k, v in txdf.groupby("from_address")}
            if include_to_address and ("to_address" in txdf.columns):
                by_to = {k: v for k, v in txdf.groupby("to_address") if k is not None}

        actions_by_idx: Dict[int, List[Dict[str, object]]] = {}

        for tup in df_b.itertuples(index=False):
            idx = getattr(tup, "_idx", None)
            actor = getattr(tup, actor_col, None)
            w0 = getattr(tup, start_col, None)
            w1 = getattr(tup, end_col, None)

            actor = _normalize_addr(actor)
            if idx is None or actor is None or w0 is None or w1 is None:
                continue

            parts = []
            if actor in by_from:
                parts.append(by_from[actor])
            if include_to_address and actor in by_to:
                parts.append(by_to[actor])
            if not parts:
                actions_by_idx[idx] = []
                continue

            sub = parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)
            if "transaction_hash" in sub.columns:
                sub = sub.drop_duplicates(subset=["transaction_hash"], keep="first")

            if "block_timestamp" in sub.columns:
                sub = sub[(sub["block_timestamp"] >= w0) & (sub["block_timestamp"] < w1)]

            sort_cols = [c for c in ["block_timestamp", "block_number", "transaction_index", "transaction_hash"] if c in sub.columns]
            if sort_cols:
                sub = sub.sort_values(sort_cols)

            if max_txs_per_episode > 0 and len(sub) > max_txs_per_episode:
                sub = sub.iloc[:max_txs_per_episode]

            actions: List[Dict[str, object]] = []
            for row in sub.itertuples(index=False):
                ts = getattr(row, "block_timestamp", None)
                txh = getattr(row, "transaction_hash", None)
                to = getattr(row, "to_address", None)
                val = getattr(row, "value", None)
                inp = getattr(row, "input", None)
                method_id = None
                if isinstance(inp, str) and inp.startswith("0x") and len(inp) >= 10:
                    method_id = inp[:10]
                val_out = None
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    try:
                        val_out = int(val)
                    except Exception:
                        val_out = str(val)

                action = {
                    "ts": ts_iso(pd.Timestamp(ts) if ts is not None else None),
                    "tx_hash": str(txh).lower() if txh else None,
                    "to": to,
                    "value": val_out,
                    "method_id": method_id,
                    "chain": chain,
                    "phase": phase,
                }
                if txh and event_actions_by_tx:
                    action["events"] = event_actions_by_tx.get(str(txh).lower(), [])
                actions.append(action)

            actions_by_idx[idx] = actions

        return actions_by_idx

    # 1) dst_post actions
    dst_df = receipts[["_idx", "_chain", "_recv", "_dst_start", "_dst_end", "_dst_bucket", "_ts"]].copy()
    dst_df = dst_df.rename(columns={"_dst_start": "_window_start", "_dst_end": "_window_end", "_dst_bucket": "_bucket"})

    dst_actions_by_idx: Dict[int, List[Dict[str, object]]] = {}
    for chain, df_chain in dst_df.groupby("_chain", dropna=True):
        if chain is None:
            continue
        for _, df_b in df_chain.groupby("_bucket"):
            if df_b.empty:
                continue
            dst_actions_by_idx.update(
                _collect(
                    chain=chain,
                    df_b=df_b,
                    actor_col="_recv",
                    start_col="_window_start",
                    end_col="_window_end",
                    phase="dst_post",
                    bucket_col="_bucket",
                )
            )

    # 2) src_pre actions
    src_actions_by_idx: Dict[int, List[Dict[str, object]]] = {}
    if has_src:
        src_cols = ["_idx", "_src_chain", "_src_start", "_src_end", "_src_bucket", "_ts"]
        if "_src_addr" in receipts.columns:
            src_cols.append("_src_addr")
        if "_src_tx_hash" in receipts.columns:
            src_cols.append("_src_tx_hash")
        src_df = receipts[src_cols].copy()
        src_df = src_df.dropna(subset=["_src_chain", "_src_start", "_src_end"]).copy()
        if not src_df.empty:
            src_df = src_df.rename(
                columns={
                    "_src_chain": "_chain",
                    "_src_addr": "_addr",
                    "_src_start": "_window_start",
                    "_src_end": "_window_end",
                    "_src_bucket": "_bucket",
                }
            )
            if "_src_tx_hash" in src_df.columns:
                src_df["_src_tx_hash"] = src_df["_src_tx_hash"].astype(str).str.lower()
            for chain, df_chain in src_df.groupby("_chain", dropna=True):
                if chain is None:
                    continue
                for _, df_b in df_chain.groupby("_bucket"):
                    if df_b.empty:
                        continue
                    if "_src_tx_hash" in df_b.columns:
                        tx_hashes = (
                            df_b["_src_tx_hash"].dropna().astype(str).str.lower().unique().tolist()
                        )
                        if tx_hashes:
                            txds = tx_cache.get(chain)
                            t_start = pd.to_datetime(df_b["_window_start"].min(), utc=True)
                            t_end = pd.to_datetime(df_b["_window_end"].max(), utc=True)
                            tx_cols = [c for c in ["transaction_hash", "from_address"] if c in txds.available_cols]
                            if tx_cols:
                                tx_tbl = txds.scan_by_hashes(
                                    tx_hashes,
                                    t_start,
                                    t_end,
                                    columns=tx_cols,
                                    hash_chunk=tx_addr_chunk,
                                )
                                txdf = tx_tbl.to_pandas() if tx_tbl.num_rows else pd.DataFrame(columns=tx_cols)
                                if not txdf.empty:
                                    txdf["transaction_hash"] = txdf["transaction_hash"].astype(str).str.lower()
                                    if "from_address" in txdf.columns:
                                        txdf["from_address"] = txdf["from_address"].apply(_normalize_addr)
                                    tx_from = txdf.set_index("transaction_hash")["from_address"].to_dict()
                                    df_b = df_b.copy()
                                    df_b["_addr"] = df_b["_src_tx_hash"].map(tx_from).fillna(df_b.get("_addr"))
                    # reuse collector with actor_col name '_addr'
                    df_b2 = df_b.rename(columns={"_addr": "_recv"})
                    src_actions_by_idx.update(
                        _collect(
                            chain=chain,
                            df_b=df_b2,
                            actor_col="_recv",
                            start_col="_window_start",
                            end_col="_window_end",
                            phase="src_pre",
                            bucket_col="_bucket",
                        )
                    )

    # 3) write out
    # NOTE: _idx is unique; build a dict to avoid O(N^2) filtering.
    idx_to_receipt_ts: Dict[int, Optional[pd.Timestamp]] = {
        int(i): (ts if not pd.isna(ts) else None) for i, ts in zip(receipts["_idx"].tolist(), receipts["_ts"].tolist())
    }

    for idx in out.index:
        receipt_ts = idx_to_receipt_ts.get(int(idx))

        all_actions = (src_actions_by_idx.get(idx, []) or []) + (dst_actions_by_idx.get(idx, []) or [])
        # stable sort by ts
        all_actions = sorted(all_actions, key=lambda a: (a.get("ts") or ""))

        first_tx = all_actions[0]["tx_hash"] if all_actions else None
        last_tx = all_actions[-1]["tx_hash"] if all_actions else None
        last_ts = all_actions[-1]["ts"] if all_actions else None

        out.at[idx, "EPISODE_TX_COUNT"] = len(all_actions)
        out.at[idx, "EPISODE_ACTIONS_JSON"] = json.dumps(all_actions, ensure_ascii=False)
        out.at[idx, "EPISODE_FIRST_TX_HASH"] = first_tx
        out.at[idx, "EPISODE_LAST_TX_HASH"] = last_tx
        out.at[idx, "EPISODE_LAST_TX_TIMESTAMP"] = last_ts

        # duration: last tx ts - receipt ts
        try:
            if receipt_ts is not None and last_ts is not None:
                last_ts_pd = pd.to_datetime(last_ts, utc=True)
                receipt_ts_pd = pd.to_datetime(receipt_ts, utc=True)
                dur = int((last_ts_pd - receipt_ts_pd).total_seconds())
                out.at[idx, "EPISODE_DURATION_SEC"] = dur
        except Exception:
            pass

    return out


# ----------------------------
# File processing
# ----------------------------

def process_one_file(
    in_path: str,
    out_path: str,
    tx_cache: TxDatasetCache,
    events_cache: Optional[DecodedEventsDatasetCache],
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
    event_hash_chunk: int,
    pre_window_minutes: int,
    pre_end_offset_minutes: int,
    enrich_events: bool,
    include_to_address: bool,
    max_events_per_tx: int,
    allowed_chains: Optional[set] = None,
    limit_rows: Optional[int] = None,
) -> None:
    pf = open_parquet_file(in_path)
    cols = pf.schema.names

    auto_recv, auto_chain, auto_ts, auto_id = detect_crossdata_columns(cols)
    recv_col = recv_col or auto_recv
    chain_col = chain_col or auto_chain
    ts_col = ts_col or auto_ts
    id_col = id_col or auto_id

    missing_required = [name for name, col in [("recv_col", recv_col), ("chain_col", chain_col), ("ts_col", ts_col)] if col is None]
    if missing_required:
        print(f"[WARN] {os.path.basename(in_path)} missing columns {missing_required}; passthrough with empty episodes.", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    writer: Optional[pq.ParquetWriter] = None
    written_rows = 0

    for batch in pf.iter_batches(batch_size=batch_rows):
        tbl_in = pa.Table.from_batches([batch])

        if missing_required:
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

        need_cols = [recv_col, chain_col, ts_col]
        if id_col and id_col in tbl_in.column_names:
            need_cols.append(id_col)

        # optional src chain/address
        cols_in = tbl_in.column_names
        src_chain_col, src_addr_col, src_hash_col = detect_crossdata_source_columns(cols_in)
        if src_chain_col and src_chain_col in cols_in:
            need_cols.append(src_chain_col)
        if src_addr_col and src_addr_col in cols_in:
            need_cols.append(src_addr_col)
        if src_hash_col and src_hash_col in cols_in:
            need_cols.append(src_hash_col)

        # de-duplicate while preserving order; duplicated names can make
        # df[col] return a DataFrame (instead of Series), breaking .str ops.
        need_cols = list(dict.fromkeys(need_cols))

        df_need = tbl_in.select(need_cols).to_pandas()
        n = len(df_need)
        if n == 0:
            continue

        if limit_rows is not None and written_rows >= limit_rows:
            break
        if limit_rows is not None and written_rows + n > limit_rows:
            df_need = df_need.iloc[: (limit_rows - written_rows)]
            tbl_in = tbl_in.slice(0, len(df_need))
            n = len(df_need)

        receipts = pd.DataFrame(
            {
                "_idx": np.arange(written_rows, written_rows + n, dtype=np.int64),
                "_recv": df_need[recv_col].astype(str).str.lower(),
                "_chain": df_need[chain_col].apply(canonical_chain),
                "_ts": df_need[ts_col].apply(to_utc_ts),
            }
        )

        if src_chain_col and src_chain_col in df_need.columns:
            receipts["_src_chain"] = df_need[src_chain_col].apply(canonical_chain)
        else:
            receipts["_src_chain"] = None
        if src_addr_col and src_addr_col in df_need.columns:
            receipts["_src_addr"] = df_need[src_addr_col].astype(str).str.lower()
        else:
            receipts["_src_addr"] = None
        if src_hash_col and src_hash_col in df_need.columns:
            receipts["_src_tx_hash"] = df_need[src_hash_col].astype(str).str.lower()
        elif id_col and id_col in df_need.columns:
            receipts["_src_tx_hash"] = df_need[id_col].astype(str).str.lower()
        else:
            receipts["_src_tx_hash"] = None

        # drop invalid
        receipts = receipts[receipts["_recv"].astype(str).str.startswith("0x") & receipts["_ts"].notna()].copy()
        if allowed_chains:
            receipts = receipts[receipts["_chain"].isin(allowed_chains)].copy()

        epi = build_episodes_for_receipts(
            receipts=receipts,
            tx_cache=tx_cache,
            events_cache=events_cache,
            window_minutes=window_minutes,
            start_offset_minutes=start_offset_minutes,
            bucket_minutes=bucket_minutes,
            max_txs_per_episode=max_txs_per_episode,
            tx_addr_chunk=tx_addr_chunk,
            event_hash_chunk=event_hash_chunk,
            pre_window_minutes=pre_window_minutes,
            pre_end_offset_minutes=pre_end_offset_minutes,
            enrich_events=enrich_events,
            include_to_address=include_to_address,
            max_events_per_tx=max_events_per_tx,
        )

        # align back
        tx_count = np.zeros(n, dtype=np.int32)
        actions_json = np.array(["[]"] * n, dtype=object)
        first_hash = np.array([None] * n, dtype=object)
        last_hash = np.array([None] * n, dtype=object)
        last_ts = np.array([None] * n, dtype=object)
        duration_sec = np.array([None] * n, dtype=object)

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
    ap.add_argument("--crossdata_root", default="/home/chain1/zl/chain/crossdata")
    ap.add_argument("--tx_root", default="/home/chain1/zl/chain")
    ap.add_argument("--decoded_events_root", default=None)
    ap.add_argument("--out_root", default="/home/chain1/zl/chain/Heimdall/out/crossdata_with_episodes")
    ap.add_argument("--chains", default="eth,arbitrum,optimism")
    ap.add_argument("--recv_col", default=None)
    ap.add_argument("--chain_col", default=None)
    ap.add_argument("--ts_col", default=None)
    ap.add_argument("--id_col", default=None)

    ap.add_argument("--window_minutes", type=int, default=180)
    ap.add_argument("--start_offset_minutes", type=int, default=0)
    ap.add_argument("--pre_window_minutes", type=int, default=60)
    ap.add_argument("--pre_end_offset_minutes", type=int, default=0)
    ap.add_argument("--bucket_minutes", type=int, default=60)
    ap.add_argument("--max_txs_per_episode", type=int, default=50)
    ap.add_argument("--batch_rows", type=int, default=20000)
    ap.add_argument("--tx_addr_chunk", type=int, default=2000)
    ap.add_argument("--event_hash_chunk", type=int, default=2000)
    ap.add_argument("--limit_files", type=int, default=None)
    ap.add_argument("--limit_rows", type=int, default=None)

    # new flags
    ap.add_argument("--enrich_events", type=int, default=1, help="1=attach decoded_events into actions; 0=no")
    ap.add_argument("--include_to_address", type=int, default=1, help="1=tx where to_address==actor also included")
    ap.add_argument("--max_events_per_tx", type=int, default=100)

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
    events_root = args.decoded_events_root or args.tx_root
    events_cache = DecodedEventsDatasetCache(events_root)

    for in_path in tqdm(files, desc="episode files"):
        rel = os.path.relpath(in_path, cross_root)
        out_path = os.path.join(out_root, rel)
        process_one_file(
            in_path=in_path,
            out_path=out_path,
            tx_cache=tx_cache,
            events_cache=events_cache,
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
            event_hash_chunk=args.event_hash_chunk,
            pre_window_minutes=args.pre_window_minutes,
            pre_end_offset_minutes=args.pre_end_offset_minutes,
            enrich_events=bool(args.enrich_events),
            include_to_address=bool(args.include_to_address),
            max_events_per_tx=args.max_events_per_tx,
            allowed_chains=allowed_chains,
            limit_rows=args.limit_rows,
        )


if __name__ == "__main__":
    main()
