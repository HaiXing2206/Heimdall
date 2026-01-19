#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""episode_to_actiontrace.py

Episode -> ActionTrace 转换器
---------------------------
输入:
  - crossdata_with_episodes parquet（包含 EPISODE_ACTIONS_JSON）
  - <chain>/transactions parquet 数据集
  - <chain>/decoded_events parquet 数据集

输出:
  - 一个新的 parquet: 一行=一笔 tx（episode 展开），包含
    * episode_row (原始输入行号)
    * chain, phase
    * tx_hash, block_timestamp
    * from/to/method_id/value
    * actions_json: [CALL] + decoded_events(按 log_index)
    * transfer_delta_json: 以 focus addresses 为对象的 token 净变化（来自 Transfer 事件 + tx.value 的 native 变化）

设计要点
--------
1) 事件序列严格按 log_index 排序；
2) Transfer/Approval 支持 args_json/args/topics/data 多路解析；
3) Transfer-delta 默认对“关注地址集合”计算：
   - tx.from_address
   - (可选) episode 行里的 recv/src 地址列（通过 --focus_addr_cols 指定）

你可以把本脚本产出的 per-tx ActionTrace 直接喂给后续
(弱监督 + 深度聚类 / 表征学习) 模型。
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


# ----------------------------
# Fallback utils (standalone)
# ----------------------------
TRANSFER_TOPIC0 = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
APPROVAL_TOPIC0 = "0x8c5be1e5ebec7d5bd14f71427d1e84f3dd0314c0f7b2291e5b200ac8c7c3b925"


def _normalize_addr(x: object) -> Optional[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().lower()
    if s in ("", "none", "null", "nan"):
        return None
    return s


def _topic_addr(topic: object) -> Optional[str]:
    if topic is None:
        return None
    s = str(topic).strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    if len(s) < 40:
        return None
    return "0x" + s[-40:]


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


def _parse_json_maybe(x: object) -> Optional[object]:
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
        s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                return json.loads(s)
            except Exception:
                return None
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


# ----------------------------
# Datasets
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
            return pa.table({c: pa.array([], type=self.schema.field(c).type) for c in cols})

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
            return pa.table({c: pa.array([], type=self.schema.field(c).type) for c in cols})
        return tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote=True)


class TxDatasetCache:
    def __init__(self, tx_root: str):
        self.tx_root = os.path.expanduser(tx_root)
        self._cache: Dict[str, TxDataset] = {}

    def get(self, chain: str) -> TxDataset:
        chain = str(chain).strip().lower()
        if chain in self._cache:
            return self._cache[chain]

        tx_dir = os.path.join(self.tx_root, chain, "transactions")
        if not os.path.isdir(tx_dir):
            raise FileNotFoundError(f"[tx] 未找到 transactions 目录: {tx_dir}")

        dset = ds.dataset(tx_dir, format="parquet")
        txds = TxDataset(chain=chain, dataset=dset)
        self._cache[chain] = txds
        return txds


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
        for part in chunked(list(tx_hashes), hash_chunk):
            expr = (
                (ds.field(self.ts_field) >= ts_start)
                & (ds.field(self.ts_field) < ts_end)
                & ds.field(self.tx_hash_field).isin([h for h in part])
            )
            t = self.dataset.to_table(columns=cols, filter=expr)
            if t.num_rows:
                tables.append(t)

        if not tables:
            return pa.table({c: pa.array([], type=self.schema.field(c).type) for c in cols})
        return tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote=True)


class DecodedEventsDatasetCache:
    def __init__(self, events_root: str):
        self.events_root = os.path.expanduser(events_root)
        self._cache: Dict[str, Optional[DecodedEventsDataset]] = {}

    def get(self, chain: str) -> Optional[DecodedEventsDataset]:
        chain = str(chain).strip().lower()
        if chain in self._cache:
            return self._cache[chain]

        events_dir = os.path.join(self.events_root, chain, "decoded_events")
        if not os.path.isdir(events_dir):
            self._cache[chain] = None
            return None

        dset = ds.dataset(events_dir, format="parquet")
        evds = DecodedEventsDataset(chain=chain, dataset=dset)
        self._cache[chain] = evds
        return evds


# ----------------------------
# Decoding + delta
# ----------------------------

def decode_events_for_txs(
    evdf: pd.DataFrame,
) -> Dict[str, List[Dict[str, object]]]:
    """Return tx_hash -> sorted events list (by log_index)."""
    if evdf.empty:
        return {}

    # normalize
    if "transaction_hash" in evdf.columns:
        evdf["transaction_hash"] = evdf["transaction_hash"].astype(str).str.lower()
    if "block_timestamp" in evdf.columns:
        evdf["block_timestamp"] = pd.to_datetime(evdf["block_timestamp"], utc=True, errors="coerce")
    if "address" in evdf.columns:
        evdf["address"] = evdf["address"].apply(_normalize_addr)

    out: Dict[str, List[Dict[str, object]]] = {}

    for _, r in evdf.iterrows():
        txh = r.get("transaction_hash")
        if not txh:
            continue

        sig = r.get("event_signature")
        topic0 = r.get("event_hash")
        topics = _parse_json_maybe(r.get("topics"))
        if topics is None:
            topics = r.get("topics")
        if not isinstance(topics, list):
            topics = []

        t0 = str(topic0).lower() if topic0 is not None else (str(topics[0]).lower() if topics else "")
        ev_name = None
        if isinstance(sig, str) and sig:
            ev_name = sig.split("(", 1)[0]
        if not ev_name and t0 == TRANSFER_TOPIC0:
            ev_name = "Transfer"
        if not ev_name and t0 == APPROVAL_TOPIC0:
            ev_name = "Approval"

        ts = r.get("block_timestamp")
        ea: Dict[str, object] = {
            "ts": ts_iso(pd.Timestamp(ts) if ts is not None else None),
            "tx_hash": str(txh).lower(),
            "log_index": r.get("log_index"),
            "contract": _normalize_addr(r.get("address")),
        }
        if ev_name:
            ea["event_name"] = ev_name
        if isinstance(sig, str) and sig and (not ev_name or ev_name not in ("Transfer", "Approval")):
            ea["event_signature"] = sig

        # parse args
        args_obj = _parse_json_maybe(r.get("args_json"))
        if args_obj is None:
            args_obj = _parse_json_maybe(r.get("args"))

        # structured for Transfer/Approval
        if ev_name == "Transfer":
            token = ea.get("contract")
            if token:
                ea["token"] = token
            from_addr = None
            to_addr = None
            amount = None
            if isinstance(args_obj, list) and len(args_obj) >= 3:
                from_addr = _normalize_addr(args_obj[0])
                to_addr = _normalize_addr(args_obj[1])
                try:
                    amount = int(args_obj[2])
                except Exception:
                    amount = _parse_u256_hex(args_obj[2]) if isinstance(args_obj[2], str) else None
            elif isinstance(args_obj, dict):
                from_addr = _normalize_addr(args_obj.get("from") or args_obj.get("src") or args_obj.get("sender"))
                to_addr = _normalize_addr(args_obj.get("to") or args_obj.get("dst") or args_obj.get("recipient"))
                v = args_obj.get("value") if "value" in args_obj else args_obj.get("amount")
                try:
                    amount = int(v) if v is not None else None
                except Exception:
                    amount = _parse_u256_hex(v) if isinstance(v, str) else None

            if (from_addr is None or to_addr is None) and len(topics) >= 3:
                from_addr = from_addr or _topic_addr(topics[1])
                to_addr = to_addr or _topic_addr(topics[2])

            if amount is None:
                amount = _parse_u256_hex(r.get("data"))

            if from_addr is not None:
                ea["from"] = from_addr
            if to_addr is not None:
                ea["to"] = to_addr
            if amount is not None:
                ea["amount"] = amount

        elif ev_name == "Approval":
            token = ea.get("contract")
            if token:
                ea["token"] = token
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
                amount = _parse_u256_hex(r.get("data"))

            if owner is not None:
                ea["owner"] = owner
            if spender is not None:
                ea["spender"] = spender
            if amount is not None:
                ea["amount"] = amount

        out.setdefault(str(txh).lower(), []).append(ea)

    for txh, evs in out.items():
        evs.sort(key=lambda x: (x.get("log_index") is None, x.get("log_index", 0)))
    return out


def compute_transfer_delta(
    events: Sequence[Dict[str, object]],
    focus_addrs: Sequence[str],
    native_value: Optional[int] = None,
    native_from: Optional[str] = None,
    native_to: Optional[str] = None,
    native_token_key: str = "native",
) -> Dict[str, Dict[str, int]]:
    """Return {focus_addr: {token: net_delta}}.

    - Transfer: to += amount, from -= amount
    - Native value: native_to += value, native_from -= value

    注: 这里只基于 logs/decoded_events + tx.value，无法覆盖内部转账(traces)。
    """
    focus = [a for a in (_normalize_addr(x) for x in focus_addrs) if a]
    focus_set = set(focus)

    delta: Dict[str, Dict[str, int]] = {a: {} for a in focus}

    def _add(addr: str, token: str, d: int):
        if addr not in delta:
            delta[addr] = {}
        delta[addr][token] = int(delta[addr].get(token, 0)) + int(d)

    # ERC20 transfers
    for e in events:
        if e.get("event_name") != "Transfer":
            continue
        token = _normalize_addr(e.get("token") or e.get("contract")) or "unknown_token"
        f = _normalize_addr(e.get("from"))
        t = _normalize_addr(e.get("to"))
        amt = e.get("amount")
        try:
            amt_i = int(amt) if amt is not None else None
        except Exception:
            amt_i = None
        if amt_i is None:
            continue

        if f in focus_set:
            _add(f, token, -amt_i)
        if t in focus_set:
            _add(t, token, +amt_i)

    # native
    if native_value is not None:
        try:
            v = int(native_value)
        except Exception:
            v = None
        if v is not None and v != 0:
            nf = _normalize_addr(native_from)
            nt = _normalize_addr(native_to)
            if nf in focus_set:
                _add(nf, native_token_key, -v)
            if nt in focus_set:
                _add(nt, native_token_key, +v)

    return delta


# ----------------------------
# Conversion
# ----------------------------

def parse_episode_actions(s: object) -> List[Dict[str, object]]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    if isinstance(s, list):
        return s
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, list) else []
    except Exception:
        return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_parquet", required=True, help="crossdata_with_episodes 的某个 parquet 文件")
    ap.add_argument("--out_parquet", required=True, help="输出 parquet（per-tx）")
    ap.add_argument("--tx_root", required=True, help="包含 <chain>/transactions 的根目录")
    ap.add_argument("--decoded_events_root", default=None, help="包含 <chain>/decoded_events 的根目录，默认同 tx_root")

    ap.add_argument("--episode_actions_col", default="EPISODE_ACTIONS_JSON")
    ap.add_argument(
        "--focus_addr_cols",
        default="",
        help="额外纳入 transfer-delta 的地址列（逗号分隔），例如 'SOURCE_ADDRESS,DESTINATION_ADDRESS'。默认空=只用 tx.from_address",
    )

    ap.add_argument("--batch_rows", type=int, default=5000)
    ap.add_argument("--tx_hash_chunk", type=int, default=2000)
    ap.add_argument("--event_hash_chunk", type=int, default=2000)
    ap.add_argument("--max_events_per_tx", type=int, default=500)

    args = ap.parse_args()

    in_path = os.path.expanduser(args.in_parquet)
    out_path = os.path.expanduser(args.out_parquet)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tx_cache = TxDatasetCache(args.tx_root)
    ev_cache = DecodedEventsDatasetCache(args.decoded_events_root or args.tx_root)

    focus_cols = [c.strip() for c in str(args.focus_addr_cols).split(",") if c.strip()]

    pf = pq.ParquetFile(in_path)
    cols = pf.schema.names
    need_cols = [args.episode_actions_col]
    for c in focus_cols:
        if c in cols:
            need_cols.append(c)

    writer: Optional[pq.ParquetWriter] = None
    global_row = 0

    for batch in tqdm(pf.iter_batches(batch_size=args.batch_rows), desc="actiontrace batches"):
        tbl = pa.Table.from_batches([batch])
        df = tbl.select([c for c in need_cols if c in tbl.column_names]).to_pandas()
        if df.empty:
            global_row += tbl.num_rows
            continue

        # 1) explode episodes -> tx rows
        tx_rows: List[Dict[str, object]] = []
        per_episode_focus: Dict[int, List[str]] = {}

        for i, r in df.iterrows():
            episode_row = global_row + i

            # focus addrs from columns
            extra_focus: List[str] = []
            for c in focus_cols:
                if c in df.columns:
                    extra_focus.append(_normalize_addr(r.get(c)))
            extra_focus = [a for a in extra_focus if a and a.startswith("0x")]
            per_episode_focus[episode_row] = extra_focus

            acts = parse_episode_actions(r.get(args.episode_actions_col))
            for pos, a in enumerate(acts):
                txh = _normalize_addr(a.get("tx_hash") or a.get("transaction_hash"))
                chain = str(a.get("chain") or "").strip().lower()
                if not txh or not chain:
                    continue
                tx_rows.append(
                    {
                        "episode_row": episode_row,
                        "tx_pos": pos,
                        "chain": chain,
                        "phase": a.get("phase"),
                        "tx_hash": txh,
                        "ts": a.get("ts"),
                    }
                )

        if not tx_rows:
            global_row += tbl.num_rows
            continue

        tx_rows_df = pd.DataFrame(tx_rows)

        # 2) fetch tx meta + events per chain
        out_records: List[Dict[str, object]] = []

        for chain, g in tx_rows_df.groupby("chain"):
            tx_hashes = g["tx_hash"].dropna().astype(str).str.lower().unique().tolist()
            ts_list = pd.to_datetime(g["ts"], utc=True, errors="coerce")
            # time range fallback: use action timestamps +/- 1h
            t_min = ts_list.min()
            t_max = ts_list.max()
            if pd.isna(t_min) or pd.isna(t_max):
                # if missing, give a wide but bounded window
                t_min = pd.Timestamp("1970-01-01", tz="UTC")
                t_max = pd.Timestamp("2100-01-01", tz="UTC")
            else:
                t_min = t_min - pd.Timedelta(hours=1)
                t_max = t_max + pd.Timedelta(hours=1)

            txds = tx_cache.get(chain)
            tx_cols = [c for c in [
                "block_timestamp",
                "transaction_hash",
                "from_address",
                "to_address",
                "value",
                "input",
                "block_number",
                "transaction_index",
            ] if c in txds.available_cols]

            tx_tbl = txds.scan_by_hashes(tx_hashes, t_min, t_max, columns=tx_cols, hash_chunk=args.tx_hash_chunk)
            txdf = tx_tbl.to_pandas() if tx_tbl.num_rows else pd.DataFrame(columns=tx_cols)
            if not txdf.empty:
                txdf["transaction_hash"] = txdf["transaction_hash"].astype(str).str.lower()
                if "block_timestamp" in txdf.columns:
                    txdf["block_timestamp"] = pd.to_datetime(txdf["block_timestamp"], utc=True, errors="coerce")
                if "from_address" in txdf.columns:
                    txdf["from_address"] = txdf["from_address"].apply(_normalize_addr)
                if "to_address" in txdf.columns:
                    txdf["to_address"] = txdf["to_address"].apply(_normalize_addr)
                if "input" in txdf.columns:
                    txdf["input"] = txdf["input"].astype(str)

            # Make a lightweight hash -> dict meta mapping (avoid pandas.Series handling later)
            tx_meta: Dict[str, Dict[str, object]] = (
                txdf.set_index("transaction_hash").to_dict(orient="index") if not txdf.empty else {}
            )

            # events
            ev_map: Dict[str, List[Dict[str, object]]] = {}
            evds = ev_cache.get(chain)
            if evds is not None:
                ev_cols = [c for c in [
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
                ] if c in evds.available_cols]
                if ev_cols:
                    ev_tbl = evds.scan(tx_hashes, t_min, t_max, columns=ev_cols, hash_chunk=args.event_hash_chunk)
                    evdf = ev_tbl.to_pandas() if ev_tbl.num_rows else pd.DataFrame(columns=ev_cols)
                    if not evdf.empty:
                        # normalize topics: keep as python list if string JSON
                        if "topics" in evdf.columns:
                            evdf["topics"] = evdf["topics"].apply(lambda x: _parse_json_maybe(x) if not isinstance(x, list) else x)
                        ev_map = decode_events_for_txs(evdf)

            # 3) build output records for chain
            for row in g.itertuples(index=False):
                txh = str(row.tx_hash).lower()
                meta = tx_meta.get(txh)

                from_addr = _normalize_addr(meta.get("from_address")) if isinstance(meta, dict) else _normalize_addr(getattr(meta, "from_address", None))
                to_addr = _normalize_addr(meta.get("to_address")) if isinstance(meta, dict) else _normalize_addr(getattr(meta, "to_address", None))

                # method_id
                inp = None
                if isinstance(meta, dict):
                    inp = meta.get("input")
                elif meta is not None:
                    inp = getattr(meta, "input", None)
                method_id = None
                if isinstance(inp, str) and inp.startswith("0x") and len(inp) >= 10:
                    method_id = inp[:10]

                value = None
                if isinstance(meta, dict):
                    value = meta.get("value")
                elif meta is not None:
                    value = getattr(meta, "value", None)
                try:
                    value_i = int(value) if value is not None and not (isinstance(value, float) and np.isnan(value)) else None
                except Exception:
                    value_i = None

                bts = None
                if isinstance(meta, dict):
                    bts = meta.get("block_timestamp")
                elif meta is not None:
                    bts = getattr(meta, "block_timestamp", None)

                events = ev_map.get(txh, [])
                if args.max_events_per_tx > 0 and len(events) > args.max_events_per_tx:
                    events = events[: args.max_events_per_tx]

                # focus addrs = tx.from + extra from episode columns
                extra_focus = per_episode_focus.get(int(row.episode_row), [])
                focus_addrs = [a for a in [from_addr] + list(extra_focus) if a]

                transfer_delta = compute_transfer_delta(
                    events=events,
                    focus_addrs=focus_addrs,
                    native_value=value_i,
                    native_from=from_addr,
                    native_to=to_addr,
                    native_token_key="native",
                )

                # actions: CALL + events
                call_action = {
                    "type": "CALL",
                    "ts": ts_iso(pd.Timestamp(bts) if bts is not None else None) or row.ts,
                    "tx_hash": txh,
                    "from": from_addr,
                    "to": to_addr,
                    "method_id": method_id,
                    "value": value_i,
                    "chain": chain,
                }
                actions = [call_action] + [
                    {
                        **{k: v for k, v in e.items() if k != "tx_hash"},
                        "type": e.get("event_name") or e.get("event_signature") or "EVENT",
                        "tx_hash": txh,
                    }
                    for e in events
                ]

                out_records.append(
                    {
                        "episode_row": int(row.episode_row),
                        "tx_pos": int(row.tx_pos),
                        "chain": chain,
                        "phase": row.phase,
                        "tx_hash": txh,
                        "block_timestamp": ts_iso(pd.Timestamp(bts) if bts is not None else None) or row.ts,
                        "from_address": from_addr,
                        "to_address": to_addr,
                        "method_id": method_id,
                        "value": value_i,
                        "actions_json": json.dumps(actions, ensure_ascii=False),
                        "transfer_delta_json": json.dumps(transfer_delta, ensure_ascii=False),
                        "n_events": len(events),
                    }
                )

        if not out_records:
            global_row += tbl.num_rows
            continue

        out_df = pd.DataFrame(out_records)

        # write parquet (append)
        out_tbl = pa.Table.from_pandas(out_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, out_tbl.schema, compression="snappy")
        writer.write_table(out_tbl)

        global_row += tbl.num_rows

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
