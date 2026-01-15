# -*- coding: utf-8 -*-
"""
crossdata_schema.py

负责 crossdata 各种 schema 的字段探测与规范化：
- receiver 列
- chain 列
- timestamp 列
- optional id/tx_hash 列

注意：本模块不做任何 episode 逻辑，只做“字段识别 + canonicalize”。
"""

from __future__ import annotations

import re
from typing import Optional, Sequence, Tuple, Dict
import numpy as np
import pandas as pd

# 这些 mapping 也可以继续放这里，供 canonical_chain 使用
CHAIN_ID_TO_NAME: Dict[int, str] = {
    1: "eth",
    10: "optimism",
    56: "bsc",
    100: "gnosis",
    137: "polygon",
    250: "fantom",
    324: "zksync",
    8453: "base",
    42161: "arbitrum",
    43114: "avalanche",
}

CHAIN_SYNONYMS: Dict[str, str] = {
    "ethereum": "eth",
    "eth": "eth",
    "mainnet": "eth",
    "ethereum_mainnet": "eth",
    "arbitrum": "arbitrum",
    "arbitrumone": "arbitrum",
    "arbitrum_one": "arbitrum",
    "arb": "arbitrum",
    "optimism": "optimism",
    "op": "optimism",
    "polygon": "polygon",
    "matic": "polygon",
    "base": "base",
    "bsc": "bsc",
    "bnb": "bsc",
    "avalanche": "avalanche",
    "avax": "avalanche",
    "gnosis": "gnosis",
    "xdai": "gnosis",
    "fantom": "fantom",
    "ftm": "fantom",
    "zksync": "zksync",
}

def _norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")

def _pick_col(cols: Sequence[str], keys: Sequence[str], contains=()) -> Optional[str]:
    norm_map = {_norm_col(c): c for c in cols}
    for k in keys:
        kk = _norm_col(k)
        if kk in norm_map:
            return norm_map[kk]
    for c in cols:
        nc = _norm_col(c)
        for parts in contains:
            if all(p in nc for p in parts):
                return c
    return None

def detect_crossdata_columns(cols: Sequence[str]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    返回 (recv_col, chain_col, ts_col, id_col)
    找不到则返回 None（由调用方决定 passthrough / 或显式指定参数）
    """
    recv_col = _pick_col(
        cols,
        keys=[
            "destination_address", "dest_address", "dst_address",
            "recipient_address", "receiver_address",
            "recipient", "receiver",
            "to", "TO",
        ],
        contains=[
            ("destination", "address"),
            ("dest", "address"),
            ("dst", "address"),
            ("receive", "address"),
            ("recipient", "address"),
            ("to", "address"),
            ("to",),
        ],
    )
    if recv_col is None:
        recv_col = _pick_col(cols, keys=["to_address", "to", "TO"], contains=[("to", "address"), ("to",)])

    chain_col = _pick_col(
        cols,
        keys=[
            "destination_chain", "destination_chain_id",
            "dest_chain", "dest_chain_id",
            "dst_chain", "dst_chain_id",
            "to_chain", "to_chain_id",
            "tochain",
        ],
        contains=[
            ("destination", "chain"),
            ("dest", "chain"),
            ("dst", "chain"),
            ("to", "chain"),
        ],
    )
    if chain_col is None:
        # fallback: some crossdata only has SOURCE_CHAIN; using it as destination chain is imperfect.
        chain_col = _pick_col(cols, keys=["source_chain", "SOURCE_CHAIN"], contains=[("source", "chain")])

    ts_col = _pick_col(
        cols,
        keys=[
            "destination_timestamp", "destination_block_timestamp",
            "dest_timestamp", "dest_block_timestamp",
            "dst_timestamp", "dst_block_timestamp",
            "block_timestamp", "timestamp", "time",
        ],
        contains=[
            ("destination", "timestamp"),
            ("dest", "timestamp"),
            ("dst", "timestamp"),
            ("block", "timestamp"),
        ],
    )

    id_col = _pick_col(
        cols,
        keys=[
            "source_tx_hash", "src_tx_hash", "src_hash", "source_hash",
            "transaction_hash", "tx_hash", "hash",
        ],
        contains=[("src", "hash"), ("source", "hash"), ("tx", "hash")],
    )

    return recv_col, chain_col, ts_col, id_col

def canonical_chain(x) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return CHAIN_ID_TO_NAME.get(int(x))
    if isinstance(x, (float, np.floating)) and not np.isnan(x):
        return CHAIN_ID_TO_NAME.get(int(x))

    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none", "null"):
        return None
    if re.fullmatch(r"\d+", s):
        return CHAIN_ID_TO_NAME.get(int(s))

    k = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    k2 = k.replace("_", "")
    return CHAIN_SYNONYMS.get(k) or CHAIN_SYNONYMS.get(k2) or k

def to_utc_ts(x) -> pd.Timestamp:
    return pd.to_datetime(x, utc=True, errors="coerce")