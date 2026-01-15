#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow.parquet as pq


def pick_col(cols: List[str], cands: List[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in cols:
            return c
        if c.lower() in low:
            return low[c.lower()]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_parquet", required=True, help="episode 输出 parquet 文件路径（包含 EPISODE_* 列）")
    ap.add_argument("--out_html", required=True, help="输出 HTML 路径")
    ap.add_argument("--limit", type=int, default=2000, help="最多导出多少条 episode（tx_count>0）")
    ap.add_argument("--explorer_tx", default="", help="交易浏览器前缀，例如 https://arbiscan.io/tx/ 或 https://optimistic.etherscan.io/tx/")
    args = ap.parse_args()

    in_path = Path(args.in_parquet)
    if not in_path.exists():
        raise SystemExit(f"not found: {in_path}")

    # 只读需要的列（速度快）
    needed = [
        "EPISODE_TX_COUNT",
        "EPISODE_ACTIONS_JSON",
        "EPISODE_FIRST_TX_HASH",
        "EPISODE_LAST_TX_HASH",
        "EPISODE_LAST_TX_TIMESTAMP",
        "EPISODE_DURATION_SEC",
    ]
    tbl = pq.read_table(str(in_path))
    df = tbl.to_pandas()

    for c in needed:
        if c not in df.columns:
            raise SystemExit(f"missing column {c} in {in_path}")

    # 尽量带上原始关键信息（如果存在）
    cols = list(df.columns)
    recv_col = pick_col(cols, ["DESTINATION_ADDRESS", "RECEIVER", "TO", "to", "destination_address", "receiver"])
    chain_col = pick_col(cols, ["DESTINATION_CHAIN", "DST_CHAIN", "destination_chain", "dst_chain"])
    ts_col = pick_col(cols, ["BLOCK_TIMESTAMP", "block_timestamp", "DESTINATION_TIMESTAMP", "destination_timestamp", "timestamp"])

    keep = []
    if recv_col: keep.append(recv_col)
    if chain_col: keep.append(chain_col)
    if ts_col: keep.append(ts_col)
    keep += needed

    view = df[keep].copy()
    view = view[view["EPISODE_TX_COUNT"].fillna(0).astype(int) > 0].copy()
    if args.limit and len(view) > args.limit:
        view = view.head(args.limit).copy()

    # 每条 episode 解析 actions JSON，供前端展示
    episodes: List[Dict[str, Any]] = []
    for i, row in view.reset_index(drop=True).iterrows():
        try:
            actions = json.loads(row["EPISODE_ACTIONS_JSON"]) if isinstance(row["EPISODE_ACTIONS_JSON"], str) else []
            if not isinstance(actions, list):
                actions = []
        except Exception:
            actions = []

        ep = {
            "id": i,
            "tx_count": int(row["EPISODE_TX_COUNT"]),
            "first_tx": row.get("EPISODE_FIRST_TX_HASH"),
            "last_tx": row.get("EPISODE_LAST_TX_HASH"),
            "last_ts": row.get("EPISODE_LAST_TX_TIMESTAMP"),
            "duration_sec": (int(row["EPISODE_DURATION_SEC"]) if pd.notna(row.get("EPISODE_DURATION_SEC")) else None),
            "receiver": (row.get(recv_col) if recv_col else None),
            "dst_chain": (row.get(chain_col) if chain_col else None),
            "receipt_ts": (str(row.get(ts_col)) if ts_col else None),
            "actions": actions,
        }
        episodes.append(ep)

    # 生成静态 HTML（用 DataTables CDN）
    explorer_tx = args.explorer_tx.strip()
    if explorer_tx and not explorer_tx.endswith("/"):
        explorer_tx += "/"

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Episode Viewer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css"/>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; }}
    .wrap {{ display: flex; height: 100vh; }}
    .left {{ width: 52%; padding: 12px; overflow: auto; border-right: 1px solid #ddd; }}
    .right {{ width: 48%; padding: 12px; overflow: auto; }}
    .muted {{ color: #666; font-size: 12px; }}
    pre {{ background: #f6f6f6; padding: 10px; border-radius: 6px; overflow: auto; }}
    table.dataTable tbody tr.selected {{ background-color: #dbeafe; }}
    .kpi {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; }}
    .kv {{ display:flex; gap:8px; }}
    .kv b {{ width: 140px; display:inline-block; }}
    a {{ color: #2563eb; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
<div class="wrap">
  <div class="left">
    <div class="muted">File: {in_path}</div>
    <div class="muted">Episodes shown: {len(episodes)} (tx_count&gt;0)</div>
    <hr/>
    <table id="epTable" class="display" style="width:100%">
      <thead>
        <tr>
          <th>id</th>
          <th>dst_chain</th>
          <th>receiver</th>
          <th>tx_count</th>
          <th>duration_sec</th>
          <th>first_tx</th>
          <th>last_tx</th>
          <th>last_ts</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>

  <div class="right">
    <h3>Episode Detail</h3>
    <div id="detail" class="muted">Click a row on the left.</div>
  </div>
</div>

<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>

<script>
const EXPLORER_TX = {json.dumps(explorer_tx)};
const EPISODES = {json.dumps(episodes, ensure_ascii=False)};

function txLink(txHash) {{
  if(!txHash) return '';
  if(EXPLORER_TX) {{
    return `<a href="${{EXPLORER_TX}}${{txHash}}" target="_blank">${{txHash}}</a>`;
  }}
  return txHash;
}}

function renderDetail(ep) {{
  const actions = ep.actions || [];
  const header = `
    <div class="kpi">
      <div class="card">
        <div class="kv"><b>dst_chain</b><span>${{ep.dst_chain || ''}}</span></div>
        <div class="kv"><b>receiver</b><span>${{ep.receiver || ''}}</span></div>
        <div class="kv"><b>receipt_ts</b><span>${{ep.receipt_ts || ''}}</span></div>
      </div>
      <div class="card">
        <div class="kv"><b>tx_count</b><span>${{ep.tx_count}}</span></div>
        <div class="kv"><b>duration_sec</b><span>${{ep.duration_sec ?? ''}}</span></div>
        <div class="kv"><b>first_tx</b><span>${{txLink(ep.first_tx)}}</span></div>
        <div class="kv"><b>last_tx</b><span>${{txLink(ep.last_tx)}}</span></div>
      </div>
    </div>
    <h4>Actions</h4>
  `;

  let rows = '';
  for (const a of actions) {{
    const ts = a.ts || '';
    const txh = a.tx_hash || '';
    const to  = a.to || '';
    const val = (a.value === null || a.value === undefined) ? '' : String(a.value);
    const mid = a.method_id || '';
    rows += `<tr>
      <td>${{ts}}</td>
      <td>${{txLink(txh)}}</td>
      <td>${{to}}</td>
      <td>${{val}}</td>
      <td>${{mid}}</td>
    </tr>`;
  }}

  const table = `
    <table class="display" style="width:100%">
      <thead>
        <tr><th>ts</th><th>tx_hash</th><th>to</th><th>value</th><th>method_id</th></tr>
      </thead>
      <tbody>${{rows}}</tbody>
    </table>
  `;

  document.getElementById("detail").innerHTML = header + table;
}}

$(document).ready(function() {{
  const tbody = $("#epTable tbody");
  for (const ep of EPISODES) {{
    tbody.append(`<tr data-id="${{ep.id}}">
      <td>${{ep.id}}</td>
      <td>${{ep.dst_chain || ''}}</td>
      <td>${{ep.receiver || ''}}</td>
      <td>${{ep.tx_count}}</td>
      <td>${{ep.duration_sec ?? ''}}</td>
      <td>${{ep.first_tx || ''}}</td>
      <td>${{ep.last_tx || ''}}</td>
      <td>${{ep.last_ts || ''}}</td>
    </tr>`);
  }}

  const table = $("#epTable").DataTable({{
    pageLength: 25,
    order: [[3, "desc"]],
  }});

  $('#epTable tbody').on('click', 'tr', function() {{
    const id = parseInt($(this).attr("data-id"));
    $('#epTable tbody tr').removeClass('selected');
    $(this).addClass('selected');
    const ep = EPISODES.find(x => x.id === id);
    if (ep) renderDetail(ep);
  }});
}});
</script>
</body>
</html>
"""

    out_path = Path(args.out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"[OK] wrote: {out_path}")
    print("Open it in a browser. Example:")
    print(f"  open {out_path}")  # macOS
    print(f"  xdg-open {out_path}")  # linux


if __name__ == "__main__":
    main()