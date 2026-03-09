# Heimdall 运行说明

这个仓库目前主要包含 3 个可执行脚本：

- `src/build_episodes.py`：基于 crossdata 与链上交易/事件数据构建 episode。
- `src/episode_to_actiontrace.py`：把 episode 展开为按交易粒度的 ActionTrace。
- `tools/make_episode_viewer.py`：把 episode 导出成可浏览的 HTML 视图。

## 1) 环境准备

建议 Python 3.10+，安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas numpy pyarrow tqdm
```

## 2) 先看参数（推荐）

```bash
python src/build_episodes.py --help
python src/episode_to_actiontrace.py --help
python tools/make_episode_viewer.py --help
python inspect_tx_schema.py --help
```

## 3) 典型执行流程

### Step A：构建 episode

默认输入输出路径写在 `build_episodes.py` 里：

- `--crossdata_root` 默认 `/home/chain1/zl/chain/crossdata`
- `--tx_root` 默认 `/home/chain1/zl/chain`
- `--out_root` 默认 `/home/chain1/zl/chain/Heimdall/out/crossdata_with_episodes`

最小示例：

```bash
python src/build_episodes.py \
  --crossdata_root /your/crossdata \
  --tx_root /your/chain_root \
  --out_root /your/out/crossdata_with_episodes \
  --chains eth,arbitrum,optimism
```

如需加速调试，可限制处理规模：

```bash
python src/build_episodes.py \
  --crossdata_root /your/crossdata \
  --tx_root /your/chain_root \
  --out_root /your/out/crossdata_with_episodes \
  --limit_files 2 \
  --limit_rows 10000
```

### Step B：episode 转 ActionTrace

```bash
python src/episode_to_actiontrace.py \
  --episodes_root /your/out/crossdata_with_episodes \
  --tx_root /your/chain_root \
  --out_path /your/out/actiontrace.parquet
```

### Step C：导出 HTML 预览

```bash
python tools/make_episode_viewer.py \
  --input /your/out/crossdata_with_episodes/part-00000.parquet \
  --output /your/out/episode_viewer.html
```

## 4) 数据目录约定

脚本会按下面结构查找链数据（以 `eth` 为例）：

- `/<tx_root>/eth/transactions/*.parquet`
- `/<tx_root>/eth/decoded_events/*.parquet`（若启用事件 enrichment）

## 5) 常见问题

- 报错找不到 `transactions` 目录：确认 `--tx_root/<chain>/transactions` 是否存在。
- 处理很慢：先用 `--limit_files`/`--limit_rows` 小样本验证，再全量跑。
- 没有事件明细：可检查是否有 `decoded_events` 数据，或将 `--enrich_events 1` 打开。
