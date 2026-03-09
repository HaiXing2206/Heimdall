# Heimdall：构建 Episode 使用说明

这份文档只关注一件事：**运行 `src/build_episodes.py` 构建 episode**。

## 1) 先安装依赖

建议 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas numpy pyarrow tqdm
```

## 2) 查看可用参数

```bash
python src/build_episodes.py --help
```

## 3) 你的 crossdata 目录示例

你的数据文件看起来是按链分别放在目录里，文件名按日期切分，例如：

```bash
~/zl/chain/crossdata/arbitrum/
  2025-09-01_arbitrum.parquet
  2025-09-02_arbitrum.parquet
  ...
  2025-11-06_arbitrum.parquet
```

这类目录结构可以直接用于 `--crossdata_root`。

## 4) 直接运行构建 Episode

> 下面给的是**显式传参**写法，不依赖默认值。

```bash
python src/build_episodes.py \
  --crossdata_root /home/chain1/zl/chain/crossdata \
  --tx_root /home/chain1/zl/chain \
  --out_root /home/chain1/zl/chain/Heimdall/out/crossdata_with_episodes \
  --chains arbitrum
```

如果你想一次跑多条链：

```bash
python src/build_episodes.py \
  --crossdata_root /home/chain1/zl/chain/crossdata \
  --tx_root /home/chain1/zl/chain \
  --out_root /home/chain1/zl/chain/Heimdall/out/crossdata_with_episodes \
  --chains eth,arbitrum,optimism
```

## 5) 先小规模试跑（推荐）

先用小规模验证路径和字段是否正确，再跑全量：

```bash
python src/build_episodes.py \
  --crossdata_root /home/chain1/zl/chain/crossdata \
  --tx_root /home/chain1/zl/chain \
  --out_root /home/chain1/zl/chain/Heimdall/out/crossdata_with_episodes_debug \
  --chains arbitrum \
  --limit_files 2 \
  --limit_rows 10000
```

## 6) 输入/输出目录关系（构建 episode 必需）

对 `arbitrum` 来说，通常会读取：

- Crossdata 输入：`/home/chain1/zl/chain/crossdata/arbitrum/*.parquet`
- 交易输入：`/home/chain1/zl/chain/arbitrum/transactions/*.parquet`
- 事件输入（可选 enrichment）：`/home/chain1/zl/chain/arbitrum/decoded_events/*.parquet`
- 输出目录：`/home/chain1/zl/chain/Heimdall/out/crossdata_with_episodes/...`

## 7) 常见问题

- `ModuleNotFoundError: No module named 'numpy'`：说明依赖没装全，重新执行第 1 步。
- 报错找不到 `transactions`：检查 `--tx_root/<chain>/transactions` 是否存在。
- 运行慢：先用 `--limit_files` / `--limit_rows` 做小样本验证，再全量跑。
