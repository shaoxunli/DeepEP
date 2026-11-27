# Legacy Buffer Python API 统计基准

衡量 statistics 装饰器在真实 NPU 上的开销：**同一份包**跑两遍，奇数轮 `TRMT_LOG_ENABLE=false`（装饰器为恒等函数，等价于未接入统计），偶数轮 `TRMT_LOG_ENABLE=true`。不需要第二个仓库。

## 前置条件

- Ascend NPU ≥ 4 张（默认 `CARDS=0,1,2,3`）
- 仓库根目录已构建 native 扩展（`deep_ep/_C*.so`、`deep_ep/_clos_C*.so`）：

```bash
python3 -m pip install -v --force-reinstall --no-build-isolation --no-deps .
```

- 已安装 `torch`、`torch_npu`

## 运行

```bash
# 端到端（真实 NPU）：交替 off/on，默认 10 轮
CARDS=0,1,2,3 ROUNDS=20 bash api_statistics_benchmark/run.sh

# 纯 CPU：只看装饰器本身的固定开销
python3 api_statistics_benchmark/stat_overhead_bench.py 20000 15
```

产物：

- `api_statistics_benchmark/api_overhead.md`：各 rank/API 的均值、P90、`on-off` 增量与百分比；
- `api_statistics_benchmark/artifacts/log/buffer.log`：统计组件自身落盘的日志（验证用，可随时删除）。

## 文件

| 文件 | 说明 |
| --- | --- |
| `run.sh` | 交替 off/on 采集，内部用 `torchrun` 拉起 `benchmark.py`，最后生成报告 |
| `benchmark.py` | 单轮采集：`get_dispatch_layout` / fresh dispatch / cached dispatch / combine，预热 10 次、采集 1000 次 |
| `stat_overhead_bench.py` | CPU 微基准：空函数 + 约 1.5 µs 方法体，对比 baseline / off / on |
| `api_overhead.md` | 生成物 |
| `artifacts/` | 生成物（统计日志） |

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CARDS` | `0,1,2,3` | `ASCEND_RT_VISIBLE_DEVICES` |
| `ROUNDS` | `10` | 交替轮数（off/on 各一半） |
| `NPROC` | `4` | `torchrun --nproc-per-node` |
| `DEEPEP_PACKAGE` | `<repo>/deep_ep` | 提供 `.so` 的包目录 |
| `TRMT_LOG_PATH` | `<本目录>/artifacts/log` | 统计日志目录 |
| `TRMT_LOG_INTERVAL` | `10` | 统计汇总周期（秒） |
| `EXPECT_COMMIT` | 空 | 设置后校验当前仓库 short commit |

可选：需要和另一份代码对比时，把那份代码 checkout 到别处，用 `DEEPEP_PACKAGE=/path/to/deep_ep` 指向它即可，脚本本身不依赖固定路径。

## 测量口径

- 计时区间 = 公开 Python API 两端（含参数校验、stream 依赖、native 提交、事件构造），不含异步 kernel 执行与调用方等待；
- `off` 轮 `record_api` 返回原函数，运行期零分支；`on-off` 的差即装饰器开销；
- `run.sh` 使用 `mktemp` 创建临时包目录并在退出时清理，只有报告与统计日志留在本目录。
