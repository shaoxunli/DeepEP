# SHMEM UDMA 环境对照

`diagnose.py` 直接运行 CANN SHMEM 的 `udma_perftest`，不导入 DeepEP、Torch，也不建立 Gloo 进程组。
用途是对照两组机器的底层 PUT 吞吐，分辨单链路、初始化规模和并发流量的影响。
脚本本身只需要 Python 3.11+；设备测试需要 Ascend950、可用的 CANN/SHMEM 环境和已编译的样例。

## 准备样例

两台使用同一份 SHMEM 源码和编译产物。已有对应二进制时不必重编；缺少时在 **SHMEM 仓库**执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build.sh -examples -soc_type Ascend950
```

默认二进制为 `SHMEM_ROOT/build/bin/udma_perftest`，库目录为 `SHMEM_ROOT/build/lib`。
可用 `--binary` 和 `--library-dir` 指定实际位置。`--shmem-root` 仍需指向编译该样例的源码。
如果编译配置关闭了设备周期 profiling，CSV 会缺失/无效，脚本会报错而不会填零。

**旧样例限制：** DeepEP 当前锁定的 SHMEM `836d3fd6` 中，PE>2 的单向 PUT 错把所有非发送 PE
都当作接收方；环形 PUT 的接收校验使用了下一 PE，而实际数据来自上一 PE。EP2 时这两个问题
不暴露。脚本对 `pool/ring` 检查对应校验表达式，未识别为修正版本时停止。先跑默认 EP2 对照，
不要为了跑 EP32 删除精度检查。这里不自动升级 SHMEM，也不修改生产 kernel。
源码检查不能证明二进制由该源码构建；必须提供配套产物。报告保存源码、二进制及动态库指纹供核对。

为锁定版提供了 [host 校验补丁](perftest-host-verification.patch)，只修正 PUT 接收方和环形 PUT
来源的期望值，不修改 kernel、提交、quiet 或计时。需要 EP32 时，两台在同版本 SHMEM 源码上应用并重编样例：

```bash
# DEEPEP_ROOT 指向包含本脚本的 DeepEP 仓库；SHMEM_ROOT 指向样例源码
PATCH="$DEEPEP_ROOT/benchmarks/shmem_udma/perftest-host-verification.patch"
git -C "$SHMEM_ROOT" apply --check "$PATCH" && \
git -C "$SHMEM_ROOT" apply "$PATCH"
```

然后在 SHMEM 目录执行上面的构建命令。补丁未通过 `--check` 时不要强制应用：可能已修正，或源码
版本不同，先核对。补丁仅覆盖本脚本使用的 PUT/BI_PUT，不声明修复旧样例的其他模式。

## 第一轮：EP2 单卡对

以下命令在两台机器分别执行，**仅修改 `NODE_RANK`**。两台的 `RUN_ID`、地址、设备列表及测试参数必须一致。
`HOST0/HOST1` 是双方可达的 bootstrap 地址，不是设备 IP；不用 SSH 互信。
两台均需开放对应端口范围，并在独占窗口运行。不要在同一批设备上重叠启动多组测试。

```bash
NODE_RANK=0                       # 另一台为 1
HOST0="填写第一台机器IP"
HOST1="填写第二台机器IP"
SHMEM_ROOT="/path/to/shmem"       # 改为现场已编译的 SHMEM 源码目录
RUN_ID="shmem_udma_pair_01"       # 两台相同，重跑换名字

python3 -u benchmarks/shmem_udma/diagnose.py run \
  --shmem-root "$SHMEM_ROOT" \
  --hosts "$HOST0" "$HOST1" --node-rank "$NODE_RANK" \
  --devices0 0 --devices1 0 \
  --scenarios pair pair-bidir \
  --qps 1 --batches 0 --exponents 12 24 \
  --loops 1000 --repeats 3 --port 29100 --timeout 600 \
  --run-id "$RUN_ID" --output "${RUN_ID}_ep2_node${NODE_RANK}"
```

这会测 4 KiB、8 KiB……16 MiB，每 QP 一份数据；每个方向/模式重复 3 次，共 12 个用例。
每个用例有一次 SHMEM 初始化，13 种消息大小在该次初始化内完成。样例内置 warmup，不由本脚本更改。
正反方向由 PE→机器映射交换实现；bootstrap 始终指向实际 PE0 所在机器。
`pair-bidir` 两侧同时发送，每个方向的指标通过交换 PE0 分两次运行采集，并非同一次采样的两侧总和。

先固定公共机器为 node0，对不同对端分别跑一遍；再换 `--devices0/--devices1` 检查可疑卡。
报告中的设备号是 ACL **逻辑设备号**，保留可见设备环境变量和 `npu-smi` 输出，不擅自推断物理卡号。
首次可加 `--dry-run`，只打印计划和卡号映射，不加载库、不初始化设备。

## 第二轮：EP32 初始化规模与跨机并发

使用校验已修正的样例后，将上面命令替换为：

```bash
  --devices0 {0..15} --devices1 {0..15} \
  --scenarios pair pool ring
```

同时把 `RUN_ID` 和输出目录换成新的名字，例如 `shmem_udma_ep32_01_node0`。
两节点各 16 卡时：

| 场景 | 初始化规模 | 实际通信 |
| --- | --- | --- |
| `pair` | EP2 | 列表中第一张卡之间单向 PUT，正反方向分别跑 |
| `pair-bidir` | EP2 | 第一张卡之间双向 PUT |
| `pool` | EP32 | 32 卡全部初始化，只有 PE0→PE1 发送；其余 PE 参与样例同步 |
| `ring` | EP32 | PE0→PE1→…→PE31→PE0；交错映射使每条边都跨机 |

`ring` 是跨机环形并发，**不是 all-to-all，也不是 16 对独立双向卡对**。
交换两个节点在环中的位置，同时把被观察的 PE0 从 node0 换到 node1。
`pool` 只改变初始化规模、保持活跃卡对不变，正是判断“EP2 正常但 EP32 资源规模下异常”的对照。
把两边设备列表中想测的卡放第一位，即可在完整会话中换活跃卡对。

默认 QP=1、batch=0，先不要扩参数。需要进一步分辨并发和提交策略时，再加 `--qps 1 2 4 --batches 0 1 32`。
QP 数同时是样例 AIV block 数；每 QP 消息大小固定，因此增加 QP 也增加总流量。
batch=0 是普通 NBI 连续下发后 quiet；batch=1 每次下发后 quiet；batch>1 使用 defer/submit 分批后 quiet。
shared jetty 保持样例默认关闭，不能把这些设置直接当成 Clos 的最优策略。

## 收集、合并和快慢机组对比

每台自动生成 `<output>.tar.gz`，包含 `plan.json`、`report.json`、逐用例命令、逐 rank 日志和原始 CSV。
失败也打包已完成内容。超时只终止该脚本启动的本机进程组；另一台在自己的超时内退出。
本机完成只打印 `LOCAL_PASSED`，不能据此认定全组成功。

将两个压缩包下载解压后，在任意有 Python 3.11+ 的机器上合并：

```bash
python3 benchmarks/shmem_udma/diagnose.py merge \
  shmem_udma_pair_01_ep2_node0/report.json \
  shmem_udma_pair_01_ep2_node1/report.json \
  --output pair_group_a
```

只有两边计划、可执行文件和 SHMEM 库指纹一致，所有 rank 正常结束、每个消息大小的数据校验成功且 CSV
完整时，才输出 `GLOBAL_PASSED`、合并 `report.json` 和 `summary.csv`。
`ldd` 是启动时解析结果；若环境使用运行时加载的驱动库，还需要结合现场驱动信息分析。

用同配置测第二组机器并合并到 `pair_group_b`，然后：

```bash
python3 benchmarks/shmem_udma/diagnose.py compare \
  pair_group_a/report.json pair_group_b/report.json --output pair_comparison
```

`comparison.csv` 按场景、方向、QP、batch 和消息大小配对，给出各组带宽中位数、最小/最大值、
重复间 CV、耗时中位数及百分比变化。不剔除尖峰。保留 node0/node1 角色一致再比较；
例如让同一台公共机器一直是 node0。CSV 可直接用 Excel 打开。

## 计时和定位边界

使用 SHMEM 样例 `--metric bw` 的设备周期数据：

```text
t_us = max(PE0 各活跃 block 的测量窗口时间) / loop_count
GB/s = (每 QP 字节数 × QP 数) / (t_us × 1000)
```

窗口包含提交与 quiet 完成等待；不含 SHMEM 初始化、host 校验和 kernel 尾部全 PE 同步。
这是 **PE0 在该次运行的实际传输带宽**，不含 self，不是算子逻辑带宽，也不是所有 rank 的平均值或总和。
样例 CSV 的 `Npus=1` 来自单进程设备映射参数；真实会话规模看报告的 `pes`，不能用 `Npus` 推断 EP。

- EP2 已慢：继续定位单链路、卡号、方向及基础库/驱动差异。
- EP2 正常、`pool` 慢：关注大规模资源初始化、QP/peer 映射及共享资源影响。
- `pool` 正常、`ring` 慢：关注跨机并发争用和流量分布。
- 都正常、Clos all-to-all 仍慢：不能排除环境；需要继续看 all-to-all 并发模式、完成标志和等待阶段。

这些是缩小范围的判断，不足以单独认定坏卡、交换机故障或某台发送慢。环形测试没有覆盖全 peer 全互发。

## 本地验证

```bash
python -m pytest tests/test_shmem_udma_diagnostics.py -q
```

CPU 测试覆盖计划/卡号映射、旧样例保护、CSV 公式与完整性、双节点结果合并、子进程失败和超时清理。
CPU 测试不产生或证明 NPU 性能结果。
