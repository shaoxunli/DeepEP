# EP128 全 Clos 单次初始化摸测交接

整理日期：2026-11-11。入口：benchmarks/clos/bench_fast_sweep.py。
本文命令在 Linux 执行，接手人不需要 Windows 上的远程控制脚本。

## 1. 分支与范围

- 分支：clos-fast-sweep-ep128。
- 功能提交：9ba6e60aa4088eee956a1b551f7e88d1a58bb8fd。后续交接文档提交不改变实现。
- 脚本、私有 Host 调参入口、QP 资源并集准备均已提交。必须使用此分支的 native 扩展，仅复制脚本到旧安装环境不够。
- 直接导入 deep_ep.backends.clos.elastic.ElasticBuffer，跑的是 POD 上的全 Clos 后端，不是 POD 高吞吐算子。
- 每 rank 初始化一次 Buffer/SHMEM，后续复用工作区及 QP。EP128 是128个 rank各初始化一次，不是整个集群只创建一个 Buffer。
- 不做数值精度校验；成功仅表示性能采样及退出检查通过。
- 不把赢家写入生产默认策略，没有 --replay-best 参数。本次整理不改 kernel 或采样逻辑。

## 2. 场景与搜索范围

| Token/rank | Hidden | TopK | AIV总数 | 类型 | 总专家数 |
|---:|---:|---:|---:|---|---:|
| 8192 | 2816 | 8 | 32 | BF16 | 256 |
| 16384 | 7168 | 6 | 64 | BF16 | 256 |

固定 EP128、16台服务器、每台8卡、连续rank编号。路由为随机选择不同专家，
seed=20260823（每rank用seed+rank）。expand=0、1 GiB大页、不清理L2、无bias、无weight。
combine前置notify开启。每次调用先buffer.barrier(with_cpu_sync=True)，调用后等待事件及设备完成。
只排名cached dispatch和无权重combine；fresh dispatch仅用来生成路由handle。

搜索：
- order：rank、group、group_staggered、group_round_robin、group_window。
- workers/QPs：(1,1)、(1,2)、(2,1)。
- visit：4/8/16/32；双worker配比：1:1～8:1。
- 普通order的peers：1/2/4/8；两种group轮转的peers：1/2/4。
- 按AIV、QP和peer表容量过滤后，EP128每shape为372套配置。
- peers在普通order与group轮转下含义不同；实际payload发送核数看报告的payload_aivs。

每配置粗测2次warmup、8次正式采样，另有保护调用。每shape、每算子取前三名，
正序与反序各复测30次，用两次带宽较低值排名。
每rank共2个shape×2个算子×3个阶段=12份samples.json。
不同算子可选出不同策略，只称本候选集合、本次环境下最优。

与旧Excel区别：此脚本不清理L2、仅BF16/expand=0。目前没有L2、FP8、AIV、token的自由配置开关；
不要传入不存在的--flush-l2-cache等参数。无需为了交接修改这些条件。

## 3. 部署准备

先分配16台完整8卡机器，逐台执行hostname、npu-smi info，确认8卡健康、空闲。
有作业时协调或等待。16台hostname须不同。所有节点能访问node0的同一IP和3个不同端口。
设备映射、hostname解析和网络有问题时先处理环境，不要把初始化故障当带宽结果。

所有节点使用一致的Python、Torch/torch_npu、CANN、驱动/固件与SHMEM。
优先复用已验证的统一wheel/SDK，不要在运行期间替换。
依赖锁指定cann/shmem dev提交836d3fd6f9c2f3b10a6acac36ccfd531b104d4a7。
历史成功EP128运行的libshmem.so SHA256：
4ee4db1dff7d5368d92ac0bcbc4999e05f00b212d7c6de51e43e493d29b641b3。
源码锁与历史二进制哈希是不同证据；重编哈希可能变化，不能仅凭dev分支名声称版本完全相同。

在已取得的交接仓库中，先确认无未提交修改，再执行：
~~~bash
git status --short
git fetch origin clos-fast-sweep-ep128
git switch clos-fast-sweep-ep128
git pull --ff-only origin clos-fast-sweep-ep128
git merge-base --is-ancestor 9ba6e60aa4088eee956a1b551f7e88d1a58bb8fd HEAD
git rev-parse HEAD
~~~

构建细节见docs/build.md。新checkout示例，先激活匹配Torch/torch_npu的Python环境：
~~~bash
source /实际路径/ascend-toolkit/set_env.sh
git submodule update --init --recursive
unset SHMEM_ROOT
python scripts/build_shmem.py --if-needed
export SHMEM_ROOT="$PWD/third-party/shmem/install/shmem"
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DEEPEP_NPU_ARCH=Ascend950
python -m pip install -r requirements-build.txt
python -m pip install -v --force-reinstall --no-build-isolation --no-deps .
~~~
已有验证过的SDK则跳过SHMEM构建，显式设置SHMEM_ROOT。不要覆盖其他作业正在使用的SDK。
分发wheel安装时按构建指南从源码目录外运行，避免旧源码遮蔽安装包；同一份脚本分发到每节点。
在实际启动目录确认包路径：
~~~bash
python - <<'PY'
import importlib.util
import torch, torch_npu, deep_ep
from deep_ep.backends.clos import elastic
print("package:", deep_ep.__file__)
print("elastic:", elastic.__file__)
print("native:", importlib.util.find_spec("deep_ep._clos_C").origin)
print("devices:", torch.npu.device_count())
PY
~~~

## 4. 检查计划（不占NPU、不初始化）

~~~bash
python benchmarks/clos/bench_fast_sweep.py --ep-size 128 --plan-only > plan_ep128.json
python - <<'PY'
import json
for p in json.load(open("plan_ep128.json"))["plans"]:
    print(p["shape"], "configurations:", len(p["configs"]))
    assert len(p["configs"]) == 372
PY
~~~
应有两组：H2816/AIV32和H7168/AIV64，各372套。
可单独用--smoke --samples 3 --final-samples 5 --top 1验证链路，但这是额外一次运行和初始化；
若要求总共只初始化一次，直接执行完整任务。

## 5. 正式启动（每节点执行一次，不套shape或策略循环）

先加载CANN、Python环境。在每台仓库根目录执行下面整段。
NODE_RANK分别填0～15；MASTER_ADDR、端口、RUN_ID必须统一，SHMEM_ROOT为各台同版本SDK路径。
重跑换新RUN_ID，不复用失败目录。可用tmux保留终端会话。

~~~bash
(
set -euo pipefail
EP=128
NNODES=16
NODE_RANK=0
MASTER_ADDR="填写node0可达IP"
MASTER_PORT=29600
CONTROL_PORT=29610
SHMEM_PORT=29620
RUN_ID="clos_fast_ep128_20261111_01"
export SHMEM_ROOT="/实际路径/shmem-sdk"

(( EP == NNODES * 8 && NODE_RANK >= 0 && NODE_RANK < NNODES ))
test -f "$SHMEM_ROOT/lib/libshmem.so"
# 只清掉本shell继承的Clos实验开关，不改系统配置。
while IFS= read -r name; do
    case "$name" in DEEPEP_CLOS_*) unset "$name" ;; esac
done < <(compgen -e)
# 仅在已分配的独占8卡机器上使用。
unset ASCEND_RT_VISIBLE_DEVICES ASCEND_VISIBLE_DEVICES SHMEM_UID_SESSION_ID
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export SHMEM_UDMA_CONNECT_TIMEOUT_MS=600000
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

OUT="$PWD/$RUN_ID/node$NODE_RANK"
test ! -e "$OUT"
mkdir -p "$OUT"
trap 'code=$?; printf "%s\n" "$code" > "$OUT/launcher.exit"' EXIT
git rev-parse HEAD > "$OUT/revision.txt"
hostname > "$OUT/hostname.txt"
timeout 30 npu-smi info > "$OUT/npu_before.txt"
# 所有节点健康/空闲应已人工确认；这里的快照仅用于留档。
python -u -m torch.distributed.run \
    --nnodes "$NNODES" --nproc-per-node 8 --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
    benchmarks/clos/bench_fast_sweep.py \
    --ep-size "$EP" --experts 256 --seed 20260823 \
    --control-port "$CONTROL_PORT" --shmem-port "$SHMEM_PORT" \
    --timeout 1800 --warmup 2 --samples 8 --final-samples 30 --top 3 \
    --output "$OUT" \
    2>&1 | tee "$OUT/run.log"
)
~~~

三个端口在并行任务间也不能冲突。EP64仅改EP=64、NNODES=8、NODE_RANK=0～7，另换RUN_ID；
每shape候选变为492。
控制层使用TCPStore，不创建Gloo进程组，但保留参数/二进制一致性检查。
脚本会比较脚本、elastic.py、Clos native、kernel库、libshmem.so的哈希。

## 6. 进度与失败定位

~~~bash
OUT="$PWD/clos_fast_ep128_20261111_01/node0"
tail -f "$OUT/run.log"
~~~

- BUFFER_INIT：开始一次初始化。QP并集会增加首次成本。
- BUFFER_READY initializations=1：初始化完成，后续策略不再重新初始化。
- PHASE_BEGIN、SWEEP ... i/372、PHASE_DONE：采样及profiler导出。
- WINNER：某shape/算子的赢家，不代表任务结束。
- FAST_SWEEP_COMPLETE：rank0汇总完成，仍需检查所有节点退出码。

没进展先看所有节点最早的报错。采样缺失或kernel序列不符属于无效采样，不要删掉报错样本后宣称通过。
conflicting experimental environment需清理继承开关；哈希不同需统一部署；
SHMEM超时检查节点、设备映射、网络、依赖。
没有launcher.exit可能仍在运行或被强杀；缺rank退出文件不能验收。
失败后保留日志，处理本任务残留进程，换RUN_ID重跑；当前不支持断点续跑。

## 7. 每节点验收和打包

在各节点修改OUT和NODE_RANK后执行。此检查针对默认EP128完整任务：
~~~bash
OUT="$PWD/clos_fast_ep128_20261111_01/node0"
NODE_RANK=0
python - "$OUT" "$NODE_RANK" <<'PY'
import json, sys
from pathlib import Path
p, node = Path(sys.argv[1]), int(sys.argv[2])
assert (p/"launcher.exit").read_text().strip() == "0"
assert {x.name for x in p.glob("rank*") if x.is_dir()} == {
    f"rank{i}" for i in range(node*8, node*8+8)}
for rank in range(node*8, node*8+8):
    d = p/f"rank{rank}"
    e = json.loads((d/"exit.json").read_text())
    assert e["code"] == 0 and e["complete"] and e["buffer_initializations"] == 1
    assert e["precision_checked"] is False
    files = list(d.rglob("samples.json"))
    assert len(files) == 12, (rank, len(files))
    for f in files:
        rows = json.loads(f.read_text())
        coarse = f.parent.name.endswith("_coarse")
        assert len(rows) == (372 if coarse else 3)
        assert all(r["samples"] == (8 if coarse else 30) for r in rows)
        assert all(len(r["communication_samples_us"]) == r["samples"] for r in rows)
    print(rank, "PASS", "initializations=1", "precision_checked=false")
PY
RUN_ID=clos_fast_ep128_20261111_01
ARCHIVE="$RUN_ID-node$NODE_RANK.tar.gz"
tar -C "$PWD/$RUN_ID" -czf "$ARCHIVE" "node$NODE_RANK"
sha256sum "$ARCHIVE"
~~~
最终应128rank全部通过、16个launcher退出码均为0。smoke、EP64、单shape或自定义采样数须调整检查条件。
打包保留原始profiler、samples、identity、plan、exit及日志，不能只下载node0。
压缩包自带EP和node信息，收集到同一目录即可。

## 8. 提取四个赢家

只有node0有顶层report.json，里面已聚合全EP结果：
~~~bash
python - "$PWD/clos_fast_ep128_20261111_01/node0/report.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
assert r["ep"] == 128 and r["buffer_initializations"] == 1
for case in r["results"]:
    for op, result in case["operations"].items():
        w = result["winner"]
        print(case["shape"], op)
        print("config:", w["config"])
        print("score_GBps:", w["score_GBps"])
        for i, x in enumerate(w["repeats"]):
            print("repeat", i, "GB/s", x["GBps"], "comm_us", x["mean_us"],
                  "rank_GBps_range", x["GBps_range"])
PY
~~~

config：order为轮转方式，peers为对应order的peer参数，workers为每peer worker数，
qps为每worker QP数，visit为每次访问WQE数，weight为双worker的weight:1配比。
分别交付两个shape的cached dispatch、无权重combine，共四个赢家；保留finalists供波动比较。

十进制带宽：
GBps = Σrank(去重逻辑payload字节数，含self) / Σrank(该rank通信kernel采样均值，μs) / 1000。

外层barrier、notify、最终reduce不进分母。它不是tokens×hidden×topk×2/time的算子带宽，
也不是纯跨server链路带宽。GBps_range为各rank均值带宽范围；repeats为两次正反序复测。
无权重combine的reduce另记，当前不测独立aggregate/expand。
本脚本成功不等于精度通过。

## 9. 已有验证与边界

功能提交已有EP64、EP128全量记录。历史EP128核对了128rank各一次初始化、各12份采样，
16个launcher退出码均为0，未做精度校验。
历史带宽不作为新环境验收阈值，本轮应交付当前环境的四个赢家及全量证据。
文档整理不代表再次执行了集群测试。加入FP8、L2清理或改变shape需要另行修改、验证，
不能沿用旧报告的测试条件。
