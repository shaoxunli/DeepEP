# KernelTrace：notify、dispatch 与 combine 采集

Ascend950 Global Clos 的生产采集器默认关闭。在创建 Buffer 前，给同一 EP 组的所有 rank 设置 `DEEPEP_TRACE_RECORDER_ENABLE=1` 可以开启采集。`0` 或未设置时关闭。采集范围是 V1 `Buffer` 和 V2 `ElasticBuffer` 的 eager 调用实际启动的下列 UDMA kernel。

| op | kernel_name | 范围 | 生产日志格式 |
| --- | --- | --- | --- |
| 0 | notify_dispatch_kernel | fresh dispatch 的计数、元数据及通知阶段 | 短 |
| 1 | cached_notify_dispatch_kernel | cached dispatch 的通知同步 | 短 |
| 2 | v1_dispatch_udma_kernel | fresh / cached dispatch 通信及核内输出处理 | 长 |
| 3 | cached_notify_combine_kernel | 仅在执行路径实际启动独立 notify 时记录 | 短 |
| 4 | v1_combine_udma_kernel | 融合通信和归约 | 长 |
| 5 | v1_combine_udma_comm_kernel | 独立通信阶段 | 长 |
| 6 | v1_combine_udma_reduce_kernel | 独立归约阶段 | 短 |
| 7 | v1_combine_udma_early_kernel | 提前发送阶段 | 长 |
| 8 | v1_combine_udma_remaining_kernel | 剩余阶段 | 长 |
| 9 | v1_combine_udma_pipeline_kernel | 流水阶段 | 长 |

普通 fused/communication combine 的 Begin 同步在通信 kernel 内，其耗时包含在 combine 记录中，不额外生成 notify 记录。采集不改变策略，也不额外启动通知 kernel。layout、expand、aggregate、缓存计划重建、显式 Buffer barrier 和 NPU Graph 捕获/重放不采集。图捕获时不预留槽位，避免重放旧序号。

V1 fresh dispatch 记录 op 0 和 2，cached dispatch 记录 op 1 和 2，普通 combine 记录 op 4。V1 benchmark 在 combine 的 profiler 阶段启用独立 notify，因此这些调用还会记录 op 3。

## 运行、轮转与归档

按[构建指南](build.md)在含匹配 CANN、Torch、torch_npu 和 SHMEM SDK 的环境安装本分支的 wheel。Host 与 Device 两部分均需重新构建。在单机四卡上可使用现有 benchmark 验证。

```bash
set -e
RUN="$(mktemp -d "$PWD/kernel-trace-XXXXXX")"
export DEEPEP_TRACE_RECORDER_ENABLE=1
# 可选：在创建 Buffer 前指定本机输出目录和时间参数
# export DEEPEP_TRACE_DIR=/path/to/deepep-logs
# export DEEPEP_TRACE_POLL_INTERVAL_MS=2
# export DEEPEP_TRACE_TIMEOUT_MS=10000
python -u benchmarks/clos/bench_v1_bandwidth.py \
    --ep-size 4 --nnodes 1 --aiv 32 --dtypes bf16 --expand 0 \
    --num-tokens 2048 --hidden 6144 --num-topk 8 --experts-per-rank 32 \
    --warmup 2 --iterations 5 --no-flush-l2-cache --output "$RUN/bench"
ls -lh "${DEEPEP_TRACE_DIR:-/dockerdata/.trmt/deepep}"/trmt-deepep-trace-*.txt
```

验证 V2 时使用同样的参数运行 `benchmarks/clos/bench_v2_bandwidth.py`。

生产日志写入 `DEEPEP_TRACE_DIR` 指定的本机可写目录，未设置或为空时使用 `/dockerdata/.trmt/deepep/`。请在创建 Buffer 前设置目录，可以使用相对路径或指向目标目录的符号链接。采集开关仍由 `DEEPEP_TRACE_RECORDER_ENABLE` 控制。文件前缀、容量和文件数固定，文件名为 `trmt-deepep-trace-<commId>-<busId>-<序号>.txt`，其中：

- `commId` 是随机无符号 64 位通信组 ID 的十进制表示，文件名中不补前导零。EP rank 0 在 Buffer 初始化时调用 `secrets.randbits(64)` 生成该 ID，并通过 all-gather 同步。日志中的 `comm_id` 为同一 ID 的 JSON 整数。
- `busId` 是当前主机的 CANN 物理设备编号，和日志中的整数 `bus_id` 一致，各连接类型都直接使用该值。采集器通过 `aclrtGetPhyDevIdByUserDevId` 将 ACL 用户设备编号转换为物理设备编号。运行时缺少该符号时，使用 `aclrtGetPhyDevIdByLogicDevId`。例如 ACL 用户设备 `0` 映射到物理设备 `3` 时，日志记录 `bus_id=3`。同一通信组内，如果两个 rank 的 `busId` 相同，应使用不同的输出目录以区分文件名。无法取得物理设备编号或 CANN 设备 UUID 时，采集器会报告警告并跳过该设备。[CANN 设备标识接口](https://github.com/hicann/runtime/blob/master/docs/zh/api_ref/04_device_management.md#aclrtgetphydevidbyuserdevid)
- 序号为 `0–7`，同一 `(commId, busId)` 固定有八个文件。单文件上限为 1 GB（1,000,000,000 字节），按 UTF-8 整行统计，包含换行符。下一行将超限时，先切换到 `(i+1) % 8` 并截断目标文件，再写入该行。每组每卡的日志内容合计最多 8 GB；单组八卡最多 64 GB。

**启动时，采集器按 CANN 设备 UUID 和物理设备编号获取设备锁。** 采集器持锁扫描所选目录的 `owner-v2` 归属标记，清理设备 UUID、物理设备编号、`busId` 均匹配的旧通信组普通日志文件及归属标记。该规则也适用于通信组 ID 为 32 个小写十六进制字符的 `owner-v2` 标记。历史 `busId` 与当前物理设备编号不同的文件、旧版归属标记及对应日志、无归属日志均保留。锁文件可用只读方式打开，采集器由持锁进程正常销毁时会显式解锁。需要保留的日志请在同一设备重跑前移出该目录。

## 轮询和超时配置

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `DEEPEP_TRACE_POLL_INTERVAL_MS` | 2 ms | 生产后台线程在每次扫描前的等待时间。收集、写入和线程调度也会影响相邻扫描的实际间隔。 |
| `DEEPEP_TRACE_TIMEOUT_MS` | 10000 ms（10 秒） | 从 Host 成功提交开始计时。达到阈值后，后台扫描仍未收齐完成序号时才报告 `type=1`。 |

两个值独立设置，均填写 1～2,147,483,647 的十进制整数，单位为毫秒。未设置或为空时使用默认值。请在启动训练、创建 Buffer 前设置，修改后需重新启动。无效值会发出警告并关闭该设备在本进程中的生产采集，训练继续运行。开发诊断保留原有设置。

## 生产记录字段

文件扩展名为 `.txt`，内容是 UTF-8 JSONL。每行对应一次采样后的完成或超时观测。记录格式由 `op` 决定，同一 kernel 的 `type=1` 和 `type=2` 具有相同的字段集合。

- 长格式（`op=2,4,5,7,8,9`）输出下表全部 23 个字段。
- 短格式（`op=0,1,3,6`）省略 `ep_size`、`hidden_size`、`num_experts`、`num_tokens`、`topk` 这 5 个键，输出其余 18 个字段。

| 字段 | 含义 |
| --- | --- |
| `type` | `2` 表示所有参与 AIV 的完成序号已匹配。`1` 表示 Host 成功提交后达到配置的超时阈值，轮询时仍未收齐完成序号。每个 `gid` 至多报告一次超时；之后完成仍可输出同一 `gid` 的 `type=2`。Host 提交失败与未观察到设备完成分别处理。 |
| `gid`、`op`、`op_count` | Host 为有效票据分配的进程内序号（从 1 开始）、上表固定 kernel 编号，以及本 rank 在当前通信组中同名 kernel 的启动计数（从 0 开始）。 |
| `dispatch_count` | Python dispatch 调用计数，从 0 开始。fresh 与 cached 各按一次调用计数，同一次调用的 notify 与 dispatch 保持一致。combine 使用 handle 关联的原始 fresh dispatch 计数；不可取得时为 `null`。 |
| `bus_id`、`rank`、`nvl_rank` | CANN 物理设备编号、EP 组内 rank、按约定计算的 `rank % 16`。 |
| `num_sms` | 实际参与的 AIV 数。 |
| `ep_size`、`hidden_size`、`num_experts`、`num_tokens`、`topk` | 仅长格式。分别为 EP 组大小、API 输入特征维度、专家数、原始路由 tensor 行数及 top-k。缺少相应元数据时字段值可为 `null`。 |
| `pid`、`stream`、`comm_id` | 进程号、启动时的本地 ACL 流句柄（无符号整数）、随机无符号 64 位通信组 ID（JSON 整数）。 |
| `launch_time`、`start_time`、`end_time` | 提交前的 Host 时间、校准后的最早设备开始和最晚设备结束时间。采用 UTC+8，格式为 `YYYY-MM-DD HH:MM:SS.ffffff`。 |
| `elapsed_ns`、`start_tick`、`end_tick` | 设备起止 tick 以及两者的差值，Ascend950 的 1 GHz tick 对应纳秒。 |

超时只有 Host 已成功提交这一事实。此时 `start_time`、`end_time`、`elapsed_ns`、`start_tick` 和 `end_tick` 都是 `null`，槽位继续等待设备写入。超时由达到阈值后的后台扫描观察，业务线程无需等待。生产文件只包含 `type=1/2`，时钟校准仍用于内部换算。

长格式完成记录（`op=2`）：

```json
{"type":2,"gid":1,"op":2,"op_count":0,"dispatch_count":0,"ep_size":4,"bus_id":3,"rank":0,"nvl_rank":0,"hidden_size":6144,"num_experts":256,"num_sms":32,"num_tokens":2048,"topk":8,"pid":1864,"stream":187651357453200,"comm_id":16619284372657870125,"launch_time":"2026-11-12 15:29:04.967346","start_time":"2026-11-12 15:29:04.967558","end_time":"2026-11-12 15:29:04.967564","elapsed_ns":6000,"start_tick":120000,"end_tick":126000}
```

短格式超时记录（`op=0`）：

```json
{"type":1,"gid":2,"op":0,"op_count":0,"dispatch_count":1,"bus_id":3,"rank":0,"nvl_rank":0,"num_sms":32,"pid":1864,"stream":187651357453200,"comm_id":16619284372657870125,"launch_time":"2026-11-12 15:29:05.010000","start_time":null,"end_time":null,"elapsed_ns":null,"start_tick":null,"end_tick":null}
```

## 时间含义

| 记录 | 来源及含义 |
| --- | --- |
| 校准 | 首个启用采集的 UDMA Buffer 在当前 ACL context 创建采集器，短 kernel 读取设备 tick。Host 使用私有流包围这次采样，同步一次，保存主机单调时钟窗口的宽度和墙钟偏移。共享该采集器的 Buffer 共用同一偏移。 |
| Host 提交 | Host 在调用相应 kernel 启动封装前读取 Unix 墙钟，单位 ns。封装返回后把时间与当前线程的 ACL 状态交给日志线程。 |
| Device 完成 | 每个参与 AIV 在本核实现的入口和尾部读取 tick。`start_tick` 取各核最小值，`end_tick` 取各核最大值，覆盖本 rank 参与核的执行区间。 |

Ascend950 按 1 GHz 换算，令校准核采样为 $C_0$，Host 单调时钟包围采样的中点为 $M_0$，另一次相邻采样所得墙钟和单调时钟为 $W_a,M_a$，所有 Host 时间使用 ns。

$$
O=(W_a-M_a)+M_0-C_0,\qquad W(C)\approx C+O
$$

`start_ns` 和 `end_ns` 为上述参考墙钟。`elapsed_ns = end_tick - start_tick` 独立于墙钟偏移。跨 AIV 取极值的前提是各核 GetSystemCycle 共享计时基准；设备验收需与独立 profiler 对照。校准窗口宽度与设备速率漂移都会影响跨时钟比较，launch 到 start 的差值用于参考，不能作为精确排队时延。

## 上报和资源

Python KernelTrace 在每个进程内共享一个后台线程，管理各活跃 ACL context 的原生采集器。每轮等待后，线程调用 `KernelTrace::DrainProductionBatches()`，在释放 GIL 的情况下读取并回收槽位、格式化时间并按通信组编码 JSONL。Python 接收每个通信组的一批字节，批量写入文件。当前文件放不下整批时，按完整行切分，再轮转写入剩余日志。

生产日志写入失败后最多重试 3 次，每次重新写入前等待 3 ms。短写时从剩余位置继续，写完一行后下一行恢复重试额度。重试耗尽后，采集器保留已写完的行，尝试截掉未写完的半行并停止写入该通信组，其他通信组继续写入。

正式采集器固定分配 128 槽的 Host 映射内存，每槽预留 64 个 AIV，每个 AIV 占用 256 字节，总计约 2 MiB。核内起始 tick 保存在局部变量。结束时 dispatch/notify 复用已释放的 UB，combine/cached notify 在工作结束后使用 32 字节 UB，将 start_tick、end_tick 写入偏移 0，等待 MTE3_S 后，在偏移 128 发布 done_seq。第二次写出也用 MTE3_S 保护 UB 生命周期。combine/cached notify 在读取结束 tick 前用本核 V_S、MTE3_S 将流水完成推进到标量侧；不增加全卡或全核同步。每个 AIV 独占自己的区域。

Host 为成功预留的槽分配从 1 开始的进程内序号，kernel 按此序号发布。只有全部参与核的 done_seq 等于本轮序号才消费数据。回收后保留旧 tick 和序号，下一轮靠新序号识别数据。占用的槽会导致当前采集跳过并记录 dropped。提交异常的槽隔离到销毁。

最后一个共享采集器的 Python Buffer 释放时同步设备，排空记录并关闭日志。正常退出也会停止线程并排空记录。强制退出可能丢失末尾数据。Python 调度与 GIL 会影响日志延迟和回收速度，开销及丢样率需实测。

notify、dispatch、combine 的每次启动分别预留 ticket，独立发布 done_seq；不同阶段可以交错完成，以同一通信组、设备上的 `gid` 关联超时和完成记录。分离 combine 的阶段耗时不可直接当作完整 Python API 耗时。early 阶段只表示该 kernel 的提交工作结束，异步 UDMA 传输仍可能在进行，其完成等待由后续阶段负责。


## 开发诊断与验收

普通 `--npu` 开关对照使用上文的生产写入器，验收 `type=1/2` 格式的 `.txt` 轮转文件。带 `--diagnostic` 时，worker 使用单次试验的独立目录输出开发产物，保留 `kernel-*.jsonl` 的 `calibration`、`launch`、`end` 和 `summary` 等事件，供 `scripts/check_kernel_trace.py` 校验。

### Host 元数据与调用关联

每条启用 V2 元数据的 launch/end 记录增加 metadata_version=1。元数据在 Python API 的 Host 路径构造，C++ 在 Reserve 时为对应槽保留不可变快照；设备仍只写起止 tick 和 done_seq，Host 映射区布局和 kernel 参数均未因本次元数据扩展而改变。后台线程调用原生侧合并记录并编码，不从 NPU tensor 回读参数。

| 字段 | 含义与来源 |
| --- | --- |
| ep_size | 当前 EP 组大小 |
| num_tokens | 原始路由 tensor 的行数；不是有效通信行数或 padding 后实际有效条数 |
| hidden_size / element_bytes / dtype | API 输入特征维度（元素数）、元素字节数、数据类型；FP8 dispatch 使用 x 而非 scales |
| num_experts / topk | 模型路由专家总数、路由 tensor 列数；cached dispatch/combine 从 handle 取，combine 不带 weight 时仍保留真实 top-k |
| api_input_rows | 当前 API 输入 tensor 的行数；展开后的 combine 可以不同于 num_tokens |
| kernel_input_rows / kernel_output_rows | dispatch/combine 启动点实际传入的输入/输出 tensor 行数；notify 不附加这两个字段。固定容量输出记录的是分配行数，不冒充有效接收行数 |
| cached / do_expand | 是否 cached dispatch、是否使用展开模式，0/1 |
| token_capacity | Buffer/handle 的每 rank token 容量，与实际 token 数分开 |
| comm_id | 每次 Buffer 初始化，由 EP rank 0 生成并发给组内所有 rank 的随机无符号 64 位整数。诊断记录用十进制字符串表示，同组此次 Buffer 实例使用同一 ID |
| buffer_id | 本进程内的 Python Buffer 标识；即使共享原生采集器也独立 |
| process_id / hostname / pid | 进程生命周期 UUID、主机名、操作系统 PID；PID 可复用，因此跨文件身份使用 process_id |
| rank / global_rank | EP 组内 rank；global_rank 仅在已初始化的 PyTorch distributed 环境中获取，其他环境省略，不用组内 rank 冒充 |
| device | ACL 用户设备编号 |
| stream | 实际 kernel 启动点的 ACL stream 句柄，以十进制字符串保存，避免无符号地址精度或溢出问题；只在本进程/设备生命周期内解释 |
| aiv | 本次实际启动核数；combine 通信与归约允许不同核数 |
| api_name / api_call_id | dispatch 或 combine，以及本进程内独立的调用编号；一次 API 内 notify/通信/归约共用该编号 |
| handle_id | 路由 handle 身份；fresh dispatch 预留并随返回 handle 保存，cached dispatch/combine 复用；与执行轮次不同 |
| gid | 原有的每 kernel 事件序号；一次 API 可以对应多个 gid |

身份作用域：事件用 (process_id, gid)，API 用 (process_id, api_call_id)，handle 用 (process_id, handle_id) 关联。各 rank 的 api_call_id/handle_id 不保证一致，不能直接据此跨 rank 配对。comm_id 提供通信组关联，但没有宣称增加了全组执行轮次编号。若 handle 没有历史采集标识，首次被采集使用时分配 handle_id，不虚构其创建记录。

同一 handle 被 cached dispatch 重复使用时，handle_id 保持不变，每次生成新 api_call_id。例：

- fresh dispatch：api_call_id=A，handle_id=H；notify 与 dispatch 各有独立 gid。
- cached dispatch：api_call_id=B，handle_id=H；cached notify 与 dispatch 关联到 B。
- combine：api_call_id=C，handle_id=H；通信及延后执行的 reduce 都关联到 C。

C++ 元数据上下文采用线程局部存储和作用域恢复，异常退出也恢复上一层。Buffer 固定字段只转换一次，按 shape、dtype、API、模式等缓存最多 64 份不可变模板。每次 API 创建只含两个 ID 和模板引用的小快照，使用原生句柄恢复上下文；字段在后台展开，Reserve 只保存共享引用和本次启动的 stream/行数等标量，不逐 kernel 重复复制整张元数据表，也不保留 Python 可变字典的引用。combine continuation 另保留调用时快照，执行时恢复该快照，避免其他调用覆盖。

初始化增加一次 Host 元信息 all-gather，用于检查各 rank 的采集开关一致性并分发 comm_id；UDMA 路径即使关闭采集也参加此一致性检查。业务调用不新增集合通信或设备同步。tensor.size、dtype 和 element_size 只读取 Host 元信息。元数据构造、复制与 JSONL 增量仍有 Host 开销，不能仅据此承诺小于 1%。

只用初始化时的配置决定是否启用采集。多个 Buffer 可以共享日志线程和原生槽池，但各自的 comm_id/buffer_id 随调用快照保存。Graph 捕获/重放仍不产生 kernel trace，调用编号因此可能有空缺。

调用元数据来自 V1 和 V2 Python API。V1 保持六元素 handle 接口，Buffer 返回的 handle 保存可用的路由标量和原始 fresh `dispatch_count`。省略路由张量或传入普通六元组时，不能取得的字段按现有规则写为 `null`。开发事件中的 `device` 为 ACL 用户设备编号。采集器查询物理设备编号并将其写为生产记录的 `bus_id`。

开发产物可使用 `scripts/check_kernel_trace.py` 的 `--require-metadata` 选项校验字段完整性、launch/end 元数据相等、同一 API 的元数据一致性及 handle 路由信息稳定性。旧版 `kernel-*.jsonl` 无需该选项仍可校验。远端验证应覆盖 fresh/cached/无 weight 与带 weight 的 combine、expand=0/1、超过 128 槽的复用、不同通信/归约 AIV 数，以及多个 Buffer 的身份隔离。必须重新构建 Host 扩展；性能开销需要相同输入下 trace 开关对照。

### 生产日志与开销验收用例

`tests/runtime/test_kernel_trace.py` 同时提供 CPU 回归和可选择运行的 NPU 验收入口，不需要新建 benchmark 脚本。

CPU 回归检查元数据关联、错误字段、遗漏 API、槽位代际协议，以及开销判定的通过、失败和不确定分支：

```bash
python -m pytest tests/runtime/test_kernel_trace.py -q
```

在已安装本分支 wheel、Torch、torch_npu、pytest 的 Linux 单机 UDMA 环境，空闲的四张 NPU 上执行：

```bash
python -u tests/runtime/test_kernel_trace.py --npu --api-version v1 \
    --ep-size 4 --first-device 0 \
    --num-tokens 2048 --hidden 6144 --num-topk 8 --experts-per-rank 32 \
    --aiv 32 \
    --pairs 6 --windows 3 --iterations 100 --warmup 10 \
    --output "$PWD/kernel_trace_acceptance_$(date +%Y%m%d_%H%M%S)"
```

输出目录必须尚不存在。`--api-version` 默认为 `v2`，开发诊断沿用 V2。也可设置 `DEEPEP_RUN_TRACE_NPU=1` 后通过 pytest 运行 `test_npu_metadata_and_overhead`，该入口使用默认 V2 配置。普通 pytest 默认跳过实机测试。

此用例固定 BF16、随机路由、expand=0、无 bias，不清理 L2。依次覆盖 fresh dispatch、cached dispatch、无 weight 和带 weight 的 combine。V1 fresh 将 `get_dispatch_layout()` 计入 API 耗时，V1 combine 使用同一个 `--aiv` 配置。weight 是 combine 的 topk_weights 输入，未增加 bias。精度用独立 CPU reference 检查，发生在计时之前。

普通 `--npu` 验收不带 `--diagnostic`。每个 on 试次设置 `DEEPEP_TRACE_RECORDER_ENABLE=1`，通过 `DEEPEP_TRACE_DIR` 指向本次的 `trace/` 子目录，使用生产写入器。正式采集固定使用 128 槽，`--trace-slots` 只控制开发诊断。

验收检查所有 rank 的生产日志，按 `op` 校验长短格式、字段及类型，再核对 `type=2` 完成记录数是否与精度检查、预热和正式执行预期一致。同一 rank 的完成记录需要可由同一个时钟校准偏移换算，允许墙钟时间截断至微秒带来的误差。出现 `type=1` 时仍按超时规则检查设备时间字段，它不计入完成数。off 试次不应产生 trace。槽位代际旧值拒收由 CPU 原生协议测试另行验证。

每对 A/B 使用独立的 torchrun 进程组，交替执行 off→on、on→off，默认共 6 对、12 次进程组/Buffer 初始化。子进程按试次设置采集环境，清除继承的 PYTHONPATH，在输出目录运行以加载已安装 wheel；其他运行环境保持一致。初始化、精度检查、预热和退出排空不计入稳态开销。

计时不启用 profiler、不捕获 Graph，每个窗口连续执行多次 eager API。窗口前设备同步和 Host barrier 在计时之外，窗口末设备同步计入耗时；逐次只插入正常的事件流依赖，不额外逐次做 Host 同步。计时包含 Host 提交、设备执行、业务路径原有同步和同时运行的 Python 日志线程影响。每个窗口取所有 rank 的最大耗时，各窗口取中位数，得到一次进程试验的每 API 耗时。

每种操作独立计算配对开销：

$$
\Delta_i = 100\left(\frac{T_{\mathrm{on},i}}{T_{\mathrm{off},i}}-1\right)\%.
$$

对进程级配对样本的中位数做固定随机种子的 10,000 次 bootstrap，输出每种操作的 95% 经验区间。区间上界严格小于 1% 为 PASS；下界大于等于 1% 为 FAIL；其余为 INCONCLUSIVE。四种操作均 PASS 且精度/日志校验通过时退出码为 0，否则非零。窗口不作为独立试验增加样本数，不剔除慢样本。

这是指定 shape 和运行环境的稳态验收，不是所有训练负载的开销保证，也不保证 bootstrap 区间消除系统性频率/负载漂移。波动较大时可增加 `--pairs 12 --windows 5 --iterations 500` 并保持机器空闲。初始化校准与最后日志排空另存耗时，不被宣称满足稳态的 1% 阈值。expand=1、多 Buffer、不同拓扑和 Graph 不在这个实机入口的验收范围内。

产物如下：

- `acceptance.json`：总结果、四种操作的开销、区间及 PASS/FAIL/INCONCLUSIVE；异常时保存失败原因。
- `config.json`、`runs.json`：配置（含轮询和超时的环境变量值）、各对 A/B 的汇总原始值。
- `pairXX_on/trace/trmt-deepep-trace-*.txt`：生产写入器生成的日志；off 目录不应有 trace。
- `pairXX_on/` 和 `pairXX_off/` 下的 `rankN.json`：每 rank 每窗口耗时、初始化/退出排空耗时、wheel 路径和精度结果；`run.log` 保存 torchrun 输出。

单次 torchrun 默认超时 1800 秒，可用 `--timeout` 调整。超时或中断时终止该次创建的进程组，不影响其他任务。未实际执行 NPU 验收前，CPU 测试通过不能视为开销已满足 1%。

## 分层开销诊断

当完整采集开销明显超过阈值时，同一个测试入口支持 `--diagnostic`。这是归因工具，不是性能验收，也不修改 kernel 的打点、done_seq 或 Host 映射内存协议。

需要重新构建安装本分支的 Host 原生扩展；只复制 Python 文件不够。采集启动会检查原生扩展的版本标识（当前为 4），拒绝混用不支持这些开关的旧扩展。

| 模式 | 实际工作 | 输出 |
| --- | --- | --- |
| off | 完全关闭采集 | rank 耗时/精度 JSON |
| production | 开发对照基线，使用原有 JSONL 写入路径，清除两个诊断环境变量；无诊断调用计数或批次计数 | kernel JSONL，无 diagnostic JSON |
| record_only | 与 production 相同的前台元数据、launch、设备 tick/done_seq；不启动采集线程，仅在显式窗口边界或 destroy 时收集 | 完整 JSONL，无 diagnostic JSON |
| metadata | 仅执行 API 元数据构造、原生上下文交换与恢复；不分配采集环、不启动校准 kernel 或采集线程 | rank JSON 中的元数据调用统计 |
| native | 元数据、设备打点、Python 线程轮询原生 Drain；不把每条记录转换成 Python 字典 | 退出时的 diagnostic JSON |
| json_discard | 与 json 一样在释放 GIL 时完整编码；随后在 C++ 释放文本，只返回计数和空字符串 | 退出时的 diagnostic JSON |
| json | native 的工作，加释放 GIL 的原生 JSONL 编码；返回整批文本后丢弃 | 退出时的 diagnostic JSON |
| full | 原生批量编码和 Python JSONL 写入/刷新 | kernel JSONL 和退出时的 diagnostic JSON |

native 仍构造原有 C++ TraceRecord，使用原来的 Drain 和 done_seq 回收逻辑；每批只向 Python 返回少量计数。它不是“只有设备打点”的纯设备测试，包含原生记录处理、轮询及批次统计成本。json/full 在此基础上返回整批 JSONL 文本，不返回逐事件 Python 字典。三种模式都检查完成、launch/end 数、丢样和退出 summary，不能用关闭回收或填满槽池来减少被采集的事件。

第一轮不打开细分计时，比较端到端 API 耗时：

```bash
python -u tests/runtime/test_kernel_trace.py --npu --diagnostic \
    --ep-size 4 --first-device 0 --num-tokens 2048 --hidden 6144 \
    --num-topk 8 --experts-per-rank 32 --aiv 32 --reduce-aiv 32 \
    --rounds 3 --windows 3 --iterations 100 --warmup 10 \
    --output "$PWD/trace_diagnosis_$(date +%Y%m%d_%H%M%S)"
```

默认每轮运行全部五种模式，轮转并反转运行顺序，3 轮共 15 次独立进程组/Buffer 初始化。每种模式与同轮 off 比较；保留全部窗口和逐轮比值。需要更多证据时增加 rounds。`--modes` 可以选取子集，但必须包含 off 和至少一种开启模式。

第二轮另建目录，沿用相同参数并增加 `--diagnostic-timing`，通常 `--rounds 1` 即可获得初步分项数据。此轮在 Host 增加计时调用，只用于定位，不能混入第一轮或正式 1% 验收。两种计时设置下，full 都由原生侧编码整批文本，再交给 Python 写入。

统计说明：

- `rankN.json.metadata_statistics`：fresh_dispatch/cached_dispatch/combine 的调用数、构造与进入原生上下文的 `enter_ns`、恢复上下文的 `restore_ns`；包括准备、精度和预热调用。timing 关闭时计时值为零，不代表没有成本。
- `trace/diagnostic-*.json.counters`：polls、records、launches、ends、summaries、completed、incomplete、dropped、launch_errors。native/json 不生成逐事件 kernel JSONL，这是有意省略；full 仍做完整日志校验。
- `drain_ns`：原生 Drain 墙钟耗时，执行时释放 GIL。
- `convert_ns`：整批原生文本转 Python 字符串的墙钟耗时；native 返回空字符串。版本 1 中此字段测量逐条字典转换，含义不同。
- `drain_call_wall_ns`：Python 发起整个 drain 调用到返回的墙钟耗时，包含转换、批次计数、调度以及可能的 GIL 等待；不能把它减去 drain_ns 后直接称为 GIL 等待时间。
- `serialize_ns`、`write_flush_ns`：释放 GIL 期间原生 JSONL 编码、Python 写入及 flush 的累计墙钟耗时；只在对应模式且 timing 开启时有值。flush 不是 fsync，不表示持久化到物理磁盘。

`diagnosis.json` 汇总五种模式相对 off 的开销，`diagnostic_valid=true` 只表示诊断证据通过完整性检查，`performance_acceptance` 始终为 false。完整性失败返回非零并保留原因；性能变慢本身不使诊断任务失败。native/json/full 的计数校验针对本用例当前 eager 非展开的 notify+dispatch、communication+reduce 路径（每次 API 两个 kernel）；改变执行策略后若 kernel 数不同，应先核对实际路径，不放宽丢样检查。

诊断环境变量由测试控制器管理并在初始化时读取：`DEEPEP_TRACE_DIAGNOSTIC_MODE=record_only|metadata|native|json_discard|json|full`，`DEEPEP_TRACE_DIAGNOSTIC_TIMING=0|1`。EP 内配置必须一致，普通 `--npu` 验收会清除这两个诊断变量。`--diagnostic` 的 `production` 模式使用原有开发 JSONL 基线，以保持各诊断分层之间的比较口径。生产写入路径的开关对照使用不带 `--diagnostic` 的 `--npu`。

相邻模式的差值包含工作重叠、线程调度变化及小额诊断统计成本，不能当作各模块的精确独占时间。优先用未加细分计时的五组结果定位增量，再结合计时轮次解释。json 与 native 的差值现在主要对应原生编码和批量文本传递，不能沿用旧版的逐事件 Python 转换/编码归因。最终优化必须回到原始完整采集开关 A/B 验收。


## 普通采集对照和编码返回链路定位

诊断 full 即使关闭细分计时，仍包含每 API 调用计数、每批原生计数返回及 Python 累加。先比较开发基线 production 和诊断 full，再拆分编码与文本返回，两个步骤均沿用同一个 worker、shape、精度检查及计时边界。

```bash
# 第一步：开发基线 off/production 与诊断 off/full，六轮共 24 次进程组初始化。
python -u tests/runtime/test_kernel_trace.py --npu --diagnostic --paired-baselines \
    --modes off production full --rounds 6 \
    --ep-size 4 --first-device 0 --num-tokens 2048 --hidden 6144 \
    --num-topk 8 --experts-per-rank 32 --aiv 32 --reduce-aiv 32 \
    --windows 3 --iterations 100 --warmup 10 \
    --output "$PWD/trace_production_control_$(date +%Y%m%d_%H%M%S)"

# 第二步：编码与 Python 文本返回，六轮共 36 次进程组初始化。
python -u tests/runtime/test_kernel_trace.py --npu --diagnostic --paired-baselines \
    --modes off native json_discard json --rounds 6 \
    --ep-size 4 --first-device 0 --num-tokens 2048 --hidden 6144 \
    --num-topk 8 --experts-per-rank 32 --aiv 32 --reduce-aiv 32 \
    --windows 3 --iterations 100 --warmup 10 \
    --output "$PWD/trace_encoding_control_$(date +%Y%m%d_%H%M%S)"
```

`--paired-baselines` 为每个开启模式运行独立且相邻的 off；轮转开启模式顺序，并交替 off→mode、mode→off。`plan.json` 记录执行顺序，`runs.json` 的 pair 标识对应本模式的 off。所有试次顺序执行，不共享或并发使用 Buffer。关闭该选项时仍使用原有同轮共享 off 的方式。

`diagnosis.json.layer_increments` 提供 production→full、native→json_discard、json_discard→json 等可用对照的逐轮原始差值、各自 off 校正后的差值、中位数、范围及正增量轮数；单位 ns。这些差值不是独占耗时，不能因中位数为正就忽略符号不一致的轮次。production 与 full 都校验完整逐事件日志；production 还要求无诊断统计和诊断报告。带 production 的比较拒绝 `--diagnostic-timing`，避免把加计时的诊断路径与普通路径混为一谈。

json_discard 保留相同的设备打点、done_seq 检查、回收、2 ms 轮询和 C++ 编码函数；只改变文本离开原生层的方式。它仍返回小型计数字典和空 Python 字符串，不是无 GIL 或无 Python 线程的模式。与 json 的差值包括文本转换、传递、释放位置及调度变化，不能直接称为 GIL 等待。没有关闭回收或故意填满槽位来省略采集。

所有原生诊断批次新增 `encoded_bytes` 和 `returned_text_bytes`（UTF-8 文本字节累计）：native 两者为 0，json_discard 前者大于 0、后者为 0，json/full 两者相等且大于 0。控制器检查这些条件以及原有 completed/launch/end/dropped/incomplete，避免把未编码或错误返回文本当作有效对照。新计数仅存在于诊断路径，不改变普通采集输出。

如需详细计时，可另跑第二步的模式组合，加 `--diagnostic-timing --rounds 1` 并使用新目录；此结果仅辅助定位，不纳入主性能比较。最终低于 1% 的验收仍需运行不带 `--diagnostic` 的普通 A/B 用例。

## 只记录与后台并发收集的对照

### 固定 128 槽的短窗口对照（推荐）

为避免扩大槽位改变后台扫描和地址复用，使用 `--windowed-collection --trace-slots 128 --iterations 60 --windows 5`。每个计时窗口 60 次 API，最多产生 120 条 kernel 记录；五个窗口仍为每种 API 300 次稳态调用。record_only 全程没有后台线程，production 保留原有后台轮询。两组以及 off/metadata 使用完全相同的显式边界：

1. 全部四种 API 精度检查完成后同步并收集一次，最多 8 条记录。
2. 每种 API 的 10 次预热完成后同步并收集一次，最多 20 条记录。
3. 每个窗口记录 API 耗时后，再同步、收集、编码和 flush。这些操作不进入 API 计时；窗口内仍只有原有 API 调用、消费流依赖和结束同步。

合计每 rank 25 个显式边界（精度 1 + 预热 4 + 窗口 20），最终 destroy 再收集一次并输出 summary。没有在窗口内重置槽位、丢弃记录或关闭 kernel 打点。窗口间继续复用同一个 128 槽环，沿用 done_seq 代际检查。

```bash
python -u tests/runtime/test_kernel_trace.py --npu --diagnostic --paired-baselines \
    --windowed-collection --trace-slots 128 \
    --modes off metadata record_only production --rounds 6 \
    --ep-size 4 --first-device 0 --num-tokens 2048 --hidden 6144 \
    --num-topk 8 --experts-per-rank 32 --aiv 32 --reduce-aiv 32 \
    --windows 5 --iterations 60 --warmup 10 \
    --output "$PWD/trace_window128_control_$(date +%Y%m%d_%H%M%S)"
```

仍为 36 次进程组初始化。控制器向所有 worker 传递窗口选项，并在启动前拒绝非 128 槽或任一边界间超容量的配置。`record_only` 在每个显式边界前后检查线程不存在及收集次数，发现额外收集立即失败。

`rankN.json.collection_boundaries` 保存 25 个边界的标签和 `collect_ns`，控制器核对标签、顺序和数量。`collect_ns` 是窗口外收集调用墙钟，包含可能的锁等待、编码和 flush，不含该边界前的设备同步；不是单独的磁盘成本。record_only 的 `deferred_evidence` 应为 `thread_absent=true, drains_before_destroy=25, drains_after_destroy=26`。完整日志仍要求所有调用齐全、零丢样、零未完成，且完成数超过实际 128 槽容量。

这是固定边界下的隔离实验，production 也执行显式边界收集，不完全等同于连续训练中的纯后台采集。不要直接与先前 100 次×3 窗口或 2488 槽结果相减。重点分析同一轮的 record_only−off 与 production−record_only；主测试不要启用额外分项计时。

### 全过程仅记录的大容量对照（保留）

`record_only` 仅供诊断：保留业务线程和设备侧的采集写入，不启动 Python 轮询线程，也不做诊断的逐 API 计数。初始化仍写 calibration；从精度检查、预热到全部 API 计时完成之间不 Drain、不编码、不写事件日志。随后 `buffer.destroy()` 在计时之外同步并一次性收集、编码和落盘，保留原 done_seq 验证与完整日志校验。因此它测试的是 Host+设备的前台记录成本，不是纯设备打点成本。

本 eager 用例每 rank 的总记录槽需求为 `8 × (1 + warmup + windows × iterations)`，包含全部四类 API 的精度、预热和稳态，每 API 两个 kernel。默认参数共 2488 个槽，不能使用默认 128 槽并依赖丢样完成试验。包含 record_only 时，控制器自动为同一试验中的所有开启模式设置相同的足额容量；`--trace-slots` 可显式指定，但不足会在启动前拒绝。范围为 128～16384，默认 0 表示自动选择，其他模式默认仍为 128。

```bash
# 主对照：同容量，只记录 vs 后台并发采集；六轮共 36 次初始化。
python -u tests/runtime/test_kernel_trace.py --npu --diagnostic --paired-baselines \
    --modes off metadata record_only production --rounds 6 \
    --ep-size 4 --first-device 0 --num-tokens 2048 --hidden 6144 \
    --num-topk 8 --experts-per-rank 32 --aiv 32 --reduce-aiv 32 \
    --windows 3 --iterations 100 --warmup 10 \
    --output "$PWD/trace_record_control_$(date +%Y%m%d_%H%M%S)"

# 标准容量参考：避免将大环的扫描/缓存影响当成默认采集成本。
python -u tests/runtime/test_kernel_trace.py --npu --diagnostic --paired-baselines \
    --modes off production --trace-slots 128 --rounds 6 \
    --ep-size 4 --first-device 0 --num-tokens 2048 --hidden 6144 \
    --num-topk 8 --experts-per-rank 32 --aiv 32 --reduce-aiv 32 \
    --windows 3 --iterations 100 --warmup 10 \
    --output "$PWD/trace_default_capacity_$(date +%Y%m%d_%H%M%S)"
```

诊断容量通过初始化变量 `DEEPEP_TRACE_SLOTS` 传入原生层，未设置时为 128；不支持运行中改变。控制器清除继承值，calibration 输出实际 `slots` 和 `slot_bytes`；Host 映射区大小是二者乘积，另有 Host 槽状态和待收集元数据。扩大容量会改变后台扫描范围和内存占用，所以大环 production 结果必须与默认 128 槽参考分别标注，不直接外推为正常部署开销。

`rankN.json` 记录实际容量及 record_only 的 `deferred_evidence`：采集线程不存在、destroy 前收集次数为 0、destroy 后为 1。任一不符合、容量不匹配、丢样或未完成记录都会失败。释放资源及延后落盘耗时记录在 `destroy_and_drain_ns`，不计入 API 墙钟；该字段也包含业务资源销毁，不能全算作日志成本。record_only 与 production 都不允许细分诊断计时。

执行顺序现在每两轮使用同一轮转位置，第二轮反转，然后再轮转。两个开启模式的前两轮为 A→B、B→A，避免旧的“每轮轮转再反转”相互抵消。各模式仍使用相邻独立 off，并交替 off/on 方向；查看 plan.json 确认，不依据口头顺序假定。

重点比较 off→record_only 与 record_only→production；metadata 仅用于辅助估计 Host 上下文成本，仍带诊断计数，不与 record_only 形成完全纯净的累加层。新增对应的 layer_increments，保留每轮原始和基线校正值。前台记录变慢时再拆 Host 与设备发布；只有后台开启才明显变慢时再查并发收集、调度和 GIL。当前差值本身不能证明具体一种等待机制。

## Fresh dispatch 的 Host 计数等待与 GIL 定位

`--host-dispatch-timing` 是单独的诊断开关，必须配合 `--diagnostic`，不能与 `--diagnostic-timing` 同时使用。普通业务不启用探针；关闭时不读取时钟、查询 GIL 或分配样本，但原生入口仍有关闭状态判断。启用后按业务线程预分配样本空间，在全部 API 计时窗口结束后才转换成 Python 对象并写入 `host_dispatch_rankN.json`。它不修改 kernel、计数发布协议、同步或业务线程的 GIL 策略。

```bash
python -u tests/runtime/test_kernel_trace.py --npu --diagnostic --paired-baselines \
    --host-dispatch-timing --windowed-collection --trace-slots 128 \
    --modes off record_only production --rounds 6 \
    --ep-size 4 --first-device 0 --num-tokens 2048 --hidden 6144 \
    --num-topk 8 --experts-per-rank 32 --aiv 32 --reduce-aiv 32 \
    --windows 5 --iterations 60 --warmup 10 \
    --output "$PWD/trace_host_counts_$(date +%Y%m%d_%H%M%S)"
```

六轮共 24 次进程组初始化，每个模式（包括 off）均启用相同探针，保留相邻独立 off 和交替顺序。需要重新编译 Host 扩展：`host_dispatch_probe_version=3`；采集诊断接口版本仍为 4。每 rank 导出 622 次原生 dispatch 调用，fresh/cached 各 311 次，分别标记精度、预热、稳态窗口及窗口内序号。每种 API 的 300 次稳态样本用于分段汇总，不把同一进程内的样本当作 300 次独立实验。

| 分段 | 含义 |
| --- | --- |
| `native_prepare` | 原生 dispatch 入口到 notify 提交包装函数前 |
| `notify_launch_call` | notify 提交包装函数耗时；cached 时也包含该段内的计划重建 |
| `counts_wait` | 进入计数读取到返回；包括读取路径内部的准备和返回成本 |
| `first_count_wait` | 映射内存路径中，进入读取到观察到第一个总接收计数 |
| `remaining_counts` | 观察到总计数后，继续读取全部专家计数 |
| `counts_postprocess` | 计数读取返回后的校验、列表处理，到输出准备前 |
| `outputs_prepare` | 输出 tensor 和相关输出元数据的准备 |
| `wait_end_to_dispatch` | 计数返回到 dispatch 提交包装函数前，覆盖上述后处理和输出准备 |
| `dispatch_launch_call` | dispatch 提交包装函数耗时，含其内部准备和采集提交工作 |
| `native_scope` | 原生 dispatch 函数作用域的总耗时 |

这些区间存在包含关系，不能全部相加。使用 C++ `steady_clock` 纳秒值，只用于同一进程内分段相减，不与日志墙钟或 NPU tick 直接相减。提交函数耗时不是设备 kernel 耗时；Python layout 和原生函数外的 Python 包装工作不在 `native_scope` 内。`api_call_id` 在已有采集上下文存在时可关联 trace，off 等没有上下文的样本为 -1，使用 rank、调用类型、窗口和序号对照。

`wait_path=1` 表示映射内存轮询，2 表示 `Tensor.cpu()` 回读，3 表示固定容量不等待，4 表示 cached 不等待。本测试要求 fresh 实际走 1 或 2，cached 走 4。计数就绪表示 Host 已观察到数据，不证明整个 notify kernel 已结束，也不提供设备实际发布计数的精确时刻。

GIL 状态通过当前线程的 `PyGILState_Check()` 获取，不使用 heartbeat 推测，也不获取或释放业务线程的 GIL。初始化自检分别在持有 GIL 和显式释放 GIL 的两个控制区调用同一个探针，标准 CPython 应返回 1/0。自由线程构建暂标 -1（不支持判定），不能将其解释为未持有 GIL。

映射轮询在入口、首次观察到某个计数未就绪、总计数就绪、所有计数就绪及出口采样 GIL；不在每次自旋迭代读取时间或查询 GIL。`first_pending` 没出现表示没有观察到未就绪，不填造等待样本；它也可能出现在总计数就绪之后。`Tensor.cpu()` 路径只观察调用前后，不能根据这两个值推断其内部始终持有或释放 GIL。采样时刻的值记录在 `gil_held`，未执行的点为 -1，稳态汇总包含各状态计数。

先逐 rank、逐轮检查 `counts_wait` 是否随采集开启而增加，再比较 `outputs_prepare` 和 `wait_end_to_dispatch`。GIL 为 0 时不能将等待增长归因于业务线程持有 GIL；仍可能存在 CPU 调度或设备计数发布延迟。GIL 为 1 也仅说明持有状态，不能单凭此证明它解释了全部劣化。

`diagnosis.json` 将该运行标记为 `instrumented=true, performance_acceptance=false`。探针本身有开销，因此该运行只用于定位，不能作为 <1% 验收或与先前未插桩耗时直接相减。控制器校验 GIL 自检、样本数量、阶段顺序、实际等待路径和零丢样；CPU 回归还覆盖关闭时无探针调用、容量溢出、异常退出及错误证据拒绝。

### Notify 计数发布的设备分段

在上述命令中增加 `--notify-device-timing`，其余参数不变。这是独立的第二轮定位：所有模式（包括 off）增加同样的设备诊断，不能与未加此选项的历史结果直接相减。必须同时启用 `--host-dispatch-timing`；Host 和 device 均须重新构建，因为 `DispatchTilingData` 新增了诊断地址和序号，原有运行时 ABI 大小检查仍然生效。

只在 fresh 的映射计数发布路径启用，cached、固定容量和图捕获不打这些点。测试遇到 fresh 回退到 `Tensor.cpu()` 会拒绝结果，而不是悄悄缺失设备样本。每条 Host 样本新增 `notify_device`：

| tick | core 0 的观察位置 |
| --- | --- |
| `entry` | notify 初始化上下文之后、算法主体之前，与原有 trace 起点一致 |
| `counts_ready` | 原有本地全核同步之后、准备向 Host 发布计数之前 |
| `publish_begin` | 第一批计数的 DataCopyPad 提交之前，已完成打包及 V_MTE3 等待 |
| `publish_end` | 第一批计数拷贝的原有 MTE3_S 等待完成之后 |

第一批包含总接收计数，也可能包含全部专家计数，因此最后一个区间不是单个 4 字节计数的传输时间。`entry→counts_ready` 包含 notify 中的计算、访存和同步等待，不能单独称为计算成本。`publish_end` 不是 CPU 实际读到计数的时刻，也不是整个 notify 结束时刻。

导出 `device_entry_to_counts_ready`、`device_count_pack`、`device_first_count_copy`、`device_entry_to_publish_begin/end` 的同卡 tick 差。当前 Ascend950 使用 1 GHz 系统计数器，差值单位为 ns；与 Host `counts_wait` 分别比较各模式的增量，不跨时钟直接相减，不据此计算精确的“发布到 CPU 观察”延迟。

内存与同步约定：

- 初始化时分配独立 Host 映射区，每个原生调用保留 256 字节，当前 622 条共 159232 字节/rank。它不改变普通 trace 的 128 槽容量。
- 只有 core 0 写本次调用的独占地址：偏移 0 写四个 uint64 tick（32 字节），偏移 32～95 写八类同步耗时、96～159 写八类同步次数，偏移 224 写本轮 `done_seq`（32 字节）。不在诊断运行内复用地址，序号为该 Host 样本的 ordinal。
- 四个时间戳及同步累计量先保存在局部变量中。原有计数发布循环全部完成后，才复用已排空、32 字节对齐的 accumulator UB 前 32 字节；沿用 S_MTE3/MTE3_S 发布 payload，再发布 done_seq。不增加全卡或全核同步。
- 普通采集线程不扫描这个诊断区，CPU 业务不等待诊断 done_seq。所有计时窗口结束后显式导出，设备同步后核对序号、读取并释放映射区；异常清理同样先确保设备不再写入。
- 新增 tick 和两次 32 字节诊断写入仍可能扰动 notify 尾部及后续调用，只能用于定位。不能以“异步”宣称零开销。

分析优先逐 rank、逐轮、逐窗口比较 Host `native_prepare`、`notify_launch_call`、`counts_wait` 和设备上述区间。若设备 `entry→publish_begin` 也明显增长，继续定位 notify 内部及对端到达；若设备区间不增长而 Host 等待增长，仍需区分 kernel 入队/启动、CPU 进入等待时刻变化和 CPU 观察调度，不能直接断言是 CPU 调度。

CPU 测试覆盖序号不匹配、漏报、时间倒序、cached 意外报告及回读路径误用；Linux 原生协议测试模拟独立槽写入、完成标志和重新配置后的清零。模拟不替代真实设备的编译、精度和内存可见性验证。

### Fresh 总耗时对齐与 notify 同步分解

为避免拿平均 rank 分段解释最慢 rank 总耗时，增加 `--align-fresh-timing`。必须配合 `--host-dispatch-timing`，所有模式包括 off 都使用相同插桩。当前原生探针版本为 3，Host/device 都须重新构建；采集接口版本仍为 4。

```bash
python -u tests/runtime/test_kernel_trace.py --npu --diagnostic --paired-baselines \
    --host-dispatch-timing --notify-device-timing --align-fresh-timing \
    --windowed-collection --trace-slots 128 \
    --modes off record_only production --rounds 6 \
    --ep-size 4 --first-device 0 --num-tokens 2048 --hidden 6144 \
    --num-topk 8 --experts-per-rank 32 --aiv 32 --reduce-aiv 32 \
    --windows 5 --iterations 60 --warmup 10 \
    --output "$PWD/trace_fresh_aligned_$(date +%Y%m%d_%H%M%S)"
```

原性能指标保持为每个窗口内“连续调用、消费流依赖提交、窗口末尾设备同步”的总墙钟除以调用次数；每窗口取最大 rank，再对窗口取中位数。没有逐次增加设备同步。新增逐次 `api_host_ns` 只测 Python dispatch 调用到返回，不表示该次设备通信已完成；`consume_ns` 测原有消费流依赖提交和本次返回对象释放；不会额外持有输出跨到下次调用。

`rankN.json.fresh_timing` 保存逐次 Host 调用、消费依赖提交、循环记账和末尾同步耗时。控制器按 rank/窗口/iteration 将其与 Host 原生样本一一关联，输出 `fresh_alignment.json`：

- `trials` 保存每个 rank 的窗口总耗时、逐次关联的 ordinal/API ID、Host 分段和设备分段。
- Host ledger 使用互不重叠的原生区间；加上 Python/原生边界残差、consume、循环记账、窗口末尾同步后，必须与该 rank 窗口总耗时一致。边界残差为 Host 调用时长减原生作用域时长，包含 Python layout、包装及探针边界等工作，不能称为纯 Python 或 GIL 成本。两种 Host 计时器只比较时长，不直接相减绝对时间戳；残差原样保留。
- `comparisons` 使用本模式相邻独立 off。每窗口记录 `off_max_rank`、`on_max_rank` 和实际最大值之差；`critical_ledger_delta_ns` 比较这两个实际获选 rank 的 ledger，允许最慢 rank 变化。
- `rank_deltas` 同时保留所有固定 rank 的总耗时、Host 分段和设备区间增量。不要把“不同获选 rank 的分段差”误称为“某个固定 rank 的变化”。
- 每轮 `aggregate_percent` 严格沿用原先的中位数窗口最大值口径。各分段中位差不保证能加成总指标中位差；逐窗口 ledger 才用于核对加和。

窗口末尾同步单列，不平均摊成某次 kernel 的执行时长。设备区间与 Host 等待重叠，仅作为旁证，不参与 Host ledger 加和。新增 Python 时钟读取和记账有开销，该运行依然不是 <1% 验收，也不能直接套用历史未插桩的劣化百分比。

notify 的 core 0 在原有八处同步调用前后读取 tick，循环内的调用累计耗时和次数：

| 分类 | 原有同步位置 |
| --- | --- |
| `entry_collective` | 条件性的 workspace 复用入口跨 rank 同步 |
| `channel_counts_join` | 本地 channel 计数写入后的全核同步 |
| `channel_prefix_join` | channel 前缀处理后的全核同步 |
| `counts_collective` | rank/expert 计数发布后的跨 rank 同步 |
| `gather_join` | 批量收集远端计数后的全核同步，取决于算法路径 |
| `expert_reuse_join` | 后续 expert tile 复用 scratch 前的全核同步，累计 |
| `expert_ready_join` | 后续 expert tile 装入后的全核同步，累计 |
| `final_join` | Host 计数发布之前原有的全核同步 |

`notify_device.sync_ns/sync_calls` 保存八类累计量；未执行的条件分支为 0 次。每个参与 core 仍执行原来的同步，只有 core 0 读取诊断时钟，不新增同步点。`device_sync_total` 为这些区间之和，`device_ready_other` 为 `entry→counts_ready` 扣除这些区间后的剩余量。后者仍包含计算、访存、流水事件等待和诊断指令，不能称为纯计算时间；同步区间包含同步实现自身执行成本，不能称为纯对端等待。core 0 的同步耗时也不代表所有 core 的最大耗时。

同步统计沿用前述独占诊断槽，payload 共 160 字节，done_seq 块位于 224～255；各 32 字节写入互不重叠，全部 payload 排空后才发布 done_seq。复用原有 accumulator 前 32 字节，逐块写出，不扩大 UB、不更改生产 trace 环容量。数据仍在业务计数发布后写出，测试结束后读取；额外诊断写入可能扰动后续推进，需结合对照解释。

分析顺序：先报告本轮 fresh 总耗时劣化及获选 rank，再展示对应窗口闭合的 Host 增量，最后用该 rank 的设备同步分类解释等待变化。如果只能定位到 `device_ready_other`，或条件同步次数变化，应保留这一不确定性，不按平均占比分摊。

### Fresh Python/绑定边界残差细分

在上一节的命令上增加 `--python-dispatch-timing`，必须同时带 `--align-fresh-timing`。所有模式（包括 off）都启用相同的外层探针。该探针仅记录 fresh，cached 不占用样本；初始化预分配时间戳行，业务期间只保存整数，不保存 tensor/handle，不进行文件输出，也不启动新线程。普通业务没有配置探针时不读取这些 Python 时钟，仍有关闭状态判断。

本次仅改 Python 诊断和用例，未修改 kernel、同步及流依赖。Host 原生探针版本仍为 3，采集诊断版本仍为 4；Python `DispatchPythonProbe.version=1`。重新安装匹配的 Python 包，不能只更新 benchmark 文件。

新增 `python_dispatch_rankN.json`，当前 fixture 每 rank 为 311 条（精度 1、预热 10、稳态 300），与原有 622 条 Host 原生报告里的 fresh 子集按调用顺序、窗口和 iteration 关联。控制器检查数量、序号、必需点、时间顺序及元数据区间包含关系；漏打点或加载旧包装代码会失败。

`fresh_alignment.json` 中原 `python_and_native_boundary` 被以下互不重叠的 ledger 项替代：

| 项目 | 含义 |
| --- | --- |
| `py_checks_and_policy` | 参数/容量校验、策略选择及前置准备 |
| `py_dependency_event` | 原有依赖事件获取或记录 |
| `py_trace_context_create` | trace context 对象创建；元数据生成器主体尚未执行 |
| `py_trace_context_enter` | 上下文进入，包含元数据构造和原生 push |
| `py_stream_enter` | 原有通信流上下文进入及依赖处理 |
| `py_layout_prepare` / `py_layout_call` | layout 前边界和 get_dispatch_layout 调用/返回字段处理；不是设备 layout kernel 时长 |
| `py_config_and_args` | Config 与 dispatch 参数组装 |
| `py_native_binding_residual` | Python 中测得的原生方法调用时长减去已测 C++ dispatch 作用域；包含 launch plan 参数求值、绑定转换及原生作用域外的工作，不代表纯绑定或 GIL 成本 |
| `py_result_epilogue` | 返回值拆包、可选 expand、路由索引 clone 等 |
| `py_stream_exit` / `py_trace_context_exit` | 流上下文退出、trace 元数据恢复 |
| `py_handle_and_record` | EPHandle 构造及相关 stream 记录 |
| `py_output_record` / `py_output_pack` | 输出 stream 记录与输出包装 |
| `py_finish_event` / `py_return_tail` | 原有完成事件提交及返回前边界 |
| `py_outer_boundary` | benchmark 测得的调用时长减 Python 函数内部起止区间，保留装饰器/返回/探针边界等剩余成本 |

原生 dispatch 内部 ledger 保留，新的 Python ledger 与其合并后继续核对每窗口总耗时。`py_native_binding_residual` 和 `py_outer_boundary` 是时长之差，原样保留正负值；不跨 Python/C++ 绝对时间戳相减。窗口尾部同步仍单列，不新增逐次设备同步，也不额外持有返回对象跨到下一次调用。

元数据进一步单列为 `metadata_nested_ns` / `metadata_nested_mean_ns`，包括 identity 获取、shape/context key 与模板查找、缓存未命中时的模板构造/原生转换、动态 API ID、原生 push、原生 restore。off 没有 trace session，原始点为 -1，汇总这些嵌套项为 0。它们已经包含在 `py_trace_context_enter/exit` 内，不再加到 ledger；缓存命中时 template_build 只包含分支和打点边界成本，不代表仍在重建模板。

读取报告时先确认本轮总劣化和获选最慢 rank，再看原残差主要落在哪个新项。若仍集中在 `py_native_binding_residual`，只能继续向绑定/原生方法外层追踪；不能直接认定 GIL 竞争。新增 Python 时钟和方法调用会产生诊断成本，metadata 内层点只在有 trace session 时执行，因此不能把它的全部增量当成未插桩的业务成本，也不以本轮结果作 <1% 验收。

CPU 回归实际调用 fresh/cached 包装，检查采集开关、模板复用和异常时上下文恢复；另覆盖缺样、乱序、错误 off 元数据、探针容量、未完成导出以及控制器参数传递和总账加和。

## 元数据缓存与原生批量编码的验证

缓存只保留标量和原生快照，不保留用户 tensor 或 handle 对象。shape、dtype、容量或模式变化会使用新模板；超过 64 种时清空模板缓存。正在执行或延迟执行的 kernel 持有 shared_ptr 父快照引用，不受缓存清空影响。上下文仍是线程局部的，嵌套调用及异常退出使用原生句柄恢复。

每次调用仍读取 Host tensor 属性、更新 ID、预留槽位和记录 launch 时间，并非零开销。kernel 的 tick、done_seq 和满槽跳过机制保持不变。Python 管理后台线程和退出排空。开发和生产路径各自在 C++ 编码对应的 JSONL 格式。Python 写入编码后的内容，并管理生产日志的文件轮转。

重新构建安装 Host 扩展后，先运行本文 CPU 回归，再运行不带 `--diagnostic` 的 `--npu` 正式采集开关 A/B。检查精度、生产日志字段、完成记录数和实际开销。诊断的 json/full 已采用新编码实现，分项不能直接套用旧版本含义。

本次 EP4 实测中，V1 和 V2 各完成六对开关试次，精度均通过。V1 日志完整，Fresh dispatch 配对中位开销为 +11.01%，未通过 1% 验收。V2 Fresh dispatch 配对中位开销为 +10.73%，判定为 `INCONCLUSIVE`。V2 有两个开启试次共丢弃 19 条 Combine 记录，日志完整性未通过。

### 调用上下文的轻量化

每次 API 调用的 `api_call_id`、`handle_id` 在原生快照中直接保存为整数，不再为两个字段构造 map 节点；排空时才展开为原来的 JSON 字段。每次调用仍分配独立的不可变原生快照，保留父模板引用。不能原地覆盖正在被延迟 kernel 或收集器使用的上下文。

Python 将两个 ID 的分配合并到一次锁操作，以只读 Mapping 代替动态字典加 ChainMap；每次分别读取一次输入及路由 shape，以 dtype 缓存 element bytes。缓存只含标量，不缓存 tensor 身份，也不保留 tensor/handle 引用；shape、dtype、容量和模式变化仍参与模板选择。异常退出、嵌套上下文及线程局部恢复保持原有语义。

此修改不改变 kernel 打点、done_seq、槽位容量、同步或收集线程。Python 包与 Host 扩展必须一起更新，原生 `kernel_trace_metadata_version` 为 2；不匹配时启用采集会明确报错。采集关闭时不要求创建元数据 session。

诊断中 API ID 分配现包含在 identity 区间，dynamic_ids 区间主要剩余只读视图构造；不要直接把优化前后的这两个子项各自相减解释收益。首先对两版分别运行无细分探针的 off/on 配对验收，并交错两版执行顺序，比较 fresh/cached 绝对时长和相对各自 off 的开销。需要定位时再单独运行分段诊断。探针结果不替代未插桩的 <1% 验收，也不能将未修改的 handle 构造或同步等待宣称为本次已优化。

CPU 回归覆盖 shape/dtype 变化、模板复用、不持有 tensor、旧扩展拒绝、嵌套与异常恢复；Linux 原生协议测试覆盖标量 ID 覆盖父字段、快照离开作用域后的排空及槽位多轮复用。Windows 上的语法编译不替代 Linux 原生测试和 NPU 验证。
