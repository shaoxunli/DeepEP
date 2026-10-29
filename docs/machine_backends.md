# POD 与全 Clos 后端

一个安装包包含两个独立实现，共用 `dependencies.lock.json` 固定的 CANN SHMEM
`4dd80e2129da42bd773982c2bb3758af1ed0bba8`。POD 保留本仓已有的 HT、LL、
实验组件和接口限制。全 Clos 实现来自 ascend_deepep 的 `deepep` 分支
`c09040328c853a011dfe37c74463c44723db4476`，包含 V1 Buffer、V2 ElasticBuffer、
MTE/UDMA、fresh/cached、FP8/BF16、expand/aggregate 以及原有资源接口。
各实现继续使用自己的 kernel、工作区、QP 调度和 handle，功能范围不互相推导。

## 构建

继续使用本仓的[编译安装流程](build.md)，同一 Python 环境需有匹配的 Torch/torch_npu。
不额外安装第二份 SHMEM，不运行原 ascend_deepep 的安装脚本。
构建产物包括 `_C`、`_clos_C` 和各自的 device library；二者加载同一 SDK 的 `libshmem.so`。
本迁移构建面向 Ascend950，统一使用 64-bit top-k 索引。

## 选择后端

在调用方设置当前 NPU 后构造 Buffer。两个类均接受：

```python
buffer = deep_ep.ElasticBuffer(
    group=ep_group,
    num_max_tokens_per_rank=8192,
    hidden=6144,
    num_topk=8,
    explicitly_destroy=True,
    machine_type="auto",  # auto / pod / clos_only
)
print(buffer.machine_type, buffer.machine_selection)
```

`auto` 读取当前物理 device 对应的 `/etc/hccl_rootinfo.json` 及其中引用的 topo 文件。
`TOPO_FILE_DESC` 按 topo 边声明解释；有 Clos 且无 NPU Mesh 的设备选择 `clos_only`。
Mesh+Clos 还需已核对的 POD board 规则（0x03/0x07）。缺省文件缺失时只回退到这些
DCMI board 规则；DCMI 查询超时为 5 秒，不重试。未知 board、缺失端点、文件矛盾或
不完整拓扑报错，要求显式指定，不能仅凭“16 卡”或某一 EID 位推断机型。
`rootinfo_path="/path/to/rootinfo.json"` 可指定文件；显式文件错误不会悄悄回退。
这些信息是拓扑声明，不证明物理连接健康或带宽利用率。

现场没有可靠 rootinfo 时使用 `machine_type="clos_only"` 或 `"pod"`。显式选择跳过
自动探测，但不修改 SHMEM 自身的拓扑初始化；依赖的驱动/SDK 仍须支持该环境。
传入 EP ProcessGroup/MPI communicator 时，构造阶段先核对全组的后端选择和探测错误，
再进入 SHMEM 初始化。保留 POD 的显式 rank/world_size/MPI 启动方式；没有 EP group 时
不增加框架进程组，调用方须保证全体 rank 选择一致。

后端只在构造时选择一次。全 Clos 按实际 EP group 的 rank 数工作，保留单服务器
16 卡，不加入 `rank // 8` 的分组假设。rank ID 和物理 device ID 不是同一个编号。

显式选择 Clos 的 V1 示例：

```python
config = deep_ep.Config(num_compute_units=32)
num_bytes = config.get_scaleup_buffer_size_hint(6144 * 2, ep_group.size(), 8192)
buffer = deep_ep.Buffer(
    ep_group, num_scaleup_bytes=num_bytes, explicitly_destroy=True,
    machine_type="clos_only", transport="udma",
)
# 原 V1 get_dispatch_layout -> dispatch -> combine 调用和返回语义保持不变。
```

V2 的 `get_buffer_size_hint(..., machine_type=...)` 也须选择同一后端；容量查询是
非 collective，不增加跨 rank 同步。直接让构造函数按 shape 计算可避免选错 hint。
POD 保留其 endpoint/MPI bootstrap；Clos 保留默认 UID 与显式 `shmem_ip_port` 方式。

## 参数与生命周期

公共调用优先使用 `num_compute_units`、`num_device_timeout_secs`、`num_scaleup_bytes`、
`num_scaleout_bytes`、`get_scaleup_buffer_size_hint`。`num_sms`、`num_gpu_timeout_secs`、
`num_nvl_bytes`、`num_rdma_bytes` 和 Config 的旧 chunk 名仍作为兼容别名接受；
同一参数用新旧名给出不同值时抛错。别名不会让某后端原本不支持的参数变得可用。

同一进程不能并存 POD 和 Clos SHMEM 会话。初始化/销毁使用共享锁，native 端检查
SDK 初始化状态，禁止覆盖另一后端的 QP/heap 配置。必须在整个 EP group 中按相同顺序
完成操作、`destroy()` 后再重建。handle 不能跨 Buffer 或后端复用；公共 EventHandle
可作为 Clos 的 previous_event/epilogue 依赖，等待语义和原有 stream 顺序保留。

`ascend_deepep` 保留为迁入实现的兼容导入路径；新代码通过 `deep_ep` 选择后端。
导入、CPU 契约检查、编译成功都不等于 NPU 精度或性能验证通过。
