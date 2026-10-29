#include "csrc/ops/dup_server.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include <torch_npu/csrc/aten/common/from_blob.h>

#include "ascend_deepep/platform.h"
#include "ascend_deepep/kernels/types.h"
#include "csrc/launcher.h"
#include "csrc/shmem_instance.h"

namespace {

constexpr int64_t kRanksPerServer = 8;
constexpr int64_t kRanksPerFrame = 64;
constexpr int64_t kServersPerFrame = kRanksPerFrame / kRanksPerServer;
constexpr int64_t kCompletionSignalBytes = 512;
constexpr int64_t kWorkspaceAlignment = 512;
constexpr int64_t kUbAlignment = 32;
constexpr int64_t kUdmaWqeBufferBytes = 128 * 64;
constexpr int64_t kReceiverTraceRecordBytes = static_cast<int64_t>(ascend_deepep::kDupServerReceiverTraceRecordBytes);
constexpr int64_t kMaxNotifyWorkersPerGateway = 2;
constexpr int64_t kMaxDupServerTopk = 16;
constexpr int64_t kMaxNotifyPackWorkers = 32;
constexpr int64_t kGatewayPolicySourceLane = 0;
constexpr int64_t kRemoteGetNumAiv = 16;
constexpr uint32_t kInvalidPowerOfTwoShift = 32;
constexpr auto kTopkIdxScalarType = c10::CppTypeToScalarType<ascend_deepep::topk_idx_t>::value;

void CheckNpuContiguous(const torch::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be an NPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

int64_t CheckedMul(int64_t lhs, int64_t rhs, const char* name)
{
    TORCH_CHECK(lhs >= 0 && rhs >= 0, name, " factors must be nonnegative");
    TORCH_CHECK(lhs == 0 || rhs <= std::numeric_limits<int64_t>::max() / lhs, name, " overflows int64");
    return lhs * rhs;
}

int64_t CheckedAdd(int64_t lhs, int64_t rhs, const char* name)
{
    TORCH_CHECK(lhs >= 0 && rhs >= 0, name, " values must be nonnegative");
    TORCH_CHECK(lhs <= std::numeric_limits<int64_t>::max() - rhs, name, " overflows int64");
    return lhs + rhs;
}

int64_t AlignUp(int64_t value, int64_t alignment)
{
    TORCH_CHECK(value >= 0, "alignment input must be nonnegative");
    TORCH_CHECK(value <= std::numeric_limits<int64_t>::max() - (alignment - 1), "aligned value overflows int64");
    return (value + alignment - 1) / alignment * alignment;
}

uint32_t CheckedUint32(int64_t value, const char* name)
{
    TORCH_CHECK(value >= 0 && static_cast<uint64_t>(value) <= std::numeric_limits<uint32_t>::max(), name,
                " must fit uint32");
    return static_cast<uint32_t>(value);
}

torch::Tensor MakeShmemTensor(void* symmetric_buffer, std::vector<int64_t> sizes, at::ScalarType dtype)
{
    TORCH_CHECK(symmetric_buffer != nullptr, "symmetric_buffer must not be null");
    return at_npu::native::from_blob(symmetric_buffer, sizes, [](void*) {}, at::TensorOptions().dtype(dtype));
}

struct DupServerWorkspaceLayout {
    int64_t num_servers;
    int64_t payload_owner_count;
    int64_t hidden_bytes;
    int64_t chunk_capacity;
    int64_t completion_signal_offset;
    int64_t producer_signal_offset;
    int64_t receiver_trace_offset;
    int64_t workspace_bytes;
};

struct DupServerNotifyWorkspaceLayout {
    int64_t gateway_prefix_offset;
    int64_t gateway_prefix_stride;
    int64_t gateway_route_offset;
    int64_t gateway_route_stride;
    int64_t gateway_route_lane_offset;
    int64_t gateway_route_lane_stride;
    int64_t gateway_ready_offset;
    int64_t rank_prefix_offset;
    int64_t rank_prefix_stride;
    int64_t rank_route_offset;
    int64_t rank_route_stride;
    int64_t rank_route_lane_offset;
    int64_t rank_route_lane_stride;
    int64_t rank_ready_offset;
    int64_t workspace_bytes;
};

struct DupServerUbLayout {
    int64_t copy_tile_bytes;
};

int64_t CeilDiv(int64_t value, int64_t divisor)
{
    TORCH_CHECK(value >= 0 && divisor > 0, "invalid ceil-div operands");
    return value / divisor + (value % divisor != 0);
}

void CheckDupServerTopology(int64_t ranks_per_server, int world_size)
{
    TORCH_CHECK(ranks_per_server == kRanksPerServer, "duplicate-server currently requires 8 ranks/server");
    TORCH_CHECK(world_size >= kRanksPerServer && world_size % kRanksPerServer == 0,
                "duplicate-server requires world_size to be a positive multiple of 8");
    TORCH_CHECK(world_size / ranks_per_server <= 64, "duplicate-server supports at most 64 servers (512 ranks)");
}

void CheckGatewayPolicy(int64_t gateway_policy)
{
    TORCH_CHECK(gateway_policy == kGatewayPolicySourceLane,
                "gateway_policy must be 0 (source_lane); other policies are not supported");
}

int64_t MaxFirstWavePeers(int64_t num_servers, int64_t max_inflight_same_frame_servers)
{
    int64_t max_wave_peers = 0;
    for (int64_t frame_begin = 0; frame_begin < num_servers; frame_begin += kServersPerFrame) {
        const int64_t frame_servers = std::min(kServersPerFrame, num_servers - frame_begin);
        const int64_t cross_frame_peers = num_servers - frame_servers;
        const int64_t same_frame_peers = std::min(max_inflight_same_frame_servers, frame_servers - 1);
        max_wave_peers = std::max(max_wave_peers, cross_frame_peers + same_frame_peers);
    }
    return max_wave_peers;
}

uint32_t PowerOfTwoShift(int64_t value)
{
    if (value <= 0 || (value & (value - 1)) != 0) {
        return kInvalidPowerOfTwoShift;
    }

    uint32_t shift = 0;
    while (value > 1) {
        value >>= 1;
        ++shift;
    }
    return shift;
}

DupServerWorkspaceLayout BuildWorkspaceLayout(int64_t world_size, int64_t ranks_per_server, int64_t token_capacity,
                                              int64_t hidden, int64_t num_tokens_per_chunk, int64_t gateway_policy)
{
    CheckGatewayPolicy(gateway_policy);
    const int64_t num_servers = world_size / ranks_per_server;
    const int64_t payload_owner_count = num_servers;
    const int64_t hidden_bytes = CheckedMul(hidden, static_cast<int64_t>(sizeof(c10::BFloat16)), "hidden bytes");
    const int64_t chunk_capacity = CeilDiv(token_capacity, num_tokens_per_chunk);
    TORCH_CHECK(hidden_bytes % 32 == 0, "duplicate-server scale-out requires a 32-byte-aligned BF16 row");
    const int64_t payload_slots = CheckedMul(payload_owner_count, token_capacity, "gateway payload slots");
    const int64_t payload_bytes = CheckedMul(payload_slots, hidden_bytes, "gateway payload bytes");
    const int64_t completion_signal_offset = AlignUp(payload_bytes, kWorkspaceAlignment);
    const int64_t completion_signal_slots = CheckedMul(payload_owner_count, chunk_capacity, "completion signal slots");
    const int64_t completion_signal_bytes =
        CheckedMul(completion_signal_slots, kCompletionSignalBytes, "completion signal bytes");
    const int64_t producer_signal_offset =
        CheckedAdd(completion_signal_offset, completion_signal_bytes, "producer signal offset");
    const int64_t producer_signal_slots = CheckedMul(num_servers, chunk_capacity, "producer signal slots");
    const int64_t producer_signal_bytes =
        CheckedMul(producer_signal_slots, kCompletionSignalBytes, "producer signal bytes");
    const int64_t receiver_trace_offset = AlignUp(
        CheckedAdd(producer_signal_offset, producer_signal_bytes, "receiver trace offset"), kWorkspaceAlignment);
    // Each of the eight receiver AIVs owns one disjoint compact-trace slice in local symmetric GM.
    const int64_t receiver_trace_source_count = num_servers;
    const int64_t receiver_trace_events_per_core =
        CheckedMul(receiver_trace_source_count, chunk_capacity, "receiver trace events per core");
    TORCH_CHECK(receiver_trace_events_per_core <=
                    std::numeric_limits<uint32_t>::max() / ascend_deepep::kDupServerReceiverTraceFields,
                "receiver trace element index must fit uint32");
    const int64_t receiver_trace_bytes =
        CheckedMul(CheckedMul(ranks_per_server, receiver_trace_events_per_core, "receiver trace events"),
                   kReceiverTraceRecordBytes, "receiver trace bytes");
    const int64_t workspace_bytes =
        CheckedAdd(receiver_trace_offset, receiver_trace_bytes, "duplicate-server workspace");
    return {num_servers,           payload_owner_count,      hidden_bytes,
            chunk_capacity,        completion_signal_offset, producer_signal_offset,
            receiver_trace_offset, workspace_bytes};
}

DupServerNotifyWorkspaceLayout BuildNotifyWorkspaceLayout(int64_t workspace_offset, int64_t world_size,
                                                          int64_t ranks_per_server, int64_t token_capacity,
                                                          int64_t num_topk, int64_t num_experts,
                                                          int64_t num_tokens_per_chunk)
{
    const int64_t num_servers = world_size / ranks_per_server;
    const int64_t experts_per_server = num_experts / num_servers;
    const int64_t num_local_experts = num_experts / world_size;
    const int64_t send_route_capacity = CheckedMul(token_capacity, num_topk, "notify send route capacity");
    const int64_t chunk_capacity = CeilDiv(token_capacity, num_tokens_per_chunk);

    const int64_t gateway_prefix_offset = AlignUp(workspace_offset, kWorkspaceAlignment);
    const int64_t gateway_prefix_stride =
        AlignUp(CheckedMul(experts_per_server + 1 + chunk_capacity, sizeof(int32_t), "gateway prefix and chunk row"),
                kWorkspaceAlignment);
    const int64_t gateway_prefix_bytes = CheckedMul(num_servers, gateway_prefix_stride, "gateway prefix inbox");
    const int64_t gateway_route_offset =
        CheckedAdd(gateway_prefix_offset, gateway_prefix_bytes, "gateway route offset");
    const int64_t gateway_route_stride = CheckedMul(send_route_capacity, sizeof(int32_t), "gateway route row");
    const int64_t gateway_route_bytes = CheckedMul(num_servers, gateway_route_stride, "gateway route inbox");
    const int64_t gateway_route_lane_offset =
        AlignUp(CheckedAdd(gateway_route_offset, gateway_route_bytes, "gateway route end"), kWorkspaceAlignment);
    const int64_t gateway_route_lane_stride = AlignUp(send_route_capacity, kWorkspaceAlignment);
    const int64_t gateway_route_lane_bytes =
        CheckedMul(num_servers, gateway_route_lane_stride, "gateway route lane inbox");
    const int64_t gateway_ready_offset =
        CheckedAdd(gateway_route_lane_offset, gateway_route_lane_bytes, "gateway route lane end");
    const int64_t gateway_ready_bytes = CheckedMul(num_servers, kCompletionSignalBytes, "gateway ready signals");

    const int64_t rank_prefix_offset = CheckedAdd(gateway_ready_offset, gateway_ready_bytes, "rank prefix offset");
    const int64_t rank_prefix_stride =
        AlignUp(CheckedMul(num_local_experts + 1 + chunk_capacity, sizeof(int32_t), "rank prefix and chunk row"),
                kWorkspaceAlignment);
    const int64_t rank_prefix_bytes = CheckedMul(world_size, rank_prefix_stride, "rank prefix inbox");
    const int64_t rank_route_offset = CheckedAdd(rank_prefix_offset, rank_prefix_bytes, "rank route offset");
    const int64_t rank_route_stride = CheckedMul(send_route_capacity, sizeof(int32_t), "rank route row");
    const int64_t rank_route_bytes = CheckedMul(world_size, rank_route_stride, "rank route inbox");
    const int64_t rank_route_lane_offset =
        AlignUp(CheckedAdd(rank_route_offset, rank_route_bytes, "rank route end"), kWorkspaceAlignment);
    const int64_t rank_route_lane_stride = AlignUp(send_route_capacity, kWorkspaceAlignment);
    const int64_t rank_route_lane_bytes = CheckedMul(world_size, rank_route_lane_stride, "rank route lane inbox");
    const int64_t rank_ready_offset = CheckedAdd(rank_route_lane_offset, rank_route_lane_bytes, "rank route lane end");
    const int64_t rank_ready_bytes = CheckedMul(world_size, kCompletionSignalBytes, "rank ready signals");
    const int64_t workspace_bytes =
        CheckedAdd(rank_ready_offset, rank_ready_bytes, "duplicate-server notify workspace");

    return {gateway_prefix_offset,     gateway_prefix_stride,     gateway_route_offset, gateway_route_stride,
            gateway_route_lane_offset, gateway_route_lane_stride, gateway_ready_offset, rank_prefix_offset,
            rank_prefix_stride,        rank_route_offset,         rank_route_stride,    rank_route_lane_offset,
            rank_route_lane_stride,    rank_ready_offset,         workspace_bytes};
}

DupServerUbLayout BuildUbLayout(int64_t num_servers, int64_t num_local_experts, bool use_urma_get)
{
    const int64_t source_count = num_servers;
    const int64_t next_chunk_bytes =
        AlignUp(CheckedMul(source_count, static_cast<int64_t>(sizeof(uint32_t)), "next chunk UB bytes"), kUbAlignment);
    const int64_t route_cursor_elements = CheckedMul(source_count, num_local_experts, "route cursor UB elements");
    const int64_t route_cursor_bytes =
        AlignUp(CheckedMul(route_cursor_elements, static_cast<int64_t>(sizeof(uint32_t)), "route cursor UB bytes"),
                kUbAlignment);
    const int64_t segment_end_bytes = route_cursor_bytes;
    const int64_t route_tile_elements = ascend_deepep::kDupServerSourceLaneRouteTileElements;
    const int64_t route_position_tile_bytes =
        AlignUp(CheckedMul(route_tile_elements, static_cast<int64_t>(sizeof(int32_t)), "route position tile UB bytes"),
                kUbAlignment);
    const int64_t route_metadata_bytes =
        CheckedAdd(CheckedAdd(next_chunk_bytes, route_cursor_bytes, "route cursor metadata UB bytes"),
                   CheckedAdd(segment_end_bytes, route_position_tile_bytes, "route scan metadata UB bytes"),
                   "route metadata UB bytes");
    const int64_t reserved_bytes = route_metadata_bytes;
    const int64_t minimum_required_bytes = CheckedAdd(reserved_bytes, 2 * kUbAlignment, "minimum receiver UB bytes");
    TORCH_CHECK(minimum_required_bytes <= ascend_deepep::platform::kKernelUbBytes,
                "duplicate-server receiver requires ", reserved_bytes,
                " UB bytes of route metadata and leaves no room for ping-pong payload tiles");
    const int64_t copy_tile_bytes =
        (ascend_deepep::platform::kKernelUbBytes - reserved_bytes) / 2 / kUbAlignment * kUbAlignment;
    if (use_urma_get) {
        TORCH_CHECK(copy_tile_bytes * 2 >= kUdmaWqeBufferBytes, "duplicate-server scale-up UB head requires ",
                    kUdmaWqeBufferBytes, " bytes for URMA GET WQEs, got ", copy_tile_bytes * 2);
    }
    return {copy_tile_bytes};
}

void CheckWorkspaceCapacity(const DupServerWorkspaceLayout& layout, std::size_t symmetric_buffer_bytes)
{
    TORCH_CHECK(static_cast<uint64_t>(layout.workspace_bytes) <= symmetric_buffer_bytes,
                "duplicate-server scale-out requires ", layout.workspace_bytes, " bytes of symmetric workspace, got ",
                symmetric_buffer_bytes);
}

}  // namespace

namespace ascend_deepep::ops {

DupServerNotifyPrepareResult DupServerNotifyPrepare(torch::Tensor topk_idx, c10::optional<torch::Tensor> topk_weights,
                                                    int64_t num_experts, int64_t token_capacity,
                                                    int64_t ranks_per_server, int64_t num_tokens_per_chunk,
                                                    int64_t num_aiv, int64_t gateway_policy, int rank, int world_size,
                                                    int num_device_aiv, bool enable_timer,
                                                    std::size_t symmetric_buffer_bytes, ShmemInstance& shmem,
                                                    void* symmetric_buffer)
{
    CheckNpuContiguous(topk_idx, "topk_idx");
    TORCH_CHECK(topk_idx.dim() == 2, "topk_idx must have shape [num_tokens, num_topk]");
    TORCH_CHECK(topk_idx.scalar_type() == kTopkIdxScalarType, "topk_idx must have dtype deep_ep.topk_idx_t");
    if (topk_weights.has_value()) {
        CheckNpuContiguous(*topk_weights, "topk_weights");
        TORCH_CHECK(topk_weights->scalar_type() == torch::kFloat32, "topk_weights must have dtype float32");
        TORCH_CHECK(topk_weights->sizes() == topk_idx.sizes(), "topk_weights must have the same shape as topk_idx");
        TORCH_CHECK(topk_weights->device() == topk_idx.device(), "topk_idx and topk_weights must be on one NPU");
    }

    CheckDupServerTopology(ranks_per_server, world_size);
    CheckGatewayPolicy(gateway_policy);
    TORCH_CHECK(num_experts > 0 && num_experts % world_size == 0, "num_experts must be divisible by world_size");
    TORCH_CHECK(token_capacity > 0, "token_capacity must be positive");
    TORCH_CHECK(num_tokens_per_chunk > 0 && num_tokens_per_chunk <= std::numeric_limits<int32_t>::max(),
                "num_tokens_per_chunk must be in int32 range");
    TORCH_CHECK(num_aiv > 0 && num_aiv <= num_device_aiv, "num_aiv must be in (0, device AIV count]");
    TORCH_CHECK(num_aiv >= ranks_per_server + 1,
                "duplicate-server notify requires at least one producer and one gateway worker per server-local rank");
    TORCH_CHECK(rank >= 0 && rank < world_size, "invalid duplicate-server notify rank");

    const int64_t num_tokens = topk_idx.size(0);
    const int64_t num_topk = topk_idx.size(1);
    const int64_t num_servers = world_size / ranks_per_server;
    const int64_t experts_per_server = num_experts / num_servers;
    const int64_t num_token_chunks = CeilDiv(num_tokens, num_tokens_per_chunk);
    const int64_t chunk_capacity = CeilDiv(token_capacity, num_tokens_per_chunk);
    const int64_t chunk_scratch_elements = CheckedMul(num_servers, chunk_capacity, "notify chunk scratch elements");
    TORCH_CHECK(chunk_scratch_elements <= std::numeric_limits<int32_t>::max(),
                "notify chunk metadata must fit int32 indexing");
    TORCH_CHECK(num_tokens > 0, "num_tokens must be positive");
    TORCH_CHECK(num_topk > 0 && num_topk <= kMaxDupServerTopk, "num_topk must be in [1, 16]");
    TORCH_CHECK(token_capacity >= num_tokens, "token_capacity must be at least num_tokens");
    TORCH_CHECK(token_capacity <= std::numeric_limits<int32_t>::max() / num_topk,
                "token_capacity * num_topk must fit int32");

    const int64_t block_dim = num_aiv;
    const int64_t expert_stride = AlignUp(num_experts, kUbAlignment / static_cast<int64_t>(sizeof(int32_t)));
    const int64_t server_stride = AlignUp(num_servers, kUbAlignment / static_cast<int64_t>(sizeof(int32_t)));
    const int64_t scratch_stride = CheckedAdd(expert_stride, server_stride, "notify scratch stride");
    const int64_t state_elems = CheckedAdd(expert_stride, CheckedMul(2, server_stride, "notify server state elements"),
                                           "notify state elements");
    const int64_t max_tokens_per_core = (num_tokens + block_dim - 1) / block_dim;
    const int64_t source_server_mask_bytes =
        AlignUp(CheckedMul(max_tokens_per_core, sizeof(uint64_t), "notify local server-mask bytes"), kUbAlignment);
    int64_t pack_workers = kMaxNotifyPackWorkers;
    int64_t metadata_bytes = 0;
    while (pack_workers > 0) {
        const int64_t histogram_elems = CheckedMul(pack_workers, scratch_stride, "notify histogram elements");
        metadata_bytes = CheckedAdd(CheckedMul(CheckedAdd(state_elems, histogram_elems, "notify metadata elements"),
                                               sizeof(int32_t), "notify metadata bytes"),
                                    source_server_mask_bytes, "notify metadata and local server-mask bytes");
        if (metadata_bytes <= ascend_deepep::platform::kKernelUbBytes) {
            break;
        }
        pack_workers /= 2;
    }
    TORCH_CHECK(pack_workers > 0 && metadata_bytes <= ascend_deepep::platform::kKernelUbBytes,
                "duplicate-server notify metadata does not fit in ", ascend_deepep::platform::kKernelUbBytes,
                " UB bytes; num_experts=", num_experts, ", num_servers=", num_servers, ", num_topk=", num_topk);

    const int64_t local_routes = CheckedMul(max_tokens_per_core, num_topk, "notify local routes");
    const int64_t topk_stage_bytes =
        AlignUp(CheckedMul(local_routes, sizeof(ascend_deepep::topk_idx_t), "notify top-k stage bytes"), kUbAlignment);
    const int64_t weight_stage_bytes =
        topk_weights.has_value()
            ? AlignUp(CheckedMul(local_routes, sizeof(float), "notify weight stage bytes"), kUbAlignment)
            : 0;
    const int64_t input_stage_bytes = CheckedAdd(topk_stage_bytes, weight_stage_bytes, "notify input stage bytes");
    const bool inputs_in_ub = input_stage_bytes <= ascend_deepep::platform::kKernelUbBytes - metadata_bytes;

    const int64_t send_route_capacity = CheckedMul(token_capacity, num_topk, "send route capacity");
    TORCH_CHECK(send_route_capacity <= std::numeric_limits<uint32_t>::max() / sizeof(int32_t),
                "route metadata bytes must fit uint32");
    const int64_t recv_route_capacity = CheckedMul(world_size, send_route_capacity, "receive route capacity");
    TORCH_CHECK(recv_route_capacity <= std::numeric_limits<int32_t>::max(),
                "world_size * token_capacity * num_topk must fit int32 receive offsets");
    const int64_t num_local_experts = num_experts / world_size;
    TORCH_CHECK(symmetric_buffer != nullptr, "symmetric_buffer must not be null");
    TORCH_CHECK(symmetric_buffer_bytes <= static_cast<std::size_t>(std::numeric_limits<int64_t>::max()),
                "symmetric buffer size exceeds int64");
    const DupServerNotifyWorkspaceLayout relative_workspace = BuildNotifyWorkspaceLayout(
        0, world_size, ranks_per_server, token_capacity, num_topk, num_experts, num_tokens_per_chunk);
    TORCH_CHECK(relative_workspace.workspace_bytes <= static_cast<int64_t>(symmetric_buffer_bytes),
                "duplicate-server notify workspace exceeds the symmetric buffer");
    const int64_t notify_workspace_offset =
        (static_cast<int64_t>(symmetric_buffer_bytes) - relative_workspace.workspace_bytes) / kWorkspaceAlignment *
        kWorkspaceAlignment;
    const DupServerNotifyWorkspaceLayout notify_workspace =
        BuildNotifyWorkspaceLayout(notify_workspace_offset, world_size, ranks_per_server, token_capacity, num_topk,
                                   num_experts, num_tokens_per_chunk);
    TORCH_CHECK(notify_workspace.workspace_bytes <= static_cast<int64_t>(symmetric_buffer_bytes),
                "duplicate-server notify workspace layout exceeds the symmetric buffer");
    auto int_options = topk_idx.options().dtype(torch::kInt32);
    auto send_token_count = torch::empty({num_servers}, int_options);
    auto send_token_ids = torch::full({num_servers, token_capacity}, -1, int_options);
    auto lane_options = topk_idx.options().dtype(torch::kUInt8);
    auto send_gateway_lane = torch::full({num_servers, token_capacity}, 0xff, lane_options);
    auto send_route_count = torch::empty({num_servers}, int_options);
    auto send_expert_prefix = torch::empty({num_servers, experts_per_server + 1}, int_options);
    auto packed_route_position = torch::full({send_route_capacity}, -1, int_options);
    auto packed_gateway_lane = torch::full({send_route_capacity}, 0xff, lane_options);
    auto source_server_mask = torch::empty({num_tokens}, topk_idx.options().dtype(torch::kInt64));
    const int64_t scratch_elements = CheckedAdd(CheckedMul(block_dim, scratch_stride, "notify count scratch elements"),
                                                chunk_scratch_elements, "notify scratch elements");
    auto scratch_buffer = torch::empty({scratch_elements}, int_options);
    c10::optional<torch::Tensor> packed_topk_weights = c10::nullopt;
    if (topk_weights.has_value()) {
        packed_topk_weights = torch::zeros({num_servers, send_route_capacity}, topk_weights->options());
    }
    auto recv_route_position = torch::empty({recv_route_capacity}, int_options);
    auto recv_gateway_lane = torch::full({recv_route_capacity}, 0xff, lane_options);
    auto recv_segment_offset = torch::zeros({num_local_experts * world_size + 1}, int_options);
    auto recv_weight_segment_begin = torch::zeros({num_local_experts * world_size}, int_options);
    auto recv_chunk_token_end = torch::full({world_size, chunk_capacity}, num_tokens, int_options);
    auto gateway_route_count = torch::zeros({num_servers}, int_options);
    auto gateway_route_position =
        MakeShmemTensor(reinterpret_cast<uint8_t*>(symmetric_buffer) + notify_workspace.gateway_route_offset,
                        {num_servers, send_route_capacity}, torch::kInt32);
    auto gateway_slot_mask = torch::zeros({num_servers, token_capacity}, int_options);
    auto notify_status = torch::zeros({1}, int_options);

    // 通信阶段拆成两组 AIV：producer 固定目标 server，gateway worker 固定板内目标 lane。
    // 每个 lane 最少一个、最多两个 gateway worker；两个 worker 分别使用 QP0/QP1。
    int64_t notify_producer_workers = std::min(num_servers, std::max<int64_t>(1, block_dim / 2));
    int64_t notify_gateway_workers =
        std::min(kMaxNotifyWorkersPerGateway * ranks_per_server, block_dim - notify_producer_workers);
    if (notify_gateway_workers < ranks_per_server) {
        notify_gateway_workers = ranks_per_server;
        notify_producer_workers = block_dim - notify_gateway_workers;
    }

    const ascend_deepep::DupServerNotifyPrepareTilingData tiling{
        CheckedUint32(rank, "rank"),
        CheckedUint32(world_size, "world_size"),
        CheckedUint32(ranks_per_server, "ranks_per_server"),
        CheckedUint32(num_servers, "num_servers"),
        CheckedUint32(num_tokens, "num_tokens"),
        CheckedUint32(token_capacity, "token_capacity"),
        CheckedUint32(num_topk, "num_topk"),
        PowerOfTwoShift(num_topk),
        CheckedUint32(num_experts, "num_experts"),
        CheckedUint32(experts_per_server, "experts_per_server"),
        PowerOfTwoShift(experts_per_server),
        CheckedUint32(send_route_capacity, "send_route_capacity"),
        CheckedUint32(recv_route_capacity, "recv_route_capacity"),
        CheckedUint32(block_dim, "block_dim"),
        CheckedUint32(expert_stride, "expert_stride"),
        CheckedUint32(server_stride, "server_stride"),
        CheckedUint32(scratch_stride, "scratch_stride"),
        CheckedUint32(max_tokens_per_core, "max_tokens_per_core"),
        CheckedUint32(pack_workers, "pack_workers"),
        inputs_in_ub ? 1U : 0U,
        topk_weights.has_value() ? 1U : 0U,
        enable_timer ? 1U : 0U,
        CheckedUint32(gateway_policy, "gateway_policy"),
        CheckedUint32(num_tokens_per_chunk, "num_tokens_per_chunk"),
        CheckedUint32(num_token_chunks, "num_token_chunks"),
        CheckedUint32(chunk_capacity, "chunk_capacity"),
        CheckedUint32(num_local_experts, "num_local_experts"),
        CheckedUint32(notify_producer_workers, "notify_producer_workers"),
        CheckedUint32(notify_gateway_workers, "notify_gateway_workers"),
        CheckedUint32(notify_workspace.gateway_prefix_stride, "gateway_prefix_stride"),
        CheckedUint32(notify_workspace.gateway_route_stride, "gateway_route_stride"),
        CheckedUint32(notify_workspace.gateway_route_lane_stride, "gateway_route_lane_stride"),
        CheckedUint32(notify_workspace.rank_prefix_stride, "rank_prefix_stride"),
        CheckedUint32(notify_workspace.rank_route_stride, "rank_route_stride"),
        CheckedUint32(notify_workspace.rank_route_lane_stride, "rank_route_lane_stride"),
        static_cast<uint64_t>(notify_workspace.gateway_prefix_offset),
        static_cast<uint64_t>(notify_workspace.gateway_route_offset),
        static_cast<uint64_t>(notify_workspace.gateway_route_lane_offset),
        static_cast<uint64_t>(notify_workspace.gateway_ready_offset),
        static_cast<uint64_t>(notify_workspace.rank_prefix_offset),
        static_cast<uint64_t>(notify_workspace.rank_route_offset),
        static_cast<uint64_t>(notify_workspace.rank_route_lane_offset),
        static_cast<uint64_t>(notify_workspace.rank_ready_offset),
        static_cast<uint64_t>(notify_workspace.workspace_bytes)};

    LaunchDupServerNotifyPrepareKernel(topk_idx, topk_weights, send_token_count, send_token_ids, send_gateway_lane,
                                       send_route_count, send_expert_prefix, packed_route_position, packed_gateway_lane,
                                       source_server_mask, packed_topk_weights, recv_route_position, recv_gateway_lane,
                                       recv_segment_offset, recv_weight_segment_begin, recv_chunk_token_end,
                                       gateway_route_count, gateway_slot_mask, notify_status, scratch_buffer,
                                       symmetric_buffer, tiling, shmem.FftsAddr());
    return {send_token_count,          send_token_ids,        send_gateway_lane,   send_route_count,
            send_expert_prefix,        packed_route_position, packed_gateway_lane, source_server_mask,
            packed_topk_weights,       recv_route_position,   recv_gateway_lane,   recv_segment_offset,
            recv_weight_segment_begin, recv_chunk_token_end,  gateway_route_count, gateway_route_position,
            gateway_slot_mask,         notify_status};
}

void DupServerNotifyScaleOut(int64_t token_capacity, int64_t hidden, int64_t ranks_per_server,
                             int64_t num_tokens_per_chunk, int64_t num_aiv, int64_t gateway_policy, int rank,
                             int world_size, int num_device_aiv, std::size_t symmetric_buffer_bytes,
                             ShmemInstance& shmem, void* symmetric_buffer)
{
    CheckDupServerTopology(ranks_per_server, world_size);
    CheckGatewayPolicy(gateway_policy);
    TORCH_CHECK(rank >= 0 && rank < world_size, "invalid duplicate-server scale-out rank");
    TORCH_CHECK(token_capacity > 0 && hidden > 0, "token_capacity and hidden must be positive");
    TORCH_CHECK(num_tokens_per_chunk > 0, "num_tokens_per_chunk must be positive");
    TORCH_CHECK(num_aiv > 0 && num_aiv <= num_device_aiv, "num_aiv must be in (0, device AIV count]");

    const DupServerWorkspaceLayout workspace = BuildWorkspaceLayout(world_size, ranks_per_server, token_capacity,
                                                                    hidden, num_tokens_per_chunk, gateway_policy);
    CheckWorkspaceCapacity(workspace, symmetric_buffer_bytes);
    const ascend_deepep::DupServerScaleOutTilingData tiling{
        CheckedUint32(rank, "rank"),
        CheckedUint32(world_size, "world_size"),
        CheckedUint32(ranks_per_server, "ranks_per_server"),
        CheckedUint32(workspace.num_servers, "num_servers"),
        0U,
        CheckedUint32(token_capacity, "token_capacity"),
        CheckedUint32(hidden, "hidden"),
        CheckedUint32(workspace.hidden_bytes, "hidden_bytes"),
        CheckedUint32(num_aiv, "block_dim"),
        0U,
        0U,
        0U,
        kInvalidPowerOfTwoShift,
        0U,
        CheckedUint32(num_tokens_per_chunk, "num_tokens_per_chunk"),
        CheckedUint32(workspace.chunk_capacity, "num_token_chunks"),
        CheckedUint32(workspace.chunk_capacity, "chunk_capacity"),
        0U,
        0U,
        0U,
        CheckedUint32(gateway_policy, "gateway_policy"),
        0U,  // remote_get_servers: Notify does not select a dispatch transport
        0U,  // x_buffer_offset: payload starts at the symmetric-buffer base
        static_cast<uint64_t>(workspace.completion_signal_offset),
        static_cast<uint64_t>(workspace.producer_signal_offset),
        static_cast<uint64_t>(workspace.receiver_trace_offset),
        static_cast<uint64_t>(workspace.workspace_bytes)};
    LaunchDupServerNotifyScaleOutKernel(symmetric_buffer, tiling, shmem.FftsAddr());
}

torch::Tensor DupServerDispatchScaleOut(torch::Tensor x, torch::Tensor send_token_count, torch::Tensor send_token_ids,
                                        torch::Tensor send_gateway_lane, int64_t ranks_per_server,
                                        int64_t max_inflight_same_frame_servers, int64_t num_tokens_per_chunk,
                                        int64_t num_aiv, bool enable_timer, int64_t gateway_policy, int rank,
                                        int world_size, int num_device_aiv, std::size_t symmetric_buffer_bytes,
                                        ShmemInstance& shmem, void* symmetric_buffer)
{
    CheckNpuContiguous(x, "x");
    CheckNpuContiguous(send_token_count, "send_token_count");
    CheckNpuContiguous(send_token_ids, "send_token_ids");
    CheckNpuContiguous(send_gateway_lane, "send_gateway_lane");
    TORCH_CHECK(x.dim() == 2, "x must have shape [num_tokens, hidden]");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must have dtype bfloat16");
    TORCH_CHECK(send_token_count.scalar_type() == torch::kInt32, "send_token_count must have dtype int32");
    TORCH_CHECK(send_token_ids.scalar_type() == torch::kInt32, "send_token_ids must have dtype int32");
    TORCH_CHECK(send_gateway_lane.scalar_type() == torch::kUInt8, "send_gateway_lane must have dtype uint8");
    TORCH_CHECK(x.device() == send_token_count.device() && x.device() == send_token_ids.device() &&
                    x.device() == send_gateway_lane.device(),
                "x and scale-out metadata must be on the same NPU");
    CheckDupServerTopology(ranks_per_server, world_size);
    CheckGatewayPolicy(gateway_policy);
    TORCH_CHECK(rank >= 0 && rank < world_size, "invalid duplicate-server scale-out rank");
    TORCH_CHECK(max_inflight_same_frame_servers > 0 && max_inflight_same_frame_servers < kServersPerFrame,
                "max_inflight_same_frame_servers must be in [1, 7]");
    TORCH_CHECK(num_aiv > 0 && num_aiv <= num_device_aiv, "num_aiv must be in (0, device AIV count]");
    TORCH_CHECK(num_tokens_per_chunk > 0, "num_tokens_per_chunk must be positive");

    const int64_t num_servers = world_size / ranks_per_server;
    TORCH_CHECK(send_token_count.dim() == 1 && send_token_count.size(0) == num_servers,
                "send_token_count must have shape [num_servers]");
    TORCH_CHECK(send_token_ids.dim() == 2 && send_token_ids.size(0) == num_servers,
                "send_token_ids must have shape [num_servers, token_capacity]");
    TORCH_CHECK(send_gateway_lane.sizes() == send_token_ids.sizes(),
                "send_gateway_lane must have the same shape as send_token_ids");
    const int64_t num_tokens = x.size(0);
    const int64_t hidden = x.size(1);
    const int64_t token_capacity = send_token_ids.size(1);
    TORCH_CHECK(num_tokens > 0 && hidden > 0, "x dimensions must be positive");
    TORCH_CHECK(token_capacity >= num_tokens, "send_token_ids capacity must be at least num_tokens");
    TORCH_CHECK(num_tokens <= std::numeric_limits<int32_t>::max(), "num_tokens must fit int32 token IDs");

    const DupServerWorkspaceLayout workspace = BuildWorkspaceLayout(world_size, ranks_per_server, token_capacity,
                                                                    hidden, num_tokens_per_chunk, gateway_policy);
    CheckWorkspaceCapacity(workspace, symmetric_buffer_bytes);

    const int64_t max_wave_peers = MaxFirstWavePeers(num_servers, max_inflight_same_frame_servers);
    const int64_t required_num_aiv = std::max<int64_t>(1, max_wave_peers);
    TORCH_CHECK(num_aiv >= required_num_aiv, "duplicate-server scale-out requires at least ", required_num_aiv,
                " AIVs so every wave-0 peer has one producer");
    const int64_t block_dim = required_num_aiv;
    const ascend_deepep::DupServerScaleOutTilingData tiling{
        CheckedUint32(rank, "rank"),
        CheckedUint32(world_size, "world_size"),
        CheckedUint32(ranks_per_server, "ranks_per_server"),
        CheckedUint32(num_servers, "num_servers"),
        CheckedUint32(num_tokens, "num_tokens"),
        CheckedUint32(token_capacity, "token_capacity"),
        CheckedUint32(hidden, "hidden"),
        CheckedUint32(workspace.hidden_bytes, "hidden_bytes"),
        CheckedUint32(block_dim, "block_dim"),
        CheckedUint32(max_inflight_same_frame_servers, "max_inflight_same_frame_servers"),
        0U,
        0U,
        kInvalidPowerOfTwoShift,
        0U,
        CheckedUint32(num_tokens_per_chunk, "num_tokens_per_chunk"),
        CheckedUint32(CeilDiv(num_tokens, num_tokens_per_chunk), "num_token_chunks"),
        CheckedUint32(workspace.chunk_capacity, "chunk_capacity"),
        0U,
        0U,
        enable_timer ? 1U : 0U,
        CheckedUint32(gateway_policy, "gateway_policy"),
        0U,  // remote_get_servers: scale-out-only entry keeps the legacy path
        0U,  // x_buffer_offset: payload starts at the symmetric-buffer base
        static_cast<uint64_t>(workspace.completion_signal_offset),
        static_cast<uint64_t>(workspace.producer_signal_offset),
        static_cast<uint64_t>(workspace.receiver_trace_offset),
        static_cast<uint64_t>(workspace.workspace_bytes)};

    auto gateway_x =
        MakeShmemTensor(symmetric_buffer, {workspace.payload_owner_count, token_capacity, hidden}, torch::kBFloat16);
    LaunchDupServerDispatchScaleOutKernel(x, send_token_count, send_token_ids, send_gateway_lane, symmetric_buffer,
                                          tiling, shmem.FftsAddr());
    return gateway_x;
}

torch::Tensor DupServerDispatch(torch::Tensor x, torch::Tensor send_token_count, torch::Tensor send_token_ids,
                                torch::Tensor send_gateway_lane, torch::Tensor recv_route_position,
                                torch::Tensor recv_gateway_lane, torch::Tensor recv_src_rank,
                                torch::Tensor recv_segment_offset, int64_t num_topk, int64_t ranks_per_server,
                                int64_t max_inflight_same_frame_servers, int64_t num_tokens_per_chunk, int64_t num_aiv,
                                bool use_urma_get, bool enable_timer, int64_t gateway_policy,
                                int64_t remote_get_servers, bool remote_get_rx_tx_gate, int rank, int world_size,
                                int num_device_aiv, std::size_t symmetric_buffer_bytes, ShmemInstance& shmem,
                                void* symmetric_buffer)
{
    CheckNpuContiguous(x, "x");
    CheckNpuContiguous(send_token_count, "send_token_count");
    CheckNpuContiguous(send_token_ids, "send_token_ids");
    CheckNpuContiguous(send_gateway_lane, "send_gateway_lane");
    CheckNpuContiguous(recv_route_position, "recv_route_position");
    CheckNpuContiguous(recv_gateway_lane, "recv_gateway_lane");
    CheckNpuContiguous(recv_src_rank, "recv_src_rank");
    CheckNpuContiguous(recv_segment_offset, "recv_segment_offset");
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == torch::kBFloat16, "x must be a 2D bfloat16 tensor");
    TORCH_CHECK(send_token_count.dim() == 1 && send_token_count.scalar_type() == torch::kInt32,
                "send_token_count must be a 1D int32 tensor");
    TORCH_CHECK(send_token_ids.dim() == 2 && send_token_ids.scalar_type() == torch::kInt32,
                "send_token_ids must be a 2D int32 tensor");
    TORCH_CHECK(send_gateway_lane.dim() == 2 && send_gateway_lane.scalar_type() == torch::kUInt8,
                "send_gateway_lane must be a 2D uint8 tensor");
    TORCH_CHECK(recv_route_position.dim() == 1 && recv_route_position.scalar_type() == torch::kInt32,
                "recv_route_position must be a 1D int32 tensor");
    TORCH_CHECK(recv_gateway_lane.dim() == 1 && recv_gateway_lane.scalar_type() == torch::kUInt8,
                "recv_gateway_lane must be a 1D uint8 tensor");
    TORCH_CHECK(recv_src_rank.dim() == 1 && recv_src_rank.scalar_type() == torch::kInt32,
                "recv_src_rank must be a 1D int32 tensor");
    TORCH_CHECK(recv_segment_offset.dim() == 1 && recv_segment_offset.scalar_type() == torch::kInt32,
                "recv_segment_offset must be a 1D int32 tensor");
    TORCH_CHECK(x.device() == send_token_count.device() && x.device() == send_token_ids.device() &&
                    x.device() == send_gateway_lane.device() && x.device() == recv_route_position.device() &&
                    x.device() == recv_gateway_lane.device() && x.device() == recv_src_rank.device() &&
                    x.device() == recv_segment_offset.device(),
                "x and duplicate-server metadata must be on the same NPU");
    TORCH_CHECK(send_gateway_lane.sizes() == send_token_ids.sizes(),
                "send_gateway_lane must have the same shape as send_token_ids");
    TORCH_CHECK(recv_route_position.numel() == recv_src_rank.numel(),
                "recv_route_position and recv_src_rank must have the same length");
    TORCH_CHECK(recv_route_position.numel() == recv_gateway_lane.numel(),
                "recv_route_position and recv_gateway_lane must have the same length");
    CheckDupServerTopology(ranks_per_server, world_size);
    CheckGatewayPolicy(gateway_policy);
    TORCH_CHECK(rank >= 0 && rank < world_size, "invalid duplicate-server dispatch rank");
    TORCH_CHECK(num_topk > 0, "num_topk must be positive");
    TORCH_CHECK(max_inflight_same_frame_servers > 0 && max_inflight_same_frame_servers < kServersPerFrame,
                "max_inflight_same_frame_servers must be in [1, 7]");
    TORCH_CHECK(num_aiv > 0 && num_aiv <= num_device_aiv, "num_aiv must be in (0, device AIV count]");
    TORCH_CHECK(num_tokens_per_chunk > 0, "num_tokens_per_chunk must be positive");

    const int64_t num_servers = world_size / ranks_per_server;
    TORCH_CHECK(remote_get_servers >= 0, "remote_get_servers must be nonnegative");
    TORCH_CHECK(!remote_get_rx_tx_gate || remote_get_servers > 0,
                "remote_get_rx_tx_gate requires remote_get_servers > 0");
    if (remote_get_servers != 0) {
        TORCH_CHECK(world_size == 32 || world_size == 64 || world_size == 128,
                    "remote_get_servers requires world_size in {32, 64, 128}; "
                    "world_size=8 has no remote server");
        TORCH_CHECK(gateway_policy == kGatewayPolicySourceLane,
                    "remote_get_servers is only supported by gateway_policy='source_lane'");
        const int64_t max_remote_get_servers = num_servers / 2 - 1;
        TORCH_CHECK(remote_get_servers <= max_remote_get_servers, "remote_get_servers must be in [0, ",
                    max_remote_get_servers, "] for world_size=", world_size);
    }
    TORCH_CHECK(recv_segment_offset.numel() > 1 && (recv_segment_offset.numel() - 1) % world_size == 0,
                "recv_segment_offset must have shape [num_local_experts * world_size + 1]");
    const int64_t num_local_experts = (recv_segment_offset.numel() - 1) / world_size;
    TORCH_CHECK(send_token_count.size(0) == num_servers, "send_token_count must have shape [num_servers]");
    TORCH_CHECK(send_token_ids.size(0) == num_servers, "send_token_ids must have shape [num_servers, token_capacity]");
    const int64_t num_tokens = x.size(0);
    const int64_t hidden = x.size(1);
    const int64_t token_capacity = send_token_ids.size(1);
    const int64_t num_recv_tokens = recv_route_position.numel();
    TORCH_CHECK(num_tokens > 0 && hidden > 0, "x dimensions must be positive");
    TORCH_CHECK(token_capacity >= num_tokens, "send_token_ids capacity must be at least num_tokens");
    TORCH_CHECK(num_tokens <= std::numeric_limits<int32_t>::max() / num_topk,
                "num_tokens * num_topk must fit int32 route positions");
    TORCH_CHECK(num_recv_tokens <= std::numeric_limits<uint32_t>::max(), "num_recv_tokens must fit uint32");

    const DupServerWorkspaceLayout workspace = BuildWorkspaceLayout(world_size, ranks_per_server, token_capacity,
                                                                    hidden, num_tokens_per_chunk, gateway_policy);
    CheckWorkspaceCapacity(workspace, symmetric_buffer_bytes);
    if (remote_get_servers != 0) {
        TORCH_CHECK(num_tokens_per_chunk <= std::numeric_limits<uint32_t>::max() / workspace.hidden_bytes,
                    "remote-GET local publication chunk bytes must fit uint32");
    }
    int64_t block_dim = kRemoteGetNumAiv;
    if (remote_get_servers == 0) {
        const int64_t max_wave_peers = MaxFirstWavePeers(num_servers, max_inflight_same_frame_servers);
        const int64_t producer_count = max_wave_peers + 1;
        TORCH_CHECK(num_aiv >= producer_count + ranks_per_server, "duplicate-server dispatch requires at least ",
                    producer_count + ranks_per_server, " AIVs: ", producer_count,
                    " fixed producers and one scale-up worker per gateway");
        block_dim = producer_count + ranks_per_server;
    } else {
        TORCH_CHECK(num_aiv >= kRemoteGetNumAiv, "duplicate-server remote-GET dispatch requires at least 16 AIVs");
    }
    const int64_t num_token_chunks = CeilDiv(num_tokens, num_tokens_per_chunk);
    const DupServerUbLayout ub = BuildUbLayout(num_servers, num_local_experts, use_urma_get);
    const ascend_deepep::DupServerScaleOutTilingData tiling{
        CheckedUint32(rank, "rank"),
        CheckedUint32(world_size, "world_size"),
        CheckedUint32(ranks_per_server, "ranks_per_server"),
        CheckedUint32(num_servers, "num_servers"),
        CheckedUint32(num_tokens, "num_tokens"),
        CheckedUint32(token_capacity, "token_capacity"),
        CheckedUint32(hidden, "hidden"),
        CheckedUint32(workspace.hidden_bytes, "hidden_bytes"),
        CheckedUint32(block_dim, "block_dim"),
        CheckedUint32(max_inflight_same_frame_servers, "max_inflight_same_frame_servers"),
        CheckedUint32(num_recv_tokens, "num_recv_tokens"),
        CheckedUint32(num_topk, "num_topk"),
        PowerOfTwoShift(num_topk),
        CheckedUint32(num_local_experts, "num_local_experts"),
        CheckedUint32(num_tokens_per_chunk, "num_tokens_per_chunk"),
        CheckedUint32(num_token_chunks, "num_token_chunks"),
        CheckedUint32(workspace.chunk_capacity, "chunk_capacity"),
        CheckedUint32(ub.copy_tile_bytes, "copy_tile_bytes"),
        use_urma_get ? 1U : 0U,
        enable_timer ? 1U : 0U,
        CheckedUint32(gateway_policy, "gateway_policy"),
        CheckedUint32(remote_get_servers, "remote_get_servers"),
        0U,  // x_buffer_offset: payload starts at the symmetric-buffer base
        static_cast<uint64_t>(workspace.completion_signal_offset),
        static_cast<uint64_t>(workspace.producer_signal_offset),
        static_cast<uint64_t>(workspace.receiver_trace_offset),
        static_cast<uint64_t>(workspace.workspace_bytes),
        remote_get_rx_tx_gate ? 1U : 0U};

    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    LaunchDupServerDispatchKernel(x, send_token_count, send_token_ids, send_gateway_lane, recv_route_position,
                                  recv_gateway_lane, recv_src_rank, recv_segment_offset, recv_x, symmetric_buffer,
                                  tiling, shmem.FftsAddr());
    return recv_x;
}

}  // namespace ascend_deepep::ops
