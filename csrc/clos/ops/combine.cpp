#include "ops/combine.h"

#include <cstdint>
#include <chrono>
#include <limits>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <unistd.h>

#include "common/types.h"
#include "common/platform.h"
#include "common/udma_schedule.h"
#include "ops/launcher.h"
#include "runtime/buffer_runtime.h"
#include "ops/stream_allocator.h"
#include "runtime/shmem_instance.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"

namespace {

constexpr std::size_t kMaxCombineProfileLogBytes = 64U * 1024U * 1024U;
thread_local bool capture_combine_profile_log = false;
thread_local std::string combine_profile_log;

constexpr int64_t kMaxCombineAiv = ascend_deepep::platform::kMaxNumAiv;
constexpr int64_t kMaxCombineTopk = 128;
constexpr int64_t kCombineUbBytes = ascend_deepep::platform::kKernelUbBytes;
constexpr int64_t kCombineAlignmentBytes = 32;
constexpr int64_t kCombineWorkspaceBaseAlignment = 512;
constexpr int64_t kCombineUdmaCompletionSignalBytes = ascend_deepep::udma::kSignalBytes;
constexpr int64_t kCombineUdmaWqeScratchBytes = ascend_deepep::udma::kBatchWqes * ascend_deepep::udma::kWqeBytes;
constexpr uint32_t kCombineDtypeBfloat16 = 0;
constexpr uint32_t kCombineDtypeFloat16 = 1;

struct CombineWorkspaceLayout {
    int64_t token_capacity;
    int64_t x_row_stride;
    int64_t topk_weights_row_stride;
    int64_t x_buffer_offset;
    int64_t topk_weights_buffer_offset;
    int64_t udma_completion_signal_offset;
    int64_t total_bytes;
    int64_t end_offset;
};

int64_t AlignUpInt64(int64_t value, int64_t alignment)
{
    TORCH_CHECK(value >= 0, "combine alignment input must be nonnegative");
    TORCH_CHECK(alignment > 0, "combine alignment must be positive");
    TORCH_CHECK(value <= std::numeric_limits<int64_t>::max() - alignment + 1, "combine alignment overflow");
    return (value + alignment - 1) / alignment * alignment;
}

int64_t CheckedMulInt64(int64_t lhs, int64_t rhs, const char* name)
{
    TORCH_CHECK(lhs >= 0 && rhs >= 0, name, " factors must be nonnegative");
    TORCH_CHECK(lhs == 0 || rhs <= std::numeric_limits<int64_t>::max() / lhs, name, " size overflow");
    return lhs * rhs;
}

CombineWorkspaceLayout MakeCombineWorkspaceLayout(int64_t base_offset, int64_t world_size, int64_t num_tokens,
                                                  int64_t hidden_bytes, int64_t topk_weights_row_bytes, bool use_udma,
                                                  int64_t symmetric_buffer_bytes)
{
    TORCH_CHECK(base_offset >= 0 && base_offset <= symmetric_buffer_bytes,
                "combine workspace base must be within the symmetric buffer");
    TORCH_CHECK(base_offset % kCombineWorkspaceBaseAlignment == 0, "combine workspace base must be 512-byte aligned");
    TORCH_CHECK(num_tokens > 0, "combine workspace requires at least one original token");
    const int64_t x_row_stride = AlignUpInt64(hidden_bytes, kCombineAlignmentBytes);
    const int64_t topk_weights_row_stride =
        topk_weights_row_bytes > 0 ? AlignUpInt64(topk_weights_row_bytes, kCombineAlignmentBytes) : 0;
    const int64_t bytes_per_rank_token = x_row_stride + topk_weights_row_stride;
    TORCH_CHECK(bytes_per_rank_token > 0, "combine workspace requires a nonempty token row");

    const int64_t token_capacity = num_tokens;

    const int64_t total_slots = CheckedMulInt64(world_size, token_capacity, "combine workspace slots");
    const int64_t x_buffer_bytes = CheckedMulInt64(total_slots, x_row_stride, "combine x workspace");
    const int64_t topk_weights_buffer_bytes =
        CheckedMulInt64(total_slots, topk_weights_row_stride, "combine top-k weights workspace");
    TORCH_CHECK(x_buffer_bytes <= std::numeric_limits<int64_t>::max() - topk_weights_buffer_bytes,
                "combine workspace size overflow");
    const int64_t payload_bytes = x_buffer_bytes + topk_weights_buffer_bytes;
    TORCH_CHECK(base_offset <= std::numeric_limits<int64_t>::max() - payload_bytes,
                "combine workspace end offset overflows int64");
    const int64_t payload_end = base_offset + payload_bytes;
    // Same fixed marker header as dispatch; resetting it cannot overwrite
    // a slower rank's previous payload/notify workspace.
    const int64_t udma_completion_signal_offset = use_udma ? 0 : payload_end;
    const int64_t end_offset = payload_end;
    return {token_capacity,
            x_row_stride,
            topk_weights_row_stride,
            base_offset,
            base_offset + x_buffer_bytes,
            udma_completion_signal_offset,
            end_offset - base_offset,
            end_offset};
}

void CheckNpuContiguous(const torch::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1, name, " must be an NPU tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void CheckOptionalNpuContiguous(const c10::optional<torch::Tensor>& tensor, const char* name)
{
    if (tensor.has_value()) {
        CheckNpuContiguous(*tensor, name);
    }
}

int64_t ResolveNumAiv(int64_t num_aiv)
{
    TORCH_CHECK(num_aiv >= 0, "num_aiv must be greater than or equal to zero");
    const int64_t resolved = num_aiv == 0 ? 32 : num_aiv;
    TORCH_CHECK(resolved % 2 == 0, "num_aiv must be even");
    TORCH_CHECK(resolved <= kMaxCombineAiv, "num_aiv must not exceed ", kMaxCombineAiv, ", got ", resolved);
    return resolved;
}

}  // namespace

namespace ascend_deepep::ops {

void BeginCombineProfileLogCapture()
{
    TORCH_CHECK(!capture_combine_profile_log, "Combine profile log capture cannot be nested");
    combine_profile_log.clear();
    // Allocate before warmup/sampling, not while a rank is preparing its launch.
    combine_profile_log.reserve(kMaxCombineProfileLogBytes);
    capture_combine_profile_log = true;
}

void AppendCombineProfileLog(const std::string& message)
{
    if (capture_combine_profile_log) {
        TORCH_CHECK(message.size() <= kMaxCombineProfileLogBytes - combine_profile_log.size(),
                    "Combine diagnostic log exceeded 64 MiB; reduce samples and drain between operators");
        combine_profile_log.append(message);
        return;
    }
    TORCH_CHECK(::write(STDERR_FILENO, message.data(), message.size()) == static_cast<ssize_t>(message.size()),
                "Failed to write Clos combine profiling record");
}

std::string EndCombineProfileLogCapture()
{
    TORCH_CHECK(capture_combine_profile_log, "Combine profile log capture is not active on this thread");
    capture_combine_profile_log = false;
    std::string result;
    result.swap(combine_profile_log);
    return result;
}

CombineEpilogue::CombineEpilogue(std::function<CombineResult()> launch)
    : launch_(std::move(launch)),
      stream_id_(c10_npu::getCurrentNPUStream().id()),
      device_index_(c10_npu::getCurrentNPUStream().device_index())
{
}

void CombineEpilogue::SetOwner(c10::intrusive_ptr<runtime::BufferRuntime> owner)
{
    owner_ = std::move(owner);
}

CombineResult CombineEpilogue::Run()
{
    TORCH_CHECK(!consumed_, "combine epilogue has already been consumed");
    TORCH_CHECK(owner_ && owner_->IsInitialized(), "combine epilogue Buffer is destroyed");
    const auto stream = c10_npu::getCurrentNPUStream();
    TORCH_CHECK(stream.device_index() == device_index_ && stream.id() == stream_id_,
                "combine epilogue must run on its original device and communication stream");
    consumed_ = true;  // A failed launch must not be silently retried.
    return launch_();
}

std::tuple<torch::Tensor, c10::optional<torch::Tensor>> V1Combine(
    torch::Tensor x, c10::optional<torch::Tensor> topk_weights, c10::optional<torch::Tensor> bias_0,
    c10::optional<torch::Tensor> bias_1, torch::Tensor src_idx, torch::Tensor rank_prefix_matrix,
    torch::Tensor channel_prefix_matrix, torch::Tensor send_head, int64_t num_aiv,
    int64_t num_max_nvl_chunked_send_tokens, int64_t num_max_nvl_chunked_recv_tokens, int rank, int world_size,
    std::size_t symmetric_buffer_bytes, std::size_t notify_workspace_bytes, bool use_udma, ShmemInstance& shmem,
    void* symmetric_buffer, c10::optional<int64_t> allocation_stream_id, c10::optional<int64_t> num_max_tokens_per_rank,
    int64_t udma_peers_per_worker, int64_t udma_wqes_per_visit, c10::intrusive_ptr<CombineEpilogue>* epilogue,
    c10::optional<torch::Tensor> expanded_x, c10::optional<torch::Tensor> single_source_slots,
    std::function<void()> aggregate, c10::optional<torch::Tensor> pipeline_mapping,
    c10::optional<torch::Tensor> pipeline_values, int64_t pipeline_chunk_tokens)
{
    StreamAllocator allocator(allocation_stream_id);
    allocator.Record({x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix, send_head,
                      expanded_x, single_source_slots});
    CheckNpuContiguous(x, "x");
    CheckNpuContiguous(src_idx, "src_idx");
    CheckNpuContiguous(rank_prefix_matrix, "rank_prefix_matrix");
    CheckNpuContiguous(channel_prefix_matrix, "channel_prefix_matrix");
    CheckNpuContiguous(send_head, "send_head");
    CheckOptionalNpuContiguous(topk_weights, "topk_weights");
    CheckOptionalNpuContiguous(bias_0, "bias_0");
    CheckOptionalNpuContiguous(bias_1, "bias_1");

    TORCH_CHECK(world_size > 0, "world_size must be greater than zero");
    TORCH_CHECK(rank >= 0 && rank < world_size, "rank must be in [0, world_size)");
    TORCH_CHECK(x.dim() == 2, "x must have shape [num_recv_tokens, hidden]");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 || x.scalar_type() == torch::kFloat16,
                "x must be bfloat16 or float16");
    TORCH_CHECK(src_idx.dim() == 1, "src_idx must have shape [num_recv_tokens]");
    TORCH_CHECK(src_idx.scalar_type() == torch::kInt32, "src_idx must be int32");
    TORCH_CHECK(src_idx.size(0) == x.size(0), "src_idx and x must have the same num_recv_tokens");
    TORCH_CHECK(rank_prefix_matrix.dim() == 2, "rank_prefix_matrix must have shape [num_ranks, num_ranks]");
    TORCH_CHECK(channel_prefix_matrix.dim() == 2, "channel_prefix_matrix must have shape [num_ranks, num_channels]");
    TORCH_CHECK(send_head.dim() == 2, "send_head must have shape [num_tokens, num_ranks]");
    TORCH_CHECK(rank_prefix_matrix.scalar_type() == torch::kInt32, "rank_prefix_matrix must be int32");
    TORCH_CHECK(channel_prefix_matrix.scalar_type() == torch::kInt32, "channel_prefix_matrix must be int32");
    TORCH_CHECK(send_head.scalar_type() == torch::kInt32, "send_head must be int32");
    TORCH_CHECK(rank_prefix_matrix.size(0) == world_size, "rank_prefix_matrix first dim must equal world_size");
    TORCH_CHECK(rank_prefix_matrix.size(1) == world_size, "rank_prefix_matrix second dim must equal world_size");
    TORCH_CHECK(channel_prefix_matrix.size(0) == world_size, "channel_prefix_matrix first dim must equal world_size");
    TORCH_CHECK(send_head.size(1) == world_size, "send_head second dim must equal world_size");
    TORCH_CHECK(num_max_nvl_chunked_send_tokens > 0, "num_max_nvl_chunked_send_tokens must be greater than zero");
    TORCH_CHECK(num_max_nvl_chunked_recv_tokens > 0, "num_max_nvl_chunked_recv_tokens must be greater than zero");
    TORCH_CHECK(num_max_nvl_chunked_send_tokens < num_max_nvl_chunked_recv_tokens,
                "num_max_nvl_chunked_send_tokens must be smaller than num_max_nvl_chunked_recv_tokens");

    const int64_t resolved_num_aiv = ResolveNumAiv(num_aiv);
    const int64_t num_channels = resolved_num_aiv;
    if (ascend_deepep::udma::GroupOwnerOrder(shmem.CombinePeerOrder()) && !shmem.AutoShapePolicy())
        udma_peers_per_worker = shmem.PeersPerGroupPerWorker();
    if (use_udma) {
        TORCH_CHECK(shmem.SharedJetty(), "V1 UDMA requires a V1 shared-Jetty Buffer runtime");
        TORCH_CHECK(udma_peers_per_worker == 1 || udma_peers_per_worker == 2 || udma_peers_per_worker == 4 ||
                        udma_peers_per_worker == 8,
                    "udma_peers_per_worker must be 1, 2, 4 or 8");
        TORCH_CHECK(udma_wqes_per_visit == 4 || udma_wqes_per_visit == 8 || udma_wqes_per_visit == 16 ||
                        udma_wqes_per_visit == 32,
                    "udma_wqes_per_visit must be 4, 8, 16 or 32");
        const int64_t workers = shmem.TokenSchedule()
                                    ? ascend_deepep::udma::TokenWorkers(static_cast<uint32_t>(resolved_num_aiv))
                                    : shmem.WorkersPerPeer() *
                                          ascend_deepep::udma::Workers(world_size, udma_peers_per_worker,
                                                                       shmem.CombinePeerOrder(), shmem.RanksPerGroup());
        TORCH_CHECK(workers <= resolved_num_aiv && workers * shmem.QpsPerWorker() <= ascend_deepep::udma::kQpCount,
                    "UDMA peer grouping exceeds the AIV or 32-QP pool budget");
    }
    TORCH_CHECK(channel_prefix_matrix.size(1) == num_channels,
                "channel_prefix_matrix second dim must equal num_channels");

    const int64_t num_tokens = send_head.size(0);
    const int64_t token_capacity = num_max_tokens_per_rank.value_or(num_tokens);
    TORCH_CHECK(token_capacity > 0 && num_tokens <= token_capacity,
                "combine requires a positive common capacity >= local token count");
    const int64_t num_recv_tokens = x.size(0);
    const int64_t hidden = x.size(1);
    const int64_t x_element_bytes = x.element_size();
    const int64_t hidden_bytes = CheckedMulInt64(hidden, x_element_bytes, "combine hidden row");
    TORCH_CHECK(hidden_bytes > 0, "combine hidden row must not be empty");
    const int64_t communication_ub_bytes = kCombineUbBytes - (use_udma ? kCombineUdmaWqeScratchBytes : 0);
    TORCH_CHECK(AlignUpInt64(hidden_bytes, kCombineAlignmentBytes) <= communication_ub_bytes,
                "one combine token row requires ", hidden_bytes,
                " bytes plus alignment, but the combine staging area provides ", communication_ub_bytes, " bytes");
    // Opt-in A/B: the ordinary V2 UDMA reduction owns disjoint token ranges
    // and overwrites every output element, including tokens with no contributors.
    // This does not skip SHMEM flag resets or UB accumulator initialization.
    const char* skip_zero_env = std::getenv("DEEPEP_CLOS_COMBINE_SKIP_OUTPUT_ZERO");
    TORCH_CHECK(!skip_zero_env || !*skip_zero_env || std::strcmp(skip_zero_env, "0") == 0 ||
                    std::strcmp(skip_zero_env, "1") == 0,
                "DEEPEP_CLOS_COMBINE_SKIP_OUTPUT_ZERO must be 0 or 1");
    const bool skip_output_zero = skip_zero_env && std::strcmp(skip_zero_env, "1") == 0;
    TORCH_CHECK(!skip_output_zero ||
                    (use_udma && epilogue && !aggregate && !pipeline_chunk_tokens && !shmem.TokenSchedule()),
                "Skipping output zero requires the ordinary V2 UDMA communication/reduction path");
    const char* arrival_env = std::getenv("DEEPEP_CLOS_COMBINE_ARRIVAL_PROFILE");
    const bool arrival_requested = arrival_env && std::strcmp(arrival_env, "1") == 0;
    const auto host_now_ns = []() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    };
    const int64_t allocation_begin_ns = arrival_requested ? host_now_ns() : 0;
    auto recv_x = skip_output_zero ? allocator.Empty({num_tokens, hidden}, x.options())
                                   : allocator.Zeros({num_tokens, hidden}, x.options());

    c10::optional<torch::Tensor> recv_topk_weights = c10::nullopt;
    int64_t num_topk = 0;
    if (topk_weights.has_value()) {
        TORCH_CHECK(topk_weights->dim() == 2, "topk_weights must have shape [num_tokens, num_topk]");
        TORCH_CHECK(topk_weights->scalar_type() == torch::kFloat32, "topk_weights must be float32");
        TORCH_CHECK(topk_weights->size(0) == num_recv_tokens, "topk_weights and x must have the same num_recv_tokens");
        num_topk = topk_weights->size(1);
        TORCH_CHECK(num_topk > 0, "topk_weights second dim must be greater than zero");
        TORCH_CHECK(num_topk <= kMaxCombineTopk, "num_topk must not exceed ", kMaxCombineTopk,
                    " for the combine UB layout");
        recv_topk_weights = skip_output_zero ? allocator.Empty({num_tokens, num_topk}, topk_weights->options())
                                            : allocator.Zeros({num_tokens, num_topk}, topk_weights->options());
    }
    const int64_t allocation_end_ns = arrival_requested ? host_now_ns() : 0;
    if (bias_0.has_value()) {
        TORCH_CHECK(bias_0->dim() == 2 && bias_0->size(0) == num_tokens && bias_0->size(1) == hidden,
                    "bias_0 must have shape [num_tokens, hidden]");
        TORCH_CHECK(bias_0->scalar_type() == x.scalar_type(), "bias_0 must have the same dtype as x");
    }
    if (bias_1.has_value()) {
        TORCH_CHECK(bias_1->dim() == 2 && bias_1->size(0) == num_tokens && bias_1->size(1) == hidden,
                    "bias_1 must have shape [num_tokens, hidden]");
        TORCH_CHECK(bias_1->scalar_type() == x.scalar_type(), "bias_1 must have the same dtype as x");
    }

    TORCH_CHECK(num_tokens <= std::numeric_limits<int32_t>::max(), "num_tokens must fit int32 combine indexing");
    TORCH_CHECK(num_recv_tokens <= std::numeric_limits<int32_t>::max(),
                "num_recv_tokens must fit int32 combine indexing");
    TORCH_CHECK(hidden <= std::numeric_limits<int32_t>::max(), "hidden must fit int32 combine indexing");

    const int64_t topk_weights_row_bytes =
        CheckedMulInt64(num_topk, static_cast<int64_t>(sizeof(float)), "combine top-k weights row");
    TORCH_CHECK(notify_workspace_bytes <= static_cast<std::size_t>(std::numeric_limits<int64_t>::max()),
                "Buffer notify workspace reservation exceeds int64");
    const int64_t control_bytes =
        use_udma ? CheckedMulInt64(world_size, kCombineUdmaCompletionSignalBytes, "UDMA control header") : 0;
    TORCH_CHECK(static_cast<int64_t>(notify_workspace_bytes) <= std::numeric_limits<int64_t>::max() - control_bytes,
                "combine workspace base overflows int64");
    const int64_t workspace_base_offset = control_bytes + static_cast<int64_t>(notify_workspace_bytes);
    const CombineWorkspaceLayout workspace = MakeCombineWorkspaceLayout(
        workspace_base_offset, world_size, token_capacity, hidden_bytes,
        topk_weights.has_value() ? topk_weights_row_bytes : 0, use_udma, static_cast<int64_t>(symmetric_buffer_bytes));
    TORCH_CHECK(num_tokens <= workspace.token_capacity, "combine requires capacity for ", num_tokens,
                " original tokens per rank, but the symmetric workspace only provides ", workspace.token_capacity);
    TORCH_CHECK(workspace.token_capacity <= std::numeric_limits<uint32_t>::max(),
                "combine token capacity must fit uint32 tiling");
    TORCH_CHECK(workspace.end_offset <= static_cast<int64_t>(symmetric_buffer_bytes),
                "combine workspace exceeds the symmetric buffer");
    TORCH_CHECK(expanded_x.has_value() == single_source_slots.has_value(),
                "both single-source combine inputs must be provided");
    TORCH_CHECK(!aggregate || (expanded_x.has_value() && use_udma && epilogue != nullptr),
                "early single-source combine requires expanded UDMA inputs and an epilogue");
    ascend_deepep::CombineTilingData tiling{};
    if (expanded_x.has_value()) {
        TORCH_CHECK(use_udma && epilogue != nullptr, "single-source combine requires the UDMA V2 path");
        CheckNpuContiguous(*expanded_x, "expanded_x");
        CheckNpuContiguous(*single_source_slots, "single_source_slots");
        TORCH_CHECK(expanded_x->device() == x.device() && expanded_x->scalar_type() == x.scalar_type() &&
                        expanded_x->dim() == 2 && expanded_x->size(1) == hidden &&
                        expanded_x->size(0) <= std::numeric_limits<int32_t>::max(),
                    "invalid expanded source");
        TORCH_CHECK(single_source_slots->device() == x.device() &&
                        single_source_slots->scalar_type() == torch::kInt32 && single_source_slots->dim() == 1 &&
                        single_source_slots->size(0) == num_recv_tokens,
                    "invalid single-source slots");
        tiling.expanded_x = reinterpret_cast<uint64_t>(expanded_x->data_ptr());
        tiling.single_source_slots = reinterpret_cast<uint64_t>(single_source_slots->data_ptr());
        tiling.num_expanded_tokens = static_cast<uint32_t>(expanded_x->size(0));
    }
    tiling.rank = static_cast<uint32_t>(rank);
    tiling.world_size = static_cast<uint32_t>(world_size);
    tiling.num_tokens = static_cast<uint32_t>(num_tokens);
    tiling.num_recv_tokens = static_cast<uint32_t>(num_recv_tokens);
    tiling.hidden = static_cast<uint32_t>(hidden);
    tiling.num_topk = static_cast<uint32_t>(num_topk);
    tiling.block_dim = static_cast<uint32_t>(resolved_num_aiv);
    tiling.num_channels = static_cast<uint32_t>(num_channels);
    tiling.udma_peer_order = shmem.CombinePeerOrder();
    tiling.udma_ranks_per_group = shmem.RanksPerGroup();
    tiling.udma_workers_per_peer = shmem.WorkersPerPeer();
    tiling.udma_qps_per_worker = shmem.QpsPerWorker();
    tiling.udma_cross_lane0_weight = shmem.CrossLane0Weight();
    tiling.udma_peers_per_worker = static_cast<uint32_t>(udma_peers_per_worker);
    tiling.udma_wqes_per_visit = static_cast<uint32_t>(udma_wqes_per_visit);
    tiling.udma_token_schedule = use_udma && shmem.TokenSchedule();
    tiling.has_topk_weights = topk_weights.has_value() ? 1U : 0U;
    tiling.has_bias_0 = bias_0.has_value() ? 1U : 0U;
    tiling.has_bias_1 = bias_1.has_value() ? 1U : 0U;
    tiling.x_dtype = x.scalar_type() == torch::kBFloat16 ? kCombineDtypeBfloat16 : kCombineDtypeFloat16;
    tiling.x_element_bytes = static_cast<uint32_t>(x_element_bytes);
    tiling.token_capacity = static_cast<uint32_t>(workspace.token_capacity);
    tiling.x_row_stride = static_cast<uint32_t>(workspace.x_row_stride);
    tiling.topk_weights_row_stride = static_cast<uint32_t>(workspace.topk_weights_row_stride);
    tiling.x_buffer_offset = static_cast<uint64_t>(workspace.x_buffer_offset);
    tiling.topk_weights_buffer_offset = static_cast<uint64_t>(workspace.topk_weights_buffer_offset);
    tiling.udma_completion_signal_offset = static_cast<uint64_t>(workspace.udma_completion_signal_offset);
    tiling.combine_workspace_bytes = static_cast<uint64_t>(workspace.total_bytes);

    // Opt-in diagnostics only. Each AIV owns one ordinary-GM row; blocking
    // readback below keeps it alive and never changes SHMEM/output ownership.
    auto diagnostic_enabled = [](const char* name) {
        const char* value = std::getenv(name);
        if (!value || !*value || std::strcmp(value, "0") == 0) return false;
        TORCH_CHECK(std::strcmp(value, "1") == 0, name, " must be 0 or 1");
        return true;
    };
    const bool coarse_profile = diagnostic_enabled("DEEPEP_CLOS_COMM_STAGE_PROFILE");
    const bool detailed_profile = diagnostic_enabled("DEEPEP_CLOS_COMBINE_PROFILE");
    const bool arrival_profile = diagnostic_enabled("DEEPEP_CLOS_COMBINE_ARRIVAL_PROFILE");
    TORCH_CHECK(!arrival_profile || (world_size == 64 && resolved_num_aiv == 32 && !detailed_profile),
                "Combine arrival profiling requires EP64/AIV32 and excludes detailed profiling");
    const bool notify_before_combine = diagnostic_enabled("DEEPEP_CLOS_COMBINE_PRE_NOTIFY");
    if (notify_before_combine) {
        TORCH_CHECK(use_udma && epilogue && !aggregate && !pipeline_chunk_tokens && !shmem.TokenSchedule(),
                    "Pre-combine notify requires the ordinary V2 UDMA communication path");
    }
    torch::Tensor profile_output;
    const uint32_t profile_words = arrival_profile ? ascend_deepep::kCombineArrivalProfileWords :
                                                    ascend_deepep::kDispatchProfileWords;
    if (coarse_profile || detailed_profile || arrival_profile) {
        TORCH_CHECK(c10_npu::currentStreamCaptureStatusMayInitCtx() == c10_npu::CaptureStatus::None,
                    "Combine profiling performs a CPU readback and cannot be graph-captured");
        TORCH_CHECK(use_udma && epilogue && !aggregate && !pipeline_chunk_tokens && !shmem.TokenSchedule(),
                    "Combine profiling requires the ordinary V2 UDMA communication path");
        profile_output = allocator.Empty({resolved_num_aiv, profile_words},
                                         x.options().dtype(torch::kInt64));
        tiling.udma_profile_output = reinterpret_cast<uint64_t>(profile_output.data_ptr());
        tiling.udma_profile_detailed = arrival_profile ? 2U : uint32_t(detailed_profile && !coarse_profile);
    }

    c10::optional<torch::Tensor> pipeline_ready;
    if (pipeline_chunk_tokens) {
        TORCH_CHECK(shmem.QpsPerWorker() == 1, "aggregate pipeline requires one QP per worker");
        TORCH_CHECK(use_udma && epilogue && !aggregate && pipeline_mapping.has_value() && expanded_x.has_value(),
                    "invalid aggregate pipeline mode");
        TORCH_CHECK(!shmem.TokenSchedule() || (resolved_num_aiv == 32 && world_size <= 8),
                    "self-send aggregate pipeline requires 32 AIV and EP <= 8");
        TORCH_CHECK(pipeline_chunk_tokens == 32 || pipeline_chunk_tokens == 64, "invalid pipeline chunk size");
        TORCH_CHECK(resolved_num_aiv == 32 || resolved_num_aiv == 64, "pipeline requires 32/64 AIV");
        TORCH_CHECK(
            shmem.TokenSchedule() ||
                shmem.WorkersPerPeer() * ascend_deepep::udma::Workers(world_size, udma_peers_per_worker,
                                                                      shmem.CombinePeerOrder(), shmem.RanksPerGroup()) <
                    resolved_num_aiv,
            "aggregate pipeline needs at least one producer core");
        TORCH_CHECK(num_recv_tokens <= std::numeric_limits<int32_t>::max() - 64,
                    "pipeline receive rows exceed signed device range");
        CheckNpuContiguous(*pipeline_mapping, "pipeline_mapping");
        TORCH_CHECK(pipeline_mapping->scalar_type() == torch::kInt32 && pipeline_mapping->dim() == 2 &&
                        pipeline_mapping->size(0) == num_recv_tokens && pipeline_mapping->size(1) > 0 &&
                        pipeline_mapping->size(1) <= 128,
                    "invalid aggregate pipeline mapping");
        TORCH_CHECK(pipeline_values.has_value() == topk_weights.has_value(), "pipeline values mismatch");
        if (!shmem.TokenSchedule()) {
            const int64_t chunks = (num_recv_tokens + pipeline_chunk_tokens - 1) / pipeline_chunk_tokens;
            const int64_t ready_bytes = CheckedMulInt64(std::max<int64_t>(chunks, 1), 512, "pipeline ready bytes");
            pipeline_ready = allocator.Empty({(ready_bytes + 511) / 8}, x.options().dtype(torch::kInt64));
        }
        allocator.Record({pipeline_mapping, pipeline_values, pipeline_ready});
        tiling.pipeline_mapping = reinterpret_cast<uint64_t>(pipeline_mapping->data_ptr());
        tiling.pipeline_values = pipeline_values ? reinterpret_cast<uint64_t>(pipeline_values->data_ptr()) : 0;
        tiling.pipeline_ready =
            pipeline_ready ? (reinterpret_cast<uint64_t>(pipeline_ready->data_ptr()) + 511U) / 512U * 512U : 0;
        tiling.pipeline_chunk_tokens = static_cast<uint32_t>(pipeline_chunk_tokens);
        tiling.pipeline_topk = static_cast<uint32_t>(pipeline_mapping->size(1));
    }
    const auto ffts_addr = shmem.FftsAddr();
    // Capture the prepared tensors/tiling by value once; no recomputation or
    // allocation between communication and reduction, and no host synchronize.
    auto launch = [=](CombinePhase phase) {
        // Explicit odr-use retains GM source owners, not just tiling addresses.
        (void)expanded_x;
        (void)single_source_slots;
        (void)pipeline_mapping;
        (void)pipeline_values;
        (void)pipeline_ready;
        // Experimental entry alignment: all ranks launch the same AIV count.
        // Queue directly before communication, with no host wait in between.
        // Begin still resets payload flags and performs its own collective.
        if (notify_before_combine && phase == CombinePhase::Communication)
            LaunchElasticBarrierKernel(resolved_num_aiv, ffts_addr);
        const int64_t launch_begin_ns = arrival_profile ? host_now_ns() : 0;
        LaunchV1CombineKernel(x, topk_weights, bias_0, bias_1, src_idx, rank_prefix_matrix, channel_prefix_matrix,
                              send_head, recv_x, recv_topk_weights, tiling, symmetric_buffer, ffts_addr, use_udma,
                              phase);
        const int64_t launch_end_ns = arrival_profile ? host_now_ns() : 0;

        if (profile_output.defined() && phase == CombinePhase::Communication) {
            const auto cpu = profile_output.cpu();
            const auto* values = cpu.data_ptr<int64_t>();
            if (arrival_profile) {
                std::ostringstream host;
                host << "[CLOS_COMBINE_HOST] rank=" << rank << " weighted=" << tiling.has_topk_weights
                     << " allocation_begin_ns=" << allocation_begin_ns << " allocation_end_ns=" << allocation_end_ns
                     << " launch_begin_ns=" << launch_begin_ns << " launch_end_ns=" << launch_end_ns << '\n';
                const auto message = host.str();
                AppendCombineProfileLog(message);
            }
            constexpr const char* names[] = {
                "begin_ns", "metadata_ns", "setup_ns", "plan_ns", "send_ns", "build_ns", "credit_ns",
                "submit_ns", "publish_ns", "incoming_ns", "quiet_ns", "barrier_ns", "refills", "hits",
                "wqes", "batches", "origin_cycle", "remote_wait_ns", "local_sync_ns", "credit_checks",
                "credit_stalls", "send_records", "scale_pack_ns", "scale_submit_ns", "scale_wait_ns",
                "self_ns", "output_ns", "total_ns", "kernel_entry_cycle", "begin_entry_cycle",
                "reset_done_cycle", "sync_entry_cycle", "local_enter_cycle", "local_exit_cycle",
                "generation_ready_cycle", "round0_send_cycle", "round0_wait_cycle", "round0_end_cycle",
                "round0_source_plus1", "round1_send_cycle", "round1_wait_cycle", "round1_end_cycle",
                "round1_source_plus1", "sync_exit_cycle", "sync_radix", "sync_phases", "sync_generation",
                "arrival_reserved"};
            for (int64_t core = 0; core < resolved_num_aiv; ++core) {
                std::ostringstream line;
                line << "[CLOS_COMBINE_PROFILE] rank=" << rank << " core=" << core
                     << " weighted=" << tiling.has_topk_weights << " detailed=" << tiling.udma_profile_detailed;
                const uint32_t named_words = arrival_profile ? sizeof(names) / sizeof(names[0]) : 28U;
                for (uint32_t word = 0; word < named_words; ++word)
                    line << ' ' << names[word] << '=' << values[core * profile_words + word];
                const auto message = line.str() + '\n';
                AppendCombineProfileLog(message);
            }
        }

        return CombineResult{recv_x, recv_topk_weights};
    };
    if (epilogue != nullptr) {
        *epilogue = c10::make_intrusive<CombineEpilogue>([launch] { return launch(CombinePhase::Reduction); });
        if (pipeline_chunk_tokens) {
            launch(CombinePhase::Pipeline);
        } else if (aggregate) {
            launch(CombinePhase::EarlySingles);
            aggregate();
            launch(CombinePhase::Remaining);
        } else {
            launch(CombinePhase::Communication);
        }
    } else {
        launch(CombinePhase::Fused);
    }

    return {recv_x, recv_topk_weights};
}

}  // namespace ascend_deepep::ops
