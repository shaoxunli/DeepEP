#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>

#include <torch/extension.h>

class ShmemInstance;

namespace ascend_deepep::ops {

using DupServerNotifyPrepareResult = std::tuple<torch::Tensor,                 // send_token_count
                                                torch::Tensor,                 // send_token_ids
                                                torch::Tensor,                 // send_gateway_lane
                                                torch::Tensor,                 // send_route_count
                                                torch::Tensor,                 // send_expert_prefix
                                                torch::Tensor,                 // packed_route_position
                                                torch::Tensor,                 // packed_gateway_lane
                                                torch::Tensor,                 // source_server_mask
                                                c10::optional<torch::Tensor>,  // packed_topk_weights
                                                torch::Tensor,                 // recv_route_position
                                                torch::Tensor,                 // recv_gateway_lane
                                                torch::Tensor,                 // recv_segment_offset
                                                torch::Tensor,                 // recv_weight_segment_begin
                                                torch::Tensor,                 // recv_chunk_token_end
                                                torch::Tensor,                 // gateway_route_count
                                                torch::Tensor,                 // gateway_route_position
                                                torch::Tensor,                 // gateway_slot_mask
                                                torch::Tensor>;                // notify_status

DupServerNotifyPrepareResult DupServerNotifyPrepare(torch::Tensor topk_idx, c10::optional<torch::Tensor> topk_weights,
                                                    int64_t num_experts, int64_t token_capacity,
                                                    int64_t ranks_per_server, int64_t num_tokens_per_chunk,
                                                    int64_t num_aiv, int64_t gateway_policy, int rank, int world_size,
                                                    int num_device_aiv, bool enable_timer,
                                                    std::size_t symmetric_buffer_bytes, ShmemInstance& shmem,
                                                    void* symmetric_buffer);

void DupServerNotifyScaleOut(int64_t token_capacity, int64_t hidden, int64_t ranks_per_server,
                             int64_t num_tokens_per_chunk, int64_t num_aiv, int64_t gateway_policy, int rank,
                             int world_size, int num_device_aiv, std::size_t symmetric_buffer_bytes,
                             ShmemInstance& shmem, void* symmetric_buffer);

torch::Tensor DupServerDispatchScaleOut(torch::Tensor x, torch::Tensor send_token_count, torch::Tensor send_token_ids,
                                        torch::Tensor send_gateway_lane, int64_t ranks_per_server,
                                        int64_t max_inflight_same_frame_servers, int64_t num_tokens_per_chunk,
                                        int64_t num_aiv, bool enable_timer, int64_t gateway_policy, int rank,
                                        int world_size, int num_device_aiv, std::size_t symmetric_buffer_bytes,
                                        ShmemInstance& shmem, void* symmetric_buffer);

torch::Tensor DupServerDispatch(torch::Tensor x, torch::Tensor send_token_count, torch::Tensor send_token_ids,
                                torch::Tensor send_gateway_lane, torch::Tensor recv_route_position,
                                torch::Tensor recv_gateway_lane, torch::Tensor recv_src_rank,
                                torch::Tensor recv_segment_offset, int64_t num_topk, int64_t ranks_per_server,
                                int64_t max_inflight_same_frame_servers, int64_t num_tokens_per_chunk, int64_t num_aiv,
                                bool use_urma_get, bool enable_timer, int64_t gateway_policy,
                                int64_t remote_get_servers, bool remote_get_rx_tx_gate, int rank, int world_size,
                                int num_device_aiv, std::size_t symmetric_buffer_bytes, ShmemInstance& shmem,
                                void* symmetric_buffer);

}  // namespace ascend_deepep::ops
