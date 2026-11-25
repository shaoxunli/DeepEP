# Dispatch output-only consumer stream recording experiment

Base: `5276899c5108fce5c7f049e2436e96ae3d04ec42`, the installed library used
in the EP4 allocator lifetime probe. This branch changes Python stream tracking
only. Native kernels, allocation streams, sending policies, clone behavior and
resource ownership remain unchanged.

Direct/Global Clos dispatch now records only returned `recv_x`, `recv_topk_idx`
and `recv_topk_weights` on the caller stream and on subsequent event-wait streams.
For FP8, the scales are part of returned `recv_x` and remain protected as well.
`handle.num_recv_tokens_per_expert_list` remains an ordinary CPU list; it requires
no stream registration. Both fresh/cached and expanded/non-expanded dispatch use
this output-only consumer selection.

Inputs and directly tracked temporary resources retain communication-stream
protection. The returned event continues to own all the same resources, including
the handle. Early event destruction still uses the existing producer registrations.
Combine, Direct V1, Mesh+Clos and public EventOverlap construction retain their
previous behavior; no new public parameter or environment switch is introduced.

Other handle tensors no longer receive automatic caller/consumer-stream recording
from dispatch. This is an experimental contract: if application code reads those
NPU tensors on a compute stream, it must protect their lifetime on that stream.
Reading the CPU expert-count list and creating a new tensor from it does not read
the original NPU count tensor. Device waits alone do not protect later consumers
against premature memory reuse.

## Remote validation

Compare this branch against the base above, using the same Torch/torch_npu, SHMEM,
device placement and routing. Use EP4, tokens=4096, hidden=2816, topk=8,
experts=256, BF16 with FP32 routing weights, AIV32, expand=0 and FC1 width1536.
Explicitly pass `allocate_on_comm_stream=True` in BOTH versions to match the
customer setting. The original allocator probe omitted it and used False.

Run the normal-release path (A) of the allocator-lifetime harness on both builds;
do not confuse the library comparison with its A/B lifetime intervention. Keep
one Buffer per rank, warmup5 and20 measured calls. Capture4 separate profiler
calls per rank. Record actual loaded Python hashes and verify native library
hashes match between builds.

Report Host dispatch duration, old-result release duration, FC1 Host submission,
allocation time, allocator record/query counts when visible, and full window time
including cleanup. Keep formal and profiler timing separate. Preserve outliers.

Validate fresh and cached output values, weights and expert counts against the
existing independent reference; exercise combine reuse of the handle. Also check
event discard/release, multiple consumer streams for the returned data, and old
handle disposal under allocator churn. Check an expanded case for regression.
Do not claim precision or asynchronous lifetime safety from CPU mocks alone.
