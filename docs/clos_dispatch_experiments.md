# Clos dispatch sender experiments

These opt-in experiments target the UDMA **peer** schedule. Both switches
default to `0`; unset or empty values also disable them. `1` enables them.
Other values are rejected. Set the same
environment on every rank before calling dispatch. The cache flag is captured
with the launch tiling during graph capture; changing the environment does not
alter an already captured graph. Diagnostic profiling rejects graph capture
because it explicitly reads counters back to the CPU.

```bash
export DEEPEP_CLOS_DISPATCH_WQE_CACHE=1
export DEEPEP_CLOS_DISPATCH_PROFILE=0
```

The cache does not change row ownership, peer rotation, visit limits, QP choice,
weighted lane boundaries, unsignaled payload submission, or completion markers.
It is experimental, not an assertion of improved performance. Measure both
settings in the same build, and also compare the disabled setting with the
original binary to check instrumentation/state overhead.

## Cache ownership and lifetime

Each payload AIV owns eight direct-mapped entries of 32 WQEs (16 KiB of UB).
The cache key checks the local peer/QP index, next GM plan pointer, destination
row, field, and unconsumed length. A sender instance fixes source/output bases;
entries never survive a kernel invocation. Prefetch is bounded by the current
tile's remaining WQEs. FP8 payload and scale are separate WQEs; a 32-WQE entry
is not necessarily 32 tokens.

The UB ledger, in bytes, is:

| Region | Offset / size |
|---|---|
| Aggregate WQE batch | 0 / 2048 |
| Aggregate span descriptors | 2048 / 1024 |
| Two SDK headers per peer/QP | 3072 / `Capacity * Qps * 128` |
| Output descriptors | after headers / `Capacity * Qps * 64` |
| Epoch / refill descriptor | after outputs / 32 |
| Optional WQE cache | after refill descriptor / 16384 |
| Token list windows | after cache / `Capacity * 256 * 4` |

All starts and strides are 32-byte aligned. The largest supported template
(`Capacity=64`, `Qps=2`) occupies 109600 bytes, checked at compile time against
the 216 KiB arena. Metadata and output-copy phases reuse the arena only after
their established completion dependencies.

Scalar refill descriptors precede SIMT construction through `S_V`; `V_S`
completes construction before the entry becomes available. A pending-reader
bitset prevents replacement of an entry referenced by an aggregate batch.
That batch is gathered by SIMT, ordered through `V_MTE3`, and copied to the SQ
before reuse. The existing submission helper drains both wrapped SQ copy spans
before publishing the doorbell. SQ checkpoints can overwrite the aggregate
buffer, but cannot overwrite cache entries. Source payload remains live until
the existing terminal-marker and quiet protocol completes.

The cache needs no GM allocation or symmetric workspace layout changes. Its
flags use the previously reserved 32-bit tiling word. Diagnostic profiling adds
an optional GM pointer to the tiling ABI; rebuild host and device libraries
together. Every launched AIV still participates
in Begin, incoming synchronization, and the final collective, including AIVs
with no payload work. Combine and the token schedule remain unchanged.

## Diagnostic timing

For a short diagnostic run, set `DEEPEP_CLOS_DISPATCH_PROFILE=1`. Each AIV emits
one private 256-byte GM row per dispatch invocation, including empty payload
AIVs. At most 16 KiB is allocated through the existing stream allocator and
remains live until the CPU readback finishes. After the final communication
collective, each AIV reuses the drained WQE scratch for its own record, orders
scalar writes with `S_MTE3`, and drains the GM store with `MTE3_S`. The Host
emits `[CLOS_DISPATCH_PROFILE]` lines after readback, independent of device
printf support. `cached=0/1` distinguishes fresh/cached dispatch. Ascend950's
system counter is 1 GHz; fields ending in `_ns` contain system-counter deltas.

Top-level intervals are `begin_ns`, `metadata_ns`, `setup_ns`, `plan_ns`,
`send_ns`, `publish_ns`, `incoming_ns`, `quiet_ns`, and `barrier_ns`. They cover
the measured path through the final communication collective, not the entire
API or output epilogue. Scale packing, top-k initialization, and output address
preparation before Begin are also outside these intervals. Begin includes
marker reset and the entry collective;
incoming includes the existing local AIV synchronization. Descriptor setup
includes sender construction and output descriptor checks. Metadata helpers
can run asynchronously; their recorded issue time is not device completion.

`build_ns`, `credit_ns`, and `submit_ns` are nested sender counters inside
`send_ns` and the final flush in `publish_ns`. Do **not** add them to the
top-level intervals. Credit includes SQ checks and pressure checkpoint waits;
In diagnostic mode, an additional `V_S` pair drains SIMT before the scalar
construction timer stops; `V_MTE3` alone only orders the copy engine. This
drain is absent from performance mode. Submit covers UB-to-SQ copy and
doorbell issue, not payload delivery. `refills`,
`hits`, `wqes`, and `batches` diagnose cache reuse and short submissions.

Counter-store overhead is outside these counters but inside kernel duration;
The diagnostic SIMT drain, CPU readback, and logging also perturb overlap and
subsequent arrivals. Profiling runs must
not supply headline bandwidth.
Keep `PROFILE=0` for bandwidth A/B runs. Analyze sender and metadata AIVs
separately; do not sum independent per-stage rank/core maxima into an API time.

## Validation matrix

The host regression executes the actual WQE builders and sender methods against
an independent expected-address sequence. It covers cache hits and collisions,
short tails, payload/scale alternation, visits 4/8/16/32, one/two QPs, shared and
distinct SQ states, repeated invocations, and checkpoint scratch overwrites.
The existing bounded-queue regression separately covers SQ/producer-counter
wrap and completion checkpoints. These are CPU checks, not NPU precision proof.

Before enabling the cache by default, run NPU precision and repeated fresh/
cached dispatch with BF16 and FP8, masked and uneven routes, empty peers, and
dispatch/combine reuse. For performance, first compare equal cross-server
payloads with equal sender cores, huge pages, and L2 settings, then repeat full
fresh/cached dispatch to detect costs shifted between stages.
