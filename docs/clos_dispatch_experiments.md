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
with no payload work. The cache does not change combine or the token schedule.

## Diagnostic timing

For a short diagnostic run, set `DEEPEP_CLOS_DISPATCH_PROFILE=1`. Each AIV emits
one private 256-byte GM row per dispatch invocation, including empty payload
AIVs. At most 16 KiB is allocated through the existing stream allocator and
remains live until the CPU readback finishes. After the final communication
collective and output-copy drain, each AIV reuses the drained WQE scratch for its own record, orders
scalar writes with `S_MTE3`, and drains the GM store with `MTE3_S`. The Host
emits `[CLOS_DISPATCH_PROFILE]` lines after readback, independent of device
printf support. `cached=0/1` distinguishes fresh/cached dispatch. Ascend950's
system counter is 1 GHz; fields ending in `_ns` contain system-counter deltas.

Top-level intervals are `begin_ns`, `metadata_ns`, `setup_ns`, `plan_ns`,
`send_ns`, `publish_ns`, `incoming_ns`, `quiet_ns`, `barrier_ns`, and `output_ns`.
`output_ns` covers the kernel's output-copy/handle tail after the final
communication collective. `total_ns` spans Begin entry through this tail, before
record storage; it is not the entire API. Scale packing, top-k initialization, and output address
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

### Matched dispatch/combine stages

Set `DEEPEP_CLOS_COMM_STAGE_PROFILE=1` for coarse timing of both operators.
It takes precedence over detailed profiling and skips per-batch timestamps and
the additional SIMT `V_S` drain. For detailed timing instead, leave that switch
off and set `DEEPEP_CLOS_DISPATCH_PROFILE=1` and
`DEEPEP_CLOS_COMBINE_PROFILE=1`. These switches default off, accept only `0/1`,
and perform blocking CPU readback; graph capture is unsupported. Use the same
settings on every rank. Keep dispatch tracing off for the matched coarse test.

Combine profiling supports ordinary V2 UDMA communication, excluding token
and aggregate-pipeline schedules. Rebuild both host and device libraries for
the new combine tiling fields. `[CLOS_COMBINE_PROFILE]` records use the same
32-word per-AIV layout; `weighted=0/1` identifies the variant. `self_ns` covers
the self-rank copy and its final DMA drain. `setup_ns` includes optional SIMT
weight issue before constructing the payload sender. `total_ns` stops before
record storage and before the separate V2 reduction kernel.

Both operators record `remote_wait_ns` and `local_sync_ns` separately inside
`incoming_ns`; do not count these twice. `send_ns` covers the scheduler/payload
loop and `publish_ns` includes any final staged batch plus terminal markers.
Neither means the payload has finished transferring: the existing incoming
wait, Quiet and final synchronization still establish completion.

Keep raw records per invocation/rank/AIV. Compare one complete core record at
a time, not a sum of independently selected stage maxima. Compare diagnostic
kernel distributions with profiling disabled; CPU readback changes host pacing
and detailed timing adds synchronization. Do not attribute a performance gap
to address layout from these timers alone.

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

## Independent peer cursors

`DEEPEP_CLOS_DISPATCH_PEER_CURSOR=1` opts into independent per-peer cursors.
Default is `0`. Each visit still issues at most the configured WQE count and
uses the existing sender/QP assignment and unsignaled batch protocol. A peer
advances through its immutable channel lists without waiting for the other
peers to finish the same input-token tile. Unweighted channel ownership and
weighted whole-peer row intersections are unchanged. No extra payload is sent.

With this mode, `send_ns` includes cursor advancement and channel-prefix reads;
compare the full kernel or `plan_ns + send_ns`, rather than treating a reduced
`plan_ns` alone as a speedup. Cache and cursor switches are independent; start
cursor comparisons with the WQE cache disabled.

## Completion observations

`DEEPEP_CLOS_DISPATCH_TRACE=1` implies `PROFILE=1` and adds per-peer/QP and
per-incoming-marker records. It is off by default and rejects graph capture.
The ordinary GM stride in u64 words per AIV is
`32 + world*qps*12 + ceil(world*workers*qps/cores)*8`; each record is 32-byte
aligned. At most 128 incoming slots per AIV are supported. Allocation/readback
lifetime follows the execution stream, with no added SHMEM region. Only
initialized owned records are read, including skips for self markers.

The summary adds `remote_wait_ns` and `local_sync_ns`, splitting the existing
`incoming_ns` interval; `credit_checks` and `credit_stalls` count SQ admission
checks and actual pressure checkpoints. They do not add new waits or CQEs.
`origin_cycle` is the per-AIV system counter immediately after Begin.

`CLOS_DISPATCH_SEND` records peer, physical QP, first/last payload posting,
SO posting, Quiet return, WQE/batch counts, pressure count/time, and an SQ
alias index local to that AIV. A batch that contains several peers contributes
one batch observation to each peer; do not sum these as global doorbell counts.
Peers sharing an SQ inherit the same Quiet observation, not separate CQEs.
`CLOS_DISPATCH_RECEIVE` records the first poll and ready observation for each
source/worker marker, plus poll count. Offsets are relative to the AIV origin.

These are software observations. Quiet follows incoming waits and therefore
bounds completion from above; it does not timestamp NIC completion. A marker
ready on its first poll arrived at or before the observation. Sequential
polling can delay observation of later sources. Compare clocks only within one
device; never subtract sender and receiver clocks on different ranks. Trace
collection perturbs scheduling and must not provide headline bandwidth.

The cursor CPU regression executes the actual scheduler against independently
calculated output-row ownership, covering empty and skew peers/channels,
partial sends, payload/scales, weighted/unweighted lanes and varied visits.
NPU precision and performance evidence must still be collected separately.

## Huge-page receive placement

`DEEPEP_CLOS_DISPATCH_RECV_WORKSPACE=1` redirects dispatch payload and scale
receive targets to the existing 1 GiB-page workspace, then copies them to owned
output tensors on the execution stream. Default is `0`. This is a diagnostic
experiment, not a zero-copy API: metadata destinations, kernel code, tiling ABI,
peer row ownership, and completion protocol are unchanged. It requires the
UDMA peer schedule and huge pages; zero-copy outputs, graph capture, and
workspace-backed payload inputs are rejected before notification.

The existing checked layout reserves disjoint, 32-byte-aligned payload and
scale regions for the common receive capacity. The actual output count cannot
exceed this capacity. Remote writes and local copies use the existing output
mailbox and row ownership. Kernel completion, including incoming completion,
Quiet, and epilogue, precedes the materialization copies on the same stream.
The public completion event follows the copies. Subsequent serialized calls
and their entry rendezvous protect workspace reuse; returned tensors retain
ordinary allocator ownership and survive later dispatch/combine calls. No new
UB storage or device synchronization is introduced.

One `CLOS_DISPATCH_RECV_WORKSPACE` record per process confirms the enabled
path. The bandwidth benchmark checks that an earlier fresh result survives a
cached dispatch with different payload. Compare identical huge-page settings
with this switch off/on. Report both communication kernel duration and the
additional materialization cost (`other_us` in the profiler), as well as the
device/API interval where available. A shorter communication kernel alone is
not an end-to-end speedup, and this comparison changes receive placement as
well as the memory mapping; it does not isolate page size alone.

## SIMT scale transport

`DEEPEP_CLOS_DISPATCH_SIMT_SCALES=1` opts into SIMT remote stores for dispatch
scales. Default is `0`; set it consistently on every rank. It requires the UDMA
peer schedule and 32-bit scale elements; calls without scales keep their normal path. FP32 and int32 scales
are transferred bit for bit, using the existing input packing for strided
scales. Payload remains on UDMA. The dispatch row scheduler and WQE cache omit
the scale WQE when this switch is enabled.

Metadata AIVs own complete logical channels and write scales to the existing
symmetric receive region. Self rows retain the MTE copy. The existing final
collective completes the SIMT writes before each AIV copies its disjoint valid
scale rows to the returned tensor. No extra cross-rank collective or workspace
allocation is added. The ordinary output tensor retains its own storage across
subsequent calls. Zero-copy outputs already alias this region and need no copy.

This changes both the transport and the scale receive placement. Compare the
same binary with the switch off/on, keeping page size, payload placement,
schedule, routing and shape fixed. The final scale copy is inside the dispatch
communication kernel and must remain in its timing. Bandwidth includes the
actual deduplicated payload and scale bytes, including self; it is not a pure
SIMT-link bandwidth measurement. This experimental switch does not change
combine or the default automatic policy.

## Manual cross-server lane weights

`udma_cross_lane0_weight` accepts integer weights from 1 through 8. The default
remains 1; the manual bandwidth benchmark keeps its previous default of 3 and
accepts the range through `--tuning-cross-lane0-weight`. Manual non-unit weights require two workers per peer, one QP per worker,
and an explicit contiguous physical-server group size. The EP64 POD automatic
policy selects weight 4 after validating eight contiguous ranks per host; other
topologies retain weight 1. See the [policy contract](machine_backends.md#全-clos-的默认调度与手动调优). All
ranks must agree on this immutable constructor setting during bootstrap.

For a cross-server peer with `N` actual receive rows, lane 0 owns
`[0, floor(uint64(N) * weight / (weight + 1)))` and lane 1 owns the remainder.
Within the same server the weighted path retains a 1:1 split. Weight 1 retains
the existing channel-based scheduling path. Empty lanes still publish their
completion markers. This changes row ownership between the two workers, not
the output row order, payload bytes, QP allocation, workspace layout, or
completion protocol. Compare effective configurations and verify cross-server
precision; a single-server run cannot validate the non-unit cross-server split.


## Automatic packed-scale selection

The EP64 POD row-byte policy can enable the existing packed-scale overlap without
setting experimental environment variables. It requires owned mailbox outputs,
1 GiB-page workspace, at least 32 AIVs, one-byte payload with 32-bit scales,
payload row width >= 4096 bytes and payload plus scale row width <= 8192 bytes.
Before enabling it, the host checks peer staging capacity, isolated ready-flag
capacity, absence of workspace-backed inputs, and graph/zero-copy restrictions.
Unsupported calls use the original scale path. This does not enable WQE cache,
peer cursor, receive-workspace copies or diagnostic profiling.

Enabled manual scale switches take precedence over automatic selection and retain their
existing validation. `udma_auto_policy=False` disables automatic packed scales.
`CLOS_SCALE_PIPELINE` records `automatic`, `active`, producer count and AIV count
when the pipeline is selected. Host row-byte policy affects visit only; the
existing kernel producer publication, incoming completion and Quiet remain intact.
