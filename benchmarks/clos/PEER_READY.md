# Experimental peer-ready protocol

Branch: `shaoxunli-peer-ready-notify`, based on `0ba82ce`.
Enable with `DEEPEP_CLOS_PEER_READY=1`; unset or `0` retains the collective protocol.
Rebuild **both host and device** libraries (tiling ABI changed). This is opt-in
until CANN compilation, device precision, skew stress and performance pass.

## Contract and supported path

- All ranks execute the same ordered sequence of dispatch/combine calls on one
  Buffer's serialized communication stream, with one ordered host caller. Epoch advances per actual API call,
  not per training step or handle. Full MoE recomputation adds cached dispatch /
  combine calls; expert-only recomputation adds no communication calls.
- Ranks may start arbitrarily late within the timeout. They may not reorder or
  omit calls. Epoch does not detect a mismatched operator/handle sequence.
- Ordinary BF16 UDMA peer schedule only. Token schedule, graph capture, FP8,
  zero-copy outputs, and expanded combine using single-source/early-single/
  aggregate pipeline inputs are rejected. The validation script uses expand=0.
  Inputs backed by the communication workspace are not supported. Do not call
  the private split epilogue API and launch another operation before reduction.
- A mode switch requires drained streams and an external all-rank rendezvous;
  the benchmark does this between A/B modes. Never toggle per rank independently.
- Existing send ownership, rank/RR/group/staggered order, P/W/Q, visit, metadata
  and self-copy AIV assignment, communication and reduction core counts stay
  unchanged. Unready peers are skipped within that order and revisited. Existing
  channel/list-window boundaries remain: a late peer can still delay advancement
  to the next window. This does not promise whole-message progress to every
  ready peer ahead of every late peer.
- No implicit rescheduling of metadata to new cores. With no spare metadata
  AIVs, the existing metadata-before-payload phase can still delay sends. Fresh
  metadata scatters are gated per destination; this can add scanning/VF overhead.

## Memory and ordering ledger

Per Buffer, reuse the independent, zero-initialized 2 MiB `comm_meta` allocation.
It is **not** the payload workspace or dispatch-output mailbox. Each rank stores:

| Table | Address relative to comm_meta | Writer / consumer |
|---|---|---|
| Ready | `(0*EP + peer)*512` | peer writes its readiness; local senders poll |
| Metadata done | `(1*EP + source)*512` | source writes after SIMT completion; local receiver polls |
| UDMA done | `(2*EP + source)*512 + lane*8` | source SO marker per worker/QP; local receiver polls |

Capacity is `3*EP*512` bytes; each source has up to 64 UDMA completion words.
512-byte isolation prevents unrelated publishers sharing a control cache line.
No per-iteration clear: epochs are monotonically increasing 64-bit values, with
host overflow rejection. Control notifications use MTE3 to mapped peer memory,
not a payload SQ or remote load loop. There are two small pushes per peer per
call (ready and metadata completion), plus existing per-QP payload completion
writes. Long launch gaps increase **local** polling, not remote reads.

UB uses existing resource scratch at byte zero: at most 512 bytes in Begin, 8
bytes in each control publisher. S_MTE3 / MTE3_S protect scalar writes and MTE3
reads. Begin finishes before sender initialization. Final notification reuses
scratch only after all sender Flush/Publish staging readers have drained.
Sender templates/descriptors and list-window offsets remain unchanged. Ready
caches and emitted-peer bitsets are invocation-local scalar arrays.

1. Finish previous operation's local consumers on the same stream. Initialize
   receive metadata and publish this invocation's output descriptors. Drain
   SIMT/MTE writes, then all local AIVs join before publishing ready(epoch).
2. Sender reads **its own** ready[peer] slot, with cache invalidation. Only ready
   peers receive payload or metadata. Output mailbox descriptors are loaded
   lazily after that peer is ready, never eagerly from an unready peer.
3. Payload SO markers carry epoch, including empty peers. Keep each sender's
   Quiet: receipt readiness does not mean source buffers are released.
4. VFs retain their fences and V_S drains. Every AIV executes DataSyncBarrier ALL
   and a local join before delegated MTE3 metadata-done pushes. Receivers poll
   their own payload and metadata completion tables and join locally.
5. Only after local copying/reduction and stream ordering may the next call
   announce ready. Next-epoch writes cannot overwrite this epoch's receive data
   or done marker before that receiver announces next ready. A newer ready value
   is safe; payload completion is checked against the exact epoch.

Fresh notify retains its existing publication collective. Its old entry
collective is disabled in this mode: previous operation completion already
requires every peer to have entered its communication phase (and therefore
finished the preceding notify), and the new ready gate protects payload reuse.
Cached notify launch is omitted. Communication entry/tail global collectives
are replaced with the above protocol; local SyncAll and transfer completion
remain. Startup, teardown and benchmark preparation barriers are outside this
claim. Separate combine profiling notify is rejected in this mode because it
would reintroduce a global entry barrier.

The CPU interleaving model checks ownership/epoch reuse under delayed messages;
it cannot prove hardware cache, MTE or SIMT visibility. In particular the
cross-pipeline fence-to-notification ordering must pass NPU skew stress before
using this mode for training. No end-to-end speedup is assumed.

## Remote validation

One process group and one Buffer per rank for the entire A/B/A sweep:

```bash
torchrun --standalone --nproc_per_node=8 benchmarks/clos/bench_peer_ready.py \
  --tokens 4096 --hidden 2816 --experts 256 --topk 8 --num-sms 8 \
  --recompute none full --skew-us 0 200 2000 --flush-l2 --profile \
  --warmup 10 --samples 30 --output results/peer_ready_ep8
```

Use `--no-huge-pages` only if matching the baseline environment. On multiple
nodes provide normal torchrun rendezvous settings and consistent control/SHMEM
ports. Start with `--precision-only`; then repeat the full command with a new
output directory. The script checks native protocol version before Buffer init.

Precision is mandatory: independent CPU BF16 payload and FP32 weight references,
changing payload each invocation, no weighting of BF16 payload, random/masked/
hot/empty routes, one empty source, repeated handles, forward/backward and full
recompute, rotating late ranks. No precision checks run inside performance
samples. Expert-only recompute has the same communication sequence as none;
this script does not implement FFN computation or autograd.

Performance uses ABBA (`0,1,1,0`), identical routes and policies, and separates:

- Unprofiled sequences: clear L2 then external alignment before sequence start;
  inject measured host delays before selected calls, with no intermediate
  all-rank barrier. Report per-rank samples and max-rank sequence completion
  window including final rendezvous. These are imbalance/latency metrics, not
  link bandwidth. Host sleep is not cycle-precise; actual delay is recorded.
- Optional independent profiler pass: clear L2 then align before **each** API,
  use two guard samples at each end. Report communication, layout, notify and
  reduce separately. Logical bandwidth is deduplicated BF16 rows including self
  / communication kernel time; weights and indices are excluded. Combine has no
  separate notify launch here (zero means absent; embedded waits are in comm).
  Do not add separately aggregated stages to infer full API time.

Record DeepEP SHA, SHMEM SHA and loaded library hashes on every node, topology,
large-page state, clocks and external jobs. `manifest.json`, raw rank JSON and
`summary.json` retain configuration and sample IDs. Rebuild/install success and
all-rank precision must precede any performance claim.

## Local validation at implementation time

Windows CPU checks: targeted protocol/fused-notify/copy/API compatibility suite
340 passed, 5 skipped (compiler unavailable on default PATH). A broader API
run passed 514 tests, with 237 skips for missing torch/NPU/compiler or existing
unsupported cases. Explicit C++17 compilation of the production control helper
bodies and the existing cached-dispatch/copy oracle succeeded; executing those
binaries was blocked by Windows application control (4551). Ruff, CLI help and
whitespace checks passed. These are not CANN compilation or device results.
