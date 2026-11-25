# Dispatch allocator lifetime probe

This experiment uses the installed DeepEP wheel. It changes no kernel,
record_stream call, allocator setting, or completion protocol. It does not require
recompilation. Run from this benchmark directory, with PYTHONPATH configured to use
the intended installed wheel rather than a different source checkout. Check the
recorded deep_ep_file and dispatch_file before accepting the results.

```bash
git fetch origin dispatch-allocator-lifetime-probe
git switch --track origin/dispatch-allocator-lifetime-probe
cd benchmarks/clos/allocator_lifetime
```

## Intervention

- A: after fresh dispatch returns, release the preceding dispatch result before FC1.
- B: at the same point, retain the preceding result until the bounded window ends.
- The retained tuple includes receive tensors, handle (and its referenced metadata),
  and event. Fixed input payload/routes/weights remain alive in both groups.
- Each window starts with a priming dispatch result. Neither group forces garbage
  collection or empties the allocator cache. There is no Host synchronization between
  dispatch and FC1. After FC1 submission, queue the dispatch event wait and drain both
  streams. Align ranks before each subsequent sample, outside timing.
- B retains at most window-size old results plus the current result. All work is
  completed before window cleanup. This preserves safety but intentionally changes
  release timing and allocation pressure.

This is a controlled synthetic release point, not a reproduction of the customer's
whole training graph. It cannot capture internal temporaries freed before dispatch
returns. Failure to reduce event counts is therefore inconclusive. A win may include
changed allocation pressure; correlate the event counts, query durations and MatMul
submission timeline before attributing it to allocator event processing.

## Default EP4 case

EP4, BF16, tokens=4096, hidden=2816, topk=8, experts=256, AIV=32,
FP32 routing weights, expand=0, fresh dispatch with default handle copy. Independent
FC1 is torch.mm([4096,2816], [2816,1536]); no backward or training golden check.
These dimensions target the old fa trace's shared-FC1 example. This is not a full MLP.

Use a quiet node with four cards. Use the same installed build and SHMEM for A/B;
record the actual package paths, native library hashes and dependency commit. The
script writes Python identity/config and exposed DEEPEP/SHMEM allocator environment
settings for each rank. It does not change sender policy. Keep huge-page mode identical
between groups; the default is off, matching the earlier small EP4 Host experiment.
If the validated environment uses 1 GiB pages, add --huge-pages to both runs.

Formal run (20 samples/group/rank, one Buffer per rank, order A then B):

```bash
torchrun --standalone --nproc-per-node=4 bench_allocator_release.py \
  --output results/formal --warmup 5 --windows 5 --window-size 4
```

Independent short profiler run (4 samples/group/rank):

```bash
torchrun --standalone --nproc-per-node=4 bench_allocator_release.py \
  --output results/profile --warmup 5 --windows 1 --window-size 4 --profile
```

Use new output paths on every run. `--first-device` selects a contiguous device range;
do not combine it with an incompatible device-visibility mapping. Multi-node torchrun
also works when WORLD_SIZE=4; all ranks must use identical arguments and libraries.

To instead match the earlier EP4 Host-cache experiment use these extra arguments in
both commands: `--tokens 8192 --hidden 1536 --topk 16 --experts 512`. Choose FC1 output
width from that customer's model; do not compare those timings to the H2816 default.

## Read the result

- `summary.json`: per matched iteration maximum over ranks, then mean/median/P95/max.
- `rank*/samples.csv`: dispatch Host duration, release duration, FC1 Host call,
  return-to-FC1-call-end, total Host submission and completed pair duration; allocated
  and reserved bytes. Completed pair includes the final synchronization; it is not a
  training-step or communication-kernel duration.
- Profiler markers: `allocator_probe/A|B/wNNN/sNNN/{dispatch,release_previous,fc1,drain}`.
  Compare runtime event recording inside release_previous, event queries and
  empty_tensor inside fc1. Correlate FC1's launch with its MatMul kernel to locate
  actual submission/start. Host perf_counter values do not supply device start times.
- Formal timing has no profiler ranges. Do not mix its timing with profiler timings.

Extract each rank's exported trace_view.json:

```bash
python summarize_trace.py /path/to/rank0/trace_view.json \
  /path/to/rank1/trace_view.json /path/to/rank2/trace_view.json \
  /path/to/rank3/trace_view.json --output results/allocator_events.csv
```

The extractor preserves each API spelling separately; never add wrapper and nested
runtime API counts. It only attributes calls contained on the same Host thread as
the marker. If event APIs are absent, verify runtime API capture before interpreting
zeros. Inspect first versus later samples and individual ranks, not just pooled means.

Evidence supporting the hypothesis: B reduces event recording, subsequent queries
and FC1 allocation/Host submission delay together. No such reduction means the
experiment has not established the cause; do not remove record_stream based on it.

This experiment has only local syntax, CLI and synthetic trace-parser validation.
NPU execution, installed API compatibility and runtime correctness remain unvalidated.
