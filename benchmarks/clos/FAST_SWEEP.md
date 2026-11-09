# Clos peer strategy sweep

`bench_fast_sweep.py` is an experimental performance-only runner for EP64/128,
with eight contiguous ranks per server. It skips numerical validation. Reports
always contain `precision_checked: false`; a successful run is not a precision
pass. The ordinary API behavior is unchanged unless
`DEEPEP_CLOS_TUNING_SESSION=1` is set before constructing the buffer.

The two BF16 shapes are T8192/H2816/K8/AIV32 and T16384/H7168/K6/AIV64, with 256
experts, random distinct experts per token, 1 GiB pages, no L2 eviction and
`DEEPEP_CLOS_COMBINE_PRE_NOTIFY=1`. Each sample starts with `buffer.barrier`.
Only cached dispatch and unweighted combine are ranked. Communication bandwidth
uses rank-deduplicated payload bytes including self, in decimal GB/s. Notify and
final reduction are recorded separately and excluded from that bandwidth.

## Runtime ownership

There is one Buffer and one SHMEM initialization per rank. The explicit byte
capacity is the maximum of both shape hints at the maximum token capacity;
hidden/top-k are not bound in the constructor. Each shape creates a separate
fresh routing handle at its actual AIV count. Output tensors and routing plans
are retained until device work completes, then released before the next shape.

The benchmark reserves the union of existing peer schedules' QP masks. Every
candidate retains the original kernel's exclusive AIV-to-SQ ownership and memory
layout. A private native setter validates the next configuration without
initializing another SHMEM instance, drains all local streams and aligns all
ranks before changing scheduling fields. All ranks validate the same plan and
binary hashes via TCPStore before entering the sweep. Switching is serialized;
concurrent API calls on this experimental buffer are unsupported. No UB layout,
kernel, completion protocol or production default is changed.

Prebuilding a larger QP union can make the first initialization more expensive.
It does not cure environment/channel failures. A timeout or missing profiler
sample fails the run; it never promotes an incomplete configuration to winner.

## Run

Build/install this branch first, with the same CANN/SHMEM on every node. Start the
following on all nodes, setting a unique shared run name, reachable master and
different node ranks. Use EP=64 and NNODES=8 for eight servers, or EP=128 and
NNODES=16 for sixteen servers. Remove inherited experimental `DEEPEP_CLOS_*` overrides.

```bash
EP=128
NNODES=16
NODE_RANK=0
MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to node0's reachable IP}"
RUN_ID=clos_fast_ep128_01
export SHMEM_ROOT=/path/to/shmem
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:${LD_LIBRARY_PATH:-}"
export SHMEM_UDMA_CONNECT_TIMEOUT_MS=600000
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
set -o pipefail
python -u -m torch.distributed.run \
  --nnodes "$NNODES" --nproc-per-node 8 --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port 29600 \
  benchmarks/clos/bench_fast_sweep.py --ep-size "$EP" \
  --control-port 29610 --shmem-port 29620 \
  --output "$RUN_ID/node$NODE_RANK" 2>&1 | tee "${RUN_ID}_node${NODE_RANK}.log"
```

`--plan-only` prints the plan without importing torch. `--smoke` selects five
configurations covering every order and all worker/QP combinations, using both
full shapes. `--shape 2816` or `--shape 7168` selects one shape. `--weights 3 4`
and `--visits 4 8` allow a smaller explicitly restricted search.

The full candidate set has 492 configurations per shape at EP64 and 372 at
EP128. Each configuration uses two warmups, eight measured calls and guard calls.
The three best configurations per shape/operator are rerun forward and reverse,
30 measured calls each. The winner maximizes the lower bandwidth of its two
confirmation runs. This is the best of the measured candidate set, not proof of
a global optimum. Different operators may select different configurations.

Profiler export is per shape/operator/phase, not per candidate. Timing ranges
are computed from each rank's mean communication-kernel duration. Ranking uses
`sum(rank logical bytes) / sum(rank mean microseconds) / 1000`; rank bandwidth
ranges are also retained. `report.json` contains the winners, confirmation runs
and full coarse ranking. Per-rank manifests, samples, kernel CSVs, identities and
exit files preserve the evidence. Every rank must finish with exit code zero and
`buffer_initializations: 1` before accepting the report.
