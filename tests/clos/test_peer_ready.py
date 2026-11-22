"""Protocol interleavings and benchmark accounting; no claim about NPU fences."""

import importlib.util
from pathlib import Path
import random

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("peer_bench", ROOT / "benchmarks/clos/bench_peer_ready.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def simulate(seed, world, calls, gate=True):
    """Single-slot receiver with delayed writes and arbitrary rank start order.

    Events have independent delivery times, including delayed ready messages.
    Payload and metadata precede their own completion publication. Quiet must
    release the source before it can publish the next epoch. Never clear slots.
    """
    rng = random.Random(seed)
    epoch, consumed = [0] * world, [0] * world
    ready = [[0] * world for _ in range(world)]
    done = [[[0, 0] for _ in range(world)] for _ in range(world)]
    slots = [[[None, None] for _ in range(world)] for _ in range(world)]
    sent = [set() for _ in range(world)]
    outstanding = [0] * world
    events = []
    now, reads, pushes = 0, 0, 0
    # Rank zero is extremely late on alternating operations. No global launch
    # synchronization is inserted, and recomputation is just more ordered calls.
    available = [rng.randrange(100) for _ in range(world)]
    while min(consumed) < len(calls):
        now += 1
        assert now < 100000, "deadlock"
        rng.shuffle(events)
        for event in events[:]:
            deadline, kind, source, target, value, field = event
            if deadline > now:
                continue
            events.remove(event)
            if kind == "ready":
                # The source cannot advance again until the receiver consumed
                # its previous done, so same-pair ready writes cannot overtake.
                assert ready[target][source] <= value
                ready[target][source] = value
            else:
                assert epoch[target] == value and consumed[target] < value, "overwrite before receiver ready"
                count = (source * 7 + target * 3 + value) % 5  # includes empty peers
                slots[target][source][field] = (calls[value - 1], value, source, target, field, count)
                done[target][source][field] = value
                outstanding[source] -= 1
        order = list(range(world))
        rng.shuffle(order)
        for rank in order:
            if epoch[rank] == consumed[rank] and epoch[rank] < len(calls) and now >= available[rank]:
                epoch[rank] += 1
                sent[rank].clear()
                for target in range(world):
                    events.append((now + rng.randrange(1, 9), "ready", rank, target, epoch[rank], 0))
                    pushes += 1
            if epoch[rank] == consumed[rank]:
                continue
            e = epoch[rank]
            for peer in order:
                if peer in sent[rank]:
                    continue
                reads += 1  # Every readiness read is from this rank's own array.
                if gate and ready[rank][peer] < e:
                    continue
                sent[rank].add(peer)
                for field in (0, 1):
                    events.append((now + rng.randrange(1, 13), "data_done", rank, peer, e, field))
                    outstanding[rank] += 1
            if len(sent[rank]) == world and outstanding[rank] == 0 and all(v == [e, e] for v in done[rank]):
                for source in range(world):
                    for field in (0, 1):
                        expected = (calls[e - 1], e, source, rank, field, (source * 7 + rank * 3 + e) % 5)
                        assert slots[rank][source][field] == expected
                consumed[rank] = e
                available[rank] = now + (300 if rank == 0 and e % 2 else rng.randrange(9))
    assert pushes == world * world * len(calls)  # Independent of polling duration.
    return reads


@pytest.mark.parametrize("world", [2, 4, 8, 16])
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("recompute", ["none", "expert", "full"])
def test_skew_recompute_single_slot_reuse(world, seed, recompute):
    calls = [op for _, op in bench.stages(recompute)] * 3
    assert simulate(seed, world, calls) > 0


def test_oracle_detects_missing_ready_gate():
    with pytest.raises(AssertionError, match="overwrite before receiver ready"):
        simulate(0, 8, ["fresh", "combine", "cached", "combine"] * 2, gate=False)


def test_expert_recompute_adds_no_communication():
    assert bench.stages("expert") == bench.stages("none")
    assert len(bench.stages("full")) == len(bench.stages("none")) + 2


def test_timing_aggregates_slowest_rank_per_sample():
    assert bench.aggregate([[1, 100], [90, 2]]) == 95


def test_kernel_parser_does_not_charge_preparation_barrier_to_notify(tmp_path):
    lines = ["Name,Duration(us)"]
    for _ in range(6):
        lines += [
            "cached_notify_combine_kernel,10000",
            "v1_combine_udma_comm_kernel,200",
            "v1_combine_udma_reduce_kernel,70",
        ]
    (tmp_path / "kernel_details.csv").write_text("\n".join(lines))
    assert bench.parse_profile(tmp_path, "combine", 6) == {
        "comm": [200.0, 200.0],
        "reduce": [70.0, 70.0],
        "notify": [0, 0],
        "layout": [0, 0],
    }
    with pytest.raises(RuntimeError, match="expected 7"):
        bench.parse_profile(tmp_path, "combine", 7)


def test_cached_notify_can_be_absent_but_comm_cannot(tmp_path):
    (tmp_path / "kernel_details.csv").write_text("Name,Duration(us)\n" + "v1_dispatch_udma_kernel,200\n" * 6)
    assert bench.parse_profile(tmp_path, "cached", 6)["notify"] == [0, 0]
    with pytest.raises(RuntimeError, match="fresh notify"):
        bench.parse_profile(tmp_path, "fresh", 6)


def test_control_layout_fits_ep256_and_ep1024():
    for world in (8, 256, 1024):
        intervals = [(table * world + peer) * 512 for table in range(3) for peer in range(world)]
        assert len(set(intervals)) == len(intervals)
        assert all(address % 512 == 0 for address in intervals)
        assert intervals[-1] + 64 * 8 <= 2 * 1024 * 1024


def test_production_control_primitives(tmp_path):
    """Execute real control helper bodies with local/remote memory ownership mocks."""
    import os
    import shutil
    import subprocess
    from tests.utils.cpp import cpp_functions

    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("C++17 compiler required for control-helper CPU test")
    control = (ROOT / "csrc/direct/kernels/common/peer_ready.h").read_text()
    funcs = cpp_functions(
        control,
        "PeerSlot",
        "PeerLoad",
        "PeerReady",
        "WaitPeerReady",
        "PushPeerEpoch",
        "PublishMetadataDone",
        "WaitMetadataDone",
    )
    shared = (ROOT / "csrc/direct/kernels/common/udma_shared_sender.h").read_text()
    funcs += cpp_functions(shared, "Signal", "Begin")
    harness = r"""
#include <cstdint>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <array>
#define __aicore__
#define __gm__
using GM_ADDR = uint8_t*;
constexpr unsigned kPeerSlotBytes=512, kSignalBytes=512, world=8, cores=8, bytes=3*world*512;
constexpr int EVENT_ID4=4;
static std::array<std::array<uint64_t,bytes/8>,world> memory{};
static unsigned current_rank, remote_writes, global_barriers;
static uint64_t cycle;
void trap() { throw std::runtime_error("timeout"); }
void dcci_cacheline(uint8_t*) {}
void aclshmemx_sync_vec_all() { ++global_barriers; }
void* aclshmem_ptr(void* p, uint32_t peer) {
  auto offset=reinterpret_cast<uint8_t*>(p)-reinterpret_cast<uint8_t*>(memory[current_rank].data());
  assert(offset>=0 && offset<bytes);
  return reinterpret_cast<uint8_t*>(memory[peer].data())+offset;
}
namespace AscendC {
enum class HardEvent { S_MTE3, MTE3_S };
enum class MemDsbT { ALL };
static int credits[2]{};
template<HardEvent E> void SetFlag(int) { assert(credits[int(E)]++==0); }
template<HardEvent E> void WaitFlag(int) { assert(--credits[int(E)]==0); }
template<MemDsbT> void DataSyncBarrier() {}
template<bool> void SyncAll() {}
using ::printf;
uint64_t GetSystemCycle() { return ++cycle; }
template<class T> struct GlobalTensor {
 T* ptr{};
 void SetGlobalBuffer(T* p) { ptr=p; }
 T GetValue(unsigned i) const {
   auto base=reinterpret_cast<uintptr_t>(memory[current_rank].data());
   auto addr=reinterpret_cast<uintptr_t>(ptr+i);
   assert(addr>=base && addr<base+bytes); // polling MUST remain local
   return ptr[i];
 }
};
template<class T> struct LocalTensor { T* ptr; void SetValue(unsigned i,T v) { ptr[i]=v; } };
struct DataCopyExtParams { unsigned blocks,bytes,a,b,c; };
template<class T> void DataCopyPad(GlobalTensor<T> dst,LocalTensor<T> src,DataCopyExtParams p) {
 assert(p.blocks==1 && p.bytes<=512); std::memcpy(dst.ptr,src.ptr,p.bytes); ++remote_writes;
}
}
struct Tiling { uint64_t peer_control,peer_epoch,npu_timeout_cycles; unsigned rank,world_size; };
// FUNCTIONS
int main() {
  std::array<std::array<uint64_t,64>,cores> ub{};
  for(uint64_t epoch=1;epoch<9;++epoch) {
    for(current_rank=0;current_rank<world;++current_rank) {
      auto control=reinterpret_cast<uint64_t>(memory[current_rank].data());
      Tiling td{control,epoch,100000,current_rank,world};
      auto untouched=*PeerSlot(control,world,2,(current_rank+1)%world);
      assert(PeerReady(td,current_rank,cycle));
      assert(PeerReady(td,(current_rank+1)%world,cycle) == ((current_rank+1)%world < current_rank));
      for(unsigned core=0;core<cores;++core)
        Begin(nullptr,0,current_rank,world,core,cores,{ub[core].data()},4,control,epoch);
      assert(*PeerSlot(control,world,2,current_rank)==epoch);
      assert(*PeerSlot(control,world,2,(current_rank+1)%world)==untouched);
    }
    for(current_rank=0;current_rank<world;++current_rank) {
      auto control=reinterpret_cast<uint64_t>(memory[current_rank].data());
      Tiling td{control,epoch,100000,current_rank,world};
      for(unsigned p=0;p<world;++p) { assert(PeerReady(td,p,cycle)); WaitPeerReady(td,p); }
      for(unsigned core=0;core<cores;++core) PublishMetadataDone(td,core,cores,{ub[core].data()});
    }
    for(current_rank=0;current_rank<world;++current_rank) {
      Tiling td{reinterpret_cast<uint64_t>(memory[current_rank].data()),epoch,100000,current_rank,world};
      for(unsigned core=0;core<cores;++core) WaitMetadataDone(td,core,cores);
    }
  }
  assert(global_barriers==0);
  assert(remote_writes==8*(world+2*world*world));
  current_rank=0;
  Tiling td{reinterpret_cast<uint64_t>(memory[0].data()),9,0,0,world};
  bool threw=false;
  try { PeerReady(td,1,cycle); } catch(const std::runtime_error&) { threw=true; }
  assert(threw);
}
""".replace("// FUNCTIONS", funcs)
    source = tmp_path / "control.cpp"
    source.write_text(harness)
    executable = tmp_path / ("control.exe" if os.name == "nt" else "control")
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(executable)], check=True, timeout=180)
    try:
        subprocess.run([str(executable)], check=True, timeout=30)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 4551:
            pytest.skip("compiled, but Windows application control blocked harness execution (4551)")
        raise
