"""Host routing regressions; dtype-only doubles never execute device operations."""

from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_packed_dispatch_scales_preserve_bits_and_peer_order(tmp_path):
    """Execute the actual packing VF on CPU against an independently routed reference."""
    import shutil
    from tests.utils.cpp import cpp_functions

    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    header = (ROOT / "csrc/clos/kernels/dispatch/dispatch_scale_pack.h").read_text()
    method = cpp_functions(header, "PackDispatchPeerScalesVf", "PackV1DispatchScalesSimtVf")
    source = tmp_path / "pack.cpp"
    source.write_text(r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>
#define __simt_vf__
#define __launch_bounds__(n)
#define __gm__
constexpr unsigned kDispatchScalePackThreads=128;
struct {unsigned x;} threadIdx;
void asc_threadfence() {}
namespace __asc_simt_vf {void __trap() {throw std::runtime_error("bounds");}}
''' + method + r'''
int main() {
 for(unsigned tokens:{0,1,17,257}) for(unsigned scales:{1,22,56,129})
 for(unsigned channels:{1,3,16}) for(unsigned cores:{1,8,16,32,48,64}) for(unsigned stride:{1,2}) for(unsigned column_major:{0,1}) {
  const unsigned world=8,rank=2,ct=(tokens+channels-1)/channels,ps=(ct+7)/8*8;
  const uint64_t ts=column_major ? 1U : uint64_t(scales)*stride+3;
  const uint64_t ss=column_major ? tokens+3U : stride;
  const uint64_t storage=tokens ? (tokens-1)*ts+(scales-1)*ss+1 : 0;
  std::vector<uint32_t> input(storage,0xabcdef01),out(world*tokens*scales,0xdeadbeef),expected=out;
  std::vector<int32_t> plan(world*channels*ps,-1),prefix(world*channels);
  for(unsigned t=0;t<tokens;++t) for(unsigned s=0;s<scales;++s)
   input[t*ts+s*ss]=0x80000000U^(t*65537U+s*17U); // Includes nonnumeric float bit patterns.
  for(unsigned peer=0;peer<world;++peer) {
   unsigned ordinal=0;
   for(unsigned ch=0;ch<channels;++ch) {
    unsigned local=0;
    for(unsigned t=ch*ct;t<std::min(tokens,(ch+1)*ct);++t) {
     if(peer==1 || (t+peer)%3) continue; // Completely empty peer and irregular tails.
     plan[(peer*channels+ch)*ps+local++]=t;
     if(peer!=rank) for(unsigned s=0;s<scales;++s)
      expected[(peer*tokens+ordinal)*scales+s]=input[t*ts+s*ss];
     ++ordinal;
    }
    prefix[peer*channels+ch]=ordinal;
   }
  }
  for(unsigned core=0;core<cores;++core) for(threadIdx.x=0;threadIdx.x<128;++threadIdx.x)
   PackDispatchPeerScalesVf(input.data(),out.data(),plan.data(),prefix.data(),rank,world,channels,ps,
                           tokens,scales,ts,ss,core,cores);
  assert(out==expected); // Also checks untouched self and unused capacity.
  std::vector<uint32_t> normalized(tokens*scales,0xdeadbeef);
  for(unsigned core=0;core<cores;++core) for(threadIdx.x=0;threadIdx.x<128;++threadIdx.x)
   PackV1DispatchScalesSimtVf(input.data(),normalized.data(),tokens,scales,ts,ss,core,cores);
  for(unsigned t=0;t<tokens;++t)for(unsigned col=0;col<scales;++col)
   assert(normalized[t*scales+col]==input[t*ts+col*ss]);
  if(!out.empty()) {out[0]^=1;assert(out!=expected);} // Reference check must detect corruption.
 }
}
''')
    binary = tmp_path / "pack"
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_packed_scale_ready_waits_for_delayed_producers_and_reuse(tmp_path):
    """Exercise actual local ready-store/wait helpers; NPU tests cover device visibility."""
    import shutil
    from tests.utils.cpp import cpp_functions

    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    header = (ROOT / "csrc/clos/kernels/dispatch/udma_dispatch_schedule.h").read_text()
    methods = cpp_functions(header, "StorePackedScaleReady", "WaitPackedScaleReady")
    source = tmp_path / "ready.cpp"
    source.write_text(r'''
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>
#define __aicore__
#define __gm__
constexpr int EVENT_ID4=4;
constexpr uint64_t kIncomingTimeoutCycles=50000000;
unsigned metadata_first=16;
unsigned MetadataFirst(unsigned,unsigned,unsigned,unsigned,unsigned,unsigned){return metadata_first;}
struct DispatchTilingData {
 unsigned world_size=64,udma_peers_per_worker=1,udma_peer_order=4,udma_ranks_per_group=8,udma_workers_per_peer=2,rank=0;
 uint64_t packed_scale_ready;
};
std::atomic<uint64_t> invalidations{0};
void dcci_cacheline(uint8_t*){++invalidations;}
void trap(){throw std::runtime_error("ready timeout");}
namespace AscendC {
 enum class HardEvent{S_MTE3,MTE3_S};
 template<HardEvent> void SetFlag(int){}
 template<HardEvent> void WaitFlag(int){}
 struct DataCopyExtParams{unsigned count,bytes,a,b,c;};
 template<class T>struct LocalTensor{T* p;void SetValue(unsigned i,T v){p[i]=v;}};
 template<class T>struct GlobalTensor{
  T* p;
  void SetGlobalBuffer(T* q){p=q;}
  T GetValue(unsigned i){return reinterpret_cast<std::atomic<T>*>(p+i)->load();}
 };
 template<class T>void DataCopyPad(GlobalTensor<T> out,LocalTensor<T> in,DataCopyExtParams copy){
  assert(copy.count==1 && copy.bytes==8);
  reinterpret_cast<std::atomic<T>*>(out.p)->store(in.p[0]);
 }
 uint64_t GetSystemCycle(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
 template<class... T>void printf(const char*,T...){}
}
struct Resource{uint64_t value=0;template<class T>AscendC::LocalTensor<T> GetBufferByByte(unsigned offset){assert(offset==0);return {reinterpret_cast<T*>(&value)};}};
struct alignas(512) Slot{std::atomic<uint64_t> value{0};unsigned char padding[504]{};};
static_assert(sizeof(Slot)==512);
''' + methods + r'''
int main(){
 std::array<Slot,64> slots;
 std::array<Resource,64> resources;
 std::array<std::atomic<uint64_t>,64> produced{};
 DispatchTilingData td;td.packed_scale_ready=reinterpret_cast<uint64_t>(slots.data());
 for(unsigned first:{16,32}){
  metadata_first=first;
  for(unsigned round=1;round<=3;++round){
   for(unsigned c=0;c<64;++c)StorePackedScaleReady(resources[c],td,c,0);
   std::vector<std::thread> workers;
   for(unsigned c=first;c<64;++c)workers.emplace_back([&,c,round]{
    if(c==63)std::this_thread::sleep_for(std::chrono::milliseconds(3));
    produced[c].store(round+first*100);
    StorePackedScaleReady(resources[c],td,c,1);
   });
   WaitPackedScaleReady(td,0,64);
   // Check before joining: a stale flag or early return must fail.
   for(unsigned c=first;c<64;++c)assert(produced[c].load()==round+first*100);
   for(auto& w:workers)w.join();
   for(unsigned c=0;c<64;++c){
    assert(slots[c].value.load()==(c>=first));
    for(auto byte:slots[c].padding)assert(byte==0);
   }
  }
 }
 StorePackedScaleReady(resources[63],td,63,0);
 bool rejected=false;try{WaitPackedScaleReady(td,0,64);}catch(const std::runtime_error&){rejected=true;}
 assert(rejected && invalidations.load()>0);
}
''')
    binary = tmp_path / "ready"
    subprocess.run([compiler, "-std=c++17", "-O2", "-pthread", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=10)

DTYPES_ONLY = """
import sys
from types import ModuleType, SimpleNamespace
torch = ModuleType('torch')
torch.int32, torch.int64 = 'int32', 'int64'
sys.modules['torch'] = torch
"""


def run_host(code, *, dtypes=True):
    result = subprocess.run(
        [sys.executable, "-c", (DTYPES_ONLY if dtypes else "") + textwrap.dedent(code)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "first",
    ["deep_ep", "ascend_deepep"]
    + [
        f"ascend_deepep.{name}"
        for name in ("buffer", "elastic", "config", "event", "expand", "resources", "dtypes", "utils")
    ],
)
def test_legacy_first_import_is_lazy_and_uses_public_entries(first):
    run_host(
        f"""
        import importlib, sys
        importlib.import_module({first!r})
        import ascend_deepep as old, deep_ep as new
        from ascend_deepep.buffer import Buffer
        from ascend_deepep.elastic import ElasticBuffer, EPHandle
        from ascend_deepep.config import Config
        from ascend_deepep.event import EventHandle, EventOverlap
        assert Buffer is old.Buffer is new.Buffer
        assert ElasticBuffer is old.ElasticBuffer is new.ElasticBuffer
        assert EPHandle is old.EPHandle is new.EPHandle
        assert Config is old.Config is new.Config
        assert EventHandle is old.EventHandle is new.EventHandle
        assert EventOverlap is old.EventOverlap is new.EventOverlap
        assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
        assert 'deep_ep._C' not in sys.modules and 'deep_ep._clos_C' not in sys.modules
    """,
        dtypes=False,
    )


def test_legacy_extra_exports_reuse_backend_objects():
    run_host("""
        import ascend_deepep as old, deep_ep as new
        from deep_ep.backends import clos
        from ascend_deepep.expand import ExpandHandle
        from ascend_deepep.resources import Handle, Slot, SlotState
        from ascend_deepep.dtypes import topk_idx_t
        from ascend_deepep.utils import align_up
        for name in ('ExpandHandle', 'Handle', 'Slot', 'SlotState', 'topk_idx_t'):
            assert getattr(old, name) is getattr(new, name) is getattr(clos, name)
        assert align_up is clos.utils.align_up
        assert align_up(33, 32) == 64
        assert isinstance(object.__new__(clos.EPHandle), old.EPHandle)
    """)


def test_consolidated_submodules_preserve_imports_and_exports():
    run_host("""
        import importlib, importlib.util, sys
        import ascend_deepep as old
        aliases = {
            'buffer': ('deep_ep.buffer', ('Buffer',)),
            'elastic': ('deep_ep.buffers.elastic', ('ElasticBuffer', 'EPHandle')),
            'config': ('deep_ep', ('Config',)),
            'event': ('deep_ep.utils.event', ('EventHandle', 'EventOverlap')),
            'expand': ('deep_ep.backends.clos.expand', ('ExpandHandle',)),
            'resources': ('deep_ep.backends.clos.resources', ('Handle', 'Slot', 'SlotState')),
            'dtypes': ('deep_ep.backends.clos.dtypes', ('topk_idx_t',)),
            'utils': ('deep_ep.backends.clos.utils', ('align_up',)),
        }
        for name, (target, exports) in aliases.items():
            fullname = 'ascend_deepep.' + name
            module = importlib.import_module(fullname)
            assert getattr(old, name) is module is sys.modules[fullname]
            assert module.__name__ == fullname
            assert importlib.util.find_spec(fullname).name == fullname
            assert module.__all__ == list(exports)
            assert set(exports) <= set(dir(module))
            scope = {}
            exec('from ' + fullname + ' import *', scope)
            for symbol in exports:
                assert scope[symbol] is getattr(importlib.import_module(target), symbol)
            try:
                module.not_a_legacy_export
            except AttributeError:
                pass
            else:
                raise AssertionError('unknown compatibility export accepted')
        saved = old.buffer
        importlib.reload(old)
        assert old.buffer is saved
        scope = {'__package__': 'ascend_deepep'}
        exec('from .buffer import Buffer', scope)
        assert scope['Buffer'] is old.Buffer
        assert 'deep_ep._C' not in sys.modules and 'deep_ep._clos_C' not in sys.modules
    """)


@pytest.mark.parametrize("ep", [2, 4, 8, 16, 32, 64, 128])
def test_public_config_and_helpers_work_with_auto_policy(ep):
    run_host(f"""
        import ascend_deepep as old, deep_ep as new
        from deep_ep.backends.clos.config import Config as InternalConfig
        from deep_ep.backends.clos.udma_policy import resolve_udma_policy
        policy = resolve_udma_policy(True, {ep}, 'udma', None, None, None, None, None)
        for config in (None, new.Config(), old.Config(), InternalConfig(),
                       new.Buffer.get_dispatch_config({ep}), new.Buffer.get_combine_config({ep})):
            resolved = policy.resolve_config(config, 32)
            assert (resolved.num_aiv or 32) == 32
            assert resolved.udma_peers_per_worker == policy.peers_per_worker
        assert new.Config().num_aiv == 32
        assert not new.Config()._udma_peers_explicit
        assert not new.Config()._udma_wqes_explicit
        for keyword in ('num_aiv', 'num_sms', 'num_compute_units'):
            resolved = policy.resolve_config(new.Config(**{{keyword: 64}}), 32)
            assert resolved.num_aiv == 64
        config = new.Config(32, 8, 512, 16, 1024, udma_peers_per_worker=2, udma_wqes_per_visit=16)
        resolved = policy.resolve_config(config, 32)
        assert (resolved.num_max_nvl_chunked_send_tokens, resolved.num_max_nvl_chunked_recv_tokens,
                resolved.num_max_rdma_chunked_send_tokens, resolved.num_max_rdma_chunked_recv_tokens) == (8, 512, 16, 1024)
        assert config.udma_peers_per_worker == 2 and config.udma_wqes_per_visit == 16
        assert config.get_nvl_buffer_size_hint(12288, {ep}, 129) == config._impl.get_nvl_buffer_size_hint(12288, {ep}, 129)
        for bad in (object(), SimpleNamespace(_impl=config._impl)):
            try:
                policy.resolve_config(bad, 32)
            except TypeError:
                pass
            else:
                raise AssertionError('unrelated configuration accepted')
    """)


@pytest.mark.parametrize("operation", ["dispatch", "combine"])
def test_v1_public_config_reaches_native_argument_conversion(operation):
    run_host(f"""
        import deep_ep
        from deep_ep.backends.clos import buffer as m
        from deep_ep.backends.clos.udma_policy import resolve_udma_policy
        from deep_ep._clos_adapter import Buffer
        class Converted(Exception): pass
        def convert(config):
            assert config.num_aiv == 32
            assert config.udma_peers_per_worker == 1
            raise Converted
        m._config_to_native_args = convert
        m._get_allocation_stream_id = lambda _: 0
        m._is_capturing = lambda: False
        obj = SimpleNamespace(_ensure_alive=lambda: None, num_max_tokens_per_rank=None,
            _udma_auto_policy=resolve_udma_policy(True, 8, 'udma', None, None, None, None, None), rank=0)
        kwargs = {{'config': deep_ep.Config()}}
        if {operation!r} == 'combine': kwargs['handle'] = ()
        try:
            getattr(Buffer, {operation!r})(obj, object(), **kwargs)
        except Converted:
            pass
        else:
            raise AssertionError('public Config did not reach native argument conversion')
    """)


@pytest.mark.parametrize("operation", ["fresh", "cached", "combine"])
@pytest.mark.parametrize("requested", [None, 0, 8, 32, 64])
def test_v2_core_defaults_and_handle_inheritance(operation, requested):
    run_host(f"""
        from deep_ep.backends.clos import elastic as m
        from deep_ep.backends.clos.udma_policy import resolve_udma_policy
        from deep_ep._clos_adapter import ElasticBuffer
        from contextlib import contextmanager
        class ReadyForDevice(Exception): pass
        counts = []
        policy = resolve_udma_policy(True, 8, 'udma', None, None, None, None, None)
        def resolve(count):
            counts.append(count)
            return policy.for_num_aiv(count)
        @contextmanager
        def stop_before_device(*args):
            raise ReadyForDevice
            yield
        for name in ('_validate_dispatch_scope', '_validate_dispatch_tensors', '_validate_combine_tensors'):
            setattr(m, name, lambda *a: None)
        m._get_allocation_stream_id = lambda _: 0
        m._try_record_event = lambda _: None
        m._use_stream_after_event = stop_before_device
        x = SimpleNamespace(size=lambda dim: (129, 6144)[dim])
        topk = SimpleNamespace(size=lambda dim: (129, 8)[dim])
        handle = SimpleNamespace(num_sms=64, num_experts=256, num_max_tokens_per_rank=129,
            expert_alignment=1, do_expand=False, topk_idx=topk,
            _dispatch_layout=(None,) * 6)
        def no_estimate(*args): raise AssertionError('default still uses theoretical estimator')
        obj = SimpleNamespace(_ensure_alive=lambda: None, _validate_handle=lambda _: None,
            use_fp8_dispatch=False, num_max_tokens_per_rank=129, num_ranks=8, hidden=6144, num_topk=8,
            get_buffer_size_hint=lambda *a: 0, group=None, allow_hybrid_mode=True,
            num_bytes=1, num_device_aiv=64, get_theoretical_num_sms=no_estimate,
            _udma_auto_policy=SimpleNamespace(for_num_aiv=resolve), transport='udma',
            _communication_stream=None, _unpack_bias=lambda _: (None, None),
            _udma_config=SimpleNamespace(udma_peers_per_worker=1, udma_wqes_per_visit=8))
        requested = {requested!r}
        mismatch = {operation!r} != 'fresh' and requested in (8, 32)
        kwargs = {{}} if requested is None else {{'num_sms': requested}}
        try:
            if {operation!r} == 'combine':
                ElasticBuffer.combine(obj, x, handle, **kwargs)
            elif {operation!r} == 'cached':
                ElasticBuffer.dispatch(obj, x, handle=handle, **kwargs)
            else:
                ElasticBuffer.dispatch(obj, x, topk_idx=topk, num_experts=256, **kwargs)
        except ReadyForDevice:
            assert not mismatch
            assert counts == [({requested} or 32) if {operation!r} == 'fresh' else 64]
        except ValueError as exc:
            assert mismatch and 'must match' in str(exc), str(exc)
        else:
            raise AssertionError('did not stop at native submission boundary')
    """)


@pytest.mark.parametrize(
    "ep,cores,peers,workers",
    [
        (4, 2, 4, 2),
        (8, 2, 8, 2),
        (8, 8, 2, 2),
        (16, 2, 8, 1),
        (16, 8, 2, 1),
        (32, 4, 8, 1),
        (32, 8, 4, 1),
        (64, 8, 8, 1),
        (128, 20, 8, 1),
    ],
)
def test_low_core_policy_preserves_core_count_and_runtime_qp_lanes(ep, cores, peers, workers):
    run_host(f"""
        import deep_ep
        from dataclasses import FrozenInstanceError
        from deep_ep.backends.clos.udma_policy import resolve_udma_policy
        base = resolve_udma_policy(True, {ep}, 'udma', None, None, None, None, None)
        before = base.signature()
        selected = base.for_num_aiv({cores})
        assert (selected.peers_per_worker, selected.workers_per_peer) == ({peers}, {workers})
        assert selected.workers_per_peer == base.workers_per_peer
        assert selected.qps_per_worker == base.qps_per_worker == 1
        assert selected.wqes_per_visit == base.wqes_per_visit
        assert selected.send_workers <= {cores}
        for config in (deep_ep.Config({cores}), base.config({cores})):
            effective = base.resolve_config(config, 32)
            assert effective.num_aiv == {cores}
            assert effective.udma_peers_per_worker == {peers}
        assert base.signature() == before  # A later handle cannot change earlier handles' schedule.
        assert base.for_num_aiv({cores}) == selected
        assert base.for_num_aiv(32) == base
        assert base.for_num_aiv(64) == base
        try: selected.peers_per_worker = 1
        except FrozenInstanceError: pass
        else: raise AssertionError('policy is mutable')
    """)


def test_auto_policy_budget_boundaries_and_manual_opt_out():
    run_host("""
        from deep_ep.backends.clos.udma_policy import resolve_udma_policy
        for ep in range(2, 257):
            base = resolve_udma_policy(True, ep, 'udma', None, None, None, None, None)
            for cores in range(2, 65, 2):
                feasible = ((ep - 1 + 7) // 8) * base.workers_per_peer <= min(cores, 32)
                try: chosen = base.for_num_aiv(cores)
                except ValueError:
                    assert not feasible, (ep, cores)
                else:
                    assert feasible and chosen.send_workers <= min(cores, 32), (ep, cores)
                    assert chosen.peers_per_worker in (1, 2, 4, 8)
                    assert chosen.config(cores).num_aiv == cores
        base = resolve_udma_policy(True, 32, 'udma', None, None, None, None, None)
        assert base.for_num_aiv(0) == base
        for bad in (False, True, 8.0, -2, 1, 3, 66, 2):
            try: base.for_num_aiv(bad)
            except (TypeError, ValueError): pass
            else: raise AssertionError(('invalid/unrepresentable budget accepted', bad))
        assert base.for_num_aiv(0) == base
        assert resolve_udma_policy(False, 16, 'udma', 'peer', 'rank', 'rank', None, None,
                                   workers=1, qps=1, peers=1, wqes=8) is None
    """)


def test_bandwidth_reports_the_operation_policy_not_the_buffer_default():
    run_host("""
        sys.path.insert(0, 'benchmarks/clos')
        from bench_v2_bandwidth import effective_udma_config
        from deep_ep.backends.clos.config import Config
        from deep_ep.backends.clos.udma_policy import resolve_udma_policy
        base = resolve_udma_policy(True, 8, 'udma', None, None, None, None, None)
        buffer = SimpleNamespace(udma_auto_policy=base, udma_schedule='peer',
            udma_dispatch_peer_order='rank', udma_combine_peer_order='rank',
            udma_ranks_per_group=None, udma_peers_per_group_per_worker=None,
            udma_workers_per_peer=2, udma_qps_per_worker=1)
        for config in (effective_udma_config(buffer, num_aiv=8), effective_udma_config(buffer, Config(8))):
            assert config['udma_peers_per_worker'] == 2
            assert config['udma_workers_per_peer'] == 2
            assert config['udma_wqes_per_visit'] == 8
        assert effective_udma_config(buffer, num_aiv=32)['udma_peers_per_worker'] == 1
        assert base.peers_per_worker == 1
    """)


def test_tuning_arguments_preserve_manual_policy_and_reject_oversubscription():
    run_host("""
        sys.path.insert(0, 'benchmarks/clos')
        from bench_v2_bandwidth import arguments, worker_command
        from pathlib import Path
        common = ['--ep-size', '32', '--nnodes', '4', '--master-addr', '192.0.2.1', '--run-id', 'test']
        a = arguments(common + ['--tuning-order', 'group_round_robin', '--tuning-peers', '1'])
        assert not a.udma_auto_policy and a.tuning_visits == [8,16,32]
        cmd = worker_command(a, dict(name='bf16_expand0_aiv32_visit8',aiv=32,dtype='bf16',expand=0), Path('/tmp/test'))
        assert cmd[cmd.index('--tuning-order')+1] == 'group_round_robin'
        assert cmd[cmd.index('--tuning-cross-lane0-weight')+1] == '3'
        for weight in range(1,9):
            weighted = arguments(common + ['--tuning-order','group_round_robin',
                                           '--tuning-cross-lane0-weight',str(weight)])
            cmd = worker_command(weighted, dict(name='bf16_expand0_aiv32_visit8',aiv=32,dtype='bf16',expand=0), Path('/tmp/test'))
            assert cmd[cmd.index('--tuning-cross-lane0-weight')+1] == str(weight)
        for invalid in (['--tuning-order','rank','--tuning-peers','2','--aiv','20'],
                        ['--tuning-order','rank','--tuning-workers','1'],
                        ['--tuning-order','group_round_robin','--tuning-cross-lane0-weight','0'],
                        ['--tuning-order','group_round_robin','--tuning-cross-lane0-weight','9'],
                        ['--tuning-order','group_round_robin','--tuning-peers','8']):
            try: arguments(common + invalid)
            except SystemExit as e: assert e.code == 2
            else: raise AssertionError(invalid)
        assert arguments(common).udma_auto_policy
    """)


def test_peer_row_split_covers_channel_boundaries(tmp_path):
    import shutil

    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    source = tmp_path / "split.cpp"
    source.write_text(
        r"""#include "csrc/clos/common/udma_schedule.h"
#include <cassert>
#include <cstdint>
using ascend_deepep::udma::PeerRowBoundary;
int main() {
  for (uint32_t count = 0; count < 4096; ++count) {
    for (uint32_t peer : {1U, 8U, 31U}) {
      for (uint32_t weight : {1U, 2U, 3U, 4U, 5U, 8U}) {
        uint32_t cut = PeerRowBoundary(count, 1, 2, 0, peer, 8, weight);
        assert(cut == count * (peer < 8 ? 1 : weight) / (peer < 8 ? 2 : weight + 1));
        assert(PeerRowBoundary(count, 0, 2, 0, peer, 8, weight) == 0);
        assert(PeerRowBoundary(count, 2, 2, 0, peer, 8, weight) == count);
        assert(PeerRowBoundary(count, 1, 1, 0, peer, 8, 1) == count);
        // Uneven channels and empty tails: intersections exactly tile the whole peer interval.
        uint32_t hits[4096]{};
        for (uint32_t lane = 0; lane < 2; ++lane) {
          uint32_t a = lane ? cut : 0, b = lane ? count : cut;
          for (uint32_t previous = 0; previous < count; previous += 17) {
            uint32_t n = count - previous < 17 ? count - previous : 17;
            uint32_t skip = a > previous ? a - previous : 0;
            uint32_t consumed = skip < n ? skip : n;
            uint32_t stop = b > previous ? b - previous : 0;
            uint32_t total = stop < n ? stop : n;
            assert(consumed <= total);
            for (uint32_t i = consumed; i < total; ++i) ++hits[previous + i];
          }
        }
        for (uint32_t i = 0; i < count; ++i) assert(hits[i] == 1);
      }
    }
  }
  for (uint32_t count : {0x7fffffffU, 0xfffffffeU, 0xffffffffU})
    for (uint32_t weight = 1; weight <= 8; ++weight) {
      const auto cut = PeerRowBoundary(count, 1, 2, 0, 8, 8, weight);
      assert(cut == uint64_t(count) * weight / (weight + 1));
      assert(cut <= count && cut + (count - cut) == count);
    }
}
""".replace("#include <cassert>", "#include <cassert>\n#include <initializer_list>")
    )
    binary = tmp_path / "split"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=30)


def test_dispatch_wqe_cache_matches_direct_sender(tmp_path):
    """Run the actual SIMT builders and sender methods with a synchronous UB/SQ double."""
    import shutil
    from tests.utils.cpp import cpp_functions

    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    text = (ROOT / "csrc/clos/kernels/dispatch/udma_dispatch_sender.h").read_text()
    builders = cpp_functions(text, "BuildDispatchWqes", "GatherDispatchCachedWqes")
    methods = cpp_functions(
        text.replace("Now() const", "Now()").replace("Tracing() const", "Tracing()"),
        "PutRows",
        "Flush",
        "BeginVisit",
        "PutPackedScales",
        "Credit",
        "Now",
        "PutCachedRows",
        "TraceRows",
        "Tracing",
    )
    methods = methods.replace("private:", "public:")
    source = tmp_path / "cache.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <array>
#include <algorithm>
#define __aicore__
#define __simt_vf__
#define __launch_bounds__(n)
#define __gm__
#define __ubuf__
struct { unsigned x; } threadIdx;
struct dim3 { explicit dim3(unsigned) {} };
namespace __asc_simt_vf { void __trap() { throw std::runtime_error("bad WQE"); } }
void trap() { __asc_simt_vf::__trap(); }
namespace AscendC {
enum class HardEvent {S_V,V_MTE3,V_S};
template<HardEvent> void SetFlag(unsigned) {}
template<HardEvent> void WaitFlag(unsigned) {}
}
template<auto F, class... A> void asc_vf_call(dim3, A... a) {
    for (unsigned i=0;i<32;++i) {threadIdx.x=i;F(a...);}
}
constexpr unsigned kBatchWqes=32,kWqeBytes=64,kDispatchCacheEntries=8,kOutputRecordWords=8;
constexpr unsigned kDispatchSimtScales=16;
constexpr unsigned kDispatchPackedScales=32;
constexpr unsigned kRequestsOffset=2048,kHeadersOffset=3072,EVENT_ID5=5;
struct TD { unsigned hidden=7,x_element_bytes=2,num_scales=3,scale_element_bytes=4,
    num_tokens=128,has_x_scales=0,dispatch_experiment_flags=0;
    uint64_t packed_peer_scales=0x50000000; };
struct DispatchProfile {
    bool enabled=true,trace=true; uint64_t credit_checks=0,credit_stalls=0; uint64_t build_ns=0,submit_ns=0,credit_ns=0,wqes=0,batches=0,refills=0,hits=0;
    uint64_t scale_submit_ns=0;
    uint64_t Now() {return 0;} uint64_t Elapsed() {return 0;} void Add(uint64_t&,uint64_t) {}
};
using Wqe=std::array<uint64_t,8>;
template<unsigned Capacity,unsigned Qps> struct SenderT {
    struct Peer { uint64_t state; unsigned peer; }; Peer peers_[Capacity*Qps];
    uint8_t* scratch_; unsigned count_=Capacity,qp_=0,last_=0,credits=0;
    uint64_t active_state_=0;
    std::vector<Wqe> sent;
    unsigned pending_=0;
    std::vector<std::array<uint64_t,4>> bulk;
    void Flush() {
        assert(pending_==1);
        std::array<uint64_t,4> r;std::memcpy(r.data(),scratch_+kRequestsOffset,32);
        bulk.push_back(r);pending_=0;
    }
    unsigned QueuePeer(unsigned p) {return qp_*Capacity+p;}
    void SelectQp(unsigned q) {qp_=q;}
    bool EnsurePayloadCredit(unsigned) {
        // Exercise the scratch overwrite performed by an SQ checkpoint.
        if (++credits%3==0) std::memset(scratch_,0xEE,2048);
        return credits%3==0;
    }
    void SubmitPayload(unsigned n) {
        for(unsigned i=0;i<n;++i) {
            Wqe w;std::memcpy(w.data(),scratch_+i*64,64);sent.push_back(w);
        }
        std::memset(scratch_,0xDD,2048); // No cache reader may depend on this region.
    }
};
"""
        + builders
        + r"""
template<unsigned Capacity,unsigned Qps> struct TestSender:SenderT<Capacity,Qps> {
    using Base=SenderT<Capacity,Qps>;
    using Base::scratch_;using Base::peers_;using Base::active_state_;using Base::last_;using Base::count_;
    static constexpr unsigned kDispatchBasesOffset=kHeadersOffset+Capacity*Qps*128;
    static constexpr unsigned kDispatchEpochOffset=kDispatchBasesOffset+Capacity*Qps*64;
    static constexpr unsigned kCacheOffset=kDispatchEpochOffset+32;
    struct CacheEntry {uint64_t next_indices=0;unsigned peer=UINT32_MAX,next_row=0,next_field=0,cursor=0,count=0;};
    TD td_; uint64_t x_=0x10000000,scales_=0x20000000;
    DispatchProfile stats;DispatchProfile* profile_=&stats;
    bool cache_enabled_; CacheEntry cache_[8]{};
    uint64_t trace_[Capacity*Qps][12]{}; unsigned trace_pending_[(Capacity*Qps+31)/32]{};
    unsigned rows_pending_=0,spans_=0,cache_readers_=0,next_qp_[Capacity]{};
    alignas(32) uint8_t arena[65536]{};
    TestSender(bool cache,bool scales):cache_enabled_(cache) {
        scratch_=arena;td_.has_x_scales=scales;
        for(unsigned p=0;p<Capacity*Qps;++p) {
            peers_[p].state=1+(p/Capacity)*3+(p%3); // Shared and distinct SQs.
            peers_[p].peer=p%Capacity;
            for(unsigned last=0;last<2;++last) {
                auto* h=reinterpret_cast<uint64_t*>(arena+kHeadersOffset+(p*2+last)*64);
                for(unsigned w=0;w<5;++w) h[w]=p*100+w;
            }
            auto* b=reinterpret_cast<uint64_t*>(arena+kDispatchBasesOffset+p*64);
            b[0]=0x30000000+p*65536;b[1]=0x40000000+p*65536;b[2]=10000;
        }
    }
"""
        + methods
        + r"""
};
template<unsigned Qps> void check(bool cache,bool scales,unsigned visit,unsigned peers,bool simt) {
    TestSender<16,Qps> s(cache,scales);
    s.td_.dispatch_experiment_flags=simt?kDispatchSimtScales:0;
    const unsigned fields=scales&&!simt?2:1;
    std::vector<Wqe> expected;
    for(unsigned epoch=0;epoch<3;++epoch) {
        int32_t tokens[16][47];unsigned cursor[16]{},field[16]{},total[16];
        for(unsigned p=0;p<16;++p) {
            total[p]=p==3?0:17+p*2;
            for(unsigned j=0;j<47;++j) tokens[p][j]=(j*7+p+epoch)%128;
        }
        bool remaining=true;
        while(remaining) {
            remaining=false;
            for(unsigned p=0;p<peers;++p) {
                unsigned left=visit;
                if(cursor[p]<total[p]) s.BeginVisit(p);
                while(left && cursor[p]<total[p]) {
                    auto available=(total[p]-cursor[p])*fields-field[p];
                    auto request=std::min(left,available);
                    unsigned row=epoch*100+cursor[p],q=s.QueuePeer(p);
                    auto n=s.PutRows(p,tokens[p]+cursor[p],row,field[p],request,available);
                    assert(n && n<=request);
                    for(unsigned j=0;j<n;++j) {
                        unsigned pos=field[p]+j,token=tokens[p][cursor[p]+pos/fields];
                        bool scale=pos%fields!=0;
                        uint64_t bytes=scale?12:14;
                        Wqe w{};for(unsigned k=0;k<5;++k)w[k]=q*100+k;
                        w[5]=(scale?0x40000000ULL:0x30000000ULL)+q*65536+(row+pos/fields)*bytes;
                        w[6]=bytes;w[7]=(scale?0x20000000ULL:0x10000000ULL)+token*bytes;
                        expected.push_back(w);
                    }
                    cursor[p]+=(field[p]+n)/fields;field[p]=(field[p]+n)%fields;left-=n;
                }
                remaining|=cursor[p]<total[p];
            }
        }
        s.Flush();
    }
    assert(s.sent==expected && s.stats.wqes==expected.size());
    uint64_t traced=0,pressures=0;
    for(auto& t:s.trace_) {traced+=t[6];pressures+=t[8];}
    assert(traced==expected.size() && pressures==s.stats.credit_stalls && s.stats.credit_checks==s.credits);
    if(cache) assert(s.stats.refills>0);
    if(cache && peers==8 && Qps==1 && visit<32) assert(s.stats.hits>0);
}
template<unsigned Qps> void check_packed() {
    TestSender<16,Qps> s(false,true);
    s.td_.dispatch_experiment_flags=kDispatchPackedScales;
    int32_t token=7;
    for(unsigned p=0;p<16;++p) for(unsigned pass=0;pass<3;++pass) {
        s.BeginVisit(p);
        const unsigned q=s.QueuePeer(p),first=pass*5,count=17,base=41;
        s.PutRows(p,&token,base+first,0,1,1); // Still pending when scale staging starts.
        auto before=s.sent.size();
        s.PutPackedScales(p,first,count,base);
        assert(s.sent.size()==before+1 && s.bulk.size()==p*3+pass+1);
        const auto r=s.bulk.back();
        assert(r[0]==0x40000000ULL+q*65536+(base+first)*12);
        assert(r[1]==0x50000000ULL+(p*128+first)*12 && r[2]==count*12 && r[3]==q);
    }
    const auto n=s.bulk.size();s.PutPackedScales(0,0,0,0);assert(s.bulk.size()==n);
    bool rejected=false;try{s.PutPackedScales(0,120,17,0);}catch(const std::runtime_error&){rejected=true;}
    assert(rejected);rejected=false;
    try{s.PutPackedScales(0,0,17,9990);}catch(const std::runtime_error&){rejected=true;}
    assert(rejected && s.stats.wqes==n*2);
    uint64_t traced=0,batches=0;
    for(const auto& t:s.trace_){traced+=t[6];batches+=t[7];}
    assert(traced==n*2 && batches==n*2);
}
int main() {
    check_packed<1>();check_packed<2>();
    for(bool cache:{false,true})for(bool scales:{false,true})for(unsigned v:{4,8,16,32})for(unsigned peers:{8,16}) {
        for(bool simt:{false,true}) {
            check<1>(cache,scales,v,peers,simt);check<2>(cache,scales,v,peers,simt);
        }
    }
}
"""
    )
    binary = tmp_path / "cache"
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_simt_dispatch_scales_preserve_bits_and_unique_rows(tmp_path):
    """Execute the real VF against a checked symmetric-memory transport double."""
    import shutil
    from tests.utils.cpp import cpp_functions

    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    vf = cpp_functions(
        (ROOT / "csrc/clos/kernels/dispatch/udma_dispatch_metadata.h").read_text(),
        "DispatchScalesVf",
    )
    source = tmp_path / "scales.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <cstdint>
#include <vector>
#include <array>
#include <stdexcept>
#define __simt_vf__
#define __launch_bounds__(n)
#define __gm__
constexpr unsigned kDispatchMetadataSimtThreads=128;
struct { unsigned x; } threadIdx;
namespace __asc_simt_vf { void __trap() { throw std::runtime_error("invalid token"); } }
void asc_threadfence() {}
constexpr unsigned W=4,C=8,T=17,S=8;
std::vector<uint32_t> local;
std::array<std::vector<uint32_t>,W> output,expected,writes;
namespace simt {
void aclshmem_uint32_p(uint32_t* ptr,uint32_t value,unsigned peer) {
    auto index=ptr-local.data();
    assert(peer<W && index>=0 && unsigned(index)<output[peer].size());
    assert(++writes[peer][index]==1);
    output[peer][index]=value;
}
}
"""
        + vf
        + r"""
int main() {
    for(unsigned width:{1,31,32,33,56,65}) for(unsigned epoch=0;epoch<3;++epoch) {
        // Different source ranks and channel lengths, including empty channels.
        std::array<std::vector<int32_t>,W> plans,channels;
        int32_t ranks[W*W]{};
        unsigned offset[W]{};
        for(unsigned src=0;src<W;++src) {
            plans[src].assign(W*C*S,-1);channels[src].resize(W*C);
            for(unsigned dst=0;dst<W;++dst) {
                unsigned count=0;
                for(unsigned ch=0;ch<C;++ch) {
                    unsigned n=0;
                    for(unsigned token=ch*3;token<(ch+1)*3 && token<T;++token)
                        if((token+src*3+dst*5+epoch)%5<2)
                            plans[src][(dst*C+ch)*S+n++]=token;
                    count+=n;channels[src][dst*C+ch]=count;
                }
                offset[dst]+=count;ranks[src*W+dst]=offset[dst];
            }
        }
        const unsigned guard=8,capacity=W*T*width+guard*2;
        local.assign(capacity,0);
        for(unsigned dst=0;dst<W;++dst) {
            output[dst].assign(capacity,0xDEADBEEF);
            expected[dst]=output[dst];writes[dst].assign(capacity,0);
        }
        for(unsigned src=0;src<W;++src) {
            std::vector<uint32_t> scales(T*width);
            for(unsigned token=0;token<T;++token)for(unsigned s=0;s<width;++s)
                scales[token*width+s]=0x7F800001U+src*0x10000+token*width+s+epoch;
            // Independent token scan reference, without interpreting the send plan.
            for(unsigned dst=0;dst<W;++dst) {
                unsigned row=src?ranks[(src-1)*W+dst]:0;
                for(unsigned token=0;token<T;++token)if((token+src*3+dst*5+epoch)%5<2) {
                    if(src!=dst)for(unsigned s=0;s<width;++s)
                        expected[dst][guard+row*width+s]=scales[token*width+s];
                    ++row;
                }
            }
            for(unsigned ch=0;ch<C;++ch)for(unsigned lane=0;lane<128;++lane) {
                threadIdx.x=lane;
                DispatchScalesVf(scales.data(),plans[src].data(),ranks,channels[src].data(),
                    reinterpret_cast<uint8_t*>(local.data()),guard*4,src,W,C,ch,S,T,width);
            }
        }
        assert(output==expected);
    }
}
"""
    )
    binary = tmp_path / "scales"
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_unsignaled_payload_wrap_and_pressure_checkpoint(tmp_path):
    """Execute the sender methods against bounded queues, including shared SQs."""
    import shutil
    from tests.utils.cpp import cpp_functions

    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    sender = (ROOT / "csrc/clos/kernels/common/udma_shared_sender.h").read_text()
    methods = cpp_functions(sender, "EnsurePayloadCredit", "SubmitPayload").replace("protected:", "public:")
    source = tmp_path / "sender.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#define __aicore__
#define __gm__
#define __ubuf__
constexpr uint32_t kBatchWqes=32, kQpCount=32, kWqeBytes=64, EVENT_ID5=5;
constexpr int PIPE_MTE3=0, kDone=1;
struct aclshmemi_udma_queue_state_t { uint32_t sq_head=0, sq_tail=0, cqe_cnt=0, cq_tail=0; };
struct aclshmemi_udma_wq_ctx_t { uint64_t state_addr, buf_addr; uint32_t depth=128; };
namespace AscendC {
enum class TPosition { VECOUT };
template<class T> struct LocalTensor { struct { uint8_t logicPos; uint64_t bufferAddr; } address_; };
}
struct Peer { uint32_t peer; uint64_t state; aclshmemi_udma_wq_ctx_t* context; };
aclshmemi_udma_queue_state_t queues[2];
uint64_t flags[64];
unsigned checkpoints=0, quiets=0, copies=0, doors=0;
bool aclshmemi_udma_sq_has_credit(aclshmemi_udma_queue_state_t* q, uint32_t n) {
    return uint32_t(q->sq_head+n-q->sq_tail)+4 < 128;
}
void trap() { throw std::runtime_error("credit failure"); }
template<class T, int Pipe, const int& Config>
void aclshmemx_udma_qp_put_nbi(T* dst, T* src, uint64_t* scratch, uint32_t n, uint32_t peer, uint32_t qp, uint32_t) {
    assert(Config==kDone && n==1 && peer<3 && qp<2);
    assert(dst==flags+kQpCount+qp && src==flags);
    auto& q=queues[qp];
    assert(aclshmemi_udma_sq_has_credit(&q,1));
    ++q.sq_head; ++q.cqe_cnt; ++checkpoints;
    *dst=*src; scratch[0]=0xDEAD; // SDK scratch reuse before payload construction.
}
void aclshmemx_udma_qp_quiet(uint32_t, uint32_t qp) {
    auto& q=queues[qp];
    assert(q.cqe_cnt!=q.cq_tail);
    q.sq_tail=q.sq_head; q.cq_tail=q.cqe_cnt; ++quiets;
}
void aclshmemi_udma_copy_wqe_from_ub(uint8_t* dst, AscendC::LocalTensor<uint8_t>& v, uint32_t n, uint32_t) {
    assert(n && n%kWqeBytes==0);
    std::memcpy(dst,reinterpret_cast<void*>(v.address_.bufferAddr),n); ++copies;
}
void aclshmemi_udma_post_send_update_info(uint32_t head, aclshmemi_udma_wq_ctx_t*, aclshmemi_udma_queue_state_t* q) {
    assert(copies>doors); q->sq_head=head; ++doors;
}
struct Sender {
    Peer peers_[3]; uint32_t last_=0, qp_=0; uint8_t* scratch_; uint64_t* checkpoint_=flags;
"""
        + methods
        + r"""
};
int main() {
    alignas(32) uint8_t scratch[32*64]{}, rings[2][128*64]{};
    aclshmemi_udma_wq_ctx_t ctx[2]{{reinterpret_cast<uint64_t>(&queues[0]),reinterpret_cast<uint64_t>(rings[0])},
                                 {reinterpret_cast<uint64_t>(&queues[1]),reinterpret_cast<uint64_t>(rings[1])}};
    Sender s{{{1,reinterpret_cast<uint64_t>(&queues[0]),&ctx[0]},
              {2,reinterpret_cast<uint64_t>(&queues[0]),&ctx[0]},
              {1,reinterpret_cast<uint64_t>(&queues[1]),&ctx[1]}},0,0,scratch};
    // Same SQ shared by two peers; independent second QP. Force ring wrap and
    // uint32_t producer-counter wrap, with different short and full batches.
    queues[0].sq_head=queues[0].sq_tail=126;
    queues[1].sq_head=queues[1].sq_tail=UINT32_MAX-15;
    flags[0]=1;
    unsigned split_copies=0;
    for (unsigned i=0;i<300;++i) {
        s.last_=i%3; s.qp_=s.last_==2?1:0;
        auto& q=queues[s.qp_];
        s.EnsurePayloadCredit(s.last_);
        auto head=q.sq_head, cq=q.cqe_cnt;
        uint32_t n=i%5?32:3;
        for (uint32_t j=0;j<n*64;++j) scratch[j]=uint8_t(i+j);
        auto before=copies;
        s.SubmitPayload(n);
        split_copies+=copies-before==2;
        assert(q.sq_head==uint32_t(head+n) && q.cqe_cnt==cq);
        for (uint32_t j=0;j<n;++j)
            assert(!std::memcmp(rings[s.qp_]+((head+j)%128)*64,scratch+j*64,64));
        // Pressure SOs never announce terminal receive readiness.
        for (unsigned j=1;j<kQpCount;++j) assert(flags[j]==0);
    }
    assert(split_copies && checkpoints && checkpoints<100 && quiets==checkpoints && doors==300);
}
"""
    )
    binary = tmp_path / "sender"
    subprocess.run([compiler, "-std=c++17", "-O2", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=30)


def test_tuning_visits_share_launcher_evidence(tmp_path):
    run_host(f"""
        sys.path.insert(0, 'benchmarks/clos')
        from bench_v2_bandwidth import publish_tuning_session
        from pathlib import Path
        import json
        root = Path({str(tmp_path)!r})
        first = root / 'bf16_expand0_aiv32_visit8'
        first.mkdir()
        (root / 'bf16_expand0_aiv32_visit16').mkdir()
        for name in ('command.json', 'resources.before.txt', 'run.log'):
            (first / name).write_text('shared evidence')
        (first / 'launcher_exit.json').write_text(json.dumps(dict(code=0, ended=123)))
        publish_tuning_session(first, [8,16])
        other = root / 'bf16_expand0_aiv32_visit16'
        assert (other/'run.log').read_text() == 'shared evidence'
        status = json.loads((other/'launcher_exit.json').read_text())
        assert status == dict(code=0, ended=123, shared_session=first.name)
        (first / 'launcher_exit.json').write_text(json.dumps(dict(code=1, ended=123)))
        try: publish_tuning_session(first, [8,16])
        except ValueError: pass
        else: raise AssertionError('failed launcher accepted')
    """)


def test_window_order_validation_and_manual_options():
    run_host("""
        from deep_ep.backends.clos.buffer import _validate_udma_peer_order as check
        from deep_ep.backends.clos.udma_policy import validate_manual_options
        for orders in [('group_window','group_window'), ('group_round_robin','group_window'), ('group_window','rank')]:
            got = check('udma', 'peer', 64, *orders, 8, 1, 2, 1)
            assert got[0] == (4 if orders[0] == 'group_window' else 3)
            validate_manual_options('peer', *orders, 8, 1, 2, 1, peers=4, wqes=4)
        for size, group in [(24,8), (64,4), (12,4)]:
            try:
                check('udma', 'peer', size, 'group_window','group_window',group,1,2,1)
            except ValueError:
                pass
            else:
                raise AssertionError('accepted unsupported POD window topology')
    """)


def test_window_order_coverage_ownership_and_qps(tmp_path):
    import shutil

    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    source = tmp_path / "window.cpp"
    source.write_text(r"""
#include <algorithm>
#include <cassert>
#include <vector>
#include "csrc/clos/common/udma_schedule.h"
using namespace ascend_deepep::udma;
int main() {
 for(unsigned world:{8,16,32,64,128,256,512}) for(unsigned rank=0;rank<world;++rank)
 for(unsigned peers:{1,2,4}) {
  if(peers*(world/8)>64) continue;
  std::vector<unsigned> seen(world);
  for(unsigned owner=0;owner<Workers(world,peers,kGroupWindowOrder,8);++owner) {
   std::vector<unsigned> actual,expected;
   auto count=WorkerCount(rank,world,owner,peers,kGroupWindowOrder,8);
   unsigned previous=0;
   for(unsigned i=0;i<count;++i) {
    unsigned peer=WorkerPeer(rank,world,owner,peers,i,kGroupWindowOrder,8);
    assert(peer<world && peer!=rank && !seen[peer]++);
    assert(Owner(rank,peer,world,peers,kGroupWindowOrder,8)==owner);
    assert(Owner(rank,peer,world,peers,kGroupRoundRobinOrder,8)==owner);
    unsigned window=PeerWindow(rank,peer,world,8);assert(window>=previous);previous=window;
    actual.push_back(peer);
    assert(PeerMask(rank,peer,world,4,4,8,peers,2)==PeerMask(rank,peer,world,3,3,8,peers,2));
   }
   // Independent POD source sequence, expanded to the owner's local positions.
   for(unsigned j=0;j<world/8;++j) {
    unsigned server;
    if(world>=128) server=(rank/64 ^ (rank%8)/4 ^ j/8)*8+(rank/8+1+j%8)%8;
    else if(world<32) server=(rank/8+1+j)%(world/8);
    else server=(rank/32 ^ j/4)*4+(rank/8+1+j%4)%4;
    for(unsigned local=owner;local<8;local+=Workers(world,peers,4,8))
     if(server*8+local!=rank) expected.push_back(server*8+local);
   }
   assert(actual==expected);
  }
  for(unsigned p=0;p<world;++p) assert(seen[p]==unsigned(p!=rank));
 }
}
""")
    binary = tmp_path / "window"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_dispatch_peer_cursors_preserve_row_ownership(tmp_path):
    """Actual cursor scheduler versus independent per-row ownership, with partial sends."""
    import shutil
    from tests.utils.cpp import cpp_functions

    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("host C++ compiler unavailable")
    header = (ROOT / "csrc/clos/kernels/dispatch/udma_dispatch_schedule.h").read_text()
    method = cpp_functions(header, "DispatchPeerCursors")
    source = tmp_path / "cursor.cpp"
    source.write_text(
        r"""
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>
#include "common/udma_schedule.h"
#define __aicore__
#define __gm__
using namespace ascend_deepep::udma;
struct DispatchTilingData {
    uint32_t rank=1,world_size=64,udma_peer_order=3,num_channels=0,udma_workers_per_peer=0,
        udma_cross_lane0_weight=0,udma_ranks_per_group=8,udma_wqes_per_visit=0;
};
struct Prefix { std::vector<int32_t> v; int32_t GetValue(unsigned i) {return v.at(i);} };
using Item=std::array<unsigned,4>;
struct Sender {
    unsigned fields,visit,left=0,calls=0,active=0;
    std::vector<Item> sent;
    unsigned Count() {return 4;} unsigned Rank(unsigned p) {return std::array<unsigned,4>{0,16,40,48}[p];}
    void Flush() {}
    void BeginVisit(unsigned p) {active=p;left=visit;}
    unsigned PutRows(unsigned p,const int32_t* indices,unsigned row,unsigned field,unsigned n,unsigned available) {
        assert(p==active && n && n<=left && n<=available);
        n=std::min(n,1+(calls++%7)); // Partial batch and split scale fields.
        for(unsigned i=0;i<n;++i) sent.push_back({Rank(p),unsigned(indices[(field+i)/fields]),row+(field+i)/fields,(field+i)%fields});
        left-=n;return n;
    }
};
"""
        + method
        + r"""
int main() {
 for(unsigned tokens:{0,1,257,1025}) for(unsigned channels:{1,3,8})
 for(unsigned workers:{1,2}) for(unsigned weight:{1,2,3,4,5,8}) for(unsigned fields:{1,2})
 for(unsigned visit:{4,8,16,32}) for(unsigned order:{3,4}) {
    DispatchTilingData td;td.udma_peer_order=order;td.num_channels=channels;td.udma_workers_per_peer=workers;
    td.udma_cross_lane0_weight=weight;td.udma_wqes_per_visit=visit;
    unsigned width=(tokens+channels-1)/channels,stride=(width+7)/8*8;
    std::vector<int32_t> plan(64*channels*stride,-999);
    Prefix rp,cp;rp.v.resize(64*64);cp.v.resize(64*channels);
    for(unsigned peer=0;peer<64;++peer) rp.v[peer]=100+peer*10000;
    std::vector<std::vector<Item>> expected(workers);
    for(unsigned p=0;p<4;++p) {
        unsigned peer=std::array<unsigned,4>{0,16,40,48}[p],total=0;std::vector<Item> all;std::vector<unsigned> chs;
        for(unsigned ch=0;ch<channels;++ch) {
            unsigned pos=0;
            for(unsigned t=ch*width;t<std::min(tokens,(ch+1)*width);++t) {
                if(p==1 || (t+peer)% (p+2)) continue; // Empty and skew peers.
                plan[(peer*channels+ch)*stride+pos++]=t;
                all.push_back({peer,t,rp.v[peer]+total++,0});chs.push_back(ch);
            }
            cp.v[peer*channels+ch]=total;
        }
        for(unsigned index=0;index<all.size();++index) {
            // Independent ownership definition; no call to PeerRowBoundary.
            unsigned owner=chs[index]%workers;
            if(weight!=1) {
                unsigned first_lane_weight=peer/8==td.rank/8?1:weight;
                unsigned split=uint64_t(total)*first_lane_weight/(first_lane_weight+workers-1);
                owner=workers==1 || index<split?0:1;
            }
            for(unsigned f=0;f<fields;++f) {auto item=all[index];item[3]=f;expected[owner].push_back(item);}
        }
    }
    for(unsigned lane=0;lane<workers;++lane) {
        Sender sender{fields,visit};
        DispatchPeerCursors<8>(sender,td,lane,plan.data(),rp,cp,stride,fields);
        if(order==4) {
            unsigned previous=0;
            for(auto item:sender.sent) {unsigned window=PeerWindow(td.rank,item[0],td.world_size,8);assert(window>=previous);previous=window;}
        }
        std::sort(sender.sent.begin(),sender.sent.end());std::sort(expected[lane].begin(),expected[lane].end());
        assert(sender.sent==expected[lane]);
    }
 }
}
"""
    )
    binary = tmp_path / "cursor"
    subprocess.run(
        [compiler, "-std=c++17", "-O2", "-I", str(ROOT / "csrc/clos"), str(source), "-o", str(binary)], check=True
    )
    subprocess.run([str(binary)], check=True)
