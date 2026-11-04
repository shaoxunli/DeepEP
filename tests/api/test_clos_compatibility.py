"""Host routing regressions; dtype-only doubles never execute device operations."""

from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[2]
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
        for invalid in (['--tuning-order','rank','--tuning-peers','2','--aiv','20'],
                        ['--tuning-order','rank','--tuning-workers','1'],
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
      for (uint32_t weight : {1U, 3U}) {
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
}
""".replace("#include <cassert>", "#include <cassert>\n#include <initializer_list>")
    )
    binary = tmp_path / "split"
    subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=30)


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
