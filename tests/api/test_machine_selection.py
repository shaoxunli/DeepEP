"""Backend selection and keyword translation, without loading an NPU runtime."""

import copy
import inspect
import subprocess
import sys
from types import SimpleNamespace

import pytest
from deep_ep import _machine
from deep_ep._compat import keyword_aliases


def topology(kind="clos"):
    levels = [{"net_type": "CLOS", "rank_addr_list": [{"addr_type": "EID", "addr": "00" * 16, "ports": ["0/0"]}]}]
    edges = [{"local_a": str(i), "local_a_ports": ["0/0"], "topo_type": "CLOS"} for i in range(16)]
    if kind == "pod":
        levels.append(dict(levels[0], net_type="MESH"))
        edges.append({"local_a": "8", "local_b": "9", "local_a_ports": ["0/1"]})
    root = {"rank_list": [{"device_id": 0, "local_id": 8, "level_list": levels}]}
    return root, {"edge_list": edges}


def test_clos_sixteen_does_not_assume_eight_card_servers():
    root, topo = topology()
    assert _machine.classify_root(root, topo, 0) == "clos_only"
    assert _machine.classify_root(*topology("pod"), 0) == "mesh_and_clos"
    root["rank_list"][0]["level_list"][0]["net_type"] = "TOPO_FILE_DESC"
    assert _machine.classify_root(root, topo, 0) == "clos_only"


@pytest.mark.parametrize("fault", ["device", "duplicate", "empty", "eid", "mismatch", "pcie"])
def test_incomplete_or_ambiguous_evidence_is_rejected(fault):
    root, topo = topology()
    rank = root["rank_list"][0]
    if fault == "device":
        rank["device_id"] = 4
    elif fault == "duplicate":
        root["rank_list"].append(copy.deepcopy(rank))
    elif fault == "empty":
        rank["level_list"] = []
    elif fault == "eid":
        rank["level_list"][0]["rank_addr_list"][0]["addr"] = "invalid"
    elif fault == "mismatch":
        rank["level_list"][0]["net_type"] = "MESH"
    else:
        topo["edge_list"] = [{"local_a": "8", "local_a_ports": ["PCIe"]}]
    with pytest.raises(ValueError):
        _machine.classify_root(root, topo, 0)


def test_override_needs_no_runtime_or_rootinfo(monkeypatch):
    monkeypatch.setattr(_machine, "detect", lambda _: pytest.fail("unexpected detection"))
    assert _machine.select("clos_only")["source"] == "explicit"
    assert _machine.select("pod")["source"] == "explicit"


def test_consensus_rejects_mixed_backends_and_local_errors(monkeypatch):
    pytest.importorskip("torch.distributed")

    class Group:
        def allgather(self, record):
            return [record, {"selection": {"machine_type": "clos_only"}, "error": None}]

    monkeypatch.setitem(sys.modules, "mpi4py.MPI", SimpleNamespace(Comm=Group))
    with pytest.raises(ValueError, match="different"):
        _machine.select("pod", Group())
    monkeypatch.setattr(_machine, "detect", lambda _: (_ for _ in ()).throw(ValueError("bad topology")))
    with pytest.raises(RuntimeError, match="collectively"):
        _machine.select("auto", Group())


def test_torch_process_group_uses_object_collective(monkeypatch):
    dist = pytest.importorskip("torch.distributed")
    group = dist.ProcessGroup(0, 1)
    assert hasattr(group, "allgather")  # The real type that exposed the MPI misclassification.
    calls = []

    def gather(output, record, *, group):
        calls.append(group)
        output[:] = [record]

    monkeypatch.setattr(dist, "get_world_size", lambda actual: 1 if actual is group else pytest.fail("wrong group"))
    monkeypatch.setattr(dist, "all_gather_object", gather)
    assert _machine.select("clos_only", group)["machine_type"] == "clos_only"
    assert calls == [group]


def test_unrelated_allgather_object_is_rejected(monkeypatch):
    pytest.importorskip("torch.distributed")
    monkeypatch.delitem(sys.modules, "mpi4py.MPI", raising=False)

    class NotACommunicator:
        def allgather(self, record):
            pytest.fail("must not call an unrelated allgather method")

    with pytest.raises(TypeError, match="ProcessGroup or MPI"):
        _machine.select("pod", NotACommunicator())
    assert "mpi4py.MPI" not in sys.modules


def test_driver_timeout_is_not_retried(monkeypatch):
    calls = []

    def run(*args, **kwargs):
        calls.append(kwargs)
        raise subprocess.TimeoutExpired(args[0], 5)

    monkeypatch.setattr(_machine.subprocess, "run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        _machine.board_id(0)
    assert len(calls) == 1 and calls[0]["timeout"] == 5


def test_unknown_board_requires_override(monkeypatch, tmp_path):
    monkeypatch.setattr(_machine, "physical_device", lambda: 0)
    monkeypatch.setattr(_machine, "board_id", lambda _: 0x46)
    # Explicit absent files are errors, never silently replaced by board guesses.
    with pytest.raises(RuntimeError, match="explicit rootinfo"):
        _machine.select("auto", rootinfo_path=tmp_path / "absent")


def test_aliases_preserve_defaults_and_reject_conflicts():
    def original(a=20, *, flag=False):
        return a, flag

    call = keyword_aliases(original, {"num_sms": "a", "num_aiv": "a"})
    assert call(num_sms=32) == (32, False)
    assert call(32, num_aiv=32, flag=True) == (32, True)
    with pytest.raises(TypeError, match="conflicting"):
        call(a=20, num_sms=32)
    with pytest.raises(TypeError, match="conflicting"):
        call(num_sms=32, num_aiv=64)
    with pytest.raises(TypeError, match="unexpected"):
        call(unknown=True)
    assert inspect.signature(call) == inspect.signature(original)


def test_backend_is_selected_once_and_constructor_defaults_are_preserved(monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "require_native", lambda _: None, raising=False)
    calls = []
    monkeypatch.setattr(_machine, "select", lambda *args: calls.append(args) or {"machine_type": "pod"})

    class ElasticBuffer(metaclass=_machine.MachineSelector):
        def __init__(self, group, num_device_timeout_secs=7):
            self.timeout = num_device_timeout_secs

        def operation(self):
            return self.timeout

    instance = ElasticBuffer(None, num_gpu_timeout_secs=9)
    assert instance.operation() == instance.operation() == 9
    assert len(calls) == 1
    assert instance.machine_type == "pod"
    monkeypatch.setitem(sys.modules, "deep_ep._clos_adapter", SimpleNamespace(buffer_class=lambda _: ElasticBuffer))
    assert isinstance(instance, ElasticBuffer)


def test_operation_aliases_do_not_inspect_signatures_on_each_call(monkeypatch):
    from deep_ep import _compat

    call = keyword_aliases(lambda a=32: a, {"num_sms": "a"})
    monkeypatch.setattr(_compat.inspect, "signature", lambda _: pytest.fail("hot-path reflection"))
    assert call(a=64) == 64
    assert call(num_sms=32) == 32
