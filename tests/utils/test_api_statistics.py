# Copyright (c) 2026, Lu Lu
# Modified by Copilot 2026

"""CPU-only verification for deep_ep.utils.api_statistics.

Covers the legacy Buffer API statistics contract: log format, aggregation,
rotation, config file, environment parsing, failure isolation and thread safety.
No Torch, torch_npu or NPU device is required.
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from deep_ep.utils import api_statistics as stats_module
from deep_ep.utils.api_statistics import ApiStatisticsWorker, get_api_statistics_worker

LOG_LINE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}), "
    r"(get_dispatch_layout|intranode_dispatch|intranode_combine), "
    r"(\d+), (\d+)$"
)


def read_rows(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = LOG_LINE.match(line)
        assert match, f"unexpected log line: {line!r}"
        rows.append((match.group(2), int(match.group(3)), int(match.group(4))))
    return rows


@pytest.fixture
def worker_factory(tmp_path, monkeypatch):
    created = []
    counter = {"value": 0}

    def make(**kwargs):
        counter["value"] += 1
        log_dir = tmp_path / f"log{counter['value']}"
        monkeypatch.setenv("TRMT_LOG_PATH", str(log_dir))
        monkeypatch.setenv("TRMT_LOG_ENABLE", "true")
        monkeypatch.delenv("TRMT_LOG_CONSOLE_OUTPUT", raising=False)
        worker = ApiStatisticsWorker(**kwargs)
        created.append(worker)
        return worker

    yield make
    for worker in created:
        worker.stop()


def test_module_import_is_side_effect_free():
    worker = get_api_statistics_worker()
    assert worker._started is False
    assert worker._thread is None


def test_decorating_does_not_create_process_worker(monkeypatch):
    monkeypatch.setattr(stats_module, "_WORKER", None)

    @stats_module.record_api("get_dispatch_layout")
    def api():
        return "ok"

    assert api() == "ok"
    assert stats_module._WORKER is None


def test_initialize_once_for_concurrent_buffers(tmp_path, monkeypatch):
    monkeypatch.setattr(stats_module, "_WORKER", None)
    monkeypatch.setenv("TRMT_LOG_PATH", str(tmp_path / "log"))
    monkeypatch.setenv("TRMT_LOG_ENABLE", "true")
    barrier = threading.Barrier(8)
    workers = []

    def construct_buffer():
        barrier.wait()
        assert stats_module.initialize_api_statistics() is True
        workers.append(stats_module.get_api_statistics_worker())

    threads = [threading.Thread(target=construct_buffer) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len({id(worker) for worker in workers}) == 1
    worker = workers[0]
    assert worker._started is True
    assert worker._thread is not None
    worker.stop()


def test_buffer_construction_initializes_statistics(monkeypatch):
    from deep_ep.backends.direct import buffer as direct_buffer

    calls = []
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 2)

    class Native:
        def __init__(self, *args):
            pass

        def mte_mapping_errors(self):
            return []

    monkeypatch.setattr(direct_buffer, "_is_capturing", lambda: False)
    monkeypatch.setattr(direct_buffer, "get_native_class", lambda _: Native)
    monkeypatch.setattr(direct_buffer, "_prepare_shmem_bootstrap",
                        lambda *args, **kwargs: ([], "test"))
    monkeypatch.setattr(direct_buffer, "get_global_comm_stream",
                        lambda: SimpleNamespace(synchronize=lambda: None))
    monkeypatch.setattr(direct_buffer, "initialize_api_statistics",
                        lambda: calls.append("initialize"))

    instance = direct_buffer.Buffer(
        group, num_bytes=128 * 1024 * 1024, transport="mte",
        explicitly_destroy=True)
    assert calls == ["initialize"]
    instance._destroyed = True


def test_new_buffer_restarts_unexpectedly_stopped_writer(worker_factory):
    worker = worker_factory(interval=3600)
    assert worker.start() is True
    old_thread = worker._thread

    # Model an unexpected writer exit without marking the worker permanently stopped.
    worker._stop.set()
    worker._wake.set()
    assert old_thread is not None
    old_thread.join(timeout=1)
    assert worker._started is False

    assert worker.start() is True
    assert worker._thread is not None
    assert worker._thread is not old_thread
    assert worker._thread.is_alive()


def test_stop_flushes_after_unexpected_writer_exit(worker_factory):
    worker = worker_factory(interval=3600)
    assert worker.start() is True
    old_thread = worker._thread
    worker.decorate("intranode_combine")(lambda: None)()

    worker._stop.set()
    worker._wake.set()
    assert old_thread is not None
    old_thread.join(timeout=1)
    assert worker._started is False

    worker.stop()
    assert read_rows(worker.log_path) == [("intranode_combine", 1,
                                         read_rows(worker.log_path)[0][2])]


def test_single_call_when_clock_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(stats_module, "_WORKER", None)
    monkeypatch.setenv("TRMT_LOG_PATH", str(tmp_path / "log"))
    monkeypatch.setenv("TRMT_LOG_ENABLE", "true")
    assert stats_module.initialize_api_statistics() is True
    calls = []
    monkeypatch.setattr(stats_module.time, "perf_counter_ns",
                        lambda: (_ for _ in ()).throw(RuntimeError("clock failed")))

    @stats_module.record_api("intranode_dispatch")
    def api():
        calls.append(1)
        return "ok"

    assert api() == "ok"
    assert calls == [1]
    stats_module.get_api_statistics_worker().stop()


def test_records_count_and_average(worker_factory):
    worker = worker_factory()
    assert worker.start() is True

    @worker.decorate("get_dispatch_layout")
    def api():
        time.sleep(0.001)
        return "ok"

    for _ in range(5):
        assert api() == "ok"

    worker.flush()
    rows = read_rows(worker.log_path)
    assert len(rows) == 1
    name, count, average = rows[0]
    assert name == "get_dispatch_layout"
    assert count == 5
    assert average >= 1_000_000


def test_decorator_preserves_signature_and_return(worker_factory):
    worker = worker_factory()
    worker.start()

    class Api:
        @worker.decorate("intranode_dispatch")
        def dispatch(self, x, *, async_finish=False):
            """dispatch docstring"""
            return x, async_finish

    api = Api()
    assert api.dispatch(3, async_finish=True) == (3, True)
    assert Api.dispatch.__name__ == "dispatch"
    assert Api.dispatch.__doc__ == "dispatch docstring"


def test_flush_only_emits_counted_apis(worker_factory):
    worker = worker_factory()
    worker.start()
    worker.decorate("intranode_combine")(lambda: None)()
    worker.flush()
    assert [row[0] for row in read_rows(worker.log_path)] == ["intranode_combine"]


def test_drain_is_delta_based(worker_factory):
    worker = worker_factory()
    worker.start()
    api = worker.decorate("get_dispatch_layout")(lambda: None)
    api()
    worker.flush()
    worker.flush()
    assert len(read_rows(worker.log_path)) == 1


def test_business_exception_is_recorded_and_propagated(worker_factory):
    worker = worker_factory()
    worker.start()

    @worker.decorate("intranode_dispatch")
    def api():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        api()

    worker.flush()
    rows = read_rows(worker.log_path)
    assert rows == [("intranode_dispatch", 1, rows[0][2])]


def test_business_call_without_statistics_still_works_when_start_fails(tmp_path, monkeypatch):
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("TRMT_LOG_PATH", str(blocked / "log"))
    monkeypatch.setenv("TRMT_LOG_ENABLE", "true")
    worker = ApiStatisticsWorker()

    with pytest.warns(RuntimeWarning):
        assert worker.start() is False

    seen = []

    @worker.decorate("get_dispatch_layout")
    def api(value):
        seen.append(value)
        return value * 2

    assert api(21) == 42
    assert seen == [21]
    assert worker.log_path.exists() is False


def test_flush_failure_disables_statistics(worker_factory):
    worker = worker_factory()
    worker.start()
    api = worker.decorate("intranode_combine")(lambda: "ok")
    assert api() == "ok"

    worker.log_path.parent.mkdir(parents=True, exist_ok=True)
    worker.log_path.mkdir()

    with pytest.warns(RuntimeWarning):
        worker.flush()

    assert worker._failed is True
    assert api() == "ok"


def test_disabled_worker_decorator_is_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("TRMT_LOG_ENABLE", "false")
    monkeypatch.setenv("TRMT_LOG_PATH", str(tmp_path / "log"))
    worker = ApiStatisticsWorker()

    def api(a, b=2):
        return a + b

    assert worker.decorate("intranode_combine")(api) is api
    assert worker.start() is False
    assert (tmp_path / "log").exists() is False


def test_rotation_creates_numbered_backups(worker_factory):
    worker = worker_factory(max_file_bytes=1)
    worker.start()
    api = worker.decorate("get_dispatch_layout")(lambda: None)
    for _ in range(3):
        api()
        worker.flush()

    names = {path.name for path in worker.log_path.parent.glob("buffer_*.log")}
    assert {"buffer_0.log", "buffer_1.log"} <= names
    assert worker.log_path.exists()


def test_rotation_keeps_at_most_five_backups(worker_factory):
    worker = worker_factory(max_file_bytes=1)
    worker.start()
    api = worker.decorate("intranode_combine")(lambda: None)
    for _ in range(12):
        api()
        worker.flush()

    numbered = sorted(
        int(path.stem.split("_")[1])
        for path in worker.log_path.parent.glob("buffer_*.log")
    )
    assert len(numbered) <= stats_module._MAX_BACKUPS
    assert numbered == list(range(numbered[0], numbered[0] + len(numbered)))
    assert numbered[-1] == 10


def test_start_writes_config_json(worker_factory):
    worker = worker_factory()
    worker.start()
    config = worker.log_path.parent.parent / "deepep.config.json"
    data = json.loads(config.read_text(encoding="utf-8"))
    assert set(data) == {"deepep_version", "timestamp"}
    assert isinstance(data["deepep_version"], str) and data["deepep_version"]
    datetime.fromisoformat(data["timestamp"])
    assert config.with_suffix(".json.lock").exists()


def test_config_write_failure_does_not_disable_statistics(
        worker_factory, tmp_path, monkeypatch):
    worker = worker_factory()
    blocked = tmp_path / "config-blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(worker, "_config_path",
                        lambda: blocked / "deepep.config.json")

    with pytest.warns(RuntimeWarning):
        assert worker.start() is True
    assert worker._failed is False

    api = worker.decorate("get_dispatch_layout")(lambda: "ok")
    assert api() == "ok"
    worker.flush()
    assert read_rows(worker.log_path)[0][1] == 1


def test_console_output(worker_factory, capsys):
    worker = worker_factory()
    worker.console_output = True
    worker.start()
    worker.decorate("intranode_combine")(lambda: None)()
    worker.flush()
    assert "intranode_combine" in capsys.readouterr().out


def test_stop_flushes_remaining(worker_factory):
    worker = worker_factory(interval=3600)
    worker.start()
    api = worker.decorate("intranode_combine")(lambda: None)
    api()
    api()
    assert worker.log_path.exists() is False
    worker.stop()
    assert read_rows(worker.log_path)[0][1] == 2
    assert worker._started is False


def test_background_flush_on_interval(worker_factory):
    worker = worker_factory(interval=1)
    worker.start()
    worker.decorate("get_dispatch_layout")(lambda: None)()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not worker.log_path.exists():
        time.sleep(0.05)
    assert worker.log_path.exists()
    assert read_rows(worker.log_path)[0][1] == 1


def test_multithreaded_counts(worker_factory):
    worker = worker_factory(interval=3600)
    worker.start()
    api = worker.decorate("intranode_dispatch")(lambda: None)
    barrier = threading.Barrier(8)

    def run():
        barrier.wait()
        for _ in range(500):
            api()

    threads = [threading.Thread(target=run) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    worker.flush()
    assert read_rows(worker.log_path)[0][1] == 4000


def test_multithreaded_counts_across_periodic_flushes(worker_factory):
    worker = worker_factory(interval=1)
    assert worker.start() is True
    api = worker.decorate("intranode_dispatch")(lambda: None)
    result_lock = threading.Lock()
    calls = 0

    def call_wave():
        barrier = threading.Barrier(5)

        def run():
            nonlocal calls
            barrier.wait()
            for _ in range(300):
                api()
            with result_lock:
                calls += 300

        threads = [threading.Thread(target=run) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def count_written():
        if not worker.log_path.exists():
            return 0
        return sum(row[1] for row in read_rows(worker.log_path)
                   if row[0] == "intranode_dispatch")

    call_wave()
    deadline = time.monotonic() + 5
    while count_written() != 1500 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert count_written() == 1500

    call_wave()
    worker.stop()

    assert calls == 3000
    assert count_written() == 3000


def test_lock_released_after_sigkill(tmp_path):
    lock_path = tmp_path / "buffer.log.lock"
    code = """
import sys, time
from pathlib import Path
from deep_ep.utils.api_statistics import _FileLock
lock = _FileLock(Path(sys.argv[1]))
lock.__enter__()
print("locked", flush=True)
time.sleep(60)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(Path(__file__).resolve().parents[2]), environment.get("PYTHONPATH"))))
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(lock_path)],
        stdout=subprocess.PIPE, text=True, env=environment)
    assert child.stdout is not None
    assert child.stdout.readline().strip() == "locked"
    child.kill()
    assert child.wait() == -9

    started = time.monotonic()
    with stats_module._FileLock(lock_path):
        elapsed = time.monotonic() - started
    assert elapsed < 1.0
    assert lock_path.exists()


def test_unknown_api_names_rejected(worker_factory):
    worker = worker_factory()
    with pytest.raises(ValueError):
        worker.statistics.record("unknown_api", 1)
    with pytest.raises(ValueError):
        worker.decorate("unknown_api")


def test_record_rejects_invalid_elapsed(worker_factory):
    worker = worker_factory()
    with pytest.raises(ValueError):
        worker.statistics.record("get_dispatch_layout", -1)
    with pytest.raises(ValueError):
        worker.statistics.record("get_dispatch_layout", 1.5)


def test_worker_record_errors_disable_without_raising(worker_factory):
    worker = worker_factory()
    worker.start()
    with pytest.warns(RuntimeWarning):
        worker.record("unknown_api", 10)
    assert worker._failed is True
    worker.decorate("get_dispatch_layout")(lambda: None)()


def test_env_parsing(monkeypatch):
    monkeypatch.setenv("TRMT_LOG_INTERVAL", "abc")
    with pytest.warns(RuntimeWarning):
        assert stats_module._interval_env() == 60.0

    monkeypatch.setenv("TRMT_LOG_INTERVAL", "0.1")
    assert stats_module._interval_env() == 1.0

    monkeypatch.setenv("TRMT_LOG_INTERVAL", "5")
    assert stats_module._interval_env() == 5.0

    for value in ("true",):
        monkeypatch.setenv("TRMT_FLAG", value)
        assert stats_module._bool_env("TRMT_FLAG", False) is True

    for value in ("false",):
        monkeypatch.setenv("TRMT_FLAG", value)
        assert stats_module._bool_env("TRMT_FLAG", True) is False

    # Case-sensitive literal match only: anything else (including TRUE, True,
    # padded values and 1/0/yes/no) warns and falls back to the default.
    for value in ("TRUE", "True", " true ", "FALSE", "False", "false ",
                  "1", "0", "yes", "no", "on", "off", "maybe", ""):
        monkeypatch.setenv("TRMT_FLAG", value)
        with pytest.warns(RuntimeWarning):
            assert stats_module._bool_env("TRMT_FLAG", True) is True
        with pytest.warns(RuntimeWarning):
            assert stats_module._bool_env("TRMT_FLAG", False) is False
