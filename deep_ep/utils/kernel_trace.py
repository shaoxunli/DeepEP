"""Process-local Python monitor for native kernel trace sources."""

from __future__ import annotations

import atexit
from contextlib import ExitStack, contextmanager, suppress
from collections.abc import Mapping
import errno
import itertools
import json
import os
from pathlib import Path
import secrets
import socket
import stat
import threading
import time
import warnings
import weakref
import uuid


def _recorder_enabled():
    return os.environ.get("DEEPEP_TRACE_RECORDER_ENABLE", "0") == "1"


def diagnostic_config():
    """Read once at initialization; an empty mode preserves the production path."""
    mode = os.environ.get("DEEPEP_TRACE_DIAGNOSTIC_MODE", "")
    timing = os.environ.get("DEEPEP_TRACE_DIAGNOSTIC_TIMING", "0")
    if mode not in ("", "record_only", "metadata", "native", "json_discard", "json", "full") or timing not in (
        "0",
        "1",
    ):
        raise ValueError("invalid KernelTrace diagnostic mode/timing")
    if timing == "1" and not mode:
        raise ValueError("diagnostic timing requires a diagnostic mode")
    if mode == "record_only" and timing == "1":
        raise ValueError("record_only must run without diagnostic timing")
    return mode, timing == "1"


class KernelTrace:
    """Own log files and a shared polling thread for active native sources."""

    _lock = threading.RLock()
    _sources = {}
    _thread = None
    _stop = threading.Event()
    _benchmark_directory = None

    @classmethod
    def attach(cls, native_module, comm_id=None):
        if not _recorder_enabled():
            return None
        directory = cls._benchmark_directory
        production = directory is None
        mode, timing = ("", False) if production else diagnostic_config()
        if mode == "metadata":
            return None
        source = native_module.get_kernel_trace()
        if source is None:
            return None
        identity = source.identity
        with cls._lock:
            current = cls._sources.get(identity)
            if current is not None:
                if (current["mode"], current["timing"], current["recorder"] is not None) != (mode, timing, production):
                    raise ValueError("cannot change trace output or diagnostic mode while a trace source is active")
                if production:
                    try:
                        current["recorder"].attach(comm_id)
                    except Exception as error:
                        warnings.warn(f"DeepEP trace recorder unavailable: {error}", RuntimeWarning)
                        return None
                    groups = current["groups"]
                    groups[comm_id] = groups.get(comm_id, 0) + 1
                current["references"] += 1
                return identity
            stream = recorder = None
            recorder_attached = False
            try:
                calibration = source.calibration()
                poll_interval_ms = source.poll_interval_ms if production else 2
                diagnostic_path = None
                if production:
                    if not callable(getattr(source, "drain_production_batches", None)):
                        raise RuntimeError("KernelTrace requires the matching rebuilt native extension")
                    recorder = _TraceRecorder.open(source, calibration["device"], comm_id)
                    recorder.attach(comm_id)
                    recorder_attached = True
                else:
                    directory = Path(directory)
                    directory.mkdir(parents=True, exist_ok=True)
                    name = f"kernel-{socket.gethostname()}-{os.getpid()}-d{calibration['device']}-{time.time_ns()}"
                    if mode in ("", "record_only", "full"):
                        stream = (directory / f"{name}.jsonl").open("x", encoding="utf-8")
                        stream.write(json.dumps(calibration) + "\n")
                        stream.flush()
                    diagnostic_path = directory / f"diagnostic-{name.removeprefix('kernel-')}.json"
                cls._sources[identity] = dict(
                    source=source,
                    recorder=recorder,
                    groups={comm_id: 1} if production else {},
                    stream=stream,
                    references=1,
                    failed=False,
                    mode=mode,
                    timing=timing,
                    statistics={},
                    diagnostic_path=diagnostic_path,
                    device=calibration["device"],
                    slots=calibration.get("slots", 128),
                    deferred_drains=0,
                )
                if mode != "record_only" and cls._thread is None:
                    cls._stop.clear()
                    cls._thread = threading.Thread(
                        target=cls._poll, args=(poll_interval_ms / 1000,), name="deepep-kernel-trace", daemon=True
                    )
                    cls._thread.start()
                return identity
            except Exception as error:
                cls._sources.pop(identity, None)
                if cls._thread is not None and not cls._thread.is_alive():
                    cls._thread = None
                if stream is not None:
                    stream.close()
                if recorder_attached:
                    recorder.detach(comm_id)
                warnings.warn(f"KernelTrace logging unavailable: {error}", RuntimeWarning)
                return None

    @staticmethod
    def _drain(entry, closing=False, explicit=False):
        # Even if another source owns a polling thread, deferred sources are never collected early.
        if entry["mode"] == "record_only":
            if not closing and not explicit:
                return
            entry["deferred_drains"] += 1
        try:
            recorder = entry["recorder"]
            if recorder is not None:
                batches = entry["source"].drain_production_batches(recorder.bus_id, closing)
                recorder.write(batches)
                return
            if entry["mode"] not in ("", "record_only"):
                KernelTrace._diagnostic_drain(entry, closing)
                return
            text = entry["source"].drain_trace_json(closing)
            if text:
                entry["stream"].write(text)
                entry["stream"].flush()
        except Exception as error:
            if not entry["failed"]:
                warnings.warn(f"KernelTrace logging failed: {error}", RuntimeWarning)
                entry["failed"] = True

    @staticmethod
    def _diagnostic_drain(entry, closing):
        mode, timing = entry["mode"], entry["timing"]
        begin = time.perf_counter_ns() if timing else 0
        batch = entry["source"].drain_trace_diagnostic(mode != "native", timing, closing, mode != "json_discard")
        if timing:
            batch["drain_call_wall_ns"] = time.perf_counter_ns() - begin
        text = batch.pop("text")
        stats = entry["statistics"]
        for key, value in batch.items():
            stats[key] = stats.get(key, 0) + value
        if mode != "full" or not text:
            return
        begin = time.perf_counter_ns() if timing else 0
        entry["stream"].write(text)
        entry["stream"].flush()
        if timing:
            stats["write_flush_ns"] = stats.get("write_flush_ns", 0) + time.perf_counter_ns() - begin

    @classmethod
    def _poll(cls, wait_seconds):
        while not cls._stop.wait(wait_seconds):
            with cls._lock:
                for identity in cls._sources:
                    cls._drain(cls._sources[identity])

    @classmethod
    def collect_boundary(cls, identity):
        """Diagnostic-only collection after the caller synchronizes, outside API timing."""
        if identity is None:
            return
        with cls._lock:
            entry = cls._sources[identity]
            cls._drain(entry, explicit=True)
            if entry["failed"]:
                raise RuntimeError("KernelTrace boundary collection failed")

    @classmethod
    def detach(cls, identity, comm_id=None, *, synchronized=False):
        with cls._lock:
            entry = cls._sources.get(identity)
            if entry is None:
                return
            recorder = entry["recorder"]
            if recorder is not None:
                groups = entry["groups"]
                if comm_id is None:
                    raise ValueError("comm_id is required when releasing a production trace group")
                group_references = groups.get(comm_id, 0)
                if not group_references:
                    return

            entry["references"] -= 1
            if not entry["references"]:
                cls._close_source(identity, entry, synchronized=synchronized)
                return

            if recorder is None:
                return

            try:
                if group_references == 1:
                    try:
                        if not synchronized:
                            entry["source"].synchronize()
                    finally:
                        cls._drain(entry)
            finally:
                if group_references == 1:
                    del groups[comm_id]
                else:
                    groups[comm_id] = group_references - 1
                recorder.detach(comm_id)

    @classmethod
    def _close_source(cls, identity, entry, *, synchronized=False):
        try:
            try:
                if not synchronized:
                    entry["source"].synchronize()
            finally:
                cls._drain(entry, closing=True)
        finally:
            try:
                recorder = entry["recorder"]
                if recorder is not None:
                    for comm_id, references in entry["groups"].items():
                        for _ in range(references):
                            recorder.detach(comm_id)
                if entry["stream"] is not None:
                    entry["stream"].close()
                if entry["mode"] not in ("", "record_only"):
                    report = dict(
                        mode=entry["mode"],
                        timing=entry["timing"],
                        pid=os.getpid(),
                        device=entry["device"],
                        failed=entry["failed"],
                        counters=entry["statistics"],
                    )
                    entry["diagnostic_path"].write_text(json.dumps(report, indent=2), encoding="utf-8")
            finally:
                del cls._sources[identity]

    @classmethod
    def shutdown(cls):
        cls._stop.set()
        thread = cls._thread
        if thread is not None:
            thread.join()
        with cls._lock:
            for identity in list(cls._sources):
                try:
                    cls._close_source(identity, cls._sources[identity])
                except Exception as error:
                    warnings.warn(f"KernelTrace shutdown failed: {error}", RuntimeWarning)
            cls._thread = None


atexit.register(KernelTrace.shutdown)


_identity_lock = threading.Lock()
_identities = itertools.count(1)
_process_id = uuid.uuid4().hex


class DispatchPythonProbe:
    """Opt-in single-thread diagnostic; retains only preallocated timestamp rows, never tensors."""

    version = 1
    points = (
        "entry",
        "dependency_begin",
        "dependency_end",
        "trace_created",
        "trace_entered",
        "stream_entered",
        "layout_begin",
        "layout_end",
        "args_ready",
        "native_end",
        "epilogue_end",
        "stream_exited",
        "trace_exited",
        "handle_end",
        "outputs_recorded",
        "finish_event_begin",
        "finish_event_end",
        "return",
        "metadata_begin",
        "identity_done",
        "template_lookup_done",
        "template_done",
        "dynamic_ready",
        "push_done",
        "restore_begin",
        "restore_done",
    )

    def __init__(self, capacity, clock=time.perf_counter_ns):
        if type(capacity) is not int or not 0 < capacity <= 1000000:
            raise ValueError("invalid Python dispatch probe capacity")
        self.rows = [[-1] * len(self.points) for _ in range(capacity)]
        self.indices = {name: i for i, name in enumerate(self.points)}
        self.clock = clock
        self.owner = threading.get_ident()
        self.used = 0
        self.active = None

    def begin(self):
        if self.owner != threading.get_ident() or self.active is not None:
            raise RuntimeError("Python dispatch probe requires non-nested calls on its owner thread")
        if self.used == len(self.rows):
            raise RuntimeError("Python dispatch probe capacity exhausted")
        self.active = self.rows[self.used]
        self.used += 1
        self.mark("entry")

    def mark(self, point):
        self.active[self.indices[point]] = self.clock()

    def finish(self):
        self.mark("return")
        self.active = None

    def take(self):
        if self.owner != threading.get_ident() or self.active is not None:
            raise RuntimeError("incomplete or wrong-thread Python dispatch probe export")
        return dict(
            version=self.version,
            samples=[
                dict(ordinal=i + 1, ticks_ns=dict(zip(self.points, row)))
                for i, row in enumerate(self.rows[: self.used])
            ],
        )


def _next_identity():
    with _identity_lock:
        return next(_identities)


def _global_rank():
    import torch.distributed as dist

    return dist.get_rank() if dist.is_available() and dist.is_initialized() else None


class _CallMetadata(Mapping):
    """Lazy read-only mapping; no per-call dict/ChainMap and no tensor ownership."""

    __slots__ = ("api", "handle", "dispatch_count", "template", "base")

    def __init__(self, api, handle, template, base, dispatch_count=-1):
        self.api, self.handle, self.template, self.base = api, handle, template, base
        self.dispatch_count = dispatch_count

    def __getitem__(self, key):
        if key == "api_call_id":
            return self.api
        if key == "handle_id":
            return self.handle
        if key == "dispatch_count" and self.dispatch_count >= 0:
            return self.dispatch_count
        if key in self.template:
            return self.template[key]
        return self.base[key]

    def __iter__(self):
        yield "api_call_id"
        yield "handle_id"
        if self.dispatch_count >= 0:
            yield "dispatch_count"
        yield from (k for k in self.template if k not in ("api_call_id", "handle_id"))
        yield from (k for k in self.base if k not in self.template and k not in ("api_call_id", "handle_id"))

    def __len__(self):
        return sum(1 for _ in self)


class KernelTraceSession:
    """Host-only metadata for one Buffer; each call is snapshotted by native Reserve."""

    def __init__(self, native, comm_id, ep_size, global_rank=None, *, recorder=False):
        if getattr(native, "kernel_trace_metadata_version", 0) != 2:
            raise RuntimeError("KernelTrace metadata requires the matching rebuilt native extension (version 2)")
        self.native = native
        self.diagnostic_mode, self.diagnostic_timing = ("", False) if recorder else diagnostic_config()
        if self.diagnostic_mode == "record_only":
            self.diagnostic_mode = ""  # Match production metadata work, without diagnostic per-call counters.
        self.diagnostic_stats = {}
        self.fields = dict(
            metadata_version=1,
            comm_id=comm_id,
            buffer_id=_next_identity(),
            process_id=_process_id,
            hostname=socket.gethostname(),
            ep_size=ep_size,
        )
        if global_rank is not None:
            self.fields["global_rank"] = global_rank
        self._dispatch_counts = itertools.count() if recorder else None
        self._base = (
            native.make_trace_metadata(self.fields, recorder=True) if recorder else native.make_trace_metadata(self.fields)
        )
        self._templates = {}
        self._element_bytes = {}

    def call_shape(
        self, api_name, x, *, num_tokens, num_experts, topk, handle=None, cached=False, expanded=False, capacity=0
    ):
        """Start a call from routing dimensions when the API has no route tensor."""
        return self.call(
            api_name,
            x,
            None,
            num_experts,
            handle,
            cached=cached,
            expanded=expanded,
            capacity=capacity,
            _route_shape=(num_tokens, topk),
        )

    @contextmanager
    def call(
        self,
        api_name,
        x,
        routes,
        num_experts,
        handle=None,
        *,
        cached=False,
        expanded=False,
        capacity=0,
        _probe=None,
        _route_shape=None,
    ):
        if _probe is not None:
            _probe.mark("metadata_begin")
        started = time.perf_counter_ns() if self.diagnostic_timing else 0
        with _identity_lock:
            handle_id = getattr(handle, "_trace_handle_id", None) if handle is not None else None
            if handle_id is None:
                handle_id = next(_identities)
                if handle is not None:
                    handle._trace_handle_id = handle_id
            api_call_id = next(_identities)
            dispatch_count = -1
            if self._dispatch_counts is not None:
                dispatch_count = (
                    next(self._dispatch_counts)
                    if api_name == "dispatch"
                    else getattr(handle, "_trace_dispatch_count", -1)
                )
        # Cache by scalar shape/context only; never retain user tensors or handles.
        if _probe is not None:
            _probe.mark("identity_done")
        x_shape, dtype = x.shape, x.dtype
        route_shape = routes.shape if _route_shape is None else _route_shape
        element_bytes = self._element_bytes.get(dtype)
        if element_bytes is None:
            element_bytes = x.element_size()
            if len(self._element_bytes) >= 64:
                self._element_bytes.clear()
            self._element_bytes[dtype] = element_bytes
        key = (
            api_name,
            x_shape[1],
            element_bytes,
            dtype,
            None if num_experts is None else int(num_experts),
            route_shape[1],
            route_shape[0],
            x_shape[0],
            int(cached),
            int(expanded),
            int(capacity),
        )
        template = self._templates.get(key)
        if _probe is not None:
            _probe.mark("template_lookup_done")
        if template is None:
            fields = {
                name: value
                for name, value in zip(
                    (
                        "api_name",
                        "hidden_size",
                        "element_bytes",
                        "dtype",
                        "num_experts",
                        "topk",
                        "num_tokens",
                        "api_input_rows",
                        "cached",
                        "do_expand",
                        "token_capacity",
                    ),
                    key,
                )
                if value is not None
            }
            fields["dtype"] = str(fields["dtype"])
            template = (fields, self.native.make_trace_metadata(fields, self._base))
            if len(self._templates) >= 64:
                self._templates.clear()  # In-flight native snapshots own their parents independently.
            self._templates[key] = template
        if _probe is not None:
            _probe.mark("template_done")
        fields_view = _CallMetadata(api_call_id, handle_id, template[0], self.fields, dispatch_count)
        if _probe is not None:
            _probe.mark("dynamic_ready")
        if self._dispatch_counts is not None:
            previous = self.native.push_trace_metadata(template[1], api_call_id, handle_id, dispatch_count)
        else:
            previous = self.native.push_trace_metadata(template[1], api_call_id, handle_id)
        if _probe is not None:
            _probe.mark("push_done")
        if self.diagnostic_mode:
            kind = ("cached_dispatch" if cached else "fresh_dispatch") if api_name == "dispatch" else api_name
            stats = self.diagnostic_stats.setdefault(kind, dict(calls=0, enter_ns=0, restore_ns=0))
            stats["calls"] += 1
            if self.diagnostic_timing:
                stats["enter_ns"] += time.perf_counter_ns() - started
        try:
            yield fields_view
        finally:
            if _probe is not None:
                _probe.mark("restore_begin")
            started = time.perf_counter_ns() if self.diagnostic_timing else 0
            self.native.restore_trace_metadata(previous)
            if _probe is not None:
                _probe.mark("restore_done")
            if self.diagnostic_timing:
                stats["restore_ns"] += time.perf_counter_ns() - started


def attach_trace_session(native, group, rank, ep_size, all_gather, *, comm=None):
    """Agree once at Buffer initialization, including disabled ranks; no per-call collective."""
    enabled = _recorder_enabled()
    production = KernelTrace._benchmark_directory is None
    mode, timing = ("", False) if production else diagnostic_config()
    # Keep the uint64 as decimal text in the native metadata labels.
    proposal = str(secrets.randbits(64)) if enabled and rank == 0 else None
    peers = all_gather(group, comm, ep_size, (enabled, proposal, mode, timing))
    if any((peer[0], peer[2], peer[3]) != (enabled, mode, timing) for peer in peers):
        raise ValueError("all EP ranks must agree on DEEPEP_TRACE_RECORDER_ENABLE and diagnostic settings")
    if not enabled:
        return None, None
    if not production and getattr(native, "kernel_trace_diagnostic_version", 0) != 4:
        raise RuntimeError("KernelTrace requires the matching rebuilt native extension")
    try:
        global_rank = _global_rank() if comm is None else None
    except (ImportError, RuntimeError):
        global_rank = None
    comm_id = peers[0][1]
    session = KernelTraceSession(native, comm_id, ep_size, global_rank, recorder=production)
    identity = None if mode == "metadata" else KernelTrace.attach(native, comm_id)
    return (identity, session) if identity is not None or mode == "metadata" else (None, None)


def detach_trace_session(identity, session, *, synchronized=False):
    if identity is not None:
        KernelTrace.detach(identity, session.fields["comm_id"], synchronized=synchronized)


_TRACE_DEFAULT_DIRECTORY = "/dockerdata/.trmt/deepep/"
_TRACE_FILE_COUNT = 8
_TRACE_FILE_LIMIT = 1_000_000_000
_TRACE_WRITE_RETRIES = 3
_TRACE_WRITE_RETRY_DELAY_S = 0.003


def _valid_trace_comm_id(comm_id):
    return (
        isinstance(comm_id, str)
        and len(comm_id) <= 20
        and comm_id.isascii()
        and comm_id.isdecimal()
        and (len(comm_id) == 1 or comm_id[0] != "0")
        and int(comm_id) < 1 << 64
    )


def _check_trace_comm_id(comm_id):
    if not _valid_trace_comm_id(comm_id):
        raise ValueError("expected a canonical decimal uint64 trace communication ID")


def _valid_trace_uuid(value):
    return isinstance(value, str) and len(value) == 32 and all(c in "0123456789abcdef" for c in value)


def _trace_file_stem(comm_id, bus_id):
    return f"trmt-deepep-trace-{comm_id}-{bus_id}-"


def _close_trace_fd(fd):
    if fd is not None:
        with suppress(OSError):
            os.close(fd)


def _open_trace_directory(directory):
    # Resolve the requested path, then open each component relative to its
    # verified parent so a concurrent symlink replacement cannot redirect us.
    path = Path(directory).resolve()
    flags = os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    current = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            try:
                child = os.open(part, flags, dir_fd=current)
            except FileNotFoundError:
                with suppress(FileExistsError):
                    os.mkdir(part, mode=0o755, dir_fd=current)
                child = os.open(part, flags, dir_fd=current)
            _close_trace_fd(current)
            current = child
        return os.open(".", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=current)
    finally:
        _close_trace_fd(current)


def _check_private_trace_file(fd, name):
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RuntimeError(f"expected a regular trace file with one link: {name}")


def _flock_trace_file(flock, fd, operation):
    while True:
        try:
            flock(fd, operation)
            return
        except InterruptedError:
            pass


def _clean_trace_history(directory, bus_id, owner_prefix):
    for marker in os.listdir(directory):
        if not marker.startswith(owner_prefix):
            continue
        comm_id = marker[len(owner_prefix) :]
        # Older owner-v2 markers used a hexadecimal communication ID.
        if not (_valid_trace_comm_id(comm_id) or _valid_trace_uuid(comm_id)):
            continue
        try:
            info = os.stat(marker, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            continue
        stem = _trace_file_stem(comm_id, bus_id)
        for index in range(_TRACE_FILE_COUNT):
            name = f"{stem}{index}.txt"
            with suppress(FileNotFoundError):
                info = os.stat(name, dir_fd=directory, follow_symlinks=False)
                if stat.S_ISREG(info.st_mode):
                    os.unlink(name, dir_fd=directory)
        with suppress(FileNotFoundError):
            os.unlink(marker, dir_fd=directory)


def _truncate_trace_file(fd, size):
    while True:
        try:
            os.ftruncate(fd, size)
            return
        except InterruptedError:
            pass


class _LogRing:
    def __init__(self, directory, comm_id, bus_id, owner_marker):
        stem = _trace_file_stem(comm_id, bus_id)
        self.names = [f"{stem}{index}.txt" for index in range(_TRACE_FILE_COUNT)]
        self.active = None
        self.index = 0
        self.size = 0
        self.references = 1
        self.failed = False

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        created = []
        try:
            for name in self.names:
                fd = os.open(name, flags | os.O_APPEND, 0o644, dir_fd=directory)
                created.append(name)
                if self.active is None:
                    self.active = fd
                else:
                    _close_trace_fd(fd)
            _close_trace_fd(os.open(owner_marker, flags, 0o644, dir_fd=directory))
        except BaseException:
            self.close()
            for name in reversed(created):
                with suppress(OSError):
                    os.unlink(name, dir_fd=directory)
            raise

    def close(self):
        _close_trace_fd(self.active)
        self.active = None

    def _rotate(self, directory):
        index = (self.index + 1) % _TRACE_FILE_COUNT
        name = self.names[index]
        flags = os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
        fd = os.open(name, flags, dir_fd=directory)
        try:
            _check_private_trace_file(fd, name)
            _truncate_trace_file(fd, 0)
        except BaseException:
            _close_trace_fd(fd)
            raise
        self.close()
        self.active, self.index, self.size = fd, index, 0

    def append(self, directory, payload):
        if not payload:
            return
        if not payload.endswith(b"\n"):
            raise ValueError("trace batch must end at a complete JSONL line")
        data = memoryview(payload)
        start = 0
        while start < len(payload):
            available = _TRACE_FILE_LIMIT - self.size
            end = len(payload)
            if end - start > available:
                # Fit as many complete lines as possible before rotating.
                end = payload.rfind(b"\n", start, start + available) + 1
                if end <= start:
                    next_end = payload.index(b"\n", start) + 1
                    if next_end - start > _TRACE_FILE_LIMIT:
                        raise RuntimeError("single trace record exceeds file limit")
                    self._rotate(directory)
                    continue
            self._write_chunk(payload, data, start, end)
            start = end

    def _write_chunk(self, payload, data, start, end):
        written = start
        retries = 0
        while written < end:
            try:
                count = os.write(self.active, data[written:end])
            except InterruptedError:
                continue
            except OSError as exc:
                error = exc
            else:
                if count > 0:
                    next_written = written + count
                    # The retry budget restarts after a complete line.
                    if retries and payload.find(b"\n", written, next_written) != -1:
                        retries = 0
                    written = next_written
                    continue
                error = OSError(errno.EIO, os.strerror(errno.EIO))
            if retries == _TRACE_WRITE_RETRIES:
                # Keep the complete records committed by a partial write.
                complete = max(start, payload.rfind(b"\n", start, written) + 1)
                if complete != written:
                    _truncate_trace_file(self.active, self.size + complete - start)
                self.size += complete - start
                detail = error.strerror or str(error)
                raise OSError(error.errno, f"write trace file {self.names[self.index]}: {detail}") from error
            retries += 1
            time.sleep(_TRACE_WRITE_RETRY_DELAY_S)
        self.size += written - start


class _TraceRecorder:
    """Manage a physical device's production logs under the monitor's lock."""

    _devices = weakref.WeakValueDictionary()
    _writers = weakref.WeakValueDictionary()

    def __init__(self, bus_id, device_uuid, directory):
        self.bus_id = bus_id
        self.owner = os.getpid()
        self.groups = {}
        self.directory = self.device_lock = None

        # Keep non-Linux and diagnostic imports independent of the writer.
        import fcntl

        self._flock = fcntl.flock
        self._unlock = fcntl.LOCK_UN
        device_key = f"{device_uuid}-{bus_id}"
        self.owner_prefix = f".trmt-deepep-trace-owner-v2-{device_key}-{bus_id}-"
        with ExitStack() as resources:
            directory_fd = _open_trace_directory(directory)
            resources.callback(_close_trace_fd, directory_fd)
            name = f".trmt-deepep-trace-device-{device_key}.lock"
            flags = os.O_RDONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
            device_lock = os.open(name, flags, 0o644, dir_fd=directory_fd)
            resources.callback(_close_trace_fd, device_lock)
            _check_private_trace_file(device_lock, name)
            _flock_trace_file(self._flock, device_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _clean_trace_history(directory_fd, bus_id, self.owner_prefix)
            resources.pop_all()  # The recorder now owns both descriptors.
            self.directory, self.device_lock = directory_fd, device_lock

    @classmethod
    def open(cls, source, device, comm_id):
        _check_trace_comm_id(comm_id)
        recorder = cls._devices.get(device)
        if recorder is None or recorder.directory is None:
            bus_id, device_uuid = source.device_identity()
            if not isinstance(bus_id, int) or not 0 <= bus_id < 1 << 32:
                raise ValueError("invalid trace physical device ID")
            if not _valid_trace_uuid(device_uuid) or device_uuid == "0" * 32:
                raise ValueError("invalid trace device UUID")
            key = (bus_id, device_uuid)
            recorder = cls._writers.get(key)
            if recorder is None or recorder.directory is None:
                directory = os.environ.get("DEEPEP_TRACE_DIR") or _TRACE_DEFAULT_DIRECTORY
                recorder = cls(bus_id, device_uuid, directory)
                cls._writers[key] = recorder
            recorder._check_owner()
            cls._devices[device] = recorder
        else:
            recorder._check_owner()
        return recorder

    def _check_owner(self):
        if self.owner != os.getpid():
            raise RuntimeError("cannot use inherited trace recorder")
        if self.directory is None:
            raise RuntimeError("trace recorder is closed")

    def close(self):
        if self.directory is None:
            return
        for group in self.groups.values():
            group.close()
        self.groups.clear()
        if self.device_lock is not None and self.owner == os.getpid():
            with suppress(OSError):
                _flock_trace_file(self._flock, self.device_lock, self._unlock)
        _close_trace_fd(self.device_lock)
        _close_trace_fd(self.directory)
        self.directory = self.device_lock = None

    def __del__(self):
        self.close()

    def attach(self, comm_id):
        self._check_owner()
        try:
            _check_trace_comm_id(comm_id)
            group = self.groups.get(comm_id)
            if group is None:
                self.groups[comm_id] = _LogRing(self.directory, comm_id, self.bus_id, self.owner_prefix + comm_id)
            else:
                group.references += 1
        except BaseException:
            if not self.groups:
                self.close()
            raise

    def detach(self, comm_id):
        self._check_owner()
        group = self.groups.get(comm_id)
        if group is not None:
            group.references -= 1
            if not group.references:
                del self.groups[comm_id]
                group.close()
                if not self.groups:
                    self.close()

    def write(self, batches):
        self._check_owner()
        failure = None
        for comm_id, payload in batches:
            group = self.groups.get(comm_id)
            if group is None or group.failed:
                continue
            try:
                group.append(self.directory, payload)
            except Exception as error:
                group.failed = True
                if failure is None:
                    failure = error
        if failure is not None:
            try:
                raise failure
            finally:
                failure = None  # Keep the exception traceback from retaining the writer.
