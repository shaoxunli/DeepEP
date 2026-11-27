"""Low-overhead process-local statistics for legacy Buffer Python APIs."""

from __future__ import annotations

import atexit
from datetime import datetime
from functools import wraps
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import warnings

_API_NAMES = ("get_dispatch_layout", "intranode_dispatch", "intranode_combine")
_API_INDEX = {name: index for index, name in enumerate(_API_NAMES)}
_DEFAULT_LOG_DIR = Path("/dockerdata/.trmt/log")
_MAX_FILE_BYTES = 100 * 1024 * 1024
_MAX_BACKUPS = 5


def _bool_env(name: str, default: bool) -> bool:
    """Parse a boolean env var; the only accepted literals are ``true``/``false``."""
    value = os.environ.get(name)
    if value is None:
        return default
    if value == "true":
        return True
    if value == "false":
        return False
    warnings.warn(f"Invalid {name}={value!r}; using {default}", RuntimeWarning)
    return default


def _interval_env() -> float:
    try:
        return max(1.0, float(os.environ.get("TRMT_LOG_INTERVAL", "60")))
    except ValueError:
        warnings.warn("Invalid TRMT_LOG_INTERVAL; using 60", RuntimeWarning)
        return 60.0


def _package_version() -> str:
    """Installed DeepEP version, or ``source`` when metadata is unavailable."""
    try:
        return importlib.metadata.version("ascend-deepep")
    except importlib.metadata.PackageNotFoundError:
        return "source"
    except Exception:
        return "source"


class ApiStatistics:
    def __init__(self) -> None:
        self._local = threading.local()
        self._states = []
        self._registry_lock = threading.Lock()

    def record(self, api_name: str, elapsed_ns: int) -> None:
        api_index = _API_INDEX.get(api_name)
        if api_index is None:
            raise ValueError(f"unsupported API statistics name: {api_name}")
        if type(elapsed_ns) is not int or elapsed_ns < 0:
            raise ValueError("elapsed_ns must be a nonnegative integer")
        self.record_index(api_index, elapsed_ns)

    def record_index(self, index: int, elapsed_ns: int) -> None:
        state = getattr(self._local, "state", None)
        if state is None:
            state = [[(0, 0) for _ in _API_NAMES], [(0, 0) for _ in _API_NAMES]]
            with self._registry_lock:
                self._states.append(state)
            self._local.state = state
        values = state[0]
        count, total = values[index]
        values[index] = (count + 1, total + elapsed_ns)

    def drain(self) -> dict[str, tuple[int, int]]:
        result = {}
        with self._registry_lock:
            for index, name in enumerate(_API_NAMES):
                count = total = 0
                for values, previous in self._states:
                    current = values[index]
                    last = previous[index]
                    count += current[0] - last[0]
                    total += current[1] - last[1]
                    previous[index] = current
                if count:
                    result[name] = (count, total)
        return result


class _FileLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        if sys.platform != "win32":
            import fcntl
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.stream is not None:
            if sys.platform != "win32":
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()


class ApiStatisticsWorker:
    def __init__(self, interval: float | None = None,
                 log_path: Path | str | None = None,
                 max_file_bytes: int = _MAX_FILE_BYTES) -> None:
        directory = Path(os.environ.get("TRMT_LOG_PATH", str(_DEFAULT_LOG_DIR)))
        self.log_path = Path(log_path) if log_path is not None else directory / "buffer.log"
        self.interval = _interval_env() if interval is None else max(1.0, float(interval))
        self.max_file_bytes = max_file_bytes
        self.console_output = _bool_env("TRMT_LOG_CONSOLE_OUTPUT", False)
        self.statistics = ApiStatistics()
        self._enabled = _bool_env("TRMT_LOG_ENABLE", True)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread = None
        self._started = False
        self._failed = False
        self._stopped = False

    def start(self) -> bool:
        if not self._enabled or self._failed or self._stopped:
            return False
        with self._lock:
            if self._started:
                thread = self._thread
                if thread is not None and thread.is_alive():
                    return True
                if thread is not None:
                    thread.join()
                self._started = False
                self._thread = None
            if self._failed or self._stopped:
                return False
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                self._write_config()
                self._stop.clear()
                self._wake.clear()
                self._thread = threading.Thread(
                    target=self._run, name="deepep-api-statistics", daemon=True)
                self._thread.start()
                self._started = True
                return True
            except Exception as error:
                self._failed = True
                warnings.warn(f"Python API statistics unavailable: {error}", RuntimeWarning)
                return False

    def record(self, api_name: str, elapsed_ns: int) -> None:
        if not self._started or not self._enabled or self._failed:
            return
        try:
            self.statistics.record(api_name, elapsed_ns)
        except Exception as error:
            self.disable(f"recording failed: {error}")

    def decorate(self, api_name: str):
        if not self._enabled:
            return lambda function: function
        api_index = _API_INDEX.get(api_name)
        if api_index is None:
            raise ValueError(f"unsupported API statistics name: {api_name}")
        clock = time.perf_counter_ns
        record_index = self.statistics.record_index

        def apply(function):
            @wraps(function)
            def wrapper(*args, **kwargs):
                if self._failed:
                    return function(*args, **kwargs)
                try:
                    if not self._started:
                        self.start()
                    if not self._started:
                        return function(*args, **kwargs)
                    started = clock()
                except Exception:
                    return function(*args, **kwargs)
                try:
                    return function(*args, **kwargs)
                finally:
                    try:
                        record_index(api_index, clock() - started)
                    except Exception:
                        pass
            return wrapper
        return apply

    def disable(self, reason: str) -> None:
        with self._lock:
            if self._failed:
                return
            self._failed = True
        warnings.warn(f"Python API statistics disabled: {reason}", RuntimeWarning)

    def stop(self) -> None:
        with self._lock:
            self._stopped = True
            thread = self._thread
            if self._started:
                self._stop.set()
                self._wake.set()
            self._started = False
            self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        self.flush()

    def flush(self) -> None:
        self._flush()

    def _config_path(self) -> Path:
        """Config sits next to the log directory (requirement: .trmt/deepep.config.json)."""
        return self.log_path.parent.parent / "deepep.config.json"

    def _write_config(self) -> None:
        """Best-effort version/timestamp record; never disables statistics."""
        try:
            config = self._config_path()
            lock_path = config.with_suffix(config.suffix + ".lock")
            payload = json.dumps({
                "deepep_version": _package_version(),
                "timestamp": datetime.now().isoformat(timespec="microseconds"),
            }, sort_keys=True)
            with _FileLock(lock_path):
                config.write_text(payload + "\n", encoding="utf-8")
        except Exception as error:
            warnings.warn(
                f"Python API statistics config not written: {error}", RuntimeWarning)

    def _run(self) -> None:
        try:
            deadline = time.monotonic() + self.interval
            while not self._stop.is_set():
                self._wake.wait(max(0.0, deadline - time.monotonic()))
                self._wake.clear()
                if self._stop.is_set():
                    break
                if time.monotonic() >= deadline:
                    self._flush()
                    deadline = time.monotonic() + self.interval
        except Exception as error:
            if not self._stop.is_set():
                warnings.warn(
                    f"Python API statistics writer stopped unexpectedly: {error}",
                    RuntimeWarning,
                )
        finally:
            with self._lock:
                if self._thread is threading.current_thread():
                    self._started = False
                    self._thread = None

    def _flush(self) -> None:
        if self._failed:
            return
        values = self.statistics.drain()
        if not values:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        text = "".join(f"{timestamp}, {name}, {count}, {total // count}\n"
                       for name, (count, total) in values.items())
        try:
            lock_path = self.log_path.with_suffix(self.log_path.suffix + ".lock")
            with _FileLock(lock_path):
                encoded = text.encode("utf-8")
                if self.log_path.exists() and self.log_path.stat().st_size + len(encoded) > self.max_file_bytes:
                    self._rotate()
                with self.log_path.open("a", encoding="utf-8") as stream:
                    stream.write(text)
            if self.console_output:
                print(text, end="", flush=True)
        except Exception as error:
            self.disable(f"flush failed: {error}")

    def _rotate(self) -> None:
        numbered = []
        pattern = re.compile(r"^buffer_(\d+)\.log$")
        for path in self.log_path.parent.iterdir():
            match = pattern.fullmatch(path.name)
            if match:
                numbered.append((int(match.group(1)), path))
        numbered.sort()
        if len(numbered) >= _MAX_BACKUPS:
            numbered[0][1].unlink()
            numbered = numbered[1:]
        next_index = numbered[-1][0] + 1 if numbered else 0
        os.replace(self.log_path, self.log_path.with_name(f"buffer_{next_index}.log"))


_WORKER = None
_WORKER_LOCK = threading.Lock()


def _get_worker() -> ApiStatisticsWorker:
    global _WORKER
    worker = _WORKER
    if worker is None:
        with _WORKER_LOCK:
            worker = _WORKER
            if worker is None:
                worker = ApiStatisticsWorker()
                _WORKER = worker
    return worker


def initialize_api_statistics() -> bool:
    """Create and start the process singleton after a legacy Buffer is ready."""
    worker = _get_worker()
    return worker.start()


def get_api_statistics_worker() -> ApiStatisticsWorker:
    """Return the process singleton, creating it without starting the writer."""
    return _get_worker()


def record_api(api_name: str):
    api_index = _API_INDEX.get(api_name)
    if api_index is None:
        raise ValueError(f"unsupported API statistics name: {api_name}")
    clock = time.perf_counter_ns

    def apply(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            worker = _WORKER
            if worker is None or worker._failed or not worker._started:
                return function(*args, **kwargs)

            try:
                started = clock()
            except Exception:
                return function(*args, **kwargs)
            try:
                return function(*args, **kwargs)
            finally:
                try:
                    worker.statistics.record_index(api_index, clock() - started)
                except Exception:
                    pass
        return wrapper
    return apply


def _stop_worker() -> None:
    worker = _WORKER
    if worker is not None:
        worker.stop()


atexit.register(_stop_worker)
