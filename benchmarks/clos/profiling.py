"""Shared profiler sampling for the Clos bandwidth entrypoints."""

import os
import sys
from functools import wraps
from typing import Any, Callable

import torch_npu


class _ProfileComplete(Exception):
    pass


def _defer_combine_profile_output(fn):
    @wraps(fn)
    def run(*args, **kwargs):
        # BufferRuntime has already loaded this module. Do not initialize native
        # libraries just to profile a Python-only callable.
        native = sys.modules.get("deep_ep._clos_C")
        diagnostic_keys = (
            "DEEPEP_CLOS_COMM_STAGE_PROFILE", "DEEPEP_CLOS_COMBINE_PROFILE",
            "DEEPEP_CLOS_COMBINE_ARRIVAL_PROFILE",
        )
        diagnostics = os.environ.get("BENCH_COMM_DIAGNOSTICS", "off") != "off" or any(
            os.environ.get(key) == "1" for key in diagnostic_keys
        )
        available = native is not None and hasattr(native, "_begin_combine_profile_log_capture")
        if diagnostics and not available:
            raise RuntimeError("Rebuild the Clos native module for deferred diagnostic logs before profiling")
        capture = diagnostics and available
        if capture:
            native._begin_combine_profile_log_capture()
        try:
            return fn(*args, **kwargs)
        finally:
            # Normal completion reaches here after prof.stop() and device sync.
            # Also preserve partial records if sampling raises an exception.
            if capture:
                records = native._end_combine_profile_log_capture()
                if records:
                    # Keep each original line separate, as in native logging;
                    # one large write could interleave ranks inside a record.
                    for line in records.splitlines(keepends=True):
                        sys.stderr.write(line)
                        sys.stderr.flush()
    return run


def _append_sample_marker(message):
    # Markers and native records share one buffer to retain their ordering.
    sys.modules["deep_ep._clos_C"]._append_combine_profile_log(message + "\n")


def _consume_profile_result(result: Any) -> None:
    if not isinstance(result, (tuple, list)) or not result:
        return
    event = result[-1]
    if event is not None and getattr(event, "event", None) is not None:
        event.current_stream_wait(release_handle=True)


@_defer_combine_profile_output
def _profile_operator(
    rank: int,
    args,
    operator: str,
    fn: Callable[[], Any],
    *,
    sample_callback=None,
    guard_iters: int = 0,
    synchronize_each_iteration: bool = False,
    before_iteration=None,
) -> None:
    if guard_iters < 0 or (guard_iters and sample_callback is None):
        raise ValueError("profile guards require a sample callback and a nonnegative count")
    os.makedirs(args.profile_dir, exist_ok=True)

    for _ in range(args.profile_warmup):
        if before_iteration is not None:
            before_iteration()
        _consume_profile_result(fn())
        if synchronize_each_iteration:
            torch_npu.npu.synchronize()
    torch_npu.npu.synchronize()

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
        msprof_tx=True,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None,
        host_sys=[],
        sys_io=False,
        sys_interconnection=False,
    )
    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0,
            warmup=1,
            active=1,
            repeat=1,
            skip_first=0,
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(args.profile_dir),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        with_flops=False,
        experimental_config=experimental_config,
    )

    prof.start()
    for step in range(2):
        guards = guard_iters if step == 1 and sample_callback is not None else 0
        for sample in range(-guards, args.profile_iters + guards):
            # Keep rank alignment outside sample markers, including warmup and guards.
            if before_iteration is not None:
                before_iteration()
            if step == 1 and sample_callback is not None:
                sample_callback(sample, lambda: _consume_profile_result(fn()))
            else:
                _consume_profile_result(fn())
            # Stream waits only order device work. Drain each invocation on the
            # host as well so released large outputs can be reused instead of
            # accumulating while Python queues subsequent iterations. Keep this
            # outside the callback's end marker and all measured kernel times.
            if synchronize_each_iteration:
                torch_npu.npu.synchronize()
        torch_npu.npu.synchronize()
        prof.step()
    prof.stop()
    torch_npu.npu.synchronize()

    if rank == 0:
        detail = (
            f"measured_iters={args.profile_iters}; guard_iters_per_side={guard_iters}; "
            f"captured_iters={args.profile_iters + 2 * guard_iters}"
            if sample_callback is not None
            else f"iters_per_step={args.profile_iters}"
        )
        print(
            f"profile operator {operator} written to {args.profile_dir}: "
            f"{detail}; host_sync_each_iteration={synchronize_each_iteration}"
        )
    raise _ProfileComplete
