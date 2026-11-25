#!/usr/bin/env python3
"""EP4 allocator-lifetime A/B probe; use the installed DeepEP wheel.

A releases the previous fresh-dispatch result immediately before FC1.
B keeps previous results alive until the end of a bounded window.
No record_stream, event, kernel, or allocator implementation is patched.
This is a synthetic lifetime intervention, not a training correctness test.
"""

import argparse
import contextlib
import csv
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import time


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=2816)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--experts", type=int, default=256)
    p.add_argument("--sms", type=int, default=32)
    p.add_argument("--fc1-out", type=int, default=1536)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--windows", type=int, default=5)
    p.add_argument("--window-size", type=int, default=4)
    p.add_argument("--order", choices=("AB", "BA"), default="AB")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--huge-pages", action="store_true")
    p.add_argument("--first-device", type=int, default=0)
    p.add_argument("--seed", type=int, default=20261125)
    args = p.parse_args()
    for key in ("tokens", "hidden", "topk", "experts", "sms", "fc1_out", "windows", "window_size"):
        if getattr(args, key) <= 0:
            p.error(f"{key} must be positive")
    if args.topk > args.experts or args.experts % 4 or args.warmup < 0:
        p.error("require topk <= experts, experts divisible by EP4, warmup >= 0")
    return args


def describe(values):
    ordered = sorted(values)
    return dict(
        mean=statistics.mean(values),
        median=statistics.median(values),
        p95=ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        max=max(values),
    )


def main():
    args = arguments()
    # Deliberately do not add a source checkout to sys.path.
    import torch
    import torch.distributed as dist
    import torch_npu
    import deep_ep
    from deep_ep.backends.global_clos import ElasticBuffer

    if int(os.environ.get("WORLD_SIZE", "0")) != 4:
        raise RuntimeError("Launch with torchrun, WORLD_SIZE=4 (EP4)")
    rank = int(os.environ["RANK"])
    device = args.first_device + int(os.environ["LOCAL_RANK"])
    torch.set_num_threads(1)
    torch.npu.set_device(device)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=300))
    dest = args.output / f"rank{rank}"
    dest.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator().manual_seed(args.seed + rank)
    # Fixed, unique top-k experts per token. Route generation is outside measurement.
    idx = torch.rand(args.tokens, args.experts, generator=generator).topk(args.topk, dim=1).indices
    idx = idx.to(dtype=deep_ep.topk_idx_t, device="npu")
    weights = torch.softmax(torch.randn(args.tokens, args.topk, generator=generator), dim=1).to("npu")
    x = torch.randn(args.tokens, args.hidden, generator=generator, dtype=torch.bfloat16).to("npu")
    matrix = torch.randn(args.fc1_out, args.hidden, generator=generator, dtype=torch.bfloat16).to("npu")
    matrix_t = matrix.t()
    buffer = ElasticBuffer(
        dist.group.WORLD,
        num_max_tokens_per_rank=args.tokens,
        hidden=args.hidden,
        num_topk=args.topk,
        transport="udma",
        explicitly_destroy=True,
        use_1g_huge_pages=args.huge_pages,
    )
    dispatch_source = inspect.getsourcefile(buffer.dispatch)
    identity = dict(
        rank=rank,
        device=device,
        torch=torch.__version__,
        torch_npu=getattr(torch_npu, "__version__", "unknown"),
        deep_ep_file=deep_ep.__file__,
        dispatch_file=dispatch_source,
        dispatch_sha256=hashlib.sha256(Path(dispatch_source).read_bytes()).hexdigest(),
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        environment={
            k: v for k, v in os.environ.items() if k.startswith(("DEEPEP_", "SHMEM_", "PYTORCH_NPU_ALLOC_CONF"))
        },
        precision_checked=False,
        l2_flush=False,
        note="Fresh dispatch + independent torch.mm; no backward, no internal temporary capture",
    )
    (dest / "identity.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")

    def dispatch():
        return buffer.dispatch(
            x,
            topk_idx=idx,
            topk_weights=weights,
            num_experts=args.experts,
            num_sms=args.sms,
            do_expand=False,
            expert_alignment=1,
            async_with_compute_stream=True,
        )

    def mark(name):
        return torch.autograd.profiler.record_function(name) if args.profile else contextlib.nullcontext()

    def synchronize_window():
        torch.npu.synchronize()
        dist.barrier()

    rows = []
    with torch.no_grad():
        for _ in range(args.warmup):
            result = dispatch()
            fc1 = torch.mm(x, matrix_t)
            result[-1].current_stream_wait()
            torch.npu.synchronize()
            del result, fc1
        synchronize_window()
        profiler = None
        if args.profile:
            profiler = torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                record_shapes=True,
                profile_memory=False,
                with_stack=False,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(dest / "profiler")),
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
                    export_type=[torch_npu.profiler.ExportType.Text],
                    data_simplification=False,
                ),
            )
            profiler.start()

        for mode in args.order:
            for window in range(args.windows):
                synchronize_window()
                # Prime a real previous result, including handle/event. Both groups
                # start with one live result; only its release timing will differ.
                previous = [dispatch()]
                previous[0][-1].current_stream_wait()
                torch.npu.synchronize()
                retained = []
                for sample in range(args.window_size):
                    # Align ranks and finish preceding work outside the timed region.
                    # There is NO synchronize/barrier between dispatch and FC1.
                    synchronize_window()
                    tag = f"allocator_probe/{mode}/w{window:03d}/s{sample:03d}"
                    start = time.perf_counter_ns()
                    with mark(tag + "/dispatch"):
                        current = dispatch()
                    returned = time.perf_counter_ns()
                    with mark(tag + "/release_previous"):
                        if mode == "B":
                            retained.extend(previous)
                        previous.clear()
                    released = time.perf_counter_ns()
                    with mark(tag + "/fc1"):
                        fc1 = torch.mm(x, matrix_t)  # Allocate output identically in A/B.
                    submitted = time.perf_counter_ns()
                    # Consume the completion event AFTER independent compute submission.
                    # Keep the event wrapper: release_handle behavior is identical in A/B.
                    with mark(tag + "/drain"):
                        current[-1].current_stream_wait()
                        torch.npu.synchronize()
                    completed = time.perf_counter_ns()
                    rows.append(
                        dict(
                            mode=mode,
                            window=window,
                            sample=sample,
                            dispatch_host_us=(returned - start) / 1000,
                            release_host_us=(released - returned) / 1000,
                            fc1_host_us=(submitted - released) / 1000,
                            return_to_fc1_end_us=(submitted - returned) / 1000,
                            pair_host_us=(submitted - start) / 1000,
                            pair_completed_us=(completed - start) / 1000,
                            allocated_bytes=torch.npu.memory_allocated(),
                            reserved_bytes=torch.npu.memory_reserved(),
                        )
                    )
                    previous.append(current)
                    del current, fc1
                # All device work is complete before dropping retained references.
                # Cleanup is outside measured samples; no empty_cache / gc.collect.
                with mark(f"allocator_probe/{mode}/w{window:03d}/cleanup"):
                    previous.clear()
                    retained.clear()
                synchronize_window()
        if profiler is not None:
            profiler.stop()
        synchronize_window()

    with (dest / "samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    gathered = [None] * 4
    dist.all_gather_object(gathered, rows)
    if rank == 0:
        metrics = [k for k in rows[0] if k.endswith("_us")]
        summary = {}
        for mode in args.order:
            aligned = [[r for r in rank_rows if r["mode"] == mode] for rank_rows in gathered]
            summary[mode] = {
                metric: describe([max(rr[i][metric] for rr in aligned) for i in range(len(aligned[0]))])
                for metric in metrics
            }
        (args.output / "summary.json").write_text(
            json.dumps(
                dict(
                    aggregation="per matched sample max across ranks, then summarize",
                    profiled=args.profile,
                    modes=summary,
                    caveat="Host return-to-fc1-end includes FC1 host call, not actual kernel start; use trace.",
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2), flush=True)
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
