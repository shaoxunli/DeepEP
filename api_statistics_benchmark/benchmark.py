"""Minimal real-NPU benchmark for the three legacy Buffer APIs."""

import json
import os
import statistics
import time

import deep_ep
import torch
import torch.distributed as dist
import torch_npu

TOKENS = 4096
HIDDEN = 2816
TOPK = 8
EXPERTS = 256
WARMUP = 10
MEASURE = 1000


def wait_completion(event):
    event.current_stream_wait()


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    print(f"worker rank={rank} before npu", flush=True)
    torch.npu.set_device(local_rank)
    device = torch.device("npu", local_rank)
    print(f"worker rank={rank} before gloo", flush=True)
    dist.init_process_group("gloo")
    print(f"worker rank={rank} gloo ready", flush=True)

    generator = torch.Generator(device=device)
    generator.manual_seed(20261126 + rank)
    x = torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16, device=device, generator=generator)
    indices = torch.randint(EXPERTS, (TOKENS, TOPK), dtype=torch.int64,
                            device=device, generator=generator)
    weights = torch.ones(TOKENS, TOPK, dtype=torch.float32, device=device)

    dispatch_config = deep_ep.Buffer.get_dispatch_config(world)
    combine_config = deep_ep.Buffer.get_combine_config(world)
    configs = (dispatch_config, combine_config)
    buffer = deep_ep.Buffer(
        dist.group.WORLD,
        num_scaleup_bytes=max(c.get_scaleup_buffer_size_hint(HIDDEN * 2, world)
                              for c in configs),
        num_scaleout_bytes=max(c.get_scaleout_buffer_size_hint(HIDDEN * 2, world)
                               for c in configs),
        explicitly_destroy=True,
        machine_type="mesh_pcie",
    )

    samples = {
        "get_dispatch_layout": [],
        "dispatch_fresh": [],
        "dispatch_cached": [],
        "intranode_combine": [],
    }

    for iteration in range(WARMUP + MEASURE):
        start_ns = time.perf_counter_ns()
        layout = buffer.get_dispatch_layout(indices, EXPERTS, async_finish=True)
        end_ns = time.perf_counter_ns()
        wait_completion(layout[-1])
        layout_ns = end_ns - start_ns

        start_ns = time.perf_counter_ns()
        fresh = buffer.dispatch(
            x,
            topk_idx=indices,
            topk_weights=weights,
            num_tokens_per_rank=layout[0],
            num_tokens_per_scaleout_rank=layout[1],
            num_tokens_per_expert=layout[2],
            is_token_in_rank=layout[3],
            config=dispatch_config,
            async_finish=True,
        )
        end_ns = time.perf_counter_ns()
        wait_completion(fresh[-1])
        fresh_ns = end_ns - start_ns

        start_ns = time.perf_counter_ns()
        cached = buffer.dispatch(
            x, handle=fresh[4], config=dispatch_config, async_finish=True)
        end_ns = time.perf_counter_ns()
        wait_completion(cached[-1])
        cached_ns = end_ns - start_ns

        start_ns = time.perf_counter_ns()
        combined = buffer.combine(
            cached[0], fresh[4], topk_weights=cached[2],
            config=combine_config, async_finish=True)
        end_ns = time.perf_counter_ns()
        wait_completion(combined[-1])
        combine_ns = end_ns - start_ns

        if iteration >= WARMUP:
            samples["get_dispatch_layout"].append(layout_ns)
            samples["dispatch_fresh"].append(fresh_ns)
            samples["dispatch_cached"].append(cached_ns)
            samples["intranode_combine"].append(combine_ns)

    buffer.destroy()
    dist.destroy_process_group()

    output = {"rank": rank}
    for api, values in samples.items():
        ordered = sorted(values)
        output[f"{api}_mean"] = statistics.fmean(values)
        output[f"{api}_p90"] = ordered[round((len(ordered) - 1) * 0.90)]
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
