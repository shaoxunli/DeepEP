"""Opt-in peer-ready A/B: one Buffer, recomputation, skew and independent CPU checks."""

import argparse
import csv
from dataclasses import asdict
import datetime
import json
import os
from pathlib import Path
import statistics
import time

ENV = "DEEPEP_CLOS_PEER_READY"


def stages(recompute):
    result = [("forward_dispatch", "fresh"), ("forward_combine", "combine")]
    if recompute == "full":
        result += [("recompute_dispatch", "cached"), ("recompute_combine", "combine")]
    return result + [("backward_dispatch", "cached"), ("backward_combine", "combine")]


def aggregate(samples):
    """Aligned sample IDs; preserve rank tails instead of averaging ranks first."""
    return statistics.mean(max(rank[i] for rank in samples) for i in range(len(samples[0])))


def parse_profile(path, operation, count):
    files = list(path.rglob("kernel_details.csv"))
    if len(files) != 1:
        raise RuntimeError(f"expected one kernel CSV under {path}, got {len(files)}")
    with files[0].open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and "Start Time(us)" in rows[0]:
        rows.sort(key=lambda r: float(r["Start Time(us)"]))
    names = {
        "comm": "v1_combine_udma_comm_kernel" if operation == "combine" else "v1_dispatch_udma_kernel",
        "reduce": "v1_combine_udma_reduce_kernel" if operation == "combine" else None,
        "notify": {"fresh": "notify_dispatch_kernel", "cached": "cached_notify_dispatch_kernel"}.get(operation),
        "layout": "dispatch_layout_kernel" if operation == "fresh" else None,
    }
    result = {}
    for stage, name in names.items():
        times = [float(r["Duration(us)"]) for r in rows if r["Name"] == name] if name else []
        if times and len(times) != count:
            raise RuntimeError(f"{name}: expected {count} launches, found {len(times)}")
        if stage == "comm" and not times or stage == "reduce" and operation == "combine" and not times:
            raise RuntimeError(f"missing required kernel {name}")
        if stage == "notify" and operation == "fresh" and not times:
            raise RuntimeError("fresh notify is missing")
        result[stage] = times[2:-2] if times else [0.0] * (count - 4)
    # The sample-preparation barrier is NOT an operator notify stage. Combine
    # entry/exit synchronization, if enabled, is inside the communication kernel.
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=2816)
    p.add_argument("--experts", type=int, default=256)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--num-sms", type=int, default=8)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--skew-us", type=int, nargs="+", default=[0, 200, 2000])
    p.add_argument("--recompute", choices=["none", "expert", "full"], nargs="+", default=["none", "full"])
    p.add_argument("--flush-l2", action="store_true")
    p.add_argument("--huge-pages", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--profile", action="store_true", help="separate aligned kernel pass, not mixed with skew timing")
    p.add_argument("--precision-only", action="store_true")
    p.add_argument("--control-port", type=int, default=29881)
    p.add_argument("--shmem-port", type=int, default=29882)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if min(a.tokens, a.hidden, a.experts, a.topk, a.samples) <= 0 or a.warmup < 0 or min(a.skew_us) < 0:
        p.error("invalid shape, sample count or delay")
    if a.num_sms < 8 or a.num_sms % 2:
        p.error("num-sms must be even and >=8")
    import torch
    import torch_npu
    import torch.distributed as dist
    from deep_ep.backends.direct.elastic import ElasticBuffer
    from deep_ep.backends.direct import buffer as bm, elastic as em
    from deep_ep.backends.direct.dtypes import topk_idx_t
    from deep_ep.backends.direct._native import load_native_module
    from bench_fast_sweep import StoreGroup
    from bench_fused_notify import store_topology_collective
    from stage_metrics import L2CacheFlush, prepare_v2_sample

    rank, world, device = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if world < 2 or world > 256 or a.experts % world or a.topk > a.experts:
        p.error("EP must be 2..256 and divide experts; topk <= experts")
    torch.npu.set_device(device)
    native = load_native_module()
    if getattr(native, "peer_ready_protocol_version", 0) != 1:
        raise RuntimeError("rebuild this branch's host AND device libraries")
    # Isolate synchronization changes. Use the same ordinary non-expanded API,
    # defaults for sender policy, and unchanged communication/reduce core count.
    for key in (
        "DEEPEP_CLOS_FUSED_LAYOUT_NOTIFY",
        "DEEPEP_CLOS_NOTIFY_HOST_PREP",
        "DEEPEP_CLOS_NOTIFY_DEFER_PLAN",
        "DEEPEP_CLOS_DISPATCH_HOST_TIMING",
        "DEEPEP_CLOS_PROFILE_COMBINE_NOTIFY",
    ):
        os.environ[key] = "0"
    store = dist.TCPStore(
        os.environ["MASTER_ADDR"],
        a.control_port,
        world,
        rank == 0,
        timeout=datetime.timedelta(seconds=900),
        wait_for_workers=True,
    )
    group = StoreGroup(store, rank, world)
    bm._bootstrap_all_gather = lambda g, c, n, value: group.gather(value)
    em._validate_group_consensus = lambda g, name, value: group.agree([name, value])
    group.agree(vars(a) | {"output": None})
    out = a.output / f"rank{rank}"
    out.mkdir(parents=True, exist_ok=False)
    (out / "manifest.json").write_text(
        json.dumps(
            dict(
                args=vars(a) | {"output": str(a.output)},
                world=world,
                rank=rank,
                native=native.__file__,
                torch=torch.__version__,
                torch_npu=torch_npu.__version__,
                env={k: v for k, v in os.environ.items() if k.startswith(("DEEPEP_", "SHMEM_"))},
            ),
            indent=2,
        )
    )
    os.environ[ENV] = "0"
    with store_topology_collective(group):
        buffer = ElasticBuffer(
            group,
            num_max_tokens_per_rank=a.tokens,
            hidden=a.hidden,
            num_topk=a.topk,
            explicitly_destroy=True,
            transport="udma",
            udma_auto_policy=True,
            use_1g_huge_pages=a.huge_pages,
            shmem_ip_port=f"tcp://{os.environ['MASTER_ADDR']}:{a.shmem_port}",
        )
    flush = L2CacheFlush(torch, buffer, a.flush_l2)
    policies, summary = {}, []

    def finish(result):
        if result[-1].event is not None:
            result[-1].current_stream_wait()
        torch.npu.synchronize()
        return result

    def mode(value, label):
        torch.npu.synchronize()
        group.agree(["mode", value, label])
        buffer.barrier(use_comm_stream=True, with_cpu_sync=True)
        os.environ[ENV] = str(value)

    def routes(owner, kind):
        g = torch.Generator().manual_seed(20261122 + owner)
        idx = torch.rand((a.tokens, a.experts), generator=g).topk(a.topk, dim=1).indices
        if kind == "masked":
            idx[::3, -1] = -1
            idx[::7] = -1
        elif kind == "hot":
            idx[:] = torch.arange(a.topk)
        elif kind == "empty":
            idx.fill_(-1)
        elif kind == "empty_local" and owner == 0:
            idx = idx[:0]
        return idx

    def input_cpu(owner, tokens, generation):
        # Exactly representable BF16 values vary across ranks, rows, columns AND
        # invocations; reusing stale payload from the previous epoch must fail.
        generator = torch.Generator().manual_seed(104729 * generation + 8191 * owner + 17)
        return torch.randint(-31, 32, (tokens, a.hidden), generator=generator).to(torch.bfloat16)

    def setup(kind):
        idxs = [routes(r, kind) for r in range(world)]
        memberships = [
            torch.stack([((v >= 0) & (v // (a.experts // world) == r)).any(1) for r in range(world)], 1) for v in idxs
        ]
        local_idx = idxs[rank].to(dtype=topk_idx_t, device="npu")
        weights_cpu = [((torch.arange(len(v)).unsqueeze(1) + torch.arange(a.topk)) % 17).float() / 16 for v in idxs]
        weights = weights_cpu[rank].npu()
        payload = sum(int(m.sum()) for m in memberships) * a.hidden * 2 / world
        return idxs, memberships, local_idx, weights_cpu, weights, payload

    def run_sequence(case, weighted, recompute, delay, sample, check=False):
        idxs, memberships, idx, weights_cpu, weights, _ = case
        weights = weights + sample / 128.0
        # Allocate before the common start; individual stages still change data.
        dispatch_inputs = [input_cpu(rank, len(idxs[rank]), sample * 8 + i).npu() for i in range(3)]
        torch.npu.synchronize()
        prepare_v2_sample(buffer, flush)
        start = time.perf_counter_ns()
        timings, lag, handle, receive, recv_weights, d = {}, {}, None, None, None, 0
        expected_x = None
        weight_mask = None
        for stage_id, (name, op) in enumerate(stages(recompute)):
            lag_start = time.perf_counter_ns()
            if delay and rank == (sample + stage_id) % world:
                time.sleep(delay / 1e6)
            lag[name] = (time.perf_counter_ns() - lag_start) / 1000
            combine_weights = recv_weights
            if check and op == "combine" and weighted:
                combine_weights = recv_weights + weight_mask * ((stage_id + 1) / 128.0)
                torch.npu.synchronize()
            before = time.perf_counter_ns()
            if op == "fresh":
                result = finish(
                    buffer.dispatch(
                        dispatch_inputs[d],
                        topk_idx=idx,
                        topk_weights=weights if weighted else None,
                        num_experts=a.experts,
                        num_sms=a.num_sms,
                        do_expand=False,
                    )
                )
            elif op == "cached":
                result = finish(buffer.dispatch(dispatch_inputs[d], handle=handle, num_sms=a.num_sms, do_expand=False))
            else:
                result = finish(
                    buffer.combine(
                        receive, handle, topk_weights=combine_weights if weighted else None, num_sms=a.num_sms
                    )
                )
            timings[name] = (time.perf_counter_ns() - before) / 1000
            if op != "combine":
                receive, recv_idx, current_weights, new_handle, _ = result
                if op == "fresh":
                    handle = new_handle
                    recv_weights = current_weights
                    weight_mask = (recv_idx >= 0).float() if weighted and check else None
                    policy = asdict(handle._udma_policy)
                    combine_policy = asdict(
                        buffer._udma_auto_policy.for_launch(
                            a.num_sms, a.hidden * 2, "weighted" if weighted else "combine", bf16=True, expanded=False
                        )
                    )
                    policy = {"dispatch": policy, "combine": combine_policy, "num_sms": a.num_sms}
                    key = str(weighted)
                    if key in policies and policies[key] != policy:
                        raise RuntimeError("sender policy changed across protocol modes")
                    policies[key] = policy
                if check:
                    expected_x = torch.cat(
                        [input_cpu(r, len(idxs[r]), sample * 8 + d)[memberships[r][:, rank]] for r in range(world)]
                    )
                    torch.testing.assert_close(receive.cpu(), expected_x, rtol=0, atol=0)
                    if op == "fresh":
                        expected_idx, expected_w = [], []
                        for r in range(world):
                            hit = memberships[r][:, rank]
                            v = idxs[r][hit]
                            owned = (v >= 0) & (v // (a.experts // world) == rank)
                            expected_idx.append(torch.where(owned, v % (a.experts // world), -1))
                            expected_w.append(torch.where(owned, weights_cpu[r][hit] + sample / 128.0, 0))
                        torch.testing.assert_close(recv_idx.cpu().long(), torch.cat(expected_idx), rtol=0, atol=0)
                        if weighted:
                            torch.testing.assert_close(recv_weights.cpu(), torch.cat(expected_w), rtol=0, atol=0)
                d += 1
            elif check:
                local_x = input_cpu(rank, len(idxs[rank]), sample * 8 + d - 1).float()
                expected = (local_x * memberships[rank].sum(1).unsqueeze(1)).to(torch.bfloat16)
                torch.testing.assert_close(result[0].cpu(), expected, rtol=0, atol=0)
                if weighted:
                    want = torch.where(idxs[rank] >= 0, weights_cpu[rank] + (sample + stage_id + 1) / 128.0, 0)
                    torch.testing.assert_close(result[1].cpu(), want, rtol=0, atol=0)
        # Include rank imbalance in the sequence window; no intermediate global
        # barrier. End rendezvous makes this a job completion metric, not BW.
        buffer.barrier(use_comm_stream=True, with_cpu_sync=True)
        elapsed = (time.perf_counter_ns() - start) / 1000
        return dict(api_us=timings, sequence_us=elapsed, injected_delay_us=lag)

    completed = False
    try:
        for kind in ("random", "masked", "hot", "empty", "empty_local"):
            case = setup(kind)
            for weighted in (False, True):
                for value in (0, 1, 1, 0):
                    mode(value, ["precision", kind, weighted])
                    for recompute in a.recompute:
                        for sample in range(2):
                            run_sequence(case, weighted, recompute, max(a.skew_us), sample, check=True)
            if rank == 0:
                print(f"PRECISION {kind}: passed", flush=True)
        (out / "precision.json").write_text(json.dumps(dict(passed=True, buffer_initializations=1)))
        if a.precision_only:
            completed = True
            return
        case = setup("random")
        for weighted in (False, True):
            for recompute in a.recompute:
                for delay in a.skew_us:
                    for turn, value in enumerate((0, 1, 1, 0)):
                        tag = f"w{int(weighted)}_{recompute}_skew{delay}_round{turn}_mode{value}"
                        mode(value, tag)
                        for sample in range(a.warmup):
                            run_sequence(case, weighted, recompute, delay, sample)
                        rows = [run_sequence(case, weighted, recompute, delay, s + a.warmup) for s in range(a.samples)]
                        (out / f"{tag}.json").write_text(json.dumps(rows, indent=2))
                        gathered = group.gather(rows)
                        row = dict(
                            tag=tag,
                            mode=value,
                            weight=weighted,
                            recompute=recompute,
                            skew_us=delay,
                            sequence_us=aggregate([[r["sequence_us"] for r in g] for g in gathered]),
                        )
                        row.update(
                            {
                                name + "_api_us": aggregate([[r["api_us"][name] for r in g] for g in gathered])
                                for name, _ in stages(recompute)
                            }
                        )
                        summary.append(row)
                        if rank == 0:
                            print(json.dumps(row), flush=True)
            if a.profile:
                # Independent aligned operator measurements. No skew, no CPU
                # reference in the measured region, two guards on either end.
                idxs, _, idx, _, weights, payload = case
                x = input_cpu(rank, len(idxs[rank]), 0).npu()
                for turn, value in enumerate((0, 1, 1, 0)):
                    mode(value, ["profile", weighted, turn])

                    def fresh():
                        return finish(
                            buffer.dispatch(
                                x,
                                topk_idx=idx,
                                topk_weights=weights if weighted else None,
                                num_experts=a.experts,
                                num_sms=a.num_sms,
                                do_expand=False,
                            )
                        )

                    recv, _, recv_w, handle, _ = fresh()
                    calls = {
                        "fresh": fresh,
                        "cached": lambda: finish(buffer.dispatch(x, handle=handle, num_sms=a.num_sms, do_expand=False)),
                        "combine": lambda: finish(
                            buffer.combine(recv, handle, topk_weights=recv_w if weighted else None, num_sms=a.num_sms)
                        ),
                    }
                    for op, invoke in calls.items():
                        for _ in range(a.warmup):
                            prepare_v2_sample(buffer, flush)
                            invoke()
                        capture = out / f"profile_w{int(weighted)}_round{turn}_{op}"
                        prof = torch_npu.profiler
                        with prof.profile(
                            activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU],
                            on_trace_ready=prof.tensorboard_trace_handler(str(capture)),
                            experimental_config=prof._ExperimentalConfig(
                                export_type=[prof.ExportType.Text],
                                profiler_level=prof.ProfilerLevel.Level0,
                                aic_metrics=prof.AiCMetrics.AiCoreNone,
                            ),
                        ):
                            for _ in range(a.samples + 4):
                                prepare_v2_sample(buffer, flush)
                                invoke()
                        metrics = parse_profile(capture, op, a.samples + 4)
                        if op == "cached" and bool(any(metrics["notify"])) == bool(value):
                            raise RuntimeError("cached notify launch does not match protocol switch")
                        gathered = group.gather(metrics)
                        row = dict(profile=True, operation=op, weight=weighted, round=turn, mode=value)
                        row.update({k + "_us": aggregate([g[k] for g in gathered]) for k in metrics})
                        row["logical_BF16_GBps"] = payload / row["comm_us"] / 1000
                        summary.append(row)
        if rank == 0:
            (a.output / "summary.json").write_text(json.dumps(dict(rows=summary, policies=policies), indent=2))
        completed = True
    finally:
        # On a rank failure let torchrun stop its peers, instead of waiting in
        # another collective before the original exception can be propagated.
        if completed:
            torch.npu.synchronize()
            group.agree("destroy")
            buffer.destroy()


if __name__ == "__main__":
    main()
