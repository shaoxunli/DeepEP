"""CPU-only accounting for bench_v2_bandwidth; decimal GB, never wire bytes."""

import csv
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path


STAGES = {
    "dispatch_layout_kernel": "layout",
    "notify_dispatch_kernel": "notify",
    "cached_notify_dispatch_kernel": "notify",
    "cached_notify_combine_kernel": "notify",
    "v1_dispatch_udma_kernel": "comm",
    "v1_combine_udma_comm_kernel": "comm",
    "v1_combine_udma_reduce_kernel": "reduce",
    "expand_kernel": "expand",
    "aggregate_kernel": "aggregate",
}
LIMITS = [
    "official_logical_bytes follows the DeepEP-Ascend test numerator: all rank-deduplicated receive "
    "rows including self; dispatch/cached use x+scales+full top-k indices/weights, combine uses BF16 x only. "
    "This is logical accounting, not actual wire bytes. Matching the numerator does not match timing boundaries.",
    "Remote application bytes exclude self, count rank-deduplicated payload and actual field writes. "
    "They are not NIC/wire bytes: packet headers, ACK/retries and SHMEM collective control traffic are unmeasured.",
    "comm_us is the complete communication kernel, including local work, synchronization, "
    "metadata and (dispatch) GM output copy; it is not an isolated NIC transfer timer.",
    "device_sum_us sums all API device kernels. device_span_us includes gaps between those kernels; "
    "caller_us is measured separately without profiler and includes host submission/event waits.",
    "combine_stage_us = aggregate_us + comm_us; excludes notify and final reduce. "
    "It is reported separately from comm_us and full API time.",
    "Algorithm bytes = T*K*H*element_bytes, including invalid top-k slots. "
    "valid_algorithm_bytes excludes invalid slots. Combine payload is always BF16.",
    "No cold-L2 flush. Repeated fixed routing/input, caller allocation, independent GM outputs. "
    "A bandwidth comparison also requires matching routing, caching, hardware and library versions.",
]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def file_identity(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return dict(path=str(path.resolve()), size=path.stat().st_size, sha256=h.hexdigest())


def route_counts(routes_by_rank, experts_per_rank):
    """Independent directed matrices: [source][destination], before device use."""
    world = len(routes_by_rank)
    rows, slots, valid = [], [], []
    for routes in routes_by_rank:
        counts, selections, valid_slots = [0] * world, [0] * world, 0
        for token in routes:
            peers = set()
            for expert in token:
                if expert == -1:
                    continue
                if not 0 <= expert < world * experts_per_rank:
                    raise ValueError(f"invalid expert {expert}")
                peer = expert // experts_per_rank
                selections[peer] += 1
                valid_slots += 1
                peers.add(peer)
            for peer in peers:
                counts[peer] += 1
        rows.append(counts)
        slots.append(selections)
        valid.append(valid_slots)
    return dict(rows=rows, slots=slots, valid_slots=valid)


def byte_ledger(counts, rank, *, tokens, hidden, topk, dtype, idx_bytes, op):
    """Matches peer sender WQE lengths and sparse SIMT stores, not padded HBM strides."""
    if dtype not in ("bf16", "fp8") or idx_bytes not in (4, 8):
        raise ValueError("unsupported byte-accounting contract")
    if op not in ("dispatch", "cached-dispatch", "combine_unweighted", "combine_weighted"):
        raise ValueError(op)
    world = len(counts["rows"])
    combine = op.startswith("combine")
    tx = [counts["rows"][p][rank] for p in range(world)] if combine else counts["rows"][rank]
    rx = counts["rows"][rank] if combine else [counts["rows"][p][rank] for p in range(world)]
    remote_rows = sum(tx) - tx[rank]
    element_bytes = 2 if combine or dtype == "bf16" else 1
    payload = remote_rows * hidden * element_bytes
    scales = remote_rows * ((hidden + 127) // 128) * 4 if not combine and dtype == "fp8" else 0
    fresh = op == "dispatch"
    remote_slots = sum(counts["slots"][rank]) - counts["slots"][rank][rank]
    # Fresh metadata: one src_idx per row, index/weight only for each valid route.
    src = remote_rows * 4 if fresh else 0
    idx = remote_slots * idx_bytes if fresh else 0
    weights = remote_rows * topk * 4 if op == "combine_weighted" else remote_slots * 4 if fresh else 0
    # Peer scheduling publishes one uint64 SO marker per non-self peer, even if empty.
    markers = (world - 1) * 8
    # DeepEP-Ascend tests/ep/test_ep.py uses handle num_recv_tokens INCLUDING self.
    # Cached dispatch deliberately keeps the same logical numerator as fresh,
    # although this benchmark's cached path only transmits x/scales.
    logical_rows = sum(counts["rows"][p][rank] for p in range(world))
    logical_row_size = (
        hidden * 2
        if combine
        else hidden * element_bytes + (((hidden + 127) // 128) * 4 if dtype == "fp8" else 0) + topk * (idx_bytes + 4)
    )
    return dict(
        remote_tx_rows=remote_rows,
        remote_rx_rows=sum(rx) - rx[rank],
        self_rows=tx[rank],
        remote_x_bytes=payload,
        remote_scale_bytes=scales,
        remote_src_idx_bytes=src,
        remote_topk_idx_bytes=idx,
        remote_weight_bytes=weights,
        remote_so_bytes=markers,
        remote_payload_bytes=payload + scales,
        remote_application_bytes=payload + scales + src + idx + weights + markers,
        official_logical_rows=logical_rows,
        official_logical_bytes_per_row=logical_row_size,
        official_logical_bytes=logical_rows * logical_row_size,
        algorithm_bytes=tokens * topk * hidden * element_bytes,
        valid_algorithm_bytes=counts["valid_slots"][rank] * hidden * element_bytes,
        wire_bytes=None,
    )


def parse_profile(root, op, expanded, iterations, device, aiv):
    """Fail closed on missing, duplicated or unrecognizable profiler samples."""
    root = Path(root)
    csvs = list(root.rglob("kernel_details.csv"))
    traces = list(root.rglob("trace_view.json"))
    infos = list(root.rglob("profiler_info*.json"))
    if len(csvs) != 1 or len(traces) != 1 or len(infos) != 1:
        raise ValueError(f"{root}: expected one kernel CSV, trace and profiler_info")
    artifacts = [file_identity(p) for p in csvs + traces + infos]
    if any(p["size"] == 0 for p in artifacts):
        raise ValueError("empty profiler artifact")
    with csvs[0].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda r: Decimal(r["Start Time(us)"].strip()))
    combine = op.startswith("combine")
    main = "v1_combine_udma_comm_kernel" if combine else "v1_dispatch_udma_kernel"
    required = [main]
    required += (
        ["cached_notify_combine_kernel", "v1_combine_udma_reduce_kernel"]
        if combine
        else ["dispatch_layout_kernel", "notify_dispatch_kernel"]
        if op == "dispatch"
        else ["cached_notify_dispatch_kernel"]
    )
    if expanded:
        required.append("aggregate_kernel" if combine else "expand_kernel")
    counts = Counter(r["Name"] for r in rows)
    if any(counts[n] != iterations for n in required):
        raise ValueError(f"{root}: unexpected kernel counts {counts}; required {required}")
    forbidden = (
        ["expand_kernel", "dispatch_layout_kernel", "notify_dispatch_kernel"]
        if combine
        else ["aggregate_kernel", "v1_combine_udma_reduce_kernel"]
    )
    if not expanded:
        forbidden += ["aggregate_kernel", "expand_kernel"]
    if op == "cached-dispatch":
        forbidden += ["dispatch_layout_kernel", "notify_dispatch_kernel"]
    if any(counts[n] for n in forbidden):
        raise ValueError(f"unexpected stages: {counts}")
    if not rows or len(rows) % iterations:
        raise ValueError("incomplete profiler iteration")
    width = len(rows) // iterations
    sequence = [r["Name"] for r in rows[:width]]
    if any([r["Name"] for r in rows[i * width : (i + 1) * width]] != sequence for i in range(iterations)):
        raise ValueError("kernel sequence changed between iterations; cannot assign samples")
    if combine and sequence.index(main) >= sequence.index("v1_combine_udma_reduce_kernel"):
        raise ValueError("reduce precedes communication")
    if expanded and (
        (combine and sequence.index("aggregate_kernel") >= sequence.index(main))
        or (not combine and sequence.index(main) >= sequence.index("expand_kernel"))
    ):
        raise ValueError("local stage order mismatch")
    samples = []
    for i in range(iterations):
        chunk = rows[i * width : (i + 1) * width]
        times = {
            name + "_us": Decimal(0) for name in ("layout", "notify", "comm", "expand", "aggregate", "reduce", "other")
        }
        for row in chunk:
            if str(row["Device_id"]).strip() != str(device) or str(row["Step Id"]).strip() != "1":
                raise ValueError("unexpected device or profiler step")
            name = row["Name"]
            duration = Decimal(row["Duration(us)"])
            if not duration.is_finite() or duration <= 0:
                raise ValueError("nonpositive/nonfinite kernel duration")
            if STAGES.get(name) in ("comm", "expand", "aggregate", "reduce") and int(row["Block Num"]) != aiv:
                raise ValueError("unexpected launch AIV count")
            times[STAGES.get(name, "other") + "_us"] += duration
        times["device_sum_us"] = sum(times.values())
        starts = [Decimal(r["Start Time(us)"].strip()) for r in chunk]
        ends = [s + Decimal(r["Duration(us)"]) for s, r in zip(starts, chunk)]
        times["device_span_us"] = max(ends) - min(starts)
        times["combine_stage_us"] = times["aggregate_us"] + times["comm_us"] if combine else None
        samples.append(dict(sample=i, **{k: None if v is None else str(v) for k, v in times.items()}))
    return dict(
        samples=samples,
        artifacts=artifacts,
        kernel_sequence=sequence,
        absent_stages=[
            k for k in ("layout", "expand", "aggregate", "reduce") if not any(STAGES.get(n) == k for n in sequence)
        ],
    )


def write_csv(path, rows):
    if not rows:
        raise ValueError("refusing to publish an empty result table")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(records):
    """Rank-local and job-wide effective throughput, never average(rank GB/s)."""
    output = []
    for op in records[0]["operators"]:
        local = []
        for record in records:
            data = record["operators"][op]
            samples = data["profile"]["samples"]
            n = len(samples)
            row = dict(rank=record["rank"], op=op, **data["bytes"])
            for key in samples[0]:
                if key != "sample":
                    row[key] = None if samples[0][key] is None else float(sum(Decimal(s[key]) for s in samples) / n)
            row["caller_us"] = sum(data["caller_us"]) / len(data["caller_us"])
            row["remote_payload_GBps"] = row["remote_payload_bytes"] / row["comm_us"] / 1000
            row["remote_application_GBps"] = row["remote_application_bytes"] / row["comm_us"] / 1000
            row["official_logical_GBps"] = row["official_logical_bytes"] / row["comm_us"] / 1000
            row["official_logical_combine_stage_GBps"] = (
                None
                if row["combine_stage_us"] is None
                else row["official_logical_bytes"] / row["combine_stage_us"] / 1000
            )
            row["algorithm_device_sum_GBps"] = row["algorithm_bytes"] / row["device_sum_us"] / 1000
            row["algorithm_device_span_GBps"] = row["algorithm_bytes"] / row["device_span_us"] / 1000
            row["algorithm_caller_GBps"] = row["algorithm_bytes"] / row["caller_us"] / 1000
            output.append(row)
            local.append(row)
        # Lockstep sample number, max-rank elapsed per operation. No cross-host clock subtraction.
        job = dict(rank="all", op=op)
        job["official_logical_rows"] = sum(r["official_logical_rows"] for r in local)
        for key in local[0]:
            if key.endswith("_bytes") and key != "wire_bytes":
                job[key] = sum(r[key] for r in local)
        for key in (
            "layout_us",
            "notify_us",
            "comm_us",
            "expand_us",
            "aggregate_us",
            "reduce_us",
            "other_us",
            "device_sum_us",
            "device_span_us",
            "combine_stage_us",
        ):
            if local[0][key] is None:
                job[key] = None
                continue
            groups = [r["operators"][op]["profile"]["samples"] for r in records]
            job[key] = float(
                sum(max(Decimal(g[i][key]) for g in groups) for i in range(len(groups[0]))) / len(groups[0])
            )
        groups = [r["operators"][op]["caller_us"] for r in records]
        job["caller_us"] = sum(max(g[i] for g in groups) for i in range(len(groups[0]))) / len(groups[0])
        for numerator, denominator, label in [
            ("official_logical_bytes", "comm_us", "official_logical_GBps"),
            ("remote_payload_bytes", "comm_us", "remote_payload_GBps"),
            ("remote_application_bytes", "comm_us", "remote_application_GBps"),
            ("algorithm_bytes", "device_sum_us", "algorithm_device_sum_GBps"),
            ("algorithm_bytes", "device_span_us", "algorithm_device_span_GBps"),
            ("algorithm_bytes", "caller_us", "algorithm_caller_GBps"),
        ]:
            job[label] = job[numerator] / job[denominator] / 1000
        job["official_logical_combine_stage_GBps"] = (
            None if job["combine_stage_us"] is None else job["official_logical_bytes"] / job["combine_stage_us"] / 1000
        )
        output.append(job)
    return output
