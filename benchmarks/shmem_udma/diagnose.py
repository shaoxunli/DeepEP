#!/usr/bin/env python3
"""Run the SHMEM udma_perftest executable, without importing DeepEP or Torch."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import tarfile
import time


SCENARIOS = ("pair", "pair-bidir", "pool", "ring")
SOURCE = Path("examples/shmem_perftest/udma_perftest")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def make_plan(args):
    devices = [args.devices0, args.devices1]
    if len(devices[0]) != len(devices[1]) or not devices[0]:
        raise ValueError("Both nodes must supply the same positive number of devices")
    for ids in devices:
        if len(set(ids)) != len(ids) or min(ids) < 0:
            raise ValueError("Device lists must contain distinct nonnegative logical device IDs")
    if not 3 <= args.exponents[0] <= args.exponents[1] <= 26:
        raise ValueError("Require 3 <= minimum exponent <= maximum exponent <= 26")
    if args.loops < 1 or args.repeats < 1 or args.timeout <= 0:
        raise ValueError("Loops, repeats and timeout must be positive")
    if any(q < 1 or q > 56 for q in args.qps) or any(b < 0 or b > 32 for b in args.batches):
        raise ValueError("QP counts must be 1..56; batches must be 0..32")
    cases = []
    for repeat in range(args.repeats):
        for scenario in dict.fromkeys(args.scenarios):
            for qp in dict.fromkeys(args.qps):
                for batch in dict.fromkeys(args.batches):
                    for leader in (0, 1):
                        count = 1 if scenario in ("pair", "pair-bidir") else len(devices[0])
                        mapping = [
                            {"rank": 2 * slot + side, "node": node, "device": devices[node][slot]}
                            for slot in range(count)
                            for side, node in enumerate((leader, 1 - leader))
                        ]
                        index = len(cases)
                        cases.append(
                            {
                                "id": f"{index:03d}_{scenario}_from{leader}_q{qp}_b{batch}_r{repeat}",
                                "scenario": scenario,
                                "repeat": repeat,
                                "leader": leader,
                                "mode": "bi_put" if scenario in ("pair-bidir", "ring") else "put",
                                "pes": 2 * count,
                                "qps": qp,
                                "batch": batch,
                                "endpoint": f"tcp://{args.hosts[leader]}:{args.port + index}",
                                "mapping": mapping,
                            }
                        )
    if args.port < 1024 or args.port + len(cases) - 1 > 65535:
        raise ValueError("Case-specific bootstrap ports must fit in 1024..65535")
    config = {"run_id": args.run_id, "exponents": args.exponents, "loops": args.loops, "cases": cases}
    config["plan_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return config


def check_source(root, plan):
    """Reject known PE>2 verifier defects; this is a source guard, not binary attestation."""
    main = (root / SOURCE / "main.cpp").read_text(encoding="utf-8")
    for flag in ("pes", "pe-id", "gnpus", "fnpu", "fpe", "metric", "batch", "block-size", "exponent-range"):
        if f'"{flag}"' not in main:
            raise ValueError(f"udma_perftest source lacks --{flag}; incompatible CLI")
    compact = re.sub(r"\s+", "", main)
    for case in plan["cases"]:
        if case["pes"] <= 2:
            continue
        expected = "(prof_pe+1)%n_pes" if case["mode"] == "put" else "(pe_id-1+n_pes)%n_pes"
        if expected not in compact:
            raise ValueError(
                f"{case['id']}: PE>2 {case['mode']} verifier is not recognized as fixed. "
                "Use --scenarios pair first, or rebuild a SHMEM sample with the corrected verifier. "
                "Do not suppress verification errors."
            )
    return {str(name): digest(root / SOURCE / name) for name in ("main.cpp", "udma_perftest_kernel.cpp")}


def command(binary, plan, case, entry):
    # Global --pe-id plus --fpe 0 works with both the old and corrected sample.
    # --gnpus 1 makes device=(global_pe % 1)+fnpu; it does NOT set the world size.
    return [
        str(binary),
        "--pes",
        str(case["pes"]),
        "--pe-id",
        str(entry["rank"]),
        "--gnpus",
        "1",
        "--fpe",
        "0",
        "--fnpu",
        str(entry["device"]),
        "--ipport",
        case["endpoint"],
        "-t",
        case["mode"],
        "-d",
        "float",
        "--exponent-range",
        *map(str, plan["exponents"]),
        "--loop-count",
        str(plan["loops"]),
        "--ub-size",
        "16",
        "--metric",
        "bw",
        "--batch",
        str(case["batch"]),
        "--block-size",
        str(case["qps"]),
    ]


def capture(argv, env=None):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, errors="replace", env=env, timeout=10)
        return {"code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def inventory(args, binary, env, sources):
    linked = capture(["ldd", str(binary)], env)
    libraries = {}
    for path in re.findall(r"=>\s+(/\S+)", linked.get("stdout", "")):
        if Path(path).is_file():
            libraries[path] = digest(path)
    if not any("shmem" in Path(p).name for p in libraries):
        raise ValueError("ldd did not resolve a SHMEM library; check --library-dir and CANN environment")
    if "not found" in linked.get("stdout", ""):
        raise ValueError("ldd reports missing libraries: " + linked["stdout"])
    return {
        "hostname": socket.gethostname(),
        "binary": str(binary),
        "binary_sha256": digest(binary),
        "sources": sources,
        "linked_libraries": libraries,
        "ldd": linked,
        "source_revision": capture(["git", "-C", str(args.shmem_root), "rev-parse", "HEAD"]),
        "source_diff": capture(["git", "-C", str(args.shmem_root), "diff", "--stat", "--", str(SOURCE)]),
        "npu_smi": capture(["npu-smi", "info"]),
        "environment": {
            k: v
            for k, v in env.items()
            if k.startswith(("SHMEM_", "ACLSHMEM_"))
            or k in ("LD_LIBRARY_PATH", "ASCEND_HOME_PATH", "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES")
        },
    }


def stop_children(children):
    # Only terminate process groups created by this invocation. Never pkill by name.
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM) if os.name == "posix" else child.terminate()
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 3
    while any(p.poll() is None for p in children) and time.monotonic() < deadline:
        time.sleep(0.05)
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL) if os.name == "posix" else child.kill()
            except ProcessLookupError:
                pass
        child.wait()


def run_case(commands, directory, env, timeout):
    children, streams = [], []
    reason = None
    try:
        for rank, argv in commands:
            stream = (directory / f"rank{rank}.log").open("w", encoding="utf-8")
            streams.append(stream)
            children.append(
                subprocess.Popen(
                    argv,
                    cwd=directory,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=os.name == "posix",
                )
            )
        deadline = time.monotonic() + timeout
        while True:
            codes = [p.poll() for p in children]
            if any(code not in (None, 0) for code in codes):
                reason = "child_failed"
                break
            if all(code is not None for code in codes):
                break
            if time.monotonic() >= deadline:
                reason = "timeout"
                break
            time.sleep(0.1)
    finally:
        stop_children(children)
        for stream in streams:
            stream.close()
    records = []
    for (rank, _), child in zip(commands, children):
        log = (directory / f"rank{rank}.log").read_text(encoding="utf-8", errors="replace")
        verified = log.count("[Verification] SUCCESS")
        records.append(
            {
                "rank": rank,
                "code": child.returncode,
                "verified_points": verified,
                "done": f"[SUCCESS] udma_perftest done in pe {rank}" in log,
                "has_error": any(s in log for s in ("[ERROR]", "[FAILED]", "[Verification] FAILED")),
            }
        )
    return {"reason": reason, "ranks": records}


def read_measurements(directory, plan, case):
    files = list((directory / "output").glob("*.csv"))
    if len(files) != 1:
        raise ValueError(f"Expected exactly one prof-PE CSV, found {len(files)} in {directory}")
    with files[0].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {1 << e for e in range(plan["exponents"][0], plan["exponents"][1] + 1)}
    measurements = []
    for row in rows:
        size, blocks = int(row["DataSize/B"]), int(row["Blocks"])
        duration = float(row["CoreMaxTime/us"])
        key = next(
            k for k in row if isinstance(k, str) and k.replace(" ", "") in ("Bandwidth/GB/s(1000)", "Bandwidth/GB/s")
        )
        bandwidth = float(row[key])
        if (
            size not in expected
            or blocks != case["qps"]
            or not all(math.isfinite(v) and v > 0 for v in (duration, bandwidth))
        ):
            raise ValueError("Invalid size/QP count/timing/bandwidth in " + str(files[0]))
        expected.remove(size)
        calculated = size * blocks / duration / 1000
        if not math.isclose(bandwidth, calculated, rel_tol=0.02, abs_tol=0.01):
            raise ValueError("CSV bandwidth disagrees with payload/time; check sample version")
        measurements.append(
            {
                "case": case["id"],
                "scenario": case["scenario"],
                "pes": case["pes"],
                "source_node": case["leader"],
                "qps": blocks,
                "batch": case["batch"],
                "repeat": case["repeat"],
                "bytes_per_qp": size,
                "bytes_per_iteration": size * blocks,
                "time_us": duration,
                "bandwidth_GBps": bandwidth,
                "prof_pe": 0,
            }
        )
    if expected:
        raise ValueError("Missing data sizes in prof-PE CSV")
    return measurements


def run(args, plan):
    if os.name != "posix":
        raise ValueError("Device runs require Linux; --dry-run and merge work without an NPU")
    root = args.shmem_root.resolve()
    binary = (args.binary or root / "build/bin/udma_perftest").resolve()
    if not binary.is_file():
        raise ValueError(f"Missing executable: {binary}; build the SHMEM examples first")
    sources = check_source(root, plan)
    env = os.environ.copy()
    libdir = (args.library_dir or root / "build/lib").resolve()
    env["LD_LIBRARY_PATH"] = str(libdir) + ":" + env.get("LD_LIBRARY_PATH", "")
    env["SHMEM_CYCLE_PROF_PE"] = "0"
    info = inventory(args, binary, env, sources)
    output = args.output.resolve()
    if output.with_name(output.name + ".tar.gz").exists():
        raise ValueError("Archive already exists; choose a fresh --output")
    output.mkdir(parents=True, exist_ok=False)
    save(output / "plan.json", plan)
    report = {"node": args.node_rank, "plan": plan, "inventory": info, "status": "running", "cases": []}
    save(output / "report.json", report)
    try:
        for case in plan["cases"]:
            directory = output / case["id"]
            (directory / "output").mkdir(parents=True)
            commands = [
                (entry["rank"], command(binary, plan, case, entry))
                for entry in case["mapping"]
                if entry["node"] == args.node_rank
            ]
            save(directory / "commands.json", commands)
            print(f"RUN {case['id']} PE={case['pes']} endpoint={case['endpoint']}", flush=True)
            result = run_case(commands, directory, env, args.timeout)
            result["id"] = case["id"]
            report["cases"].append(result)
            points = plan["exponents"][1] - plan["exponents"][0] + 1
            if result["reason"] or any(
                r["code"] != 0 or not r["done"] or r["has_error"] or r["verified_points"] != points
                for r in result["ranks"]
            ):
                raise ValueError(f"Case failed: {case['id']}; see per-rank logs")
            result["measurements"] = (
                read_measurements(directory, plan, case) if case["leader"] == args.node_rank else []
            )
            save(output / "report.json", report)
            print(f"LOCAL_CASE_PASSED {case['id']}", flush=True)
        report["status"] = "local_passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        save(output / "report.json", report)
        archive = output.with_name(output.name + ".tar.gz")
        with tarfile.open(archive, "x:gz") as package:
            package.add(output, arcname=output.name)
        print(f"ARCHIVE {archive}", flush=True)
    print("LOCAL_PASSED; merge BOTH node report.json files to validate the complete run", flush=True)


def merge(paths):
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    reports.sort(key=lambda r: r["node"])
    if [r["node"] for r in reports] != [0, 1] or any(r["status"] != "local_passed" for r in reports):
        raise ValueError("Need local_passed reports from node0 AND node1")
    if reports[0]["plan"] != reports[1]["plan"]:
        raise ValueError("Node plans differ (run ID/configuration/mapping/ports)")
    plan = reports[0]["plan"]
    fingerprints = []
    for report in reports:
        info = report["inventory"]
        fingerprints.append(
            (
                info["binary_sha256"],
                info["sources"],
                {Path(p).name: h for p, h in info["linked_libraries"].items() if "shmem" in Path(p).name},
            )
        )
    if fingerprints[0] != fingerprints[1]:
        raise ValueError("Nodes use different binary/source/SHMEM library hashes")
    rows = []
    for report in reports:
        if [c["id"] for c in report["cases"]] != [c["id"] for c in plan["cases"]]:
            raise ValueError("Incomplete case list")
        for expected, result in zip(plan["cases"], report["cases"]):
            ranks = [e["rank"] for e in expected["mapping"] if e["node"] == report["node"]]
            points = plan["exponents"][1] - plan["exponents"][0] + 1
            if (
                result["reason"]
                or [r["rank"] for r in result["ranks"]] != ranks
                or any(
                    r["code"] != 0 or not r["done"] or r["has_error"] or r["verified_points"] != points
                    for r in result["ranks"]
                )
            ):
                raise ValueError("Incomplete/failed rank verification")
            measures = result["measurements"]
            if len(measures) != (points if expected["leader"] == report["node"] else 0):
                raise ValueError("Incomplete prof-PE measurements")
            rows.extend(measures)
    return {"status": "passed", "plan": plan, "nodes": [r["inventory"] for r in reports], "measurements": rows}


def comparison(baseline, candidate):
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in (baseline, candidate)]
    if any(r["status"] != "passed" for r in reports):
        raise ValueError("Compare requires globally merged reports")
    if reports[0]["plan"]["loops"] != reports[1]["plan"]["loops"]:
        raise ValueError("Loop counts differ")
    signatures = [
        {
            (
                n["binary_sha256"],
                json.dumps(n["sources"], sort_keys=True),
                tuple(sorted((Path(p).name, h) for p, h in n["linked_libraries"].items() if "shmem" in Path(p).name)),
            )
            for n in r["nodes"]
        }
        for r in reports
    ]
    if signatures[0] != signatures[1]:
        raise ValueError("Binary/source/SHMEM hashes differ between baseline and candidate")
    columns = ("scenario", "pes", "source_node", "qps", "batch", "bytes_per_qp")
    groups = []
    for report in reports:
        grouped = {}
        for row in report["measurements"]:
            key = tuple(row[k] for k in columns)
            grouped.setdefault(key, []).append(row)
        groups.append(grouped)
    if groups[0].keys() != groups[1].keys():
        raise ValueError("Case/size sets differ; compare matching suites")
    rows = []
    for key in sorted(groups[0]):
        row = dict(zip(columns, key))
        for label, group in zip(("baseline", "candidate"), groups):
            values = [r["bandwidth_GBps"] for r in group[key]]
            row.update(
                {
                    label + "_repeats": len(values),
                    label + "_GBps_median": statistics.median(values),
                    label + "_GBps_min": min(values),
                    label + "_GBps_max": max(values),
                    label + "_time_us_median": statistics.median(r["time_us"] for r in group[key]),
                    label + "_bandwidth_cv_pct": statistics.pstdev(values) / statistics.mean(values) * 100,
                }
            )
        row["bandwidth_change_pct"] = (row["candidate_GBps_median"] / row["baseline_GBps_median"] - 1) * 100
        row["time_change_pct"] = (row["candidate_time_us_median"] / row["baseline_time_us_median"] - 1) * 100
        rows.append(row)
    return rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    launch = sub.add_parser("run")
    launch.add_argument("--shmem-root", type=Path, required=True)
    launch.add_argument("--binary", type=Path)
    launch.add_argument("--library-dir", type=Path)
    launch.add_argument("--hosts", nargs=2, required=True, help="Bootstrap addresses for node0 and node1")
    launch.add_argument("--node-rank", type=int, choices=(0, 1), required=True)
    launch.add_argument("--devices0", nargs="+", type=int, default=list(range(16)))
    launch.add_argument("--devices1", nargs="+", type=int, default=list(range(16)))
    launch.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["pair", "pair-bidir"])
    launch.add_argument("--qps", nargs="+", type=int, default=[1])
    launch.add_argument("--batches", nargs="+", type=int, default=[0])
    launch.add_argument("--exponents", nargs=2, type=int, default=[12, 24])
    launch.add_argument("--loops", type=int, default=1000)
    launch.add_argument("--repeats", type=int, default=3)
    launch.add_argument("--port", type=int, default=29100)
    launch.add_argument("--timeout", type=float, default=600)
    launch.add_argument("--run-id", required=True)
    launch.add_argument("--output", type=Path, required=True)
    launch.add_argument("--dry-run", action="store_true")
    combine = sub.add_parser("merge")
    combine.add_argument("reports", nargs=2, type=Path)
    combine.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.action == "compare":
            rows = comparison(args.baseline, args.candidate)
            args.output.mkdir(parents=True, exist_ok=False)
            write_csv(args.output / "comparison.csv", rows)
            print(f"COMPARISON {args.output / 'comparison.csv'}")
        elif args.action == "merge":
            result = merge(args.reports)
            args.output.mkdir(parents=True, exist_ok=False)
            save(args.output / "report.json", result)
            write_csv(args.output / "summary.csv", result["measurements"])
            print(f"GLOBAL_PASSED {args.output / 'summary.csv'}")
        else:
            plan = make_plan(args)
            print(f"PLAN {plan['plan_sha256']} cases={len(plan['cases'])}; match this hash on both nodes", flush=True)
            if args.dry_run:
                print(json.dumps(plan, indent=2))
            else:
                run(args, plan)
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
