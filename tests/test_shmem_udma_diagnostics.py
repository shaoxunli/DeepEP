"""CPU checks for the external SHMEM perftest launcher; no synthetic performance claims."""

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest


spec = importlib.util.spec_from_file_location(
    "diagnose", Path(__file__).resolve().parents[1] / "benchmarks/shmem_udma/diagnose.py"
)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def options(**kwargs):
    values = dict(
        devices0=list(range(16)),
        devices1=list(range(16)),
        exponents=[12, 12],
        loops=10,
        repeats=1,
        timeout=10,
        qps=[1],
        batches=[0],
        scenarios=["pair", "pool", "ring"],
        hosts=["node-a", "node-b"],
        port=29100,
        run_id="test",
    )
    values.update(kwargs)
    return argparse.Namespace(**values)


def test_plan_mapping_ports_and_reverse():
    plan = diag.make_plan(options())
    assert len(plan["cases"]) == 6
    assert len({c["endpoint"] for c in plan["cases"]}) == 6
    for case in plan["cases"]:
        assert case["mapping"][0]["node"] == case["leader"]
        assert f"node-{'a' if case['leader'] == 0 else 'b'}" in case["endpoint"]
        assert case["pes"] == (2 if case["scenario"] == "pair" else 32)
        for rank, entry in enumerate(case["mapping"]):
            assert entry["node"] != case["mapping"][(rank + 1) % case["pes"]]["node"]
            cmd = diag.command(Path("/bin/udma_perftest"), plan, case, entry)
            assert cmd[cmd.index("--pe-id") + 1] == str(rank)
            assert cmd[cmd.index("--fpe") + 1] == "0"
            assert cmd[cmd.index("--gnpus") + 1] == "1"
    assert diag.make_plan(options()) == plan


@pytest.mark.parametrize(
    "kwargs",
    [
        {"devices0": [0, 0]},
        {"devices0": [-1]},
        {"exponents": [24, 12]},
        {"qps": [0]},
        {"batches": [-1]},
        {"port": 65535},
        {"repeats": 0},
        {"timeout": 0},
    ],
)
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        diag.make_plan(options(**kwargs))


def test_old_source_verifier_is_rejected_for_full_pool(tmp_path):
    source = tmp_path / diag.SOURCE
    source.mkdir(parents=True)
    flags = '"pes" "pe-id" "gnpus" "fnpu" "fpe" "metric" "batch" "block-size" "exponent-range"'
    (source / "main.cpp").write_text(flags, encoding="utf-8")
    (source / "udma_perftest_kernel.cpp").write_text("fixture", encoding="utf-8")
    diag.check_source(tmp_path, diag.make_plan(options(scenarios=["pair", "pair-bidir"])))
    with pytest.raises(ValueError, match="verifier"):
        diag.check_source(tmp_path, diag.make_plan(options()))
    (source / "main.cpp").write_text(flags + " (prof_pe + 1) % n_pes; (pe_id - 1 + n_pes) % n_pes;", encoding="utf-8")
    diag.check_source(tmp_path, diag.make_plan(options()))


def test_csv_missing_duplicate_and_bad_formula(tmp_path):
    plan = diag.make_plan(options(qps=[2]))
    case = plan["cases"][0]
    (tmp_path / "output").mkdir()
    path = tmp_path / "output/test.csv"
    header = (
        "DataSize/B,Npus,Blocks,UBsize/KB,Bandwidth/GB/s(1000),Bandwidth/GiB/s(1024),CoreMaxTime/us,SingleCoreTime/us\n"
    )
    # Sample CSV can have extra per-core columns beyond its fixed header.
    row = "4096,1,2,16,8.192,7.629,1,1,0.9\n"
    path.write_text(header + row)
    values = diag.read_measurements(tmp_path, plan, case)
    assert values[0]["bytes_per_iteration"] == 8192
    for content in (
        header,
        header + row + row,
        header + row.replace("8.192", "16.384"),
        header + row.replace("8.192", "nan"),
    ):
        path.write_text(content)
        with pytest.raises(ValueError):
            diag.read_measurements(tmp_path, plan, case)


def reports(tmp_path):
    plan = diag.make_plan(options(scenarios=["pair"]))
    paths = []
    for node in (0, 1):
        report = {
            "node": node,
            "plan": plan,
            "status": "local_passed",
            "inventory": {"binary_sha256": "binary", "sources": {}, "linked_libraries": {"/lib/libshmem.so": "lib"}},
            "cases": [],
        }
        for case in plan["cases"]:
            rank = next(e["rank"] for e in case["mapping"] if e["node"] == node)
            report["cases"].append(
                {
                    "id": case["id"],
                    "reason": None,
                    "ranks": [{"rank": rank, "code": 0, "done": True, "has_error": False, "verified_points": 1}],
                    "measurements": [
                        {
                            "case": case["id"],
                            "scenario": "pair",
                            "pes": 2,
                            "source_node": node,
                            "qps": 1,
                            "batch": 0,
                            "bytes_per_qp": 4096,
                            "time_us": 1,
                            "bandwidth_GBps": 4.096,
                        }
                    ]
                    if node == case["leader"]
                    else [],
                }
            )
        path = tmp_path / f"node{node}.json"
        diag.save(path, report)
        paths.append(path)
    return paths


@pytest.mark.parametrize("change", ["status", "plan", "hash", "case", "rank", "samples", "verification"])
def test_merge_fails_closed(tmp_path, change):
    paths = reports(tmp_path)
    assert diag.merge(paths)["status"] == "passed"
    report = json.loads(paths[1].read_text())
    if change == "status":
        report["status"] = "failed"
    elif change == "plan":
        report["plan"]["loops"] += 1
    elif change == "hash":
        report["inventory"]["binary_sha256"] = "different"
    elif change == "case":
        report["cases"].pop()
    elif change == "rank":
        report["cases"][0]["ranks"].clear()
    elif change == "samples":
        report["cases"][1]["measurements"].clear()
    else:
        report["cases"][0]["ranks"][0]["verified_points"] = 0
    diag.save(paths[1], report)
    with pytest.raises(ValueError):
        diag.merge(paths)


def test_compare_uses_all_repeats(tmp_path):
    report = diag.merge(reports(tmp_path))
    base, new = tmp_path / "base.json", tmp_path / "new.json"
    diag.save(base, report)
    candidate = copy.deepcopy(report)
    for row in candidate["measurements"]:
        row["bandwidth_GBps"] *= 0.8
        row["time_us"] *= 1.25
    diag.save(new, candidate)
    rows = diag.comparison(base, new)
    assert rows[0]["bandwidth_change_pct"] == pytest.approx(-20)
    assert rows[0]["time_change_pct"] == pytest.approx(25)


@pytest.mark.parametrize(
    "code,timeout,reason", [("raise SystemExit(7)", 5, "child_failed"), ("import time; time.sleep(30)", 0.2, "timeout")]
)
def test_child_failure_and_timeout_cleanup(tmp_path, code, timeout, reason):
    result = diag.run_case([(0, [sys.executable, "-c", code])], tmp_path, os.environ.copy(), timeout)
    assert result["reason"] == reason
    assert result["ranks"][0]["code"] != 0


def test_successful_child_log(tmp_path):
    code = 'print("[Verification] SUCCESS"); print("[SUCCESS] udma_perftest done in pe 0")'
    result = diag.run_case([(0, [sys.executable, "-c", code])], tmp_path, os.environ.copy(), 10)
    assert result["reason"] is None
    assert result["ranks"][0]["done"]
    assert result["ranks"][0]["verified_points"] == 1
