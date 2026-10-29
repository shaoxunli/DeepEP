"""Bandwidth numerator and profiler-boundary tests; no torch/NPU dependency."""

import csv
import contextlib
import copy
import io
import json
import shutil
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks/clos"))
from stage_metrics import byte_ledger, file_identity, parse_profile, route_counts, summarize, write_json
from bench_v2_bandwidth import arguments, check_idle, local_world_size, report, run_case, suite_contract, worker_command
from bandwidth_multinode import Peer, collection_manifest, library_identity, safe_path, validate_peers


class StageMetricsTest(unittest.TestCase):
    def setUp(self):
        # rank 0: two experts on self, two on rank 1; rank 1: only sends to self.
        self.counts = route_counts([[[0, 1], [2, 3], [-1, -1]], [[2, -1], [-1, -1], [-1, -1]]], 2)

    def ledger(self, rank=0, dtype="bf16", op="dispatch"):
        return byte_ledger(self.counts, rank, tokens=3, hidden=128, topk=2, dtype=dtype, idx_bytes=8, op=op)

    def test_dedup_sparse_metadata_and_self_exclusion(self):
        self.assertEqual(self.counts["rows"], [[1, 1], [0, 1]])
        value = self.ledger()
        self.assertEqual(value["remote_x_bytes"], 256)
        self.assertEqual(value["remote_src_idx_bytes"], 4)
        self.assertEqual(value["remote_topk_idx_bytes"], 16)
        self.assertEqual(value["remote_weight_bytes"], 8)
        self.assertEqual(value["remote_so_bytes"], 8)
        self.assertEqual(value["remote_application_bytes"], 292)
        self.assertEqual(value["algorithm_bytes"], 1536)
        self.assertEqual(value["valid_algorithm_bytes"], 1024)
        self.assertIsNone(value["wire_bytes"])

    def test_fp8_scales_and_cached_omissions(self):
        value = self.ledger(dtype="fp8", op="cached-dispatch")
        self.assertEqual(value["remote_payload_bytes"], 132)
        self.assertEqual(value["remote_application_bytes"], 140)
        for key in ("remote_src_idx_bytes", "remote_topk_idx_bytes", "remote_weight_bytes"):
            self.assertEqual(value[key], 0)

    def test_official_numerator_uses_receive_rows_including_self(self):
        # Rank 0 receives one self row, though it sends two rows in total.
        value = self.ledger()
        self.assertEqual(value["official_logical_rows"], 1)
        self.assertEqual(value["official_logical_bytes_per_row"], 128 * 2 + 2 * (8 + 4))
        self.assertEqual(value["official_logical_bytes"], 280)
        # Rank 1 receives two rows: one remote and one self.
        value = self.ledger(rank=1)
        self.assertEqual(value["official_logical_bytes"], 560)
        self.assertEqual(value["remote_payload_bytes"], 0)

    def test_official_cached_counts_logical_metadata_and_combine_payload_only(self):
        fresh = self.ledger(rank=1, dtype="fp8")
        cached = self.ledger(rank=1, dtype="fp8", op="cached-dispatch")
        self.assertEqual(fresh["official_logical_bytes"], 2 * (128 + 4 + 2 * 12))
        self.assertEqual(cached["official_logical_bytes"], fresh["official_logical_bytes"])
        for op in ("combine_unweighted", "combine_weighted"):
            value = self.ledger(rank=1, dtype="fp8", op=op)
            self.assertEqual(value["official_logical_bytes"], 2 * 128 * 2)
            self.assertEqual(value["remote_x_bytes"], 128 * 2)

    def test_combine_transposes_direction_and_remains_bf16(self):
        value = self.ledger(rank=1, dtype="fp8", op="combine_weighted")
        self.assertEqual(value["remote_tx_rows"], 1)
        self.assertEqual(value["remote_rx_rows"], 0)
        self.assertEqual(value["remote_x_bytes"], 256)
        self.assertEqual(value["remote_scale_bytes"], 0)
        self.assertEqual(value["remote_weight_bytes"], 8)
        self.assertEqual(self.ledger(op="combine_unweighted")["remote_payload_bytes"], 0)

    def test_invalid_routes_and_unknown_ops_rejected(self):
        with self.assertRaises(ValueError):
            route_counts([[[4]]], 4)
        with self.assertRaises(ValueError):
            self.ledger(op="other")

    def test_resource_preflight_fails_closed(self):
        text = "| 0 | Ascend950DT | OK | 380 |\n| 1 | Ascend950DT | Warning | 380 |\n"
        text += "| No running processes found in NPU 0 |\n| No running processes found in NPU 1 |\n"
        check_idle(text, 0, 2)
        for invalid in (text.replace("Warning", "Alarm"), text.replace("in NPU 1", "in NPU 2"), ""):
            with self.assertRaises(RuntimeError):
                check_idle(invalid, 0, 2)

    def make_profile(self, root, sequence, repeat=2, device=0):
        (root / "trace_view.json").write_text("{}")
        (root / "profiler_info_1.json").write_text("{}")
        with (root / "kernel_details.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f, fieldnames=["Name", "Start Time(us)", "Duration(us)", "Device_id", "Step Id", "Block Num"]
            )
            writer.writeheader()
            for i in range(repeat):
                for j, name in enumerate(sequence):
                    writer.writerow(
                        dict(
                            Name=name,
                            **{
                                "Start Time(us)": str(i * 100 + j * 10),
                                "Duration(us)": "2.125",
                                "Device_id": str(device),
                                "Step Id": "1",
                                "Block Num": "32",
                            },
                        )
                    )

    def test_profile_preserves_repeated_auxiliary_kernels_and_gaps(self):
        sequence = [
            "zero",
            "aggregate_kernel",
            "zero",
            "cached_notify_combine_kernel",
            "v1_combine_udma_comm_kernel",
            "v1_combine_udma_reduce_kernel",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_profile(root, sequence)
            result = parse_profile(root, "combine_unweighted", True, 2, 0, 32)
            sample = result["samples"][0]
            self.assertEqual(sample["other_us"], "4.250")
            self.assertEqual(sample["device_sum_us"], "12.750")
            self.assertEqual(sample["device_span_us"], "52.125")
            self.assertEqual(sample["combine_stage_us"], "4.250")
            self.assertEqual(result["absent_stages"], ["layout", "expand"])
            with self.assertRaises(ValueError):
                parse_profile(root, "combine_unweighted", True, 3, 0, 32)
            with self.assertRaises(ValueError):
                parse_profile(root, "combine_unweighted", True, 2, 0, 64)
            with self.assertRaises(ValueError):
                parse_profile(root, "combine_unweighted", False, 2, 0, 32)

    def test_missing_reduce_is_error_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_profile(root, ["cached_notify_combine_kernel", "v1_combine_udma_comm_kernel"])
            with self.assertRaises(ValueError):
                parse_profile(root, "combine_unweighted", False, 2, 0, 32)

    def test_out_of_order_stage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_profile(
                root, ["cached_notify_combine_kernel", "v1_combine_udma_reduce_kernel", "v1_combine_udma_comm_kernel"]
            )
            with self.assertRaises(ValueError):
                parse_profile(root, "combine_unweighted", False, 2, 0, 32)

    def test_global_throughput_uses_total_bytes_and_per_sample_max(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_profile(root, ["cached_notify_dispatch_kernel", "v1_dispatch_udma_kernel"])
            profile = parse_profile(root, "cached-dispatch", False, 2, 0, 32)
            records = []
            for rank in (0, 1):
                records.append(
                    dict(
                        rank=rank,
                        operators={
                            "cached-dispatch": dict(
                                profile=profile,
                                caller_us=[10 + rank, 20 + rank],
                                bytes=self.ledger(rank=rank, op="cached-dispatch"),
                            )
                        },
                    )
                )
            summary = summarize(records)
            self.assertEqual(summary[-1]["caller_us"], 16)
            self.assertEqual(summary[-1]["remote_payload_bytes"], 256)
            self.assertAlmostEqual(summary[-1]["remote_payload_GBps"], 256 / 2.125 / 1000)


class MultinodeTest(unittest.TestCase):
    def args(self, node=0):
        return arguments(
            [
                "--ep-size",
                "16",
                "--nnodes",
                "2",
                "--node-rank",
                str(node),
                "--master-addr",
                "192.0.2.10",
                "--run-id",
                "ep16_cpu_test",
            ]
        )

    def test_ep8_command_unchanged_and_ep16_rank_mapping(self):
        case = dict(name="bf16_expand0_aiv32", dtype="bf16", expand=0, aiv=32)
        single = worker_command(arguments([]), case, Path("out"))
        self.assertIn("--standalone", single)
        self.assertIn("--nproc-per-node=8", single)
        for node in (0, 1):
            args = self.args(node)
            command = worker_command(args, case, Path("out"))
            self.assertEqual(local_world_size(args), 8)
            self.assertIn("--nnodes=2", command)
            self.assertIn("--nproc-per-node=8", command)
            self.assertIn(f"--node-rank={node}", command)
            self.assertIn("--master-port=29501", command)
            self.assertNotIn("--standalone", command)
            parsed = arguments(command[command.index("--worker") :])
            self.assertEqual(parsed.node_rank, node)
            self.assertEqual(parsed.ep_size, 16)
        self.assertEqual(suite_contract(self.args(0)), suite_contract(self.args(1)))

    def test_invalid_topologies_and_endpoints_rejected(self):
        cases = [
            ["--ep-size", "16"],
            ["--nnodes", "2"],
            ["--node-rank", "1"],
            ["--ep-size", "16", "--nnodes", "2"],
            ["--ep-size", "16", "--nnodes", "2", "--master-addr", "127.0.0.1", "--run-id", "x"],
            ["--ep-size", "16", "--nnodes", "2", "--master-addr", "192.0.2.10", "--run-id", "../x"],
            ["--master-port", "65535"],
            ["--collection-timeout", "0"],
        ]
        for argv in cases:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                arguments(argv)

    def test_shared_contract_rejects_mixed_runs_shapes_and_sources(self):
        hello = dict(node_rank=0, contract=suite_contract(self.args()), source={"sha": "a"}, topk_idx_bytes=8)
        other = copy.deepcopy(hello)
        other["node_rank"] = 1
        validate_peers(hello, other)
        for key, value in [("run_id", "other_run"), ("num_tokens", 1), ("dtypes", ["bf16"])]:
            bad = copy.deepcopy(other)
            bad["contract"][key] = value
            with self.assertRaises(ValueError):
                validate_peers(hello, bad)
        for key, value in [("source", {"sha": "b"}), ("node_rank", 0), ("topk_idx_bytes", 4)]:
            bad = dict(other, **{key: value})
            with self.assertRaises(ValueError):
                validate_peers(hello, bad)

    def pair(self):
        left, right = socket.socketpair()
        peers = Peer(left, 0, timeout=2), Peer(right, 1, timeout=2)
        self.addCleanup(peers[0].close)
        self.addCleanup(peers[1].close)
        return peers

    def parallel(self, functions):
        values, errors = {}, {}

        def run(index, function):
            try:
                values[index] = function()
            except Exception as exc:
                errors[index] = exc

        threads = [threading.Thread(target=run, args=(i, f), daemon=True) for i, f in enumerate(functions)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive(), "CPU protocol test hung")
        return values, errors

    def test_busy_or_failed_preflight_reaches_both_nodes(self):
        peers = self.pair()
        _, errors = self.parallel(
            [lambda: peers[0].require_both("preflight"), lambda: peers[1].require_both("preflight", "NPU 3 busy")]
        )
        self.assertEqual(set(errors), {0, 1})
        self.assertTrue(all("NPU 3 busy" in str(e) for e in errors.values()))

    def test_disconnect_and_stage_mismatch_stop_protocol(self):
        peers = self.pair()
        _, errors = self.parallel([lambda: peers[0].exchange("caseA", {}), lambda: peers[1].exchange("caseB", {})])
        self.assertEqual(set(errors), {0, 1})
        peers[1].close()
        with self.assertRaises((ConnectionError, OSError)):
            peers[0].receive_json()

    def test_file_transfer_is_streamed_and_hash_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "source", Path(tmp) / "collected"
            source.mkdir()
            (source / "raw.bin").write_bytes(bytes(range(256)) * 8200)
            (source / "empty.log").write_bytes(b"")
            manifest = [
                dict(relative=p.name, **{k: file_identity(p)[k] for k in ("size", "sha256")})
                for p in sorted(source.iterdir())
            ]
            peers = self.pair()
            _, errors = self.parallel(
                [
                    lambda: peers[0].transfer(source, target, None, 5),
                    lambda: peers[1].transfer(source, target, manifest, 5),
                ]
            )
            self.assertEqual(errors, {})
            for item in manifest:
                self.assertEqual(file_identity(target / item["relative"])["sha256"], item["sha256"])
            self.assertTrue(json.loads((target / "collection.json").read_text())["passed"])

    def test_transfer_corruption_and_unsafe_paths_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("../escape", "/abs", "a/../b", "a\\b", "a:b", "a//b"):
                with self.assertRaises(ValueError):
                    safe_path(root, name)
            source, target = root / "src", root / "dst"
            source.mkdir()
            (source / "file").write_bytes(b"abc")
            peers = self.pair()
            manifest = [dict(relative="file", size=3, sha256="0" * 64)]
            _, errors = self.parallel(
                [
                    lambda: peers[0].transfer(source, target, None, 5),
                    lambda: peers[1].transfer(source, target, manifest, 5),
                ]
            )
            self.assertEqual(set(errors), {0, 1})
            self.assertFalse((target / "file").exists())
            self.assertFalse((target / "collection.json").exists())

    def test_library_identity_compares_bytes_not_install_path(self):
        base = dict(
            torch="2.8",
            torch_npu="2.8",
            python_module=dict(size=2, sha256="a"),
            libraries=[dict(path="/one/libshmem.so", size=4, sha256="b")],
        )
        other = copy.deepcopy(base)
        other["libraries"][0]["path"] = "/different/libshmem.so"
        self.assertEqual(library_identity(base), library_identity(other))
        other["libraries"][0]["sha256"] = "c"
        self.assertNotEqual(library_identity(base), library_identity(other))

    def test_peer_failure_or_local_timeout_terminates_owned_child(self):
        case = dict(name="bf16_expand0_aiv32", dtype="bf16", expand=0, aiv=32)
        for failure in ("peer", "timeout", "disconnect"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                args = self.args()
                args.case_timeout = 1
                child = mock.Mock(pid=123456, returncode=-15)
                child.poll.return_value = None
                peer = mock.Mock()
                peer.exchange.return_value = dict(code=1 if failure == "peer" else None, timed_out=False)
                if failure == "disconnect":
                    peer.exchange.side_effect = ConnectionError("lost peer")
                clock = [0, 2] if failure == "timeout" else [0, 0.1]
                with (
                    mock.patch("bench_v2_bandwidth.subprocess.Popen", return_value=child),
                    mock.patch("bench_v2_bandwidth.stop_child") as stop,
                    mock.patch("bench_v2_bandwidth.time.monotonic", side_effect=clock),
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaises((RuntimeError, ConnectionError)),
                ):
                    run_case(args, case, Path(tmp), {}, peer)
                stop.assert_called_once_with(child)
                status = json.loads((Path(tmp) / "launcher_exit.json").read_text())
                self.assertTrue(status["suite_aborted"])

    def test_two_node_report_replays_all_16_raw_rank_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "node0"
            remote = Path(tmp) / "different_install" / "node1"
            case = dict(name="bf16_expand0_aiv32", dtype="bf16", expand=0, aiv=32)
            args = self.args()
            args.num_tokens, args.hidden, args.num_topk, args.iterations = 1, 128, 1, 2
            plan_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            plan = dict(format_version=2, arguments=plan_args, cases=[case], topk_idx_bytes=8)
            write_json(root / "plan.json", plan)
            write_json(root / "peer_plan.json", dict(arguments=dict(plan_args, node_rank=1)))
            routes = [[[(rank + 1) % 16 * 32]] for rank in range(16)]
            counts = route_counts(routes, 32)
            identity = dict(
                torch="2.8",
                torch_npu="2.8",
                python_module=dict(size=2, sha256="module"),
                libraries=[dict(path="/lib/libshmem.so", size=1, sha256="same")],
            )
            ops = ("dispatch", "cached-dispatch", "combine_unweighted", "combine_weighted")
            for node_path in (root, remote):
                path = node_path / case["name"]
                for name, data in [
                    ("launcher_exit.json", {"code": 0}),
                    ("command.json", {}),
                    ("resources.before.txt", {}),
                ]:
                    write_json(path / name, data)
                (path / "run.log").write_text("complete")
            for rank in range(16):
                path = (root if rank < 8 else remote) / case["name"]
                rank_dir = path / f"rank{rank}"
                write_json(rank_dir / "input_routes.json", routes[rank])
                for name in ("exit.json", "precision.json", "process.json"):
                    write_json(rank_dir / name, dict(code=0, passed=True))
                config = dict(plan_args, num_aiv=32, dtype="bf16", expand=[0], node_rank=rank // 8, output=str(path))
                operators = {}
                for op in ops:
                    profile_path = rank_dir / op / "profiler"
                    profile_path.mkdir(parents=True)
                    sequence = (
                        ["dispatch_layout_kernel", "notify_dispatch_kernel", "v1_dispatch_udma_kernel"]
                        if op == "dispatch"
                        else ["cached_notify_dispatch_kernel", "v1_dispatch_udma_kernel"]
                        if op == "cached-dispatch"
                        else [
                            "cached_notify_combine_kernel",
                            "v1_combine_udma_comm_kernel",
                            "v1_combine_udma_reduce_kernel",
                        ]
                    )
                    StageMetricsTest().make_profile(profile_path, sequence, device=rank % 8)
                    operators[op] = dict(
                        profile=parse_profile(profile_path, op, False, 2, rank % 8, 32),
                        caller_us=[10, 12],
                        bytes=byte_ledger(counts, rank, tokens=1, hidden=128, topk=1, dtype="bf16", idx_bytes=8, op=op),
                    )
                    write_json(rank_dir / op / "measurement.json", operators[op])
                write_json(
                    rank_dir / "result.json",
                    dict(
                        rank=rank,
                        device=rank % 8,
                        config=config,
                        precision_passed=True,
                        destroyed=True,
                        route_counts=counts,
                        identity=identity,
                        input_routes=file_identity(rank_dir / "input_routes.json"),
                        operators=operators,
                    ),
                )
            source = remote / case["name"]
            manifest = collection_manifest(source, range(8, 16))
            target = root / case["name"] / "peer_node1"
            for item in manifest:
                destination = target / item["relative"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / item["relative"], destination)
            write_json(target / "collection.json", dict(passed=True, files=manifest))
            with contextlib.redirect_stdout(io.StringIO()):
                report(root)
            result = json.loads((root / "report.json").read_text())
            self.assertTrue(result["passed"])
            self.assertEqual(len(result["summary"]), 17 * 4)
            row = next(r for r in result["summary"] if r["rank"] == "all" and r["op"] == "dispatch")
            self.assertEqual(row["remote_payload_bytes"], 16 * 256)
            self.assertAlmostEqual(row["remote_payload_GBps"], 16 * 256 / 2.125 / 1000)
            self.assertEqual(row["official_logical_bytes"], 16 * (256 + 12))
            self.assertAlmostEqual(row["official_logical_GBps"], 16 * (256 + 12) / 2.125 / 1000)
            self.assertEqual((root / "summary.csv").read_bytes()[:3], b"\xef\xbb\xbf")
            (target / "rank15" / "exit.json").unlink()
            with self.assertRaises(FileNotFoundError):
                report(root)


if __name__ == "__main__":
    unittest.main()
